import os
import argparse
import torch
from torch.utils.data import DataLoader
from attention_unet import AttentionUNet
from dataset_istd import ISTDDataset
from losses import UncertaintyWeightedLoss
import time

def save_checkpoint(state, path):
    torch.save(state, path)

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    train_dataset = ISTDDataset(
        args.train_shadow, args.train_gt,
        size=(args.img_size, args.img_size),
        augment=True
    )
    val_dataset = ISTDDataset(
        args.val_shadow, args.val_gt,
        size=(args.img_size, args.img_size),
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = AttentionUNet(in_channels=3, out_channels=3).to(device)

    # Loss (heuristic loss tuning)
    criterion = UncertaintyWeightedLoss(use_ssim=args.use_ssim, device=device).to(device)

    # Optimizer for both model + loss params
    params = list(model.parameters()) + [p for p in criterion.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for i, (shadow, gt, _) in enumerate(train_loader):
            shadow = shadow.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            out = model(shadow)
            loss, info = criterion(out, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % args.print_freq == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Iter {i+1}/{len(train_loader)} Loss {running_loss/(i+1):.4f} L1 {info.get('l1',0):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for shadow, gt, _ in val_loader:
                shadow = shadow.to(device)
                gt = gt.to(device)
                out = model(shadow)
                loss, _ = criterion(out, gt)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch} train_loss {running_loss/len(train_loader):.4f} val_loss {val_loss:.4f} time {(time.time()-t0):.1f}s")
        scheduler.step(val_loss)

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"attunet_epoch{epoch}.pth")
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion_state': {k: v.detach().cpu() for k, v in criterion.state_dict().items()}
        }, ckpt_path)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'criterion_state': {k: v.detach().cpu() for k, v in criterion.state_dict().items()}
            }, os.path.join(args.checkpoint_dir, "best_attunet.pth"))
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_shadow", type=str, required=True, help="Path to ISTD train_A (shadow images)")
    parser.add_argument("--train_gt", type=str, required=True, help="Path to ISTD train_C (shadow-free)")
    parser.add_argument("--val_shadow", type=str, required=True, help="Path to ISTD test_A")
    parser.add_argument("--val_gt", type=str, required=True, help="Path to ISTD test_C")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_attunet")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--use_ssim", action="store_true", help="Use SSIM + L1 loss with heuristic tuning")
    parser.add_argument("--print_freq", type=int, default=50)
    args = parser.parse_args()
    train(args)