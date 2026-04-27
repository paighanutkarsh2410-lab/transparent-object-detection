import os
import math
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image, ImageFile
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ================================
# Dataset: derives mask filename from image filename, applies joint
# geometric augmentations (image AND mask) and image-only color augs.
# ================================
class Trans10KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=224, augment=False,
                 crop_scale=(0.7, 1.0)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.crop_scale = crop_scale

        all_imgs = sorted([f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.samples = []
        skipped = 0
        for f in all_imgs:
            stem = os.path.splitext(f)[0]
            mask_name = stem + "_mask.png"
            if os.path.exists(os.path.join(mask_dir, mask_name)):
                self.samples.append((f, mask_name))
            else:
                skipped += 1
        print(f"[{image_dir}] paired {len(self.samples)} images with masks "
              f"(skipped {skipped} unmatched)")

    def __len__(self):
        return len(self.samples)

    def _random_resized_crop_params(self, w, h):
        """Pick a random crop window then resize to img_size."""
        scale = random.uniform(*self.crop_scale)
        ratio = random.uniform(0.85, 1.15)
        target_area = w * h * scale
        cw = int(round(math.sqrt(target_area * ratio)))
        ch = int(round(math.sqrt(target_area / ratio)))
        cw = max(1, min(cw, w))
        ch = max(1, min(ch, h))
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        return y, x, ch, cw  # top, left, height, width

    def __getitem__(self, idx):
        img_name, mask_name = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")

        if self.augment:
            # RandomResizedCrop applied identically to image and mask
            w, h = img.size
            top, left, ch, cw = self._random_resized_crop_params(w, h)
            img = TF.resized_crop(img, top, left, ch, cw,
                                  [self.img_size, self.img_size],
                                  interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, top, left, ch, cw,
                                   [self.img_size, self.img_size],
                                   interpolation=T.InterpolationMode.NEAREST)
            if random.random() < 0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
            # Image-only color augs (stronger ranges)
            if random.random() < 0.7:
                img = TF.adjust_brightness(img, 1 + random.uniform(-0.25, 0.25))
                img = TF.adjust_contrast(img, 1 + random.uniform(-0.25, 0.25))
                img = TF.adjust_saturation(img, 1 + random.uniform(-0.25, 0.25))
        else:
            img = TF.resize(img, [self.img_size, self.img_size])
            mask = TF.resize(mask, [self.img_size, self.img_size],
                             interpolation=T.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return img, mask


# ================================
# Model (signature unchanged so other scripts importing PyramidalNet still work)
# ================================
class PyramidalNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PyramidalNet, self).__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
        )
        feature_info = self.backbone.feature_info.channels()
        out_channels = feature_info[-1]

        self.conv1x1 = nn.Conv2d(out_channels, 512, kernel_size=1)
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upsample_final = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.backbone(x)[-1]
        features = features.permute(0, 3, 1, 2)
        x = self.conv1x1(features); x = self.relu(x)
        x = self.upsample1(x); x = self.relu(x)
        x = self.upsample2(x); x = self.relu(x)
        x = self.upsample3(x); x = self.relu(x)
        x = self.upsample4(x); x = self.relu(x)
        x = self.upsample_final(x)
        return x


# ================================
# Loss: BCEWithLogits + Dice
# ================================
class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        eps = 1e-6
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1 - (2 * intersection + eps) / (union + eps)
        return self.bce_weight * bce + self.dice_weight * dice.mean()


# ================================
# Cosine LR with linear warmup
# ================================
class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            scale = self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress))
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base * scale

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ================================
# Train + Val
# ================================
def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, accumulation_steps):
    model.train()
    total_loss, total_iou, n = 0.0, 0.0, 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="train", leave=False)
    for i, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu"):
            logits = model(imgs)
            loss = criterion(logits, masks) / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        total_loss += loss.item() * accumulation_steps * imgs.size(0)
        total_iou += iou_score(logits.detach(), masks) * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(loss=loss.item() * accumulation_steps,
                         iou=total_iou / max(n, 1),
                         lr=scheduler.get_lr())
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    for imgs, masks in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item() * imgs.size(0)
        total_iou += iou_score(logits, masks) * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n, total_iou / n


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    train_ds = Trans10KDataset(args.train_images, args.train_masks,
                               img_size=args.img_size, augment=True)

    val_datasets = [
        Trans10KDataset(args.val_images, args.val_masks,
                        img_size=args.img_size, augment=False)
    ]
    if args.val_images_hard and os.path.isdir(args.val_images_hard):
        val_datasets.append(Trans10KDataset(
            args.val_images_hard, args.val_masks_hard,
            img_size=args.img_size, augment=False))
    val_ds = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
    print(f"Combined val set size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    model = PyramidalNet(pretrained=True).to(device)
    criterion = BCEDiceLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    optimizer_steps_per_epoch = max(
        1, len(train_loader) // args.accumulation_steps)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = optimizer_steps_per_epoch * args.warmup_epochs
    scheduler = WarmupCosineLR(optimizer, warmup_steps, total_steps,
                               min_lr_ratio=0.01)
    scaler = torch.amp.GradScaler("cuda" if device == "cuda" else "cpu")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    history = {"train_loss": [], "val_loss": [],
               "train_iou": [], "val_iou": [], "lr": []}
    best_val_iou = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, args.accumulation_steps)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(val_iou)
        history["lr"].append(scheduler.get_lr())

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss {tr_loss:.4f} train_iou {tr_iou:.4f} | "
              f"val_loss {val_loss:.4f} val_iou {val_iou:.4f} | "
              f"lr {scheduler.get_lr():.2e} | "
              f"{dt:.1f}s")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> saved best (val_iou={val_iou:.4f}) to {args.save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping (no val_iou improvement for "
                      f"{args.patience} epochs).")
                break

        if device == "cuda":
            torch.cuda.empty_cache()

    plot_path = os.path.splitext(args.save_path)[0] + "_curves.png"
    _, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(history["train_iou"], label="train"); ax[1].plot(history["val_iou"], label="val")
    ax[1].set_title("IoU"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].plot(history["lr"]); ax[2].set_title("Learning rate"); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(plot_path); plt.close()
    print(f"Curves saved to {plot_path}")
    print(f"Best val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", default="E:/TOD/Dataset_shadow_free/train/images")
    parser.add_argument("--train_masks", default="E:/TOD/Dataset/train/masks")
    parser.add_argument("--val_images", default="E:/TOD/Dataset_shadow_free/validation/images")
    parser.add_argument("--val_masks", default="E:/TOD/Dataset/validation/easy/masks")
    parser.add_argument("--val_images_hard", default="E:/TOD/Dataset_shadow_free/validation_hard/images")
    parser.add_argument("--val_masks_hard", default="E:/TOD/Dataset/validation/hard/masks")
    parser.add_argument("--save_path", default="E:/TOD/trained_models/best_pyramidal_net_shadowfree_v2.pth")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    main(args)
