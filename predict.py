"""
Single-image inference: shadow removal -> segmentation -> depth -> transparency %.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --out_dir results --threshold 0.30 --no_tta

By default the script looks for model weights in ./models/ relative to this file.
Override with --att_unet and --segmentation if your weights live elsewhere.
"""
import os
import sys
import argparse
import time

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage

# Locate this script's directory so default paths work from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))

# Cache torch.hub downloads next to this script by default (overridable).
os.environ.setdefault("TORCH_HOME", os.path.join(HERE, ".torch_cache"))
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

# Make sibling modules importable when predict.py lives in repo root.
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from attention_unet import AttentionUNet  # noqa: E402
from PyramidalNet_train import PyramidalNet  # noqa: E402
from pipeline_postprocess import postprocess_mask, tta_predict  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_ATT_UNET = os.path.join(HERE, "models", "best_attunet.pth")
DEFAULT_SEG_CANDIDATES = [
    os.path.join(HERE, "models", "best_pyramidal_net_shadowfree_v2.pth"),
    os.path.join(HERE, "models", "best_pyramidal_net_shadowfree.pth"),
    # Fallbacks for the original development layout
    "E:/TOD/trained_models/best_pyramidal_net_shadowfree_v2.pth",
    "E:/TOD/trained_models/best_pyramidal_net_shadowfree.pth",
    "E:/TOD/checkpoints_attunet/best_attunet.pth",
]


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def pick_segmentation(user_path):
    if user_path:
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Segmentation checkpoint not found: {user_path}")
        return user_path
    p = find_first_existing(DEFAULT_SEG_CANDIDATES)
    if p is None:
        raise FileNotFoundError(
            "No segmentation checkpoint found. Place "
            "'best_pyramidal_net_shadowfree_v2.pth' in ./models/ next to "
            "predict.py, or pass --segmentation /path/to/weights.pth.")
    return p


def pick_att_unet(user_path):
    candidates = [user_path, DEFAULT_ATT_UNET,
                  "E:/TOD/checkpoints_attunet/best_attunet.pth"]
    p = find_first_existing(candidates)
    if p is None:
        raise FileNotFoundError(
            "No AttentionUNet checkpoint found. Place 'best_attunet.pth' in "
            "./models/ next to predict.py, or pass --att_unet /path/to/weights.pth.")
    return p


def load_shadow_remover(path):
    model = AttentionUNet(in_channels=3, out_channels=3).to(DEVICE)
    ck = torch.load(path, map_location=DEVICE)
    if isinstance(ck, dict) and 'model_state' in ck:
        ck = ck['model_state']
    model.load_state_dict(ck)
    model.eval()
    return model


def load_segmentation(path):
    model = PyramidalNet(pretrained=False).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    model.eval()
    return model


def load_depth():
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(DEVICE).eval()
    return midas


def remove_shadow(model, pil_img):
    transform = Compose([Resize((256, 256)), ToTensor()])
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x).cpu().clamp(0, 1)
    return ToPILImage()(out.squeeze()).resize(pil_img.size)


def segment(seg_model, pil_img, threshold, use_tta, post_kwargs):
    preprocess = Compose([Resize((224, 224)), ToTensor()])
    if use_tta:
        prob = tta_predict(seg_model, pil_img, preprocess, DEVICE)
    else:
        orig_w, orig_h = pil_img.size
        x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = seg_model(x)
            p = torch.sigmoid(logits).squeeze().cpu().numpy()
            if p.ndim == 3:
                p = p[0]
        prob = cv2.resize(p, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary = postprocess_mask(prob, threshold=threshold, **post_kwargs)
    return prob, binary


def estimate_depth(midas, image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (384, 384))
    transform = Compose([
        Resize(384), ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(Image.fromarray(resized)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        d = midas(x)
        d = torch.nn.functional.interpolate(
            d.unsqueeze(1), size=image_bgr.shape[:2],
            mode="bicubic", align_corners=False).squeeze().cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d


def transparency_pct(binary):
    return float(np.count_nonzero(binary)) / binary.size * 100.0


def save_visualization(orig_bgr, shadow_free_pil, binary, depth_map,
                       trans_pct, out_path, draw_depth=True):
    overlay = orig_bgr.copy()
    overlay[binary == 1] = (0, 255, 0)
    overlay = cv2.addWeighted(orig_bgr, 0.65, overlay, 0.35, 0)

    n_panels = 5 if draw_depth else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    axes[0].imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(shadow_free_pil)
    axes[1].set_title("Shadow-free"); axes[1].axis("off")
    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title("Predicted mask"); axes[2].axis("off")
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Overlay"); axes[3].axis("off")
    if draw_depth:
        axes[4].imshow(depth_map, cmap="inferno")
        axes[4].set_title("Depth map"); axes[4].axis("off")
    fig.suptitle(f"Transparency: {trans_pct:.2f}%",
                 fontsize=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def open_image_safely(path):
    """Read an image from disk, accepting RGB / RGBA / grayscale / various
    encodings. Returns (BGR ndarray, RGB PIL.Image)."""
    pil = Image.open(path)
    pil = pil.convert("RGB")  # forces RGB even for grayscale or RGBA
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, pil


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--out_dir", default=os.path.join(HERE, "predictions"),
                        help="Directory for output PNG and mask "
                             "(default: ./predictions)")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Segmentation probability threshold (default 0.30)")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation")
    parser.add_argument("--no_postprocess", action="store_true",
                        help="Disable mask cleanup (raw threshold output)")
    parser.add_argument("--min_blob_frac", type=float, default=0.005,
                        help="Drop connected components smaller than this "
                             "fraction of image area (default 0.005 = 0.5%%)")
    parser.add_argument("--no_depth", action="store_true",
                        help="Skip depth estimation (faster, no MiDaS download)")
    parser.add_argument("--att_unet", default=None,
                        help="Path to AttentionUNet checkpoint "
                             "(default: ./models/best_attunet.pth)")
    parser.add_argument("--segmentation", default=None,
                        help="Path to PyramidalNet checkpoint "
                             "(default: ./models/best_pyramidal_net_shadowfree_v2.pth)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    seg_path = pick_segmentation(args.segmentation)
    att_path = pick_att_unet(args.att_unet)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] device={DEVICE}")
    print(f"[load] shadow remover : {att_path}")
    print(f"[load] segmentation   : {seg_path}")
    shadow_model = load_shadow_remover(att_path)
    seg_model = load_segmentation(seg_path)
    midas_model = None if args.no_depth else load_depth()

    print(f"[run]  {args.image}")
    t0 = time.time()
    orig_bgr, orig_pil = open_image_safely(args.image)
    shadow_free_pil = remove_shadow(shadow_model, orig_pil)

    if args.no_postprocess:
        post_kwargs = {"min_blob_frac": 0, "close_kernel": 0, "open_kernel": 0}
    else:
        post_kwargs = {"min_blob_frac": args.min_blob_frac}

    prob, binary = segment(seg_model, shadow_free_pil,
                           threshold=args.threshold,
                           use_tta=not args.no_tta,
                           post_kwargs=post_kwargs)

    if midas_model is not None:
        sf_bgr = cv2.cvtColor(np.array(shadow_free_pil), cv2.COLOR_RGB2BGR)
        depth = estimate_depth(midas_model, sf_bgr)
    else:
        depth = np.zeros(orig_bgr.shape[:2], dtype=np.float32)

    trans = transparency_pct(binary)
    elapsed = time.time() - t0

    stem = os.path.splitext(os.path.basename(args.image))[0]
    viz_path = os.path.join(args.out_dir, f"{stem}_result.png")
    mask_path = os.path.join(args.out_dir, f"{stem}_mask.png")
    save_visualization(orig_bgr, shadow_free_pil, binary, depth, trans,
                       viz_path, draw_depth=not args.no_depth)
    cv2.imwrite(mask_path, (binary * 255).astype(np.uint8))

    print(f"[done] transparency: {trans:.2f}%  |  inference: {elapsed:.2f}s")
    print(f"       visualization -> {viz_path}")
    print(f"       binary mask   -> {mask_path}")


if __name__ == "__main__":
    main()
