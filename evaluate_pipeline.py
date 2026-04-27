"""
Evaluate the transparent-object detection pipeline on the validation set.

Runs: shadow-free image -> PyramidalNet [+ optional TTA] -> postprocess
      -> compare to ground-truth mask (Trans10K transparent class only).

Reports per-image and averaged metrics: Precision, Recall, F1, IoU,
Accuracy, MSE, MAE. Two aggregations are reported per split:
  * ALL images (rewards correct empty predictions)
  * only images that contain a transparent object

Usage:
    python evaluate_pipeline.py
    python evaluate_pipeline.py --no_tta
    python evaluate_pipeline.py --threshold 0.30 --split val_easy
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", os.path.join(HERE, ".torch_cache"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import argparse
import time
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor

from PyramidalNet_train import PyramidalNet
from pipeline_postprocess import postprocess_mask, tta_predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default split layout. Override with --val_easy_images / --val_easy_masks etc.
DEFAULT_SPLITS = {
    "val_easy": {
        "images": "E:/TOD/Dataset_shadow_free/validation/images",
        "masks":  "E:/TOD/Dataset/validation/easy/masks",
    },
    "val_hard": {
        "images": "E:/TOD/Dataset_shadow_free/validation_hard/images",
        "masks":  "E:/TOD/Dataset/validation/hard/masks",
    },
    "test_easy": {
        "images": "E:/TOD/Dataset_shadow_free/test/easy/images",
        "masks":  "E:/TOD/Dataset/test/easy/masks",
    },
    "test_hard": {
        "images": "E:/TOD/Dataset_shadow_free/test/hard/images",
        "masks":  "E:/TOD/Dataset/test/hard/masks",
    },
}

DEFAULT_SEG_CANDIDATES = [
    os.path.join(HERE, "models", "best_pyramidal_net_shadowfree_v2.pth"),
    os.path.join(HERE, "models", "best_pyramidal_net_shadowfree.pth"),
    "E:/TOD/trained_models/best_pyramidal_net_shadowfree_v2.pth",
    "E:/TOD/trained_models/best_pyramidal_net_shadowfree.pth",
]


def pick_checkpoint(user_path=None):
    if user_path:
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Checkpoint not found: {user_path}")
        return user_path
    for p in DEFAULT_SEG_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No segmentation checkpoint found. Pass --segmentation /path/to/weights.pth.")


def load_segmentation(path):
    model = PyramidalNet(pretrained=False).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def per_image_metrics(pred_bin, gt_bin):
    """
    Per-image segmentation metrics with proper empty-mask handling:
      - GT empty AND pred empty  -> perfect (P=R=F1=IoU=1)
      - GT empty AND pred has positives -> all wrong (P=R=F1=IoU=0)
      - GT non-empty -> standard formulas
    Also returns gt_has_positive so we can compute "transparent-present only" averages.
    """
    pred = pred_bin.astype(np.uint8).flatten()
    gt = gt_bin.astype(np.uint8).flatten()
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    gt_pos = int(gt.sum())
    pred_pos = int(pred.sum())

    if gt_pos == 0 and pred_pos == 0:
        precision = recall = f1 = iou = 1.0
    elif gt_pos == 0 and pred_pos > 0:
        precision = recall = f1 = iou = 0.0
    else:
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    # Cast to int32 first to avoid uint8 underflow (0 - 1 -> 255 in uint8).
    diff = pred.astype(np.int32) - gt.astype(np.int32)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return {"Precision": precision, "Recall": recall, "F1": f1,
            "IoU": iou, "Accuracy": accuracy, "MSE": mse, "MAE": mae,
            "gt_has_positive": int(gt_pos > 0)}


def build_splits(args):
    """Merge default split paths with any per-split CLI overrides."""
    splits = {k: dict(v) for k, v in DEFAULT_SPLITS.items()}
    overrides = {
        "val_easy":  (args.val_easy_images, args.val_easy_masks),
        "val_hard":  (args.val_hard_images, args.val_hard_masks),
        "test_easy": (args.test_easy_images, args.test_easy_masks),
        "test_hard": (args.test_hard_images, args.test_hard_masks),
    }
    for name, (imgs, masks) in overrides.items():
        if imgs:
            splits[name]["images"] = imgs
        if masks:
            splits[name]["masks"] = masks
    return splits


def evaluate(args):
    seg_path = pick_checkpoint(args.segmentation)
    print(f"[load] device={DEVICE}")
    print(f"[load] checkpoint: {seg_path}")
    print(f"[cfg ] threshold={args.threshold}  TTA={'off' if args.no_tta else 'on'}  "
          f"postproc={'off' if args.no_postprocess else 'on'}")

    seg_model = load_segmentation(seg_path)
    preprocess = Compose([Resize((224, 224)), ToTensor()])
    splits = build_splits(args)

    if args.split == "all":
        splits_to_run = ["val_easy", "val_hard"]
    elif args.split == "test":
        splits_to_run = ["test_easy", "test_hard"]
    else:
        splits_to_run = [args.split]

    # Drop any split whose image folder doesn't exist (e.g. shadow-free test
    # set hasn't been generated yet) instead of crashing.
    runnable = []
    for s in splits_to_run:
        if os.path.isdir(splits[s]["images"]) and os.path.isdir(splits[s]["masks"]):
            runnable.append(s)
        else:
            print(f"[skip] {s}: folder missing "
                  f"(images={splits[s]['images']}, masks={splits[s]['masks']})")
    splits_to_run = runnable
    if not splits_to_run:
        raise FileNotFoundError("No runnable splits found. Check --*_images / --*_masks paths.")

    rows = []
    for split in splits_to_run:
        cfg = splits[split]
        files = sorted([f for f in os.listdir(cfg["images"])
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        print(f"\n[run]  {split}: {len(files)} images")
        t0 = time.time()
        for fn in tqdm(files, desc=split, leave=False):
            stem = os.path.splitext(fn)[0]
            mask_path = os.path.join(cfg["masks"], stem + "_mask.png")
            if not os.path.exists(mask_path):
                continue
            pil_sf = Image.open(os.path.join(cfg["images"], fn)).convert("RGB")
            gt_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_gray is None:
                continue
            # Trans10K masks: 0 background, 76 opaque, 255 transparent.
            # The model is trained to detect ONLY transparent (>=128).
            gt_bin = (gt_gray >= 128).astype(np.uint8)

            if not args.no_tta:
                prob = tta_predict(seg_model, pil_sf, preprocess, DEVICE)
            else:
                w, h = pil_sf.size
                x = preprocess(pil_sf).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    p = torch.sigmoid(seg_model(x)).squeeze().cpu().numpy()
                    if p.ndim == 3:
                        p = p[0]
                prob = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

            if args.no_postprocess:
                pred_bin = (prob > args.threshold).astype(np.uint8)
            else:
                pred_bin = postprocess_mask(prob, threshold=args.threshold,
                                            min_blob_frac=args.min_blob_frac)

            row = {"split": split, "image": fn,
                   **per_image_metrics(pred_bin, gt_bin)}
            rows.append(row)

        print(f"[time] {split}: {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n================ AVERAGE METRICS ================")
    metric_cols = ["Precision", "Recall", "F1", "IoU", "Accuracy", "MSE", "MAE"]

    def report(sub, label):
        print(f"\n--- {label}  (n={len(sub)}) ---")
        means = sub[metric_cols].mean()
        for k in metric_cols:
            print(f"  {k:9s} {means[k]:.4f}")

    # Per split, all images
    for split in splits_to_run:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        report(sub, f"{split}: ALL images")
        sub_pos = sub[sub["gt_has_positive"] == 1]
        if len(sub_pos) > 0:
            report(sub_pos, f"{split}: only images WITH transparent objects")

    if len(splits_to_run) > 1:
        report(df, "COMBINED: ALL images")
        sub_pos = df[df["gt_has_positive"] == 1]
        if len(sub_pos) > 0:
            report(sub_pos, "COMBINED: only images WITH transparent objects")

    out_csv = "E:/TOD/evaluation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nPer-image results saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",
                        choices=["val_easy", "val_hard", "test_easy",
                                 "test_hard", "all", "test"],
                        default="all",
                        help="'all' = val_easy + val_hard, "
                             "'test' = test_easy + test_hard")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--no_postprocess", action="store_true")
    parser.add_argument("--min_blob_frac", type=float, default=0.005)
    parser.add_argument("--segmentation", default=None,
                        help="Path to PyramidalNet checkpoint")
    # Per-split path overrides
    parser.add_argument("--val_easy_images", default=None)
    parser.add_argument("--val_easy_masks",  default=None)
    parser.add_argument("--val_hard_images", default=None)
    parser.add_argument("--val_hard_masks",  default=None)
    parser.add_argument("--test_easy_images", default=None)
    parser.add_argument("--test_easy_masks",  default=None)
    parser.add_argument("--test_hard_images", default=None)
    parser.add_argument("--test_hard_masks",  default=None)
    args = parser.parse_args()
    evaluate(args)
