import cv2
import numpy as np


def postprocess_mask(prob_map, threshold, min_blob_frac=0.005,
                     close_kernel=5, open_kernel=3):
    """
    Convert a probability map to a clean binary mask.

    Steps:
      1) threshold prob_map -> binary
      2) morphological CLOSE to fill small holes (kernel = close_kernel)
      3) morphological OPEN to remove speckle noise (kernel = open_kernel)
      4) drop connected components smaller than min_blob_frac of total pixels

    Args:
        prob_map: 2D float array in [0, 1]
        threshold: float in (0, 1)
        min_blob_frac: components below this fraction of image area are removed
        close_kernel: size of closing kernel (odd int)
        open_kernel: size of opening kernel (odd int)

    Returns:
        cleaned binary mask uint8 in {0, 1}, same shape as prob_map
    """
    binary = (prob_map > threshold).astype(np.uint8)

    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (close_kernel, close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    if open_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    if min_blob_frac > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        min_pixels = max(1, int(min_blob_frac * binary.size))
        cleaned = np.zeros_like(binary)
        for lbl in range(1, n_labels):  # skip background (0)
            if stats[lbl, cv2.CC_STAT_AREA] >= min_pixels:
                cleaned[labels == lbl] = 1
        binary = cleaned

    return binary


def tta_predict(seg_model, pil_img, preprocess, device, threshold=None):
    """
    Test-time augmentation: average probabilities of the image and its
    horizontal flip. Returns a probability map at the original size.

    Args:
        seg_model: PyramidalNet (logits output, no sigmoid)
        pil_img: PIL.Image (RGB)
        preprocess: torchvision Compose (Resize + ToTensor)
        device: 'cuda' or 'cpu'
        threshold: if provided, returns (prob_map, binary_mask)

    Returns:
        prob_map at original size; or (prob_map, binary) if threshold given.
    """
    import torch
    from PIL import Image

    orig_w, orig_h = pil_img.size

    def _run(pil):
        inp = preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = seg_model(inp)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            if prob.ndim == 3:
                prob = prob[0]
        return prob

    p1 = _run(pil_img)
    p2_flipped = _run(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
    p2 = np.fliplr(p2_flipped)

    prob = 0.5 * (p1 + p2)
    prob = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    if threshold is not None:
        binary = (prob > threshold).astype(np.uint8)
        return prob, binary
    return prob
