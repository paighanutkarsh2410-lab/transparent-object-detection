import os
import torch
from PIL import Image
import torchvision.transforms as T
from attention_unet import AttentionUNet
from tqdm import tqdm  # progress bar

# -------- CONFIG --------
MODEL_PATH = "checkpoints_attunet/best_attunet.pth"  # trained model path
IMG_SIZE = 256  # must match training size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # process multiple images at once

# Paths for input & output
folders = {
    "train": {
        "input":"E:/TOD/Dataset/train/images",
        "output":"E:/TOD/Dataset_shadow_free/train/images"
    },
    "val_easy": {
        "input": "E:/TOD/Dataset/validation/easy/images",
        "output": "E:/TOD/Dataset_shadow_free/validation/images"
    },
    "val_hard": {
        "input": "E:/TOD/Dataset/validation/hard/images",
        "output": "E:/TOD/Dataset_shadow_free/validation_hard/images"
    }
}

# Create output folders
for key in folders:
    os.makedirs(folders[key]["output"], exist_ok=True)

# Load model
print(f"[load] Attention U-Net model on {DEVICE}...")
model = AttentionUNet(in_channels=3, out_channels=3).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Transforms
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])
to_pil = T.ToPILImage()

# Inference with batching
with torch.no_grad():
    for key, paths in folders.items():
        input_folder = paths["input"]
        output_folder = paths["output"]

        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[run]  Processing: {input_folder} ({len(files)} images)")

        for i in tqdm(range(0, len(files), BATCH_SIZE), desc=f"{key} split"):
            batch_files = files[i:i+BATCH_SIZE]
            batch_imgs = []
            original_sizes = []

            for filename in batch_files:
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path).convert("RGB")
                original_sizes.append(img.size)
                batch_imgs.append(transform(img))

            batch_tensor = torch.stack(batch_imgs).to(DEVICE)

            # Predict shadow-free images
            batch_out = model(batch_tensor).cpu().clamp(0, 1)

            # Save outputs
            for filename, out_img, orig_size in zip(batch_files, batch_out, original_sizes):
                out_pil = to_pil(out_img).resize(orig_size)
                out_pil.save(os.path.join(output_folder, filename))

        print(f"[done] {output_folder}")

print("[OK] Shadow-free training and validation images created successfully!")
