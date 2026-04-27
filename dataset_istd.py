import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random

class ISTDDataset(Dataset):
    def __init__(self, shadow_dir, gt_dir, size=(256, 256), augment=False):
        self.shadow_dir = shadow_dir
        self.gt_dir = gt_dir
        self.size = size
        self.augment = augment

        self.shadow_files = sorted(os.listdir(shadow_dir))
        self.gt_files = sorted(os.listdir(gt_dir))

        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor()
        ])

        self.augmentations = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])

    def __len__(self):
        return len(self.shadow_files)

    def __getitem__(self, idx):
        shadow_path = os.path.join(self.shadow_dir, self.shadow_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        shadow_img = Image.open(shadow_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.augment:
            if random.random() > 0.5:
                shadow_img = self.augmentations(shadow_img)
                gt_img = self.augmentations(gt_img)

        shadow_img = self.transform(shadow_img)
        gt_img = self.transform(gt_img)

        return shadow_img, gt_img, self.shadow_files[idx]
