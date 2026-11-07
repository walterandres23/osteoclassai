"""PyTorch dataset and transforms for knee X-rays."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .data import kl_to_task

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(size: int = 224, train: bool = False) -> T.Compose:
    """Grayscale X-ray -> 3-channel normalized tensor.

    Augmentation is anatomically plausible: horizontal flip (left/right knee look like
    mirror images), small rotation and scale, brightness/contrast. No vertical flip.
    """
    ops = [T.ToImage(), T.Grayscale(num_output_channels=3), T.Resize((size, size), antialias=True)]
    if train:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class KneeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str | Path, task: str = "kl5", transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.labels = kl_to_task(self.df.kl, task).to_numpy()
        self.transform = transform or build_transforms()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.df.path[i].replace("\\", "/")).convert("L")
        return self.transform(img), int(self.labels[i])
