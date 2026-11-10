"""PyTorch dataset and transforms for knee X-rays."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .data import kl_to_task

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomGamma(torch.nn.Module):
    """Gamma in [lo, hi] with probability p: mimics a different detector or post-processing curve."""

    def __init__(self, lo: float = 0.7, hi: float = 1.4, p: float = 0.5):
        super().__init__()
        self.lo, self.hi, self.p = lo, hi, p

    def forward(self, img):
        if random.random() < self.p:
            return T.functional.adjust_gamma(img, random.uniform(self.lo, self.hi))
        return img


class RandomBands(torch.nn.Module):
    """With probability p, blank the top and bottom of the image so that only a horizontal band of
    height in [min_frac, 1] remains: mimics wide, short crops padded to a square."""

    def __init__(self, min_frac: float = 0.55, p: float = 0.25):
        super().__init__()
        self.min_frac, self.p = min_frac, p

    def forward(self, img):
        if random.random() >= self.p:
            return img
        h = img.shape[-2]
        keep = int(h * random.uniform(self.min_frac, 1.0))
        top = random.randint(0, h - keep)
        out = img.clone()
        out[..., :top, :] = 0
        out[..., top + keep:, :] = 0
        return out


def build_transforms(size: int = 224, train: bool = False, aug: str = "base") -> T.Compose:
    """Grayscale X-ray -> 3-channel normalized tensor.

    Augmentation is anatomically plausible: horizontal flip (left/right knee look like
    mirror images), small rotation and scale, brightness/contrast. No vertical flip.

    ``aug="robust"`` widens it for other sites and scanners: zoom out to 0.6x (the knee smaller in the
    frame, with black borders), zoom in to 1.15x, blank top/bottom bands, gamma, stronger
    brightness/contrast, sharpening and blur.
    """
    if aug not in ("base", "robust"):
        raise ValueError(aug)
    ops = [T.ToImage(), T.Grayscale(num_output_channels=3), T.Resize((size, size), antialias=True)]
    if train and aug == "base":
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    elif train:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, scale=(0.6, 1.15), translate=(0.1, 0.1), fill=0),
            RandomBands(0.55, p=0.25),
            T.ColorJitter(brightness=0.25, contrast=0.3),
            RandomGamma(0.7, 1.4, p=0.5),
            T.RandomAdjustSharpness(2.0, p=0.3),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
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
