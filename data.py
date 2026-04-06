from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

from config import DataConfig


class SyntheticImageDataset(Dataset):
    """Generates synthetic images on the fly."""

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        channels: int,
        height: int,
        width: int,
        transform: Optional[Callable] = None,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.channels = channels
        self.height = height
        self.width = width
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        label = torch.randint(0, self.num_classes, (1,), generator=gen).item()
        image = torch.randn(self.channels, self.height, self.width, generator=gen)
        bias = (label / self.num_classes) * 0.5
        image = image + bias
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


class FolderImageDataset(Dataset):
    """Loads images from folder structure with subfolders as classes."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}

    def __init__(
        self,
        root: str,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.channels = channels
        self.height = height
        self.width = width
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []
        self._scan()

    def _scan(self) -> None:
        subdirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        if subdirs:
            self.class_names = [d.name for d in subdirs]
            for label, d in enumerate(subdirs):
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((str(f), label))
        else:
            self.class_names = ["default"]
            for f in sorted(self.root.iterdir()):
                if f.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(f), 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB" if self.channels == 3 else "L")
        img = img.resize((self.width, self.height), Image.BILINEAR)
        arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
        if self.channels == 3:
            arr = arr.view(self.height, self.width, 3).permute(2, 0, 1)
        else:
            arr = arr.view(1, self.height, self.width)
        arr = arr / 255.0
        if self.transform is not None:
            arr = self.transform(arr)
        return arr, label

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def load_single_image(
    image_bytes: bytes,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
) -> Tensor:
    """Load image from bytes to tensor."""
    import io as _io
    img = Image.open(_io.BytesIO(image_bytes)).convert("RGB" if channels == 3 else "L")
    img = img.resize((width, height), Image.BILINEAR)
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
    if channels == 3:
        arr = arr.view(height, width, 3).permute(2, 0, 1)
    else:
        arr = arr.view(1, height, width)
    arr = arr / 255.0
    return arr.unsqueeze(0)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            return img.flip(-1)
        return img


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            return img.flip(-2)
        return img


class RandomCrop:
    def __init__(self, size: int, padding: int = 4) -> None:
        self.size = size
        self.padding = padding

    def __call__(self, img: Tensor) -> Tensor:
        c, h, w = img.shape
        padded = torch.nn.functional.pad(img, [self.padding] * 4, mode="reflect")
        _, ph, pw = padded.shape
        top = torch.randint(0, ph - self.size + 1, (1,)).item()
        left = torch.randint(0, pw - self.size + 1, (1,)).item()
        return padded[:, top : top + self.size, left : left + self.size]


class RandomNoise:
    def __init__(self, std: float = 0.05) -> None:
        self.std = std

    def __call__(self, img: Tensor) -> Tensor:
        return img + torch.randn_like(img) * self.std


class Normalize:
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor) -> Tensor:
        return (img - self.mean) / (self.std + 1e-8)


class Cutout:
    def __init__(self, size: int = 8) -> None:
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        c, h, w = img.shape
        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()
        y1 = max(0, cy - self.size // 2)
        y2 = min(h, cy + self.size // 2)
        x1 = max(0, cx - self.size // 2)
        x2 = min(w, cx + self.size // 2)
        mask = img.clone()
        mask[:, y1:y2, x1:x2] = 0.0
        return mask


class Compose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for t in self.transforms:
            img = t(img)
        return img


def build_transforms(cfg: DataConfig, is_train: bool = True) -> Optional[Callable]:
    transforms: List[Callable] = []
    if is_train and cfg.augmentation:
        transforms.append(RandomHorizontalFlip(0.5))
        if cfg.augmentation_strength > 0.3:
            transforms.append(RandomVerticalFlip(0.2))
        transforms.append(RandomNoise(std=0.02 * cfg.augmentation_strength))
        if cfg.augmentation_strength > 0.5:
            transforms.append(Cutout(size=max(4, int(8 * cfg.augmentation_strength))))
    transforms.append(Normalize(mean=0.0, std=1.0))
    return Compose(transforms)


def build_dataloaders(
    data_cfg: DataConfig,
    model_cfg: "ModelConfig",
    distributed: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from config import ModelConfig

    train_transform = build_transforms(data_cfg, is_train=True)
    eval_transform = build_transforms(data_cfg, is_train=False)

    train_ds = SyntheticImageDataset(
        num_samples=data_cfg.num_train_samples,
        num_classes=model_cfg.num_classes,
        channels=model_cfg.input_channels,
        height=model_cfg.input_height,
        width=model_cfg.input_width,
        transform=train_transform,
        seed=seed,
    )
    val_ds = SyntheticImageDataset(
        num_samples=data_cfg.num_val_samples,
        num_classes=model_cfg.num_classes,
        channels=model_cfg.input_channels,
        height=model_cfg.input_height,
        width=model_cfg.input_width,
        transform=eval_transform,
        seed=seed + data_cfg.num_train_samples,
    )
    test_ds = SyntheticImageDataset(
        num_samples=data_cfg.num_test_samples,
        num_classes=model_cfg.num_classes,
        channels=model_cfg.input_channels,
        height=model_cfg.input_height,
        width=model_cfg.input_width,
        transform=eval_transform,
        seed=seed + data_cfg.num_train_samples + data_cfg.num_val_samples,
    )

    train_sampler = DistributedSampler(train_ds) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if distributed else None

    common = dict(
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=data_cfg.persistent_workers and data_cfg.num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **common)
    test_loader = DataLoader(test_ds, shuffle=False, sampler=test_sampler, **common)

    return train_loader, val_loader, test_loader


def build_folder_dataloaders(
    folder_path: str,
    model_cfg: "ModelConfig",
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    augmentation: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build loaders from image folder. Returns loaders and class names."""
    from config import ModelConfig, DataConfig

    dcfg = DataConfig(augmentation=augmentation)
    train_transform = build_transforms(dcfg, is_train=True)
    eval_transform = build_transforms(dcfg, is_train=False)

    full_ds = FolderImageDataset(
        root=folder_path,
        channels=model_cfg.input_channels,
        height=model_cfg.input_height,
        width=model_cfg.input_width,
        transform=None,
    )
    class_names = full_ds.class_names

    n = len(full_ds)
    n_test = max(1, int(n * test_split))
    n_val = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    gen = torch.Generator().manual_seed(seed)
    from torch.utils.data import random_split
    train_sub, val_sub, test_sub = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    train_ds = _TransformWrapper(train_sub, train_transform)
    val_ds = _TransformWrapper(val_sub, eval_transform)
    test_ds = _TransformWrapper(test_sub, eval_transform)

    common = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, class_names


class _TransformWrapper(Dataset):
    """Applies transform to a subset."""

    def __init__(self, subset, transform: Optional[Callable] = None) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
