"""Data pipeline untuk klasifikasi rempah Indonesia.

Modul ini memuat seluruh preprocessing & augmentasi (Tahap 2) dalam satu tempat
supaya notebook training (Tahap 3) memakai pipeline yang persis sama untuk
semua arsitektur CNN yang dibandingkan.

Dipakai dari notebook:

    import data_pipeline as dp

    dp.seed_everything(42)
    loaders, datasets = dp.build_dataloaders(batch_size=32)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import v2

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = PROJECT_DIR / "data_split"
SPLITS = ("train", "val", "test")

IMG_SIZE = 224          # input standar AlexNet/VGG/ResNet/MobileNet/EfficientNet-B0
RESIZE_SIZE = 256       # resize sisi pendek sebelum center crop saat evaluasi
SEED = 42

# Statistik ImageNet: dipakai karena semua model memakai bobot pretrained,
# sehingga distribusi input harus sama dengan saat model dilatih.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Statistik dataset ini sendiri (dihitung dari train split di eda.ipynb);
# relevan kalau model dilatih from scratch.
DATASET_MEAN = (0.6405, 0.5803, 0.4929)
DATASET_STD = (0.2810, 0.2935, 0.3267)


def _to_rgb(img: Image.Image) -> Image.Image:
    """Seragamkan jumlah kanal.

    EDA menemukan 25 citra bermode RGBA/palette. ToTensor/ToImage tidak
    menyeragamkan kanal sendiri, sehingga tanpa ini batch bisa berisi tensor
    4-kanal dan collate gagal. Didefinisikan di level modul (bukan lambda)
    agar tetap picklable saat DataLoader memakai num_workers > 0.
    """
    return img.convert("RGB")


def get_normalization(source: str = "imagenet"):
    if source == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if source == "dataset":
        return DATASET_MEAN, DATASET_STD
    raise ValueError(f"source tidak dikenal: {source}")


def build_transforms(
    train: bool,
    img_size: int = IMG_SIZE,
    augment: str = "basic",
    normalize: str = "imagenet",
):
    """Susun transform untuk satu split.

    train=False selalu deterministik (resize + center crop) supaya angka
    validasi/test bisa dibandingkan antar arsitektur dan antar epoch.

    augment:
        "none"  - hanya resize/crop, untuk baseline tanpa augmentasi
        "basic" - augmentasi sederhana sesuai permintaan tugas
        "strong"- basic + TrivialAugmentWide + RandomErasing
    """
    mean, std = get_normalization(normalize)
    steps = [v2.Lambda(_to_rgb)]

    if train and augment != "none":
        steps.append(
            # Crop acak sekaligus resize. scale dibatasi 0.7-1.0: objek rempah
            # umumnya memenuhi frame, crop terlalu agresif bisa memotong objek.
            v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True)
        )
        steps.append(v2.RandomHorizontalFlip(p=0.5))
        # Rempah tidak punya orientasi baku, tapi rotasi besar menambah padding
        # hitam di sudut, jadi dibatasi kecil.
        steps.append(v2.RandomRotation(degrees=15))
        # Warna adalah fitur pembeda utama (kunyit vs jahe), jadi jitter dibuat
        # konservatif: cukup untuk meniru variasi pencahayaan, tidak sampai
        # mengubah identitas warna.
        steps.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03))

        if augment == "strong":
            steps.append(v2.TrivialAugmentWide())
    else:
        steps.append(v2.Resize(int(img_size * RESIZE_SIZE / IMG_SIZE), antialias=True))
        steps.append(v2.CenterCrop(img_size))

    steps.append(v2.ToImage())
    steps.append(v2.ToDtype(torch.float32, scale=True))
    steps.append(v2.Normalize(mean=list(mean), std=list(std)))

    if train and augment == "strong":
        steps.append(v2.RandomErasing(p=0.25, scale=(0.02, 0.15)))

    return v2.Compose(steps)


def build_datasets(
    split_dir: Path | str = SPLIT_DIR,
    img_size: int = IMG_SIZE,
    augment: str = "basic",
    normalize: str = "imagenet",
):
    """ImageFolder untuk train/val/test dengan transform yang sesuai."""
    split_dir = Path(split_dir)
    ds = {
        split: datasets.ImageFolder(
            split_dir / split,
            transform=build_transforms(
                train=(split == "train"), img_size=img_size, augment=augment, normalize=normalize
            ),
        )
        for split in SPLITS
    }
    # Mapping kelas harus identik, kalau tidak label val/test tertukar saat evaluasi
    assert ds["train"].class_to_idx == ds["val"].class_to_idx == ds["test"].class_to_idx
    return ds


def class_counts(dataset) -> np.ndarray:
    return np.bincount([label for _, label in dataset.samples], minlength=len(dataset.classes))


def class_weights(dataset, scheme: str = "inverse") -> torch.Tensor:
    """Bobot per kelas untuk loss function.

    Dataset ini imbalance ringan (1.73x), jadi "inverse" sudah memadai;
    dinormalkan agar rata-rata bobot = 1 supaya skala loss tidak berubah.
    """
    counts = class_counts(dataset).astype(np.float64)
    if scheme == "inverse":
        w = counts.sum() / (len(counts) * counts)
    elif scheme == "sqrt_inverse":
        w = np.sqrt(counts.sum() / (len(counts) * counts))
    else:
        raise ValueError(f"scheme tidak dikenal: {scheme}")
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def _make_sampler(dataset, generator) -> WeightedRandomSampler:
    counts = class_counts(dataset)
    per_sample = np.array([1.0 / counts[label] for _, label in dataset.samples])
    return WeightedRandomSampler(
        weights=torch.as_tensor(per_sample, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def build_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = IMG_SIZE,
    augment: str = "basic",
    normalize: str = "imagenet",
    balanced: bool = False,
    split_dir: Path | str = SPLIT_DIR,
    seed: int = SEED,
):
    """Bangun DataLoader untuk ketiga split.

    balanced=True memakai WeightedRandomSampler agar tiap batch seimbang kelas;
    alternatifnya pakai class_weights() pada loss. Pilih salah satu, bukan keduanya.
    """
    ds = build_datasets(split_dir, img_size=img_size, augment=augment, normalize=normalize)

    generator = torch.Generator().manual_seed(seed)
    sampler = _make_sampler(ds["train"], generator) if balanced else None

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    loaders = {
        "train": DataLoader(
            ds["train"],
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,          # batch terakhir yang kecil mengganggu BatchNorm
            generator=generator,
            **common,
        ),
        "val": DataLoader(ds["val"], shuffle=False, **common),
        "test": DataLoader(ds["test"], shuffle=False, **common),
    }
    return loaders, ds


def denormalize(x: torch.Tensor, normalize: str = "imagenet") -> torch.Tensor:
    """Kembalikan tensor ternormalisasi ke rentang [0,1] untuk divisualisasikan."""
    mean, std = get_normalization(normalize)
    mean = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
