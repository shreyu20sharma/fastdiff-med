"""DermaMNIST loading: 28x28 RGB, zero-padded to 32x32, normalised to [-1, 1].

The whole dataset is ~21 MB, so we keep it as GPU-resident tensors and sample batches by
index (much faster on Colab than a DataLoader).

Set the environment variable FASTDIFF_FAKE_DATA=1 to use small random data instead
(for offline smoke tests / CI without downloading MedMNIST).
"""
import os

import numpy as np
import torch

SHORT_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = [
    "Actinic keratosis / intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesion",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevus",
    "Vascular lesion",
]
NUM_CLASSES = 7
IMG_SIZE = 32
PAD = (IMG_SIZE - 28) // 2
DEFAULT_ROOT = os.environ.get("MEDMNIST_ROOT", "data")


def _fake_split(split):
    rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
    n = {"train": 700, "val": 100, "test": 200}[split]
    # imbalanced like the real thing (nv dominant, df / vasc rare)
    p = np.array([228, 359, 769, 80, 779, 4693, 99], dtype=np.float64)
    labels = rng.choice(NUM_CLASSES, size=n, p=p / p.sum()).astype(np.int64)
    imgs = rng.integers(0, 256, size=(n, 28, 28, 3), dtype=np.uint8)
    return imgs, labels


def load_numpy(split, root=DEFAULT_ROOT):
    """Return (uint8 images N x 32 x 32 x 3, int64 labels N) for split in {train,val,test}."""
    if os.environ.get("FASTDIFF_FAKE_DATA") == "1":
        imgs, labels = _fake_split(split)
    else:
        from medmnist import DermaMNIST

        os.makedirs(root, exist_ok=True)
        try:
            ds = DermaMNIST(split=split, download=True, root=root, size=28)
        except TypeError:  # medmnist v2 has no `size` argument
            ds = DermaMNIST(split=split, download=True, root=root)
        imgs = np.asarray(ds.imgs)
        labels = np.asarray(ds.labels).reshape(-1).astype(np.int64)
        if imgs.ndim == 3:  # safety: grayscale -> RGB
            imgs = np.repeat(imgs[..., None], 3, axis=-1)
    imgs = np.pad(imgs, ((0, 0), (PAD, PAD), (PAD, PAD), (0, 0)), mode="constant")
    return imgs, labels


def uint8_to_tensor(imgs):
    """N x H x W x 3 uint8 -> N x 3 x H x W float in [-1, 1]."""
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
    return x / 127.5 - 1.0


def tensor_to_uint8(x):
    """N x 3 x H x W in [-1, 1] -> N x H x W x 3 uint8 numpy."""
    x = ((x.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def load_tensors(split, device="cpu", root=DEFAULT_ROOT):
    imgs, labels = load_numpy(split, root)
    return uint8_to_tensor(imgs).to(device), torch.from_numpy(labels).to(device)


def class_counts(labels, num_classes=NUM_CLASSES):
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    return np.bincount(labels, minlength=num_classes)


def random_flip(x):
    """Random horizontal and vertical flips (lesions have no canonical orientation)."""
    b = x.shape[0]
    h = torch.rand(b, device=x.device) < 0.5
    v = torch.rand(b, device=x.device) < 0.5
    x = torch.where(h[:, None, None, None], x.flip(3), x)
    x = torch.where(v[:, None, None, None], x.flip(2), x)
    return x


def stratified_subset(labels, frac, seed=0):
    """Indices of a class-stratified random subset (at least 1 image per class)."""
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        n = max(1, int(round(frac * len(idx))))
        keep.append(rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(keep))
