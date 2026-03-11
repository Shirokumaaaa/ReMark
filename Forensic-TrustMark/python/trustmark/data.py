from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import lightning as pl
except ImportError:
    import pytorch_lightning as pl


_PATH_COLUMNS: Tuple[str, ...] = ("img_path", "image_path", "path", "image", "img")


def read_manifest_paths(
    manifest_csv: str | Path,
    path_columns: Sequence[str] = _PATH_COLUMNS,
    check_exists: bool = False,
) -> List[str]:
    csv_path = Path(manifest_csv).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV has no header: {csv_path}")

        selected_col: Optional[str] = None
        for col in path_columns:
            if col in reader.fieldnames:
                selected_col = col
                break

        if selected_col is None:
            raise ValueError(
                f"Manifest CSV {csv_path} must contain one of columns: {path_columns}. "
                f"Got: {reader.fieldnames}"
            )

        paths: List[str] = []
        root = csv_path.parent
        for row in reader:
            raw_path = (row.get(selected_col) or "").strip()
            if not raw_path:
                continue
            p = Path(raw_path).expanduser()
            if not p.is_absolute():
                p = root / p
            if check_exists and (not p.is_file()):
                continue
            paths.append(str(p))

    if not paths:
        raise ValueError(f"No valid image path found in manifest: {csv_path}")
    return paths


class ManifestImageSecretDataset(Dataset):
    """Dataset that reads image paths from CSV and generates watermark secrets."""

    def __init__(
        self,
        manifest_csv: str | Path,
        resolution: int = 256,
        secret_len: int = 100,
        cover_key: str = "image",
        secret_key: str = "secret",
        deterministic_secret: bool = False,
        secret_seed: int = 1234,
    ):
        super().__init__()
        self.paths = read_manifest_paths(manifest_csv)
        self.secret_len = int(secret_len)
        self.cover_key = cover_key
        self.secret_key = secret_key
        self.deterministic_secret = deterministic_secret
        self.secret_seed = int(secret_seed)

        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _make_secret(self, idx: int) -> torch.Tensor:
        if self.deterministic_secret:
            g = torch.Generator()
            g.manual_seed(self.secret_seed + idx)
            return torch.randint(0, 2, (self.secret_len,), generator=g, dtype=torch.int64).float()
        return torch.randint(0, 2, (self.secret_len,), dtype=torch.int64).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # C,H,W in [0,1]
        image = image * 2.0 - 1.0  # C,H,W in [-1,1]
        image = image.permute(1, 2, 0).contiguous()  # H,W,C (matches current model get_input)

        secret = self._make_secret(idx)
        return {
            self.cover_key: image,
            self.secret_key: secret,
            "img_path": img_path,
        }


def build_manifest_dataloaders(
    train_manifest_csv: str | Path,
    val_manifest_csv: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    resolution: int = 256,
    secret_len: int = 100,
    cover_key: str = "image",
    secret_key: str = "secret",
    pin_memory: bool = True,
    train_drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ManifestImageSecretDataset(
        manifest_csv=train_manifest_csv,
        resolution=resolution,
        secret_len=secret_len,
        cover_key=cover_key,
        secret_key=secret_key,
        deterministic_secret=False,
    )
    val_dataset = ManifestImageSecretDataset(
        manifest_csv=val_manifest_csv,
        resolution=resolution,
        secret_len=secret_len,
        cover_key=cover_key,
        secret_key=secret_key,
        deterministic_secret=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train_drop_last,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


class ManifestDataModule(pl.LightningDataModule):
    """LightningDataModule wrapper for manifest-based training/validation."""

    def __init__(
        self,
        train_manifest_csv: str | Path,
        val_manifest_csv: str | Path,
        batch_size: int = 16,
        num_workers: int = 4,
        resolution: int = 256,
        secret_len: int = 100,
        cover_key: str = "image",
        secret_key: str = "secret",
        pin_memory: bool = True,
        train_drop_last: bool = True,
    ):
        super().__init__()
        self.train_manifest_csv = str(train_manifest_csv)
        self.val_manifest_csv = str(val_manifest_csv)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.resolution = int(resolution)
        self.secret_len = int(secret_len)
        self.cover_key = cover_key
        self.secret_key = secret_key
        self.pin_memory = pin_memory
        self.train_drop_last = train_drop_last

        self._train_dataset: Optional[ManifestImageSecretDataset] = None
        self._val_dataset: Optional[ManifestImageSecretDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_dataset is None:
            self._train_dataset = ManifestImageSecretDataset(
                manifest_csv=self.train_manifest_csv,
                resolution=self.resolution,
                secret_len=self.secret_len,
                cover_key=self.cover_key,
                secret_key=self.secret_key,
                deterministic_secret=False,
            )
        if self._val_dataset is None:
            self._val_dataset = ManifestImageSecretDataset(
                manifest_csv=self.val_manifest_csv,
                resolution=self.resolution,
                secret_len=self.secret_len,
                cover_key=self.cover_key,
                secret_key=self.secret_key,
                deterministic_secret=True,
            )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.train_drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
