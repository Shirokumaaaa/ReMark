from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import lpips  # perceptual loss
prec = lpips.LPIPS(net="alex")  # use AlexNet for perceptual loss
# ----------------------
#   Model components
# ----------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))

class Encoder(nn.Module):
    def __init__(self, z_channels: int = 4):
        super().__init__()
        ch = 64
        layers = [nn.Conv2d(3, ch, 3, 1, 1), nn.SiLU()]
        for i in range(3):  # 3× Down: 128→64→32
            layers += [ResBlock(ch, ch),
                       nn.Conv2d(ch, ch, 4, 2, 1),
                       nn.SiLU()]
        layers += [ResBlock(ch, ch), nn.Conv2d(ch, z_channels, 3, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, z_channels: int = 4):
        super().__init__()
        ch = 64
        layers = [nn.Conv2d(z_channels, ch, 3, 1, 1), nn.SiLU(), ResBlock(ch, ch)]
        for _ in range(3):  # 3× Upsample: 32→64→128
            layers += [nn.ConvTranspose2d(ch, ch, 4, 2, 1),
                       nn.SiLU(),
                       ResBlock(ch, ch)]
        layers += [nn.Conv2d(ch, 3, 3, 1, 1), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)

# ----------------------
#   Lightning Module
# ----------------------
class PlainAE(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, perceptual_weight: float = 0.1, z_channels: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(z_channels)
        self.decoder = Decoder(z_channels)
        self.perc = prec
        self.perc.eval().requires_grad_(False)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _step(self, batch) -> Tuple[torch.Tensor, dict]:
        x, _ = batch
        x = x * 2. - 1.  # scale to [-1,1]
        x_hat = self(x)
        l1 = F.l1_loss(x_hat, x)
        with torch.no_grad():
            perc = self.perc(x_hat, x).mean()
        loss = l1 + self.hparams.perceptual_weight * perc
        return loss, {"l1": l1, "lpips": perc}

    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch)
        self.log_dict({"train_loss": loss, **logs}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch)
        self.log_dict({"val_loss": loss, **{f"val_{k}": v for k, v in logs.items()}}, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_steps)
        return {"optimizer": opt, "lr_scheduler": sch}

# ----------------------
#   DataModule
# ----------------------
class ImageFolder128(pl.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([
            T.Resize(128, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(128),
            T.ToTensor(),
        ])

    def setup(self, stage: str | None = None):
        self.ds = ImageFolder(self.data_root, transform=self.transform)
        val_split = 0.02
        val_size = int(len(self.ds) * val_split)
        train_size = len(self.ds) - val_size
        self.train_set, self.val_set = torch.utils.data.random_split(self.ds, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

# ----------------------
#   Main entry
# ----------------------
if __name__ == "__main__":
    import argparse, datetime, json

    parser = argparse.ArgumentParser(description="Plain AE trainer (128×128)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--precision", type=str, default="16", choices=["16", "32"])
    args = parser.parse_args()

    dm = ImageFolder128(args.data_root, batch_size=args.batch_size)
    model = PlainAE(lr=args.lr)

    # Callbacks
    ckpt_cb = ModelCheckpoint(monitor="val_lpips", mode="min", save_top_k=3,
                              filename="ae-{epoch:03d}-{val_lpips:.4f}")
    early_cb = EarlyStopping(monitor="val_lpips", mode="min", patience=10)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        precision=args.precision,
        callbacks=[ckpt_cb, early_cb, lr_monitor],
        log_every_n_steps=50,
    )

    trainer.fit(model, dm)

    # Save final hyper‑params + best ckpt path
    meta = {"hparams": vars(args), "best_ckpt": ckpt_cb.best_model_path,
            "timestamp": datetime.datetime.now().isoformat()}
    out = Path(trainer.logger.save_dir) / "meta.json"
    out.write_text(json.dumps(meta, indent=2))
