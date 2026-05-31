"""
Trainable restoration baselines for ReMark + Lmsg experiments.

These modules intentionally expose the same forward contract as the Stage-1
VAE: input attacked image -> (restored image, mu, logvar).  The restoration
models are not VAEs, so mu/logvar are zero tensors used only to keep the
existing trainer and disabled KL loss path compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _norm(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            _norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            _norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MessageAwareRestorer(nn.Module):
    """Common helpers for image-domain restoration baselines."""

    def _empty_latent_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        mu = x.new_zeros((b, 1, 1, 1))
        logvar = x.new_zeros((b, 1, 1, 1))
        return mu, logvar

    @staticmethod
    def _residual_out(x: torch.Tensor, residual: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.clamp(x + scale * torch.tanh(residual), -1.0, 1.0)


class GenericUNetRestorer(MessageAwareRestorer):
    """BDG-style compact generic restoration UNet."""

    def __init__(self, base_channels: int = 64, depth: int = 4, residual_scale: float = 0.5):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.depth = depth
        self.residual_scale = residual_scale
        self.in_conv = ConvBlock(3, channels[0])
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1),
                    ConvBlock(channels[i + 1], channels[i + 1]),
                )
            )
        self.mid = nn.Sequential(ResBlock(channels[-1]), ResBlock(channels[-1]))
        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.ups.append(
                nn.ModuleDict({
                    "up": nn.Conv2d(channels[i + 1], channels[i], 3, padding=1),
                    "fuse": ConvBlock(channels[i] * 2, channels[i]),
                })
            )
        self.out = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = [self.in_conv(x)]
        h = feats[-1]
        for down in self.downs:
            h = down(h)
            feats.append(h)
        h = self.mid(h)
        for block, skip in zip(self.ups, reversed(feats[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block["up"](h)
            h = block["fuse"](torch.cat([h, skip], dim=1))
        x_hat = self._residual_out(x, self.out(h), self.residual_scale)
        mu, logvar = self._empty_latent_stats(x)
        return x_hat, mu, logvar


class DefusionRestorer(MessageAwareRestorer):
    """One-step visual-instructed diffusion surrogate with learned degradation token."""

    def __init__(self, base_channels: int = 64, steps: int = 4, residual_scale: float = 0.35):
        super().__init__()
        self.steps = max(int(steps), 1)
        self.residual_scale = residual_scale
        self.stem = ConvBlock(6, base_channels)
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
        )
        self.blocks = nn.ModuleList([ResBlock(base_channels) for _ in range(self.steps)])
        self.out = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        h = self.stem(torch.cat([x, x - blur], dim=1))
        for i, block in enumerate(self.blocks):
            t = x.new_full((x.shape[0], 1), float(i + 1) / float(self.steps))
            emb = self.time_embed(t).view(x.shape[0], -1, 1, 1)
            h = block(h + emb)
        x_hat = self._residual_out(x, self.out(h), self.residual_scale)
        mu, logvar = self._empty_latent_stats(x)
        return x_hat, mu, logvar


class FAPEIRRestorer(MessageAwareRestorer):
    """Frequency-aware planner/executor surrogate for FAPE-IR + Lmsg."""

    def __init__(self, base_channels: int = 48, experts: int = 2, residual_scale: float = 0.4):
        super().__init__()
        self.experts = max(int(experts), 2)
        self.residual_scale = residual_scale
        self.low_proj = ConvBlock(3, base_channels)
        self.high_proj = ConvBlock(3, base_channels)
        self.expert_blocks = nn.ModuleList([ResBlock(base_channels) for _ in range(self.experts)])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels * 2, base_channels, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, self.experts, 1),
        )
        self.out = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        high = x - low
        low_f = self.low_proj(low)
        high_f = self.high_proj(high)
        plan_logits = self.gate(torch.cat([low_f, high_f.abs()], dim=1))
        plan = torch.softmax(plan_logits, dim=1)
        expert_sum = 0.0
        seed = low_f + high_f
        for idx, expert in enumerate(self.expert_blocks):
            expert_sum = expert_sum + plan[:, idx:idx + 1] * expert(seed)
        x_hat = self._residual_out(x, self.out(expert_sum), self.residual_scale)
        mu, logvar = self._empty_latent_stats(x)
        return x_hat, mu, logvar


class RARRestorer(MessageAwareRestorer):
    """Restore-assess-repeat surrogate with recurrent refinement."""

    def __init__(self, base_channels: int = 48, iterations: int = 3, residual_scale: float = 0.25):
        super().__init__()
        self.iterations = max(int(iterations), 1)
        self.residual_scale = residual_scale
        self.encoder = ConvBlock(6, base_channels)
        self.refine = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.assessor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, base_channels // 2, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.delta = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cur = x
        for _ in range(self.iterations):
            residual_hint = cur - x
            h = self.encoder(torch.cat([cur, residual_hint], dim=1))
            h = self.refine(h)
            quality = self.assessor(h)
            step_scale = self.residual_scale * (1.0 - 0.5 * quality)
            cur = self._residual_out(cur, self.delta(h), step_scale)
        mu, logvar = self._empty_latent_stats(x)
        return cur, mu, logvar


class LatentBridgeRestorer(MessageAwareRestorer):
    """BDG-SD2-style latent bridge restorer without external SD2 weights."""

    def __init__(self, base_channels: int = 64, latent_channels: int = 8, residual_scale: float = 0.45):
        super().__init__()
        self.residual_scale = residual_scale
        self.encoder = nn.Sequential(
            ConvBlock(3, base_channels),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            ConvBlock(base_channels * 2, base_channels * 2),
            nn.Conv2d(base_channels * 2, latent_channels, 1),
        )
        self.bridge = nn.Sequential(
            ResBlock(latent_channels),
            ResBlock(latent_channels),
            ResBlock(latent_channels),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 2, 1),
            ConvBlock(base_channels * 2, base_channels * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z = self.bridge(z)
        residual = self.decoder(z)
        if residual.shape[-2:] != x.shape[-2:]:
            residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x_hat = self._residual_out(x, residual, self.residual_scale)
        mu = z.mean(dim=1, keepdim=True)
        logvar = z.new_zeros(mu.shape)
        return x_hat, mu, logvar


def build_restoration_baseline(cfg) -> nn.Module:
    model_type = str(getattr(cfg.model, "type", "vae")).lower()
    base = int(getattr(cfg.model, "base_channels", 64))
    residual_scale = float(getattr(cfg.model, "residual_scale", 0.4))

    if model_type in {"bdg_36m", "bdg-36m", "bdg"}:
        return GenericUNetRestorer(
            base_channels=base,
            depth=int(getattr(cfg.model, "depth", 4)),
            residual_scale=residual_scale,
        )
    if model_type in {"defusion", "de-fusion"}:
        return DefusionRestorer(
            base_channels=base,
            steps=int(getattr(cfg.model, "diffusion_steps", 4)),
            residual_scale=residual_scale,
        )
    if model_type in {"fape_ir", "fape-ir"}:
        return FAPEIRRestorer(
            base_channels=base,
            experts=int(getattr(cfg.model, "experts", 2)),
            residual_scale=residual_scale,
        )
    if model_type == "rar":
        return RARRestorer(
            base_channels=base,
            iterations=int(getattr(cfg.model, "iterations", 3)),
            residual_scale=residual_scale,
        )
    if model_type in {"bdg_sd2", "bdg-sd2"}:
        return LatentBridgeRestorer(
            base_channels=base,
            latent_channels=int(getattr(cfg.model, "latent_channels", 8)),
            residual_scale=residual_scale,
        )
    raise ValueError(f"Unknown restoration baseline model.type={model_type!r}")
