"""
Stage 2 U-Net — ReMark latent 空间修复网络 G_psi

本版本在原轻量 Latent U-Net 基础上增加了三类“旧版 Denoise 对齐能力”：
1) timestep scale-shift norm（更强时序条件注入）
2) 多尺度 attention（不只 bottleneck）
3) attention + feed-forward 的 mid-depth 堆叠（提升表达容量）
"""

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_groups(channels: int, prefer_groups: int = 8) -> int:
    g = min(prefer_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


def timestep_embedding(timesteps: torch.Tensor,
                       dim: int,
                       max_period: int = 10000) -> torch.Tensor:
    """扩散式正弦时间嵌入（sin/cos）。"""
    half = dim // 2
    if half <= 0:
        raise ValueError(f'time embedding dim must be >= 2, got {dim}')
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """
    残差块，支持:
      - time embedding 直接加和
      - scale-shift norm（对齐 diffusion U-Net 常见实现）
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 temb_dim: int = 0,
                 dropout: float = 0.0,
                 use_scale_shift_norm: bool = False,
                 groups: int = 8):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.use_scale_shift_norm = bool(use_scale_shift_norm)

        self.norm1 = nn.GroupNorm(_resolve_groups(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_resolve_groups(out_channels, groups), out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.emb_proj = None
        if temb_dim > 0:
            emb_out = out_channels * 2 if self.use_scale_shift_norm else out_channels
            self.emb_proj = nn.Linear(temb_dim, emb_out)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        emb = None
        if self.emb_proj is not None and temb is not None:
            emb = self.emb_proj(F.silu(temb))

        if self.use_scale_shift_norm and emb is not None:
            scale, shift = torch.chunk(emb, 2, dim=1)
            h = self.norm2(h)
            h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1))
            h = h + shift.unsqueeze(-1).unsqueeze(-1)
            h = F.silu(h)
        else:
            h = self.norm2(h)
            if emb is not None:
                h = h + emb.unsqueeze(-1).unsqueeze(-1)
            h = F.silu(h)

        h = self.dropout(h)
        h = self.conv2(h)
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SpatialSelfAttention2d(nn.Module):
    """2D 多头自注意力。"""

    def __init__(self, channels: int, n_heads: int = 4, groups: int = 8):
        super().__init__()
        if n_heads <= 0 or channels % n_heads != 0:
            raise ValueError(
                f'channels={channels} must be divisible by n_heads={n_heads}'
            )
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nn.GroupNorm(_resolve_groups(channels, groups), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hw = h * w
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.reshape(b, self.n_heads, self.head_dim, hw).transpose(-1, -2)
        k = k.reshape(b, self.n_heads, self.head_dim, hw).transpose(-1, -2)
        v = v.reshape(b, self.n_heads, self.head_dim, hw).transpose(-1, -2)

        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        out = self.proj(out)
        return x + out


class FeedForward2d(nn.Module):
    """1x1 Conv FFN（Transformer-style post attention FFN）。"""

    def __init__(self, channels: int, mult: int = 2, groups: int = 8):
        super().__init__()
        hidden = max(channels * mult, channels)
        self.norm = nn.GroupNorm(_resolve_groups(channels, groups), channels)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.silu(self.norm(x)))
        h = self.fc2(F.silu(h))
        return x + h


class AttentionFFNBlock(nn.Module):
    """注意力 + FFN 串联块。"""

    def __init__(self, channels: int, n_heads: int = 4, ffn_mult: int = 2, groups: int = 8):
        super().__init__()
        self.attn = SpatialSelfAttention2d(channels, n_heads=n_heads, groups=groups)
        self.ffn = FeedForward2d(channels, mult=ffn_mult, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class LatentUNet(nn.Module):
    """
    Stage2 latent 修复网络：
      z_out = z + delta
    """

    def __init__(self,
                 latent_channels: int = 48,
                 base_channels: int = 64,
                 n_levels: int = 2,
                 n_res: int = 2,
                 time_cond: bool = False,
                 time_embed_dim: int = 256,
                 use_bottleneck_attn: bool = False,
                 attn_heads: int = 4,
                 use_scale_shift_norm: bool = False,
                 dropout: float = 0.0,
                 latent_size: int = 16,
                 use_level_attn: bool = False,
                 attention_resolutions: Iterable[int] = (),
                 mid_attn_depth: int = 0,
                 ffn_mult: int = 2,
                 predict_delta: bool = False):
        super().__init__()
        self.latent_channels = latent_channels
        self.n_levels = int(n_levels)
        self.time_cond = bool(time_cond)
        # 与 legacy Denoise 对齐：默认输出“下一步绝对 latent”。
        # 若 predict_delta=true，则输出位移 delta。
        self.predict_delta = bool(predict_delta)
        self.time_embed_dim = int(time_embed_dim)
        self.latent_size = int(latent_size)
        self.use_level_attn = bool(use_level_attn)
        self.attn_resolutions = {
            int(r) for r in attention_resolutions if int(r) > 0
        }

        self.mid_attn_depth = int(mid_attn_depth)
        if self.mid_attn_depth < 0:
            raise ValueError(f'mid_attn_depth must be >= 0, got {self.mid_attn_depth}')
        if self.mid_attn_depth == 0 and bool(use_bottleneck_attn):
            # 兼容旧配置：此前只有 bottleneck 开关
            self.mid_attn_depth = 1

        if self.time_cond and self.time_embed_dim < 2:
            raise ValueError(f'time_embed_dim must be >= 2, got {self.time_embed_dim}')

        if self.time_cond:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_embed_dim, self.time_embed_dim * 4),
                nn.SiLU(),
                nn.Linear(self.time_embed_dim * 4, self.time_embed_dim),
            )
            temb_dim = self.time_embed_dim
        else:
            self.time_mlp = None
            temb_dim = 0

        ch = [base_channels * (2 ** i) for i in range(self.n_levels + 1)]
        self.input_proj = nn.Conv2d(latent_channels, ch[0], 1)

        self.enc_blocks = nn.ModuleList()
        self.enc_attn = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        for i in range(self.n_levels):
            blocks = nn.ModuleList([
                ResBlock(
                    ch[i],
                    ch[i],
                    temb_dim=temb_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                for _ in range(n_res)
            ])
            self.enc_blocks.append(blocks)

            resolution = self.latent_size // (2 ** i)
            use_attn = self.use_level_attn and (resolution in self.attn_resolutions)
            self.enc_attn.append(
                AttentionFFNBlock(ch[i], n_heads=attn_heads, ffn_mult=ffn_mult)
                if use_attn else nn.Identity()
            )

            self.enc_downs.append(
                nn.Sequential(
                    Downsample(ch[i]),
                    nn.Conv2d(ch[i], ch[i + 1], 1),
                )
            )

        self.bottleneck = nn.ModuleList([
            ResBlock(
                ch[-1],
                ch[-1],
                temb_dim=temb_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            )
            for _ in range(max(1, n_res))
        ])
        self.mid_attn_blocks = nn.ModuleList([
            AttentionFFNBlock(ch[-1], n_heads=attn_heads, ffn_mult=ffn_mult)
            for _ in range(self.mid_attn_depth)
        ])

        self.dec_ups = nn.ModuleList()
        self.dec_merges = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_attn = nn.ModuleList()
        for i in reversed(range(self.n_levels)):
            self.dec_ups.append(
                nn.Sequential(
                    Upsample(ch[i + 1]),
                    nn.Conv2d(ch[i + 1], ch[i], 1),
                )
            )
            self.dec_merges.append(nn.Conv2d(ch[i] * 2, ch[i], 1))
            self.dec_blocks.append(nn.ModuleList([
                ResBlock(
                    ch[i],
                    ch[i],
                    temb_dim=temb_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                for _ in range(n_res)
            ]))

            resolution = self.latent_size // (2 ** i)
            use_attn = self.use_level_attn and (resolution in self.attn_resolutions)
            self.dec_attn.append(
                AttentionFFNBlock(ch[i], n_heads=attn_heads, ffn_mult=ffn_mult)
                if use_attn else nn.Identity()
            )

        self.output_proj = nn.Sequential(
            nn.GroupNorm(_resolve_groups(ch[0], 8), ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], latent_channels, 1),
        )

    def _build_time_emb(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        if timestep is None:
            timestep = torch.zeros(batch_size, device=self.input_proj.weight.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.reshape(-1)
        if timestep.numel() == 1 and batch_size > 1:
            timestep = timestep.repeat(batch_size)
        if timestep.numel() != batch_size:
            raise ValueError(
                f'timestep batch mismatch: got {timestep.numel()}, expect {batch_size}'
            )
        emb = timestep_embedding(timestep, self.time_embed_dim)
        return self.time_mlp(emb)

    def forward(self, z: torch.Tensor, timestep: torch.Tensor = None) -> torch.Tensor:
        temb = None
        if self.time_cond:
            temb = self._build_time_emb(timestep, z.shape[0])

        x = self.input_proj(z)

        skips = []
        for blocks, attn, down in zip(self.enc_blocks, self.enc_attn, self.enc_downs):
            for layer in blocks:
                x = layer(x, temb)
            x = attn(x)
            skips.append(x)
            x = down(x)

        for layer in self.bottleneck:
            x = layer(x, temb)
        for layer in self.mid_attn_blocks:
            x = layer(x)

        for up, merge, blocks, attn, skip in zip(
                self.dec_ups, self.dec_merges, self.dec_blocks, self.dec_attn, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = merge(x)
            for layer in blocks:
                x = layer(x, temb)
            x = attn(x)

        delta = self.output_proj(x)
        if self.predict_delta:
            return delta
        return z + delta


def _as_int_list(values) -> List[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return [int(values)]


def build_unet(cfg, latent_channels: int) -> LatentUNet:
    """从 config 构建 LatentUNet。"""
    attn_resolutions = _as_int_list(
        getattr(cfg.model, 'attention_resolutions', [])
    )
    return LatentUNet(
        latent_channels=latent_channels,
        base_channels=int(getattr(cfg.model, 'base_channels', 64)),
        n_levels=int(getattr(cfg.model, 'n_levels', 2)),
        n_res=int(getattr(cfg.model, 'n_res', 2)),
        time_cond=bool(getattr(cfg.model, 'time_cond', False)),
        time_embed_dim=int(getattr(cfg.model, 'time_embed_dim', 256)),
        use_bottleneck_attn=bool(getattr(cfg.model, 'use_bottleneck_attn', False)),
        attn_heads=int(getattr(cfg.model, 'attn_heads', 4)),
        use_scale_shift_norm=bool(getattr(cfg.model, 'use_scale_shift_norm', False)),
        dropout=float(getattr(cfg.model, 'dropout', 0.0)),
        latent_size=int(getattr(cfg.model, 'latent_size', 16)),
        use_level_attn=bool(getattr(cfg.model, 'use_level_attn', False)),
        attention_resolutions=attn_resolutions,
        mid_attn_depth=int(getattr(cfg.model, 'mid_attn_depth', 0)),
        ffn_mult=int(getattr(cfg.model, 'ffn_mult', 2)),
        predict_delta=bool(getattr(cfg.model, 'predict_delta', False)),
    )
