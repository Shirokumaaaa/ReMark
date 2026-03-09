"""
Stage 1 VAE — ReMark 图像修复主干

架构（128×128 输入，base_channels=32，latent_channels=64）：
  128×128×3
  → 64×64×32
  → 32×32×64
  → 16×16×128
  → 8×8×256
  → μ, logvar : 8×8×64
  → sample z  : 8×8×64
  → decoder
  → 8×8×256
  → 16×16×128
  → 32×32×64
  → 64×64×32
  → 128×128×3

输入：canonical [-1,1] BCHW
输出：canonical [-1,1] BCHW（clamp 到 [-1,1]）

训练目标：X_fake → VAE → X_hat ≈ X_wm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 基础模块 ──────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """带 GroupNorm 的残差块，不改变空间尺寸"""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Downsample(nn.Module):
    """步长为 2 的卷积下采样"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """最近邻上采样 + 卷积，避免棋盘伪影"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    图像 → (mu, logvar)

    每级结构：[Conv1×1 通道升维（首级除外）] → ResBlock×n_res → Downsample
    末尾接 bottleneck ResBlock + proj 到 2×latent_channels

    n_downsample=4, base_channels=32 时通道变化：
      3 →(conv3×3)→ 32 →(res+down)→ 64 →(res+down)→ 128 →(res+down)→ 256 →(res+down)→ bottleneck(256) → mu/logvar(64)
    空间变化：128 → 64 → 32 → 16 → 8
    """

    def __init__(self, in_channels=3, base_channels=32, latent_channels=64,
                 n_downsample=4, n_res=2):
        super().__init__()
        ch_list = [base_channels * m for m in [1, 2, 4, 8][:n_downsample]]

        # 输入投影：in_channels → ch_list[0]，用 3×3 卷积保留局部信息
        layers = [nn.Conv2d(in_channels, ch_list[0], 3, padding=1)]

        for i in range(n_downsample):
            # 首级不需要通道升维（初始 conv 已完成），后续各级升维
            if i > 0:
                layers.append(nn.Conv2d(ch_list[i - 1], ch_list[i], 1))
            for _ in range(n_res):
                layers.append(ResBlock(ch_list[i]))
            layers.append(Downsample(ch_list[i]))

        # Bottleneck：在最深层多加一个 ResBlock 增强表达能力
        bot_ch = ch_list[-1]
        layers += [
            ResBlock(bot_ch),
            nn.GroupNorm(8, bot_ch),
            nn.SiLU(),
        ]

        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Conv2d(bot_ch, 2 * latent_channels, 1)

    def forward(self, x):
        h = self.encoder(x)
        params = self.proj(h)
        mu, logvar = params.chunk(2, dim=1)
        logvar = logvar.clamp(-30, 20)  # 数值稳定
        return mu, logvar


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    z → 重建图像，Encoder 的对称结构

    每级结构：ResBlock×n_res → Upsample → [Conv1×1 通道降维（末级除外）]
    末级由最终 conv 负责 ch_list[-1] → out_channels

    n_upsample=4, base_channels=32 时通道变化：
      z(64) →(conv1×1)→ 256 →(res+up)→ 128 →(res+up)→ 64 →(res+up)→ 32 →(res+up)→ 32 →(conv3×3)→ 3
    空间变化：8 → 16 → 32 → 64 → 128
    """

    def __init__(self, out_channels=3, base_channels=32, latent_channels=64,
                 n_upsample=4, n_res=2):
        super().__init__()
        ch_list = [base_channels * m for m in [8, 4, 2, 1][:n_upsample]]

        # z → 最深层通道数，再加一个 bottleneck ResBlock
        layers = [
            nn.Conv2d(latent_channels, ch_list[0], 1),
            ResBlock(ch_list[0]),
        ]

        for i in range(n_upsample):
            for _ in range(n_res):
                layers.append(ResBlock(ch_list[i]))
            layers.append(Upsample(ch_list[i]))
            # 末级不降维，由后续最终 conv 直接输出
            if i < n_upsample - 1:
                layers.append(nn.Conv2d(ch_list[i], ch_list[i + 1], 1))

        layers += [
            nn.GroupNorm(8, ch_list[-1]),
            nn.SiLU(),
            nn.Conv2d(ch_list[-1], out_channels, 3, padding=1),
            nn.Tanh(),  # 输出 [-1, 1]
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


# ── VAE ───────────────────────────────────────────────────────────────────────

class ReMark_VAE(nn.Module):
    """
    ReMark Stage 1 VAE。

    Args:
        in_channels:      3（RGB）
        base_channels:    编解码器第一级通道数
        latent_channels:  潜变量通道数（对应 config.model.latent_channels）
        n_downsample:     下采样次数（downsample_factor = 2^n_downsample）
        n_res:            每级残差块数

    Forward:
        x_fake → (x_hat, mu, logvar)

    encode / decode:
        可单独调用，用于 SLERP 潜空间插值（Stage 2）
    """

    def __init__(self, in_channels=3, base_channels=32, latent_channels=64,
                 n_downsample=4, n_res=2):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, latent_channels,
                               n_downsample, n_res)
        self.decoder = Decoder(in_channels, base_channels, latent_channels,
                               n_downsample, n_res)

    def encode(self, x) -> tuple:
        """返回 (mu, logvar)"""
        return self.encoder(x)

    def decode(self, z) -> torch.Tensor:
        """z → 图像"""
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """训练时重参数化采样；推理时直接用 mu"""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: canonical [-1,1] BCHW（攻击后的图像 X_fake）
        Returns:
            (x_hat, mu, logvar)
            x_hat: 重建图像，canonical [-1,1]
            mu/logvar: 用于 KL loss 和 SLERP
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def build_vae(cfg) -> ReMark_VAE:
    """从 config 构建 VAE，供训练脚本调用"""
    import math
    ds = getattr(cfg.model, 'downsample_factor', 16)
    n_downsample = int(math.log2(ds))
    return ReMark_VAE(
        in_channels=3,
        base_channels=getattr(cfg.model, 'base_channels', 32),
        latent_channels=getattr(cfg.model, 'latent_channels', 64),
        n_downsample=n_downsample,
        n_res=getattr(cfg.model, 'n_res', 2),
    )
