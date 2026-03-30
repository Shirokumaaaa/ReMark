"""
ReMark Stage 2 训练脚本 — Latent U-Net 水印修复

Stage 2 在 Stage 1 的 latent 空间上训练 U-Net G_ψ，
通过 SLERP 轨迹作为监督信号，迭代将受损 latent 推向原始含水印 latent。

算法（当前实现）：
  对每张含水印图 x：
    z_src ← E(fake(x))
    z_gt  ← E(x)
    {z_0, ..., z_m} ← SLERP(z_src, z_gt, m)   # 超球面轨迹
    for k = 0..m-1:
      z_in = TF_warmup(z_k, z_hat_k)           # 前期 teacher forcing，后期自回归
      z̃_{k+1} = G_ψ(z_in)
      每一步单独反传：
        L_k = λ1·w_k·L1(z̃_{k+1}, z_{k+1}) + λ2·u_k·BCE(WM_dec(VAE_dec(z̃_{k+1}|ref=fake)), msg)

关键约束：
  - VAE（Stage 1）全程冻结，不参与参数更新
  - WM-Decoder 参数冻结，但梯度可穿过（BCE 反传需要）
  - 换 Stage 1 checkpoint 必须重新训练 Stage 2（版本绑定）

启动方式：
    python train_stage2.py --config configs/stage2_unet.yaml

    # 指定 Stage 1 checkpoint（覆盖 config 中的路径）：
    python train_stage2.py --config configs/stage2_unet.yaml \\
        --stage1-ckpt runs/stage1_20260310_154310/checkpoints/vae/best.pth

    # 续训：
    python train_stage2.py --config configs/stage2_unet.yaml \\
        --resume stage2_20260311_HHMMSS
"""

import argparse
import hashlib
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from utils.config import load_config
from utils.logger import RunLogger
from data.dataset import ReMark_Dataset
from network.vae import build_vae
from network.unet_repair import build_unet
from wm_adapters.registry import build_wm_adapter
from attacks.registry import build_attack, ATTACK_REGISTRY


# ── DDP 工具（与 train_stage1.py 相同）────────────────────────────────────────

def setup_ddp():
    if 'RANK' not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── SLERP ─────────────────────────────────────────────────────────────────────

def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    """
    球面线性插值（SLERP）：在 latent 超球面上从 z0 → z1 取 t 处的点。

    t=0 → z0，t=1 → z1。
    当 θ≈0（两向量几乎平行）时退化为 LERP，避免数值不稳定。

    Args:
        z0, z1: (B, C, H, W)，Stage 1 VAE 的 mu（确定性 latent）
        t:      插值系数，[0, 1]
    Returns:
        插值后的 latent，shape 同输入
    """
    b = z0.shape[0]
    z0_flat = z0.reshape(b, -1).float()
    z1_flat = z1.reshape(b, -1).float()

    norm0 = z0_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm1 = z1_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

    cos_theta = (z0_flat / norm0 * z1_flat / norm1).sum(dim=1).clamp(-1.0, 1.0)
    theta     = cos_theta.acos()          # (B,)
    sin_theta = theta.sin()               # (B,)

    # LERP 退化条件：θ < 1e-4 rad（≈ 0.006°）
    lerp_mask = sin_theta.abs() < 1e-4   # (B,)

    # SLERP 系数
    w0 = torch.where(
        lerp_mask,
        torch.full_like(theta, 1.0 - t),
        (theta * (1.0 - t)).sin() / (sin_theta + 1e-8),
    )
    w1 = torch.where(
        lerp_mask,
        torch.full_like(theta, t),
        (theta * t).sin() / (sin_theta + 1e-8),
    )

    # 广播到空间维度
    shape = [b] + [1] * (z0.ndim - 1)
    w0 = w0.reshape(shape)
    w1 = w1.reshape(shape)

    return (w0 * z0 + w1 * z1).to(z0.dtype)


def build_slerp_trajectory(z_src: torch.Tensor,
                            z_gt: torch.Tensor,
                            n_steps: int) -> list:
    """
    构建从 z_src 到 z_gt 的 SLERP 轨迹，共 n_steps+1 个点。

    trajectory[0]      = z_src
    trajectory[n_steps] ≈ z_gt
    trajectory[k]      = slerp(z_src, z_gt, k / n_steps)

    Returns:
        list of n_steps+1 tensors，每个 shape (B, C, H, W)
    """
    if n_steps < 1:
        raise ValueError(f'n_steps must be >= 1, got {n_steps}')
    trajectory = []
    for k in range(n_steps + 1):
        t = k / n_steps
        trajectory.append(slerp(z_src, z_gt, t).detach())
    return trajectory


# ── Stage 1 Checkpoint 工具 ───────────────────────────────────────────────────

def _ckpt_hash(path: str) -> str:
    """计算 checkpoint 文件的 MD5 前 16 位，用于版本绑定。"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_stage1_vae(ckpt_path: str, cfg_s1, device: torch.device):
    """
    加载 Stage 1 VAE，全程冻结。

    Returns:
        (vae, latent_channels, ckpt_hash)
    """
    vae = build_vae(cfg_s1).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # PyTorch < 1.13 不支持 weights_only 参数
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        sd = state.get('net', state.get('vae', state))
    else:
        sd = state
    vae.load_state_dict(sd)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latent_channels = getattr(cfg_s1.model, 'latent_channels', 64)
    return vae, latent_channels, _ckpt_hash(ckpt_path)


# ── 位精度计算 ────────────────────────────────────────────────────────────────

def bit_acc(logits: torch.Tensor, messages: torch.Tensor) -> float:
    """logits: (B, L)，messages: {0,1}"""
    with torch.no_grad():
        pred = (logits > 0).float()
        return pred.eq(messages).float().mean().item()


# ── DataLoader ────────────────────────────────────────────────────────────────

def build_loaders(cfg, rank, world_size):
    def _make(csv_path, mode):
        ds = ReMark_Dataset(
            csv_path=csv_path,
            image_size=cfg.data.image_size,
            mode=mode,
            center_crop=getattr(cfg.data, 'center_crop', 0),
        )
        if world_size > 1:
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=(mode == 'train'), drop_last=(mode == 'train'),
            )
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=cfg.training.batch_size,
                sampler=sampler,
                num_workers=getattr(cfg.efficiency, 'attack_num_workers', 0),
                pin_memory=True,
            )
            return loader, sampler
        else:
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=cfg.training.batch_size,
                shuffle=(mode == 'train'),
                drop_last=(mode == 'train'),
                num_workers=getattr(cfg.efficiency, 'attack_num_workers', 0),
                pin_memory=True,
            )
            return loader, None

    train_loader, train_sampler = _make(cfg.data.train_csv, 'train')
    val_loader,   _             = _make(cfg.data.val_csv,   'val')
    return train_loader, val_loader, train_sampler


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg, cfg_s1, stage1_ckpt_path: str,
                 logger, device, rank, world_size,
                 resume_path: str = None):
        self.cfg        = cfg
        self.device     = device
        self.rank       = rank
        self.world_size = world_size
        self.main       = is_main(rank)
        self.logger     = logger

        # ── Stage 1 VAE（冻结）────────────────────────────────────────────────
        self.vae, latent_channels, s1_hash = load_stage1_vae(
            stage1_ckpt_path, cfg_s1, device
        )
        self.latent_channels = latent_channels
        self.stage1_hash     = s1_hash
        self.vae_residual_output = bool(getattr(self.vae, 'residual_output', False))
        self.vae_residual_scale = float(getattr(self.vae, 'residual_scale', 1.0))
        if self.main:
            self.logger.info(
                f'Stage1 VAE decode mode: residual_output={self.vae_residual_output} '
                f'(scale={self.vae_residual_scale:.3f})'
            )

        # ── Stage 2 U-Net（训练目标）─────────────────────────────────────────
        unet_raw = build_unet(cfg, latent_channels).to(device)
        if world_size > 1:
            self.unet = DDP(unet_raw, device_ids=[device.index])
        else:
            self.unet = unet_raw
        self._unet_module = unet_raw

        # ── WM Adapter（Decoder 参数冻结，梯度可穿过）────────────────────────
        self.wm_adapter = build_wm_adapter(cfg.wm_model, cfg)

        # ── 攻击模型（冻结）──────────────────────────────────────────────────
        self.attacks = {}
        for name in getattr(cfg.attacks, 'online', []):
            if name not in ATTACK_REGISTRY:
                if self.main:
                    logger.warning(f'Attack "{name}" not registered, skipping.')
                continue
            self.attacks[name] = build_attack(name, cfg)
            if self.main:
                logger.info(f'Online attack loaded: {name}')
        self.attack_sample_weights = {}
        sample_weights_cfg = getattr(getattr(cfg, 'attacks', None), 'sample_weights', None)
        if sample_weights_cfg is not None:
            if hasattr(sample_weights_cfg, '__dict__'):
                raw_weights = vars(sample_weights_cfg)
            elif isinstance(sample_weights_cfg, dict):
                raw_weights = sample_weights_cfg
            else:
                raw_weights = {}
            for attack_name in self.attacks.keys():
                w = float(raw_weights.get(attack_name, 1.0))
                self.attack_sample_weights[attack_name] = max(w, 0.0)
            if self.main and self.attack_sample_weights:
                self.logger.info(
                    'Attack sampling weights: ' +
                    ', '.join(f'{k}={v:.3f}' for k, v in sorted(self.attack_sample_weights.items()))
                )

        # ── Optimizer ─────────────────────────────────────────────────────────
        params = list(filter(lambda p: p.requires_grad,
                             self._unet_module.parameters()))
        self.weight_decay = float(getattr(cfg.training, 'weight_decay', 0.0))
        self.optimizer = torch.optim.Adam(
            params,
            lr=cfg.training.lr,
            betas=getattr(cfg.training, 'betas', (0.9, 0.999)),
            weight_decay=self.weight_decay,
        )
        self.scaler = GradScaler(enabled=getattr(cfg.efficiency, 'use_amp', False))
        self.use_lr_scheduler = bool(getattr(cfg.training, 'use_lr_scheduler', False))
        self.scheduler = None
        if self.use_lr_scheduler:
            lr_min_scale = float(getattr(cfg.training, 'lr_min_scale', 0.2))
            lr_min_scale = max(min(lr_min_scale, 1.0), 0.0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(int(getattr(cfg.training, 'epochs', 1)), 1),
                eta_min=float(cfg.training.lr) * lr_min_scale,
            )

        # ── 续训 ──────────────────────────────────────────────────────────────
        self.start_epoch = 0
        self.best_uplift = -1e9
        self.best_step10_acc = 0.0
        if resume_path and os.path.isfile(resume_path):
            self._load_checkpoint(resume_path)

        # ── 超参 ──────────────────────────────────────────────────────────────
        self.n_steps     = getattr(getattr(cfg, 'slerp', None), 'n_steps', 10)
        if self.n_steps < 1:
            raise ValueError(f'slerp.n_steps 必须 >= 1，当前为 {self.n_steps}')
        self.max_infer   = getattr(cfg.training, 'max_infer_steps', 10)
        self.use_amp     = getattr(cfg.efficiency, 'use_amp', False)
        self.latent_noise_std = float(getattr(cfg.training, 'latent_noise_std', 0.0))
        self.per_step_update = bool(getattr(cfg.training, 'per_step_update', True))
        # BCE 与 latent 路径不同：可选每步先做一次 BCE-only 更新，再做主损失更新。
        # 该模式仅在 per_step_update=true 时生效。
        self.separate_bce_step = bool(getattr(cfg.training, 'separate_bce_step', True))
        self.bce_only_step_scale = float(getattr(cfg.training, 'bce_only_step_scale', 1.0))
        self.bce_only_step_scale = max(self.bce_only_step_scale, 0.0)
        self.bptt_rollout = bool(getattr(cfg.training, 'bptt_rollout', False))
        if self.per_step_update and self.bptt_rollout:
            # 每步都 optimizer.step() 时不能安全保留跨步计算图，强制关闭
            self.bptt_rollout = False
        if (not self.per_step_update) and self.separate_bce_step:
            self.separate_bce_step = False
        self.tf_warmup_epochs = max(
            int(getattr(cfg.training, 'teacher_forcing_warmup_epochs', 0)), 0
        )
        self.tf_start_prob = float(getattr(cfg.training, 'teacher_forcing_start_prob', 1.0))
        self.tf_end_prob = float(getattr(cfg.training, 'teacher_forcing_end_prob', 0.0))
        self.tf_start_prob = min(max(self.tf_start_prob, 0.0), 1.0)
        self.tf_end_prob = min(max(self.tf_end_prob, 0.0), 1.0)
        self.final_step_l1_weight = max(
            float(getattr(getattr(cfg, 'losses', None), 'final_step_l1_weight', 1.0)),
            0.0,
        )
        self.final_step_bce_weight = max(
            float(getattr(getattr(cfg, 'losses', None), 'final_step_bce_weight', 1.0)),
            0.0,
        )
        # 监督域：latent | image | hybrid
        sup_mode = str(getattr(getattr(cfg, 'training', None), 'supervision_mode', 'auto')).strip().lower()
        if sup_mode not in ('auto', 'latent', 'image', 'hybrid'):
            raise ValueError(f'training.supervision_mode 仅支持 auto/latent/image/hybrid，当前: {sup_mode}')
        if sup_mode == 'auto':
            self.supervision_mode = 'hybrid' if self.vae_residual_output else 'latent'
        else:
            self.supervision_mode = sup_mode
        losses_cfg = getattr(cfg, 'losses', None)
        self.image_l1_weight = max(float(getattr(getattr(losses_cfg, 'image_l1', None), 'weight', 1.0)), 0.0)
        self.latent_aux_weight = max(float(getattr(getattr(losses_cfg, 'latent_aux', None), 'weight', 0.1)), 0.0)

        # BCE step-wise 权重：中间步弱，最后一步强；保留 legacy final_step_bce_weight 兼容项
        bce_cfg = getattr(losses_cfg, 'bce', None)
        self.bce_intermediate_weight = max(float(getattr(bce_cfg, 'intermediate_weight', 1.0)), 0.0)
        self.bce_final_weight = max(float(getattr(bce_cfg, 'final_weight', 1.0)), 0.0)

        self.reference_sensitivity_log = bool(
            getattr(getattr(cfg, 'training', None), 'reference_sensitivity_log', True)
        )
        self.direction_loss_weight = max(
            float(getattr(getattr(getattr(cfg, 'losses', None), 'direction', None), 'weight', 0.0)),
            0.0,
        )
        self.progress_loss_weight = max(
            float(getattr(getattr(getattr(cfg, 'losses', None), 'progress', None), 'weight', 0.0)),
            0.0,
        )
        self.progress_margin = float(
            getattr(getattr(getattr(cfg, 'losses', None), 'progress', None), 'margin', 0.0)
        )
        self.progress_margin = max(self.progress_margin, 0.0)
        self.step_size_loss_weight = max(
            float(getattr(getattr(getattr(cfg, 'losses', None), 'step_size', None), 'weight', 0.0)),
            0.0,
        )
        self.unet_output_blend = float(getattr(getattr(cfg, 'training', None), 'unet_output_blend', 1.0))
        self.unet_output_blend = min(max(self.unet_output_blend, 0.0), 1.0)
        self.delta_scale = float(getattr(getattr(cfg, 'training', None), 'delta_scale', 1.0))
        self.delta_scale = max(self.delta_scale, 0.0)

        move_floor_cfg = getattr(getattr(cfg, 'losses', None), 'move_floor', None)
        self.move_floor_weight = max(float(getattr(move_floor_cfg, 'weight', 0.0)), 0.0)
        self.move_floor_min_step = max(float(getattr(move_floor_cfg, 'min_step', 0.0)), 0.0)

        bce_prog_cfg = getattr(getattr(cfg, 'losses', None), 'bce_progress', None)
        self.bce_progress_weight = max(float(getattr(bce_prog_cfg, 'weight', 0.0)), 0.0)
        self.bce_progress_margin = max(float(getattr(bce_prog_cfg, 'margin', 0.0)), 0.0)

        term_anchor_cfg = getattr(getattr(cfg, 'losses', None), 'terminal_anchor', None)
        self.terminal_anchor_weight = max(float(getattr(term_anchor_cfg, 'weight', 0.0)), 0.0)

        self.enable_phase_schedule = bool(getattr(getattr(cfg, 'training', None), 'enable_phase_schedule', False))
        self.phase_a_epochs = max(int(getattr(getattr(cfg, 'training', None), 'phase_a_epochs', 0)), 0)
        self.phase_b_tf_end_prob = float(
            getattr(getattr(cfg, 'training', None), 'phase_b_teacher_forcing_end_prob', self.tf_end_prob)
        )
        self.phase_b_tf_end_prob = min(max(self.phase_b_tf_end_prob, 0.0), 1.0)
        self.bptt_horizon = max(int(getattr(getattr(cfg, 'training', None), 'bptt_horizon', 0)), 0)
        if self.per_step_update and self.bptt_horizon > 0:
            self.bptt_horizon = 0
        default_norm_step_weights = not self.per_step_update
        self.normalize_step_weights = bool(
            getattr(getattr(cfg, 'losses', None), 'normalize_step_weights', default_norm_step_weights)
        )

        # 轨迹尾段加权（对最难恢复点位加强监督）
        tail_cfg = getattr(getattr(cfg, 'losses', None), 'tail', None)
        self.tail_enabled = bool(getattr(tail_cfg, 'enabled', False))
        self.tail_last_k = max(int(getattr(tail_cfg, 'last_k', 0)), 0)
        self.tail_weight = max(float(getattr(tail_cfg, 'weight', 1.0)), 0.0)
        if self.tail_last_k <= 0 or self.tail_weight <= 0.0:
            self.tail_enabled = False
        self.step_weights = self._build_step_weights().to(self.device)

        self.progress_enabled = bool(getattr(
            getattr(cfg, 'progress', None), 'enabled', True
        ))
        self.progress_log_interval = max(
            int(getattr(getattr(cfg, 'progress', None), 'log_interval_steps', 20)),
            1,
        )
        self.preflight_enabled = bool(getattr(
            getattr(cfg, 'preflight_eval', None), 'enabled', True
        ))
        self.preflight_max_batches = int(getattr(
            getattr(cfg, 'preflight_eval', None), 'max_batches', 8
        ))
        self.curve_enabled = bool(getattr(
            getattr(getattr(cfg, 'validation', None), 'point_curve', None),
            'enabled', True
        ))
        self.curve_max_batches = int(getattr(
            getattr(getattr(cfg, 'validation', None), 'point_curve', None),
            'max_batches', 4
        ))
        curve_cfg = getattr(getattr(cfg, 'validation', None), 'point_curve', None)
        self.log_rollout_curve = bool(getattr(curve_cfg, 'log_rollout_curve', True))
        train_probe_cfg = getattr(getattr(cfg, 'validation', None), 'train_probe', None)
        self.train_probe_enabled = bool(getattr(train_probe_cfg, 'enabled', False))
        self.train_probe_max_batches = int(getattr(train_probe_cfg, 'max_batches', 0))
        self.train_probe_curve_max_batches = int(getattr(train_probe_cfg, 'curve_max_batches', 0))
        slerp_cfg = getattr(cfg, 'slerp', None)
        self.time_cond = bool(getattr(getattr(cfg, 'model', None), 'time_cond', False))
        self.time_min = float(getattr(slerp_cfg, 'time_min', 0.0))
        self.time_max = float(getattr(slerp_cfg, 'time_max', 1000.0))
        if self.time_max < self.time_min:
            raise ValueError(
                f'slerp.time_max 必须 >= slerp.time_min，当前为 '
                f'{self.time_max} < {self.time_min}'
            )
        pred_mode_cfg = str(
            getattr(getattr(cfg, 'training', None), 'unet_prediction_mode', 'auto')
        ).strip().lower()
        if pred_mode_cfg not in ('auto', 'delta', 'absolute'):
            raise ValueError(
                f'training.unet_prediction_mode 仅支持 auto/delta/absolute，当前: {pred_mode_cfg}'
            )
        if pred_mode_cfg == 'auto':
            self.unet_prediction_mode = (
                'delta' if bool(getattr(self._unet_module, 'predict_delta', False)) else 'absolute'
            )
        else:
            self.unet_prediction_mode = pred_mode_cfg

        # ── 三阶段训练调度（可选）────────────────────────────────────────────
        self.base_lr = float(getattr(cfg.training, 'lr', 1e-4))
        self.base_n_steps = int(self.n_steps)
        losses_cfg = getattr(cfg, 'losses', None)
        self.base_l1_weight = float(getattr(getattr(losses_cfg, 'l1', None), 'weight', 1.0))
        self.base_bce_weight = float(getattr(getattr(losses_cfg, 'bce', None), 'weight', 1.0))

        phase_cfg = getattr(getattr(cfg, 'training', None), 'three_phase', None)
        self.use_three_phase_schedule = bool(getattr(phase_cfg, 'enabled', False))

        self.phase1_min_epochs = max(int(getattr(phase_cfg, 'phase1_min_epochs', 0)), 0)
        self.phase1_max_epochs = max(int(getattr(phase_cfg, 'phase1_max_epochs', self.phase1_min_epochs)), 0)
        if self.phase1_max_epochs < self.phase1_min_epochs:
            self.phase1_max_epochs = self.phase1_min_epochs
        self.phase2_epochs = max(int(getattr(phase_cfg, 'phase2_epochs', 0)), 0)

        self.phase1_n_steps = max(int(getattr(phase_cfg, 'phase1_n_steps', 5)), 1)
        self.phase2_n_steps = max(int(getattr(phase_cfg, 'phase2_n_steps', self.base_n_steps)), 1)
        self.phase3_n_steps = max(int(getattr(phase_cfg, 'phase3_n_steps', self.base_n_steps)), 1)

        self.phase2_tf_end_prob = float(getattr(phase_cfg, 'phase2_tf_end_prob', 0.3))
        self.phase2_tf_end_prob = min(max(self.phase2_tf_end_prob, 0.0), 1.0)
        self.phase3_lr_scale = max(float(getattr(phase_cfg, 'phase3_lr_scale', 0.3)), 0.0)

        self.phase1_l1_weight = max(float(getattr(phase_cfg, 'phase1_l1_weight', self.base_l1_weight)), 0.0)
        self.phase1_bce_weight = max(float(getattr(phase_cfg, 'phase1_bce_weight', self.base_bce_weight)), 0.0)
        self.phase2_l1_weight = max(float(getattr(phase_cfg, 'phase2_l1_weight', self.base_l1_weight)), 0.0)
        self.phase2_bce_weight = max(float(getattr(phase_cfg, 'phase2_bce_weight', self.base_bce_weight)), 0.0)
        self.phase3_l1_weight = max(float(getattr(phase_cfg, 'phase3_l1_weight', self.base_l1_weight)), 0.0)
        self.phase3_bce_weight = max(float(getattr(phase_cfg, 'phase3_bce_weight', self.base_bce_weight)), 0.0)

        self.phase1_disable_aux_losses = bool(getattr(phase_cfg, 'phase1_disable_aux_losses', True))
        self.phase2_progress_weight = max(float(getattr(phase_cfg, 'phase2_progress_weight', self.progress_loss_weight)), 0.0)
        self.phase2_terminal_anchor_weight = max(float(getattr(phase_cfg, 'phase2_terminal_anchor_weight', self.terminal_anchor_weight)), 0.0)
        self.phase3_progress_weight = max(float(getattr(phase_cfg, 'phase3_progress_weight', self.phase2_progress_weight)), 0.0)
        self.phase3_terminal_anchor_weight = max(float(getattr(phase_cfg, 'phase3_terminal_anchor_weight', self.phase2_terminal_anchor_weight)), 0.0)

        self.phase3_tail_enabled = bool(getattr(phase_cfg, 'phase3_tail_enabled', self.tail_enabled))
        self.phase3_tail_last_k = max(int(getattr(phase_cfg, 'phase3_tail_last_k', self.tail_last_k)), 0)
        self.phase3_tail_weight = max(float(getattr(phase_cfg, 'phase3_tail_weight', self.tail_weight)), 0.0)

        self.phase1_gate_enabled = bool(getattr(phase_cfg, 'phase1_gate_enabled', True))
        self.phase1_target_uplift = float(getattr(phase_cfg, 'phase1_target_uplift', 0.05))
        self.phase1_target_step = max(int(getattr(phase_cfg, 'phase1_target_uplift_step', 5)), 1)
        self.phase1_gate_met = False
        self.phase1_exit_epoch = None

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, tag: str):
        if not self.main:
            return
        path = self.logger.checkpoint_path(tag, model='unet')
        torch.save({
            'net':          self._unet_module.state_dict(),
            'opt':          self.optimizer.state_dict(),
            'scaler':       self.scaler.state_dict(),
            'scheduler':    self.scheduler.state_dict() if self.scheduler is not None else None,
            'epoch':        epoch,
            'stage1_hash':  self.stage1_hash,   # 版本绑定
            'best_uplift':  self.best_uplift,
            'best_step10_acc': self.best_step10_acc,
            # backward compatibility
            'best_acc':     self.best_step10_acc,
        }, path)

    def _load_checkpoint(self, path: str):
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # PyTorch < 1.13 不支持 weights_only 参数
            state = torch.load(path, map_location=self.device)
        # 版本绑定校验
        saved_hash = state.get('stage1_hash', None)
        if saved_hash and saved_hash != self.stage1_hash:
            raise RuntimeError(
                f'Stage 1 checkpoint 版本不匹配！\n'
                f'  当前 Stage 1 hash: {self.stage1_hash}\n'
                f'  Stage 2 ckpt 绑定: {saved_hash}\n'
                f'  更换 Stage 1 checkpoint 必须重新训练 Stage 2。'
            )
        self._unet_module.load_state_dict(state['net'])
        self.optimizer.load_state_dict(state['opt'])
        if 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
        if self.scheduler is not None and 'scheduler' in state and state['scheduler'] is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        self.best_uplift = float(state.get('best_uplift', -1e9))
        self.best_step10_acc = float(state.get('best_step10_acc', state.get('best_acc', 0.0)))
        self.start_epoch = state.get('epoch', 0) + 1
        if self.main:
            self.logger.info(f'Resumed from {path}, start_epoch={self.start_epoch}')

    # ── 攻击辅助 ──────────────────────────────────────────────────────────────

    def _apply_attack(self, attack_name: str,
                      wm_images: torch.Tensor,
                      images: torch.Tensor,
                      batch: dict) -> torch.Tensor:
        attack = self.attacks[attack_name]
        with torch.no_grad():
            if hasattr(attack, 'attack_with_cover'):
                try:
                    return attack.attack_with_cover(wm_images, images, batch=batch)
                except TypeError:
                    return attack.attack_with_cover(wm_images, images)
            return attack(wm_images)

    def _reduce_metrics(self, metrics: dict) -> dict:
        """DDP 下跨卡聚合标量指标；支持 NaN（按有效卡数均值）。"""
        if self.world_size <= 1:
            return metrics
        reduced = {}
        for k, v in metrics.items():
            is_valid = not np.isnan(v)
            sum_t = torch.tensor(float(v) if is_valid else 0.0, device=self.device)
            cnt_t = torch.tensor(1.0 if is_valid else 0.0, device=self.device)
            dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
            denom = max(cnt_t.item(), 1.0)
            reduced[k] = sum_t.item() / denom if cnt_t.item() > 0 else float('nan')
        return reduced

    def _reduce_vector(self, values: np.ndarray) -> np.ndarray:
        """DDP 下跨卡平均向量指标。"""
        if self.world_size <= 1:
            return values
        t = torch.tensor(values, device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return t.detach().cpu().numpy()

    def _decode_latent(self, z: torch.Tensor,
                       reference_image: torch.Tensor = None) -> torch.Tensor:
        """
        解码 latent 为图像。若 Stage1 VAE 使用 residual_output，则按 Stage1 forward 方式恢复：
          x_hat = clamp(reference + residual_scale * decode(z), -1, 1)
        """
        x_hat = self.vae.decode(z)
        if self.vae_residual_output:
            if reference_image is None:
                raise RuntimeError(
                    'Stage1 VAE uses residual_output=True, but reference_image is missing.'
                )
            x_hat = torch.clamp(
                reference_image + self.vae_residual_scale * x_hat,
                -1.0, 1.0
            )
        return x_hat

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f'{h:02d}:{m:02d}:{s:02d}'
        return f'{m:02d}:{s:02d}'

    def _log_progress(self, epoch: int, phase: str, step_idx: int,
                      total_steps: int, start_ts: float):
        if not (self.main and self.progress_enabled):
            return
        done = step_idx + 1
        if (done % self.progress_log_interval != 0) and (done != total_steps):
            return
        elapsed = max(time.time() - start_ts, 1e-6)
        speed = done / elapsed
        remain = max(total_steps - done, 0)
        eta = remain / max(speed, 1e-6)
        pct = 100.0 * done / max(total_steps, 1)
        self.logger.info(
            f'Progress | epoch={epoch:03d}  phase={phase}  '
            f'step={done}/{total_steps} ({pct:.1f}%)  '
            f'speed={speed:.2f} it/s  eta={self._format_seconds(eta)}'
        )

    def _build_step_weights(self,
                            n_steps: int = None,
                            tail_enabled: bool = None,
                            tail_last_k: int = None,
                            tail_weight: float = None) -> torch.Tensor:
        """
        构造轨迹 step 权重：
          - 默认均匀加权
          - 开启 tail 后，对最后 k 个 step 赋更大权重
          - normalize_step_weights=true 时归一化到和为 1
            normalize_step_weights=false 时保持原始量纲（per-step update 更稳定）
        """
        n_steps = self.n_steps if n_steps is None else max(int(n_steps), 1)
        tail_enabled = self.tail_enabled if tail_enabled is None else bool(tail_enabled)
        tail_last_k = self.tail_last_k if tail_last_k is None else max(int(tail_last_k), 0)
        tail_weight = self.tail_weight if tail_weight is None else max(float(tail_weight), 0.0)

        weights = torch.ones(n_steps, dtype=torch.float32)
        if tail_enabled and tail_last_k > 0 and tail_weight > 0.0:
            k = min(tail_last_k, n_steps)
            weights[-k:] = float(tail_weight)
        if self.normalize_step_weights:
            denom = weights.sum().clamp(min=1e-8)
            return weights / denom
        return weights

    def _set_optimizer_lr(self, lr: float):
        lr = max(float(lr), 0.0)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _current_phase_name(self, epoch: int) -> str:
        if not self.use_three_phase_schedule:
            return 'legacy'
        if self.phase1_exit_epoch is None:
            if epoch < self.phase1_min_epochs:
                return 'phase1'
            gate_ready = (not self.phase1_gate_enabled) or self.phase1_gate_met or (epoch >= self.phase1_max_epochs)
            if not gate_ready:
                return 'phase1'
            self.phase1_exit_epoch = epoch
        if epoch < (self.phase1_exit_epoch + self.phase2_epochs):
            return 'phase2'
        return 'phase3'

    def _runtime_schedule(self, epoch: int, with_step_weights: bool = True) -> dict:
        if not self.use_three_phase_schedule:
            step_weights = self.step_weights if with_step_weights else None
            return {
                'phase': 'legacy',
                'n_steps': self.n_steps,
                'tf_prob': self._teacher_forcing_prob(epoch),
                'l1_weight': self.base_l1_weight,
                'bce_weight': self.base_bce_weight,
                'progress_weight': self.progress_loss_weight,
                'terminal_anchor_weight': self.terminal_anchor_weight,
                'direction_weight': self.direction_loss_weight,
                'step_size_weight': self.step_size_loss_weight,
                'move_floor_weight': self.move_floor_weight,
                'bce_progress_weight': self.bce_progress_weight,
                'tail_enabled': self.tail_enabled,
                'tail_last_k': self.tail_last_k,
                'tail_weight': self.tail_weight,
                'lr': float(self.optimizer.param_groups[0]['lr']),
                'step_weights': step_weights,
            }

        phase = self._current_phase_name(epoch)
        if phase == 'phase1':
            tf_prob = 1.0
            n_steps = self.phase1_n_steps
            l1_w = self.phase1_l1_weight
            bce_w = self.phase1_bce_weight
            progress_w = 0.0 if self.phase1_disable_aux_losses else self.progress_loss_weight
            terminal_w = 0.0 if self.phase1_disable_aux_losses else self.terminal_anchor_weight
            direction_w = 0.0 if self.phase1_disable_aux_losses else self.direction_loss_weight
            step_size_w = 0.0 if self.phase1_disable_aux_losses else self.step_size_loss_weight
            move_floor_w = 0.0 if self.phase1_disable_aux_losses else self.move_floor_weight
            bce_progress_w = 0.0 if self.phase1_disable_aux_losses else self.bce_progress_weight
            tail_enabled = False
            tail_last_k = 0
            tail_w = 0.0
            lr = self.base_lr
        elif phase == 'phase2':
            n_steps = self.phase2_n_steps
            if self.phase2_epochs <= 0:
                ratio = 1.0
            else:
                start = self.phase1_exit_epoch if self.phase1_exit_epoch is not None else self.phase1_min_epochs
                done = min(max(epoch - start, 0), self.phase2_epochs)
                ratio = done / float(max(self.phase2_epochs, 1))
            tf_prob = 1.0 + (self.phase2_tf_end_prob - 1.0) * ratio
            l1_w = self.phase2_l1_weight
            bce_w = self.phase2_bce_weight
            progress_w = self.phase2_progress_weight
            terminal_w = self.phase2_terminal_anchor_weight
            direction_w = self.direction_loss_weight
            step_size_w = self.step_size_loss_weight
            move_floor_w = self.move_floor_weight
            bce_progress_w = self.bce_progress_weight
            tail_enabled = False
            tail_last_k = 0
            tail_w = 0.0
            lr = self.base_lr
        else:
            tf_prob = 0.0
            n_steps = self.phase3_n_steps
            l1_w = self.phase3_l1_weight
            bce_w = self.phase3_bce_weight
            progress_w = self.phase3_progress_weight
            terminal_w = self.phase3_terminal_anchor_weight
            direction_w = self.direction_loss_weight
            step_size_w = self.step_size_loss_weight
            move_floor_w = self.move_floor_weight
            bce_progress_w = self.bce_progress_weight
            tail_enabled = self.phase3_tail_enabled
            tail_last_k = self.phase3_tail_last_k
            tail_w = self.phase3_tail_weight
            lr = self.base_lr * self.phase3_lr_scale

        tf_prob = float(min(max(tf_prob, 0.0), 1.0))
        schedule = {
            'phase': phase,
            'n_steps': int(n_steps),
            'tf_prob': tf_prob,
            'l1_weight': float(max(l1_w, 0.0)),
            'bce_weight': float(max(bce_w, 0.0)),
            'progress_weight': float(max(progress_w, 0.0)),
            'terminal_anchor_weight': float(max(terminal_w, 0.0)),
            'direction_weight': float(max(direction_w, 0.0)),
            'step_size_weight': float(max(step_size_w, 0.0)),
            'move_floor_weight': float(max(move_floor_w, 0.0)),
            'bce_progress_weight': float(max(bce_progress_w, 0.0)),
            'tail_enabled': bool(tail_enabled),
            'tail_last_k': int(max(tail_last_k, 0)),
            'tail_weight': float(max(tail_w, 0.0)),
            'lr': float(max(lr, 0.0)),
            'step_weights': None,
        }
        if with_step_weights:
            schedule['step_weights'] = self._build_step_weights(
                n_steps=schedule['n_steps'],
                tail_enabled=schedule['tail_enabled'],
                tail_last_k=schedule['tail_last_k'],
                tail_weight=schedule['tail_weight'],
            ).to(self.device)
        return schedule

    def _teacher_forcing_prob(self, epoch: int) -> float:
        """
        teacher forcing 概率线性退火：
          epoch=0                  -> start_prob
          epoch>=warmup_epochs     -> end_prob
        """
        if self.use_three_phase_schedule:
            return self._runtime_schedule(epoch, with_step_weights=False)['tf_prob']
        if self.enable_phase_schedule:
            if epoch < self.phase_a_epochs:
                return 1.0
            total_epochs = max(int(getattr(self.cfg.training, 'epochs', 1)), 1)
            phase_b_total = max(total_epochs - self.phase_a_epochs, 1)
            phase_b_epoch = min(max(epoch - self.phase_a_epochs, 0), phase_b_total)
            ratio = phase_b_epoch / float(phase_b_total)
            p = 1.0 + (self.phase_b_tf_end_prob - 1.0) * ratio
            return float(min(max(p, 0.0), 1.0))
        if self.tf_warmup_epochs <= 0:
            return 0.0
        ratio = min(max(epoch, 0), self.tf_warmup_epochs) / float(self.tf_warmup_epochs)
        p = self.tf_start_prob + (self.tf_end_prob - self.tf_start_prob) * ratio
        return float(min(max(p, 0.0), 1.0))

    def _step_timestep(self, step_idx: int,
                       total_steps: int,
                       batch_size: int) -> torch.Tensor:
        """
        将推理步映射到扩散式 timestep（默认从大到小）。
        """
        if not self.time_cond:
            return None
        if total_steps <= 1:
            t = self.time_max
        else:
            alpha = step_idx / float(total_steps - 1)
            t = self.time_max + (self.time_min - self.time_max) * alpha
        return torch.full(
            (batch_size,),
            float(t),
            device=self.device,
            dtype=torch.float32,
        )

    def _unet_step(self, z: torch.Tensor, step_idx: int, total_steps: int) -> torch.Tensor:
        if not self.time_cond:
            pred = self.unet(z)
        else:
            timestep = self._step_timestep(step_idx, total_steps, z.shape[0])
            pred = self.unet(z, timestep=timestep)

        if self.unet_prediction_mode == 'delta':
            # 预测位移：z_next = z + delta
            z_next = z + self.delta_scale * pred
        else:
            # 预测绝对 latent：z_next 向 pred 前进
            z_next = z + self.delta_scale * (pred - z)

        if self.unet_output_blend < 1.0:
            z_next = z + self.unet_output_blend * (z_next - z)
        return z_next

    @torch.no_grad()
    def _run_preflight_vae_resample(self, val_loader):
        """
        训练开始前记录 baseline：
          1) attack_raw_acc：直接 WM_dec(fake_images)
          2) stage1_acc：decode(E(fake), ref=fake) 后再 WM_dec
          3) acc_ref_wm / acc_ref_fake：同一个 E(wm) 在不同 reference 下的可解码性
        覆盖 identity + 各 deepfake attack。
        """
        if not self.preflight_enabled:
            if self.main:
                self.logger.info('Preflight VAE-resample eval: disabled by config.')
            return

        attack_items = [('identity', None)] + sorted(self.attacks.items(), key=lambda x: x[0])
        max_batches = self.preflight_max_batches

        if self.main:
            self.logger.info('=' * 72)
            self.logger.info('Preflight baseline (NO U-Net) started')
            self.logger.info(
                f'preflight_max_batches={max_batches if max_batches > 0 else "all"}'
            )

        for attack_name, _ in attack_items:
            local = torch.zeros(5, device=self.device)
            # [0]=attack_raw_correct_bits
            # [1]=stage1_correct_bits (decode(E(fake), ref=fake))
            # [2]=acc_ref_wm_correct_bits (decode(E(wm), ref=wm))
            # [3]=acc_ref_fake_correct_bits (decode(E(wm), ref=fake))
            # [4]=total_bits
            for i, batch in enumerate(val_loader):
                if max_batches > 0 and i >= max_batches:
                    break

                images = batch['image'].to(self.device)
                messages = torch.randint(
                    0, 2,
                    (images.shape[0], self.wm_adapter.message_length),
                    dtype=torch.float32, device=self.device,
                )
                wm_images = self.wm_adapter.encode(images, messages)
                if attack_name == 'identity':
                    attacked = wm_images
                else:
                    attacked = self._apply_attack(attack_name, wm_images, images, batch)

                attack_raw_logits = self.wm_adapter.decode(attacked)
                z_fake, _ = self.vae.encode(attacked)
                stage1_img = self._decode_latent(z_fake, attacked)
                stage1_logits = self.wm_adapter.decode(stage1_img)

                z_wm, _ = self.vae.encode(wm_images)
                x_ref_wm = self._decode_latent(z_wm, wm_images)
                x_ref_fake = self._decode_latent(z_wm, attacked)
                logits_ref_wm = self.wm_adapter.decode(x_ref_wm)
                logits_ref_fake = self.wm_adapter.decode(x_ref_fake)

                local[0] += (attack_raw_logits > 0).float().eq(messages).float().sum()
                local[1] += (stage1_logits > 0).float().eq(messages).float().sum()
                local[2] += (logits_ref_wm > 0).float().eq(messages).float().sum()
                local[3] += (logits_ref_fake > 0).float().eq(messages).float().sum()
                local[4] += float(messages.numel())

            if self.world_size > 1:
                dist.all_reduce(local, op=dist.ReduceOp.SUM)

            total_bits = max(local[4].item(), 1.0)
            attack_raw_acc = local[0].item() / total_bits
            stage1_acc = local[1].item() / total_bits
            acc_ref_wm = local[2].item() / total_bits
            acc_ref_fake = local[3].item() / total_bits
            if self.main:
                self.logger.info(
                    f'[Preflight-NoUNet] attack={attack_name:>12s}  '
                    f'attack_raw_acc={attack_raw_acc:.4f}  stage1_acc={stage1_acc:.4f}  '
                    f'acc_ref_wm={acc_ref_wm:.4f}  acc_ref_fake={acc_ref_fake:.4f}'
                )

        if self.main:
            self.logger.info('=' * 72)

    # ── 单步训练 ──────────────────────────────────────────────────────────────

    def _train_step(self, batch: dict, attack_name: str,
                    messages: torch.Tensor, epoch: int,
                    schedule: dict = None) -> dict:
        images = batch['image'].to(self.device)
        runtime = schedule if schedule is not None else self._runtime_schedule(epoch, with_step_weights=True)
        n_steps = int(runtime['n_steps'])
        step_weights = runtime['step_weights']
        tf_prob = float(runtime['tf_prob'])
        l1_w = float(runtime['l1_weight'])
        bce_w = float(runtime['bce_weight'])
        progress_w = float(runtime['progress_weight'])
        terminal_anchor_w = float(runtime['terminal_anchor_weight'])
        direction_w = float(runtime['direction_weight'])
        step_size_w = float(runtime['step_size_weight'])
        move_floor_w = float(runtime['move_floor_weight'])
        bce_progress_w = float(runtime['bce_progress_weight'])

        # 1. 含水印图（no_grad，VAE encoder 无需梯度）
        with torch.no_grad():
            wm_images  = self.wm_adapter.encode(images, messages)
            fake_images = self._apply_attack(attack_name, wm_images, images, batch)

        # 2. 编码两者到 latent（VAE encoder 冻结）
        with torch.no_grad():
            z_gt,  _ = self.vae.encode(wm_images)   # target: 含水印 latent
            z_src, _ = self.vae.encode(fake_images)  # source: 受损 latent

        # 3. SLERP 轨迹
        trajectory = build_slerp_trajectory(z_src, z_gt, n_steps)

        # 4. 轨迹学习（teacher forcing + 自回归 rollout）
        # 主监督完全对齐真实推理链路：
        #   z_pred -> decode(ref=fake_images) -> image/bce/progress
        total_latent_l1 = torch.tensor(0.0, device=self.device)
        total_img_l1 = torch.tensor(0.0, device=self.device)
        total_bce = torch.tensor(0.0, device=self.device)
        total_bce_mid = torch.tensor(0.0, device=self.device)
        total_bce_final = torch.tensor(0.0, device=self.device)
        total_bce_contrib = torch.tensor(0.0, device=self.device)
        total_dir = torch.tensor(0.0, device=self.device)
        total_prog = torch.tensor(0.0, device=self.device)
        total_stepmag = torch.tensor(0.0, device=self.device)
        total_dir_cos = torch.tensor(0.0, device=self.device)
        total_toward = torch.tensor(0.0, device=self.device)
        total_move_floor = torch.tensor(0.0, device=self.device)
        total_bce_prog = torch.tensor(0.0, device=self.device)
        total_anchor = torch.tensor(0.0, device=self.device)
        total_stepmove = torch.tensor(0.0, device=self.device)
        total_target_stepmove = torch.tensor(0.0, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        total_backprop_loss = torch.tensor(0.0, device=self.device)
        mid_count = 0
        final_count = 0
        z_roll = trajectory[0]
        use_short_bptt = (
            self.enable_phase_schedule
            and (not self.use_three_phase_schedule)
            and epoch >= self.phase_a_epochs
            and self.bptt_horizon > 0
            and not self.per_step_update
        )
        bptt_start = max(n_steps - self.bptt_horizon, 0)
        if not self.per_step_update:
            self.optimizer.zero_grad(set_to_none=True)

        def _forward_step_terms(z_k: torch.Tensor, z_teacher_next: torch.Tensor, step_idx: int):
            with autocast(enabled=self.use_amp):
                z_pred_local = self._unet_step(z_k, step_idx, n_steps)

            step_l1_w_local = step_weights[step_idx]
            if step_idx == n_steps - 1:
                step_l1_w_local = step_l1_w_local * self.final_step_l1_weight

            # 训练与推理保持一致：residual decode 均以 fake_images 作为 reference
            x_pred_local = self._decode_latent(z_pred_local, fake_images)
            wm_logits_local = self.wm_adapter.decode(x_pred_local)
            bce_step_local = F.binary_cross_entropy_with_logits(wm_logits_local, messages)
            img_l1_step_local = F.l1_loss(x_pred_local, wm_images)

            step_bce_w_local = step_weights[step_idx]
            if step_idx == n_steps - 1:
                step_bce_w_local = (
                    step_bce_w_local
                    * self.bce_final_weight
                    * self.final_step_bce_weight
                )
            else:
                step_bce_w_local = step_bce_w_local * self.bce_intermediate_weight

            step_move_local = torch.mean(torch.abs(z_pred_local - z_k), dim=(1, 2, 3))
            move_floor_loss_local = F.relu(self.move_floor_min_step - step_move_local).mean()

            # image-domain progress（不再使用 latent progress / latent trajectory loss）
            with torch.no_grad():
                x_curr_local = self._decode_latent(z_k.detach(), fake_images)
            dist_before_local = torch.mean(torch.abs(x_curr_local - wm_images), dim=(1, 2, 3))
            dist_after_local = torch.mean(torch.abs(x_pred_local - wm_images), dim=(1, 2, 3))
            prog_loss_local = F.relu(dist_after_local - dist_before_local + self.progress_margin).mean()
            toward_local = (dist_after_local < dist_before_local).float().mean()

            if step_idx == n_steps - 1:
                terminal_anchor_loss_local = F.l1_loss(z_pred_local, z_gt)
            else:
                terminal_anchor_loss_local = torch.tensor(0.0, device=self.device)

            latent_l1_step_local = F.l1_loss(z_pred_local, z_gt)
            recon_loss_local = l1_w * self.image_l1_weight * step_l1_w_local * img_l1_step_local
            if self.supervision_mode == 'latent':
                # 兼容旧模式：不推荐；仍避免把 trajectory 作为强目标
                recon_loss_local = recon_loss_local + self.latent_aux_weight * step_l1_w_local * latent_l1_step_local

            # 保留日志字段兼容，但 latent-direction/step-size/bce-progress 不参与训练
            dir_loss_local = torch.tensor(0.0, device=self.device)
            dir_cos_local = torch.tensor(0.0, device=self.device)
            step_size_loss_local = torch.tensor(0.0, device=self.device)
            bce_progress_loss_local = torch.tensor(0.0, device=self.device)
            target_step_move_local = torch.mean(torch.abs(z_teacher_next - z_k), dim=(1, 2, 3)).mean()

            step_aux_w_local = step_weights[step_idx]
            return {
                'z_pred': z_pred_local,
                'latent_l1_step': latent_l1_step_local,
                'img_l1_step': img_l1_step_local,
                'recon_loss': recon_loss_local,
                'bce_step': bce_step_local,
                'step_l1_w': step_l1_w_local,
                'step_bce_w': step_bce_w_local,
                'step_aux_w': step_aux_w_local,
                'move_floor_loss': move_floor_loss_local,
                'bce_progress_loss': bce_progress_loss_local,
                'terminal_anchor_loss': terminal_anchor_loss_local,
                'dir_loss': dir_loss_local,
                'prog_loss': prog_loss_local,
                'step_size_loss': step_size_loss_local,
                'dir_cos': dir_cos_local,
                'toward': toward_local,
                'step_move': step_move_local,
                'target_step_move': target_step_move_local,
            }

        for k in range(n_steps):
            z_teacher_next = trajectory[k + 1]
            if k == 0:
                z_k = trajectory[k]
            else:
                use_teacher = random.random() < tf_prob
                z_k = trajectory[k] if use_teacher else z_roll
            if self.latent_noise_std > 0:
                z_k = z_k + torch.randn_like(z_k) * self.latent_noise_std

            if self.per_step_update and self.separate_bce_step:
                # A) BCE-only step：单独优化 message 路径，避免被 latent 路径淹没
                terms_bce = _forward_step_terms(z_k, z_teacher_next, k)
                bce_only_loss = self.bce_only_step_scale * (
                    bce_w * terms_bce['step_bce_w'] * terms_bce['bce_step']
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(bce_only_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # B) Main step：优化 image-domain rollout 几何
                terms_main = _forward_step_terms(z_k, z_teacher_next, k)
                main_step_loss = (
                    terms_main['recon_loss']
                    + move_floor_w * terms_main['step_aux_w'] * terms_main['move_floor_loss']
                    + terminal_anchor_w * terms_main['terminal_anchor_loss']
                    + progress_w * terms_main['step_aux_w'] * terms_main['prog_loss']
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(main_step_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                step_loss = main_step_loss + bce_only_loss
                terms_for_log = terms_main
                bce_for_log = terms_bce
            else:
                terms_for_log = _forward_step_terms(z_k, z_teacher_next, k)
                step_loss = (
                    terms_for_log['recon_loss']
                    + bce_w * terms_for_log['step_bce_w'] * terms_for_log['bce_step']
                    + move_floor_w * terms_for_log['step_aux_w'] * terms_for_log['move_floor_loss']
                    + terminal_anchor_w * terms_for_log['terminal_anchor_loss']
                    + progress_w * terms_for_log['step_aux_w'] * terms_for_log['prog_loss']
                )
                if self.per_step_update:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(step_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_backprop_loss = total_backprop_loss + step_loss
                bce_for_log = terms_for_log

            total_latent_l1 = total_latent_l1 + terms_for_log['step_l1_w'].detach() * terms_for_log['latent_l1_step'].detach()
            total_img_l1 = total_img_l1 + terms_for_log['step_l1_w'].detach() * terms_for_log['img_l1_step'].detach()
            total_bce = total_bce + bce_for_log['step_bce_w'].detach() * bce_for_log['bce_step'].detach()
            bce_contrib_step = bce_w * bce_for_log['step_bce_w'].detach() * bce_for_log['bce_step'].detach()
            total_bce_contrib = total_bce_contrib + bce_contrib_step
            if k == (n_steps - 1):
                total_bce_final = total_bce_final + bce_for_log['bce_step'].detach()
                final_count += 1
            else:
                total_bce_mid = total_bce_mid + bce_for_log['bce_step'].detach()
                mid_count += 1
            total_dir = total_dir + terms_for_log['step_aux_w'].detach() * terms_for_log['dir_loss'].detach()
            total_prog = total_prog + terms_for_log['step_aux_w'].detach() * terms_for_log['prog_loss'].detach()
            total_stepmag = total_stepmag + terms_for_log['step_aux_w'].detach() * terms_for_log['step_size_loss'].detach()
            total_dir_cos = total_dir_cos + terms_for_log['step_aux_w'].detach() * terms_for_log['dir_cos'].mean().detach()
            total_toward = total_toward + terms_for_log['step_aux_w'].detach() * terms_for_log['toward'].detach()
            total_move_floor = total_move_floor + terms_for_log['step_aux_w'].detach() * terms_for_log['move_floor_loss'].detach()
            total_bce_prog = total_bce_prog + terms_for_log['step_aux_w'].detach() * bce_for_log['bce_progress_loss'].detach()
            total_anchor = total_anchor + terms_for_log['terminal_anchor_loss'].detach()
            total_stepmove = total_stepmove + terms_for_log['step_aux_w'].detach() * terms_for_log['step_move'].mean().detach()
            total_target_stepmove = total_target_stepmove + terms_for_log['step_aux_w'].detach() * terms_for_log['target_step_move'].detach()
            total_loss = total_loss + step_loss.detach()

            z_pred_for_roll = terms_for_log['z_pred']
            if use_short_bptt:
                if (k + 1) < bptt_start:
                    z_roll = z_pred_for_roll.detach()
                else:
                    z_roll = z_pred_for_roll
            elif self.bptt_rollout:
                z_roll = z_pred_for_roll
            else:
                z_roll = z_pred_for_roll.detach()

        if not self.per_step_update:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_backprop_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        step_move_mean = (total_stepmove / max(n_steps, 1))
        target_step_move_mean = (total_target_stepmove / max(n_steps, 1))
        step_move_ratio = step_move_mean / target_step_move_mean.clamp(min=1e-8)

        return {
            'loss': (total_loss / max(n_steps, 1)).item(),
            'l1':   (total_latent_l1 / max(n_steps, 1)).item(),
            'latent_l1': (total_latent_l1 / max(n_steps, 1)).item(),
            'img_l1': (total_img_l1 / max(n_steps, 1)).item(),
            'bce':  (total_bce / max(n_steps, 1)).item(),
            'bce_intermediate': (total_bce_mid / max(mid_count, 1)).item(),
            'bce_final': (total_bce_final / max(final_count, 1)).item(),
            'bce_contrib': (total_bce_contrib / max(n_steps, 1)).item(),
            'dir':  (total_dir / max(n_steps, 1)).item(),
            'prog': (total_prog / max(n_steps, 1)).item(),
            'stepmag': (total_stepmag / max(n_steps, 1)).item(),
            'dir_cos': (total_dir_cos / max(n_steps, 1)).item(),
            'toward': (total_toward / max(n_steps, 1)).item(),
            'move_floor': (total_move_floor / max(n_steps, 1)).item(),
            'bce_prog': (total_bce_prog / max(n_steps, 1)).item(),
            'anchor': (total_anchor / max(n_steps, 1)).item(),
            'step_move': step_move_mean.item(),
            'target_step_move': target_step_move_mean.item(),
            'step_move_ratio': step_move_ratio.item(),
            'short_bptt': float(use_short_bptt),
            'tf_prob': tf_prob,
            'n_steps': float(n_steps),
        }

    # ── 验证（自回归推理）────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_step(self, batch: dict, attack_name: str,
                  messages: torch.Tensor,
                  infer_steps: int = None) -> dict:
        images = batch['image'].to(self.device)
        infer_steps = self.max_infer if infer_steps is None else max(int(infer_steps), 1)

        wm_images   = self.wm_adapter.encode(images, messages)
        fake_images = self._apply_attack(attack_name, wm_images, images, batch)
        attack_raw_logits = self.wm_adapter.decode(fake_images)
        attack_raw_acc = bit_acc(attack_raw_logits, messages)

        # Stage1-only 基线：VAE 重采样，不经过 U-Net
        z_src, _ = self.vae.encode(fake_images)
        x_stage1 = self._decode_latent(z_src, fake_images)
        stage1_logits = self.wm_adapter.decode(x_stage1)
        stage1_acc = bit_acc(stage1_logits, messages)

        # reference-sensitivity 诊断：同一个 z_wm 在不同 ref 下可解码性差异
        z_wm, _ = self.vae.encode(wm_images)
        x_ref_wm = self._decode_latent(z_wm, wm_images)
        x_ref_fake = self._decode_latent(z_wm, fake_images)
        acc_ref_wm = bit_acc(self.wm_adapter.decode(x_ref_wm), messages)
        acc_ref_fake = bit_acc(self.wm_adapter.decode(x_ref_fake), messages)

        # Stage 2 自回归推理：从 z_src 出发，迭代 infer_steps 步
        z = z_src
        for k in range(infer_steps):
            z = self._unet_step(z, k, infer_steps)

        x_hat = self._decode_latent(z, fake_images)
        stage2_logits = self.wm_adapter.decode(x_hat)
        stage2_acc = bit_acc(stage2_logits, messages)

        # Latent L1（近似重建质量）
        z_gt, _ = self.vae.encode(wm_images)
        latent_l1 = F.l1_loss(z, z_gt).item()
        img_l1 = F.l1_loss(x_hat, wm_images).item()

        return {
            'attack_raw_acc': attack_raw_acc,
            'stage1_acc': stage1_acc,
            'stage2_acc': stage2_acc,
            'acc_ref_wm': acc_ref_wm,
            'acc_ref_fake': acc_ref_fake,
            'latent_l1': latent_l1,
            'img_l1': img_l1,
            # backward compatibility
            'raw_acc': stage1_acc,
            'acc': stage2_acc,
            'l1': latent_l1,
        }

    @torch.no_grad()
    def _val_clean_step(self, batch: dict, messages: torch.Tensor,
                        infer_steps: int = None) -> dict:
        """验证含水印图像经 VAE encode→Stage2→decode 后 ACC 是否保持。"""
        images = batch['image'].to(self.device)
        wm_images = self.wm_adapter.encode(images, messages)
        infer_steps = self.max_infer if infer_steps is None else max(int(infer_steps), 1)

        z, _ = self.vae.encode(wm_images)
        for k in range(infer_steps):
            z = self._unet_step(z, k, infer_steps)
        x_hat = self._decode_latent(z, wm_images)
        wm_logits = self.wm_adapter.decode(x_hat)
        return {'acc': bit_acc(wm_logits, messages)}

    @torch.no_grad()
    def _rollout_curve_step(self, batch: dict, attack_name: str,
                            messages: torch.Tensor,
                            infer_steps: int = None) -> np.ndarray:
        """仅记录自回归推理曲线（step 0..N）。"""
        images = batch['image'].to(self.device)
        wm_images = self.wm_adapter.encode(images, messages)
        fake_images = self._apply_attack(attack_name, wm_images, images, batch)
        infer_steps = self.max_infer if infer_steps is None else max(int(infer_steps), 1)

        z_src, _ = self.vae.encode(fake_images)

        infer_accs = []
        z = z_src
        x = self._decode_latent(z, fake_images)
        logits = self.wm_adapter.decode(x)
        infer_accs.append(bit_acc(logits, messages))

        for k in range(infer_steps):
            z = self._unet_step(z, k, infer_steps)
            x = self._decode_latent(z, fake_images)
            logits = self.wm_adapter.decode(x)
            infer_accs.append(bit_acc(logits, messages))

        return np.asarray(infer_accs, dtype=np.float32)

    def _log_rollout_curve(self, split: str, attack_name: str, curve: np.ndarray):
        if not self.main:
            return
        self.logger.info(f'rollout_curve[{split}] attack={attack_name}')
        if curve.size >= 2:
            uplift = float(curve[-1] - curve[0])
            self.logger.info(
                f'rollout_uplift[{split}] attack={attack_name}  '
                f'step0={curve[0]:.4f}  step{curve.size - 1}={curve[-1]:.4f}  '
                f'uplift={uplift:+.4f}'
            )
        for i, acc in enumerate(curve.tolist()):
            if i == 0:
                suffix = '(step0=VAE-only)'
            elif i == (curve.size - 1):
                suffix = '(last_step)'
            else:
                suffix = ''
            self.logger.info(
                f'rollout_curve[{split}] step {i}{suffix}: acc={acc:.4f}'
            )

    # ── 样例图 ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _save_sample(self, batch: dict, attack_name: str,
                     messages: torch.Tensor, epoch: int,
                     infer_steps: int = None):
        images = batch['image'].to(self.device)
        wm_images   = self.wm_adapter.encode(images, messages)
        fake_images = self._apply_attack(attack_name, wm_images, images, batch)
        infer_steps = self.max_infer if infer_steps is None else max(int(infer_steps), 1)

        z, _ = self.vae.encode(fake_images)
        for k in range(infer_steps):
            z = self._unet_step(z, k, infer_steps)
        x_hat = self._decode_latent(z, fake_images)

        self.logger.save_sample({
            'orig':    images,
            'wm':      wm_images,
            'fake':    fake_images,
            'refined': x_hat,
        }, epoch=epoch)

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader, train_sampler):
        cfg = self.cfg

        attack_names = list(self.attacks.keys())
        if not attack_names:
            raise RuntimeError('没有可用的攻击模型，请在 config attacks.online 中配置。')
        attack_weights = [max(self.attack_sample_weights.get(n, 1.0), 0.0) for n in attack_names]
        if sum(attack_weights) <= 0:
            attack_weights = [1.0 for _ in attack_names]

        self._run_preflight_vae_resample(val_loader)
        if self.world_size > 1:
            dist.barrier()

        for epoch in range(self.start_epoch, cfg.training.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            runtime = self._runtime_schedule(epoch, with_step_weights=True)
            self._set_optimizer_lr(runtime['lr'])
            infer_steps = int(runtime['n_steps'])
            if self.main:
                self.logger.info(
                    f'Epoch {epoch:03d} schedule: '
                    f'autoregressive_train=on  '
                    f'phase={runtime["phase"]}  '
                    f'n_steps={runtime["n_steps"]}  '
                    f'tf_prob={runtime["tf_prob"]:.3f}  '
                    f'l1_w={runtime["l1_weight"]:.3f}  '
                    f'bce_w={runtime["bce_weight"]:.3f}  '
                    f'progress_w={runtime["progress_weight"]:.3f}  '
                    f'anchor_w={runtime["terminal_anchor_weight"]:.3f}  '
                    f'lr={runtime["lr"]:.2e}  '
                    f'infer_steps={infer_steps}  '
                    f'short_bptt_horizon={self.bptt_horizon}  '
                    f'tail={"on" if runtime["tail_enabled"] else "off"}'
                )

            # ── Train ───────────────────────────────────────────────────────
            self._unet_module.train()
            train_metrics = {
                'loss': [], 'l1': [], 'latent_l1': [], 'img_l1': [], 'bce': [],
                'bce_intermediate': [], 'bce_final': [], 'bce_contrib': [],
                'dir': [], 'prog': [], 'stepmag': [], 'dir_cos': [], 'toward': [],
                'move_floor': [], 'bce_prog': [], 'anchor': [], 'step_move': [],
                'target_step_move': [], 'step_move_ratio': [],
                'short_bptt': [],
                'tf_prob': [],
                'n_steps': [],
            }
            train_start = time.time()
            train_steps = len(train_loader)

            for step_idx, batch in enumerate(train_loader):
                attack_name = random.choices(attack_names, weights=attack_weights, k=1)[0]
                messages = torch.randint(0, 2, (batch['image'].shape[0],
                                                self.wm_adapter.message_length),
                                         dtype=torch.float32, device=self.device)
                step_metrics = self._train_step(
                    batch, attack_name, messages, epoch, schedule=runtime
                )
                for k, v in step_metrics.items():
                    train_metrics[k].append(v)
                self._log_progress(
                    epoch=epoch,
                    phase='train',
                    step_idx=step_idx,
                    total_steps=train_steps,
                    start_ts=train_start,
                )

            train_summary = {
                k: float(np.nanmean(v)) for k, v in train_metrics.items()
            }
            train_summary = self._reduce_metrics(train_summary)

            # ── Val ─────────────────────────────────────────────────────────
            val_freq = getattr(cfg.training, 'val_freq', 1)
            if (epoch % val_freq) == 0:
                self._unet_module.eval()
                val_max = getattr(getattr(cfg, 'validation', None), 'max_batches', 8)
                val_total = len(val_loader)
                if val_max:
                    val_total = min(val_total, val_max)

                # clean 验证：wm → Stage2 → wm (ACC 不应下降)
                clean_metrics = {'acc': []}
                val_clean_start = time.time()
                for i, batch in enumerate(val_loader):
                    if val_max and i >= val_max:
                        break
                    messages = torch.randint(
                        0, 2,
                        (batch['image'].shape[0], self.wm_adapter.message_length),
                        dtype=torch.float32, device=self.device,
                    )
                    m = self._val_clean_step(batch, messages, infer_steps=infer_steps)
                    clean_metrics['acc'].append(m['acc'])
                    self._log_progress(
                        epoch=epoch,
                        phase='val-clean',
                        step_idx=i,
                        total_steps=val_total,
                        start_ts=val_clean_start,
                    )
                clean_summary = {k: float(np.mean(v)) for k, v in clean_metrics.items()}
                clean_summary = self._reduce_metrics(clean_summary)

                # attack 验证：fake → Stage2 → 检测 ACC 恢复
                val_attacks = {}
                for attack_name in attack_names:
                    atk_metrics = {
                        'attack_raw_acc': [], 'stage1_acc': [], 'stage2_acc': [],
                        'acc_ref_wm': [], 'acc_ref_fake': [],
                        'latent_l1': [], 'img_l1': [],
                        # backward compatibility
                        'raw_acc': [], 'acc': [], 'l1': [],
                    }
                    val_atk_start = time.time()
                    for i, batch in enumerate(val_loader):
                        if val_max and i >= val_max:
                            break
                        messages = torch.randint(
                            0, 2,
                            (batch['image'].shape[0], self.wm_adapter.message_length),
                            dtype=torch.float32, device=self.device,
                        )
                        m = self._val_step(batch, attack_name, messages, infer_steps=infer_steps)
                        for k, v in m.items():
                            atk_metrics[k].append(v)
                        self._log_progress(
                            epoch=epoch,
                            phase=f'val-attack:{attack_name}',
                            step_idx=i,
                            total_steps=val_total,
                            start_ts=val_atk_start,
                        )
                    val_attacks[attack_name] = {
                        k: float(np.mean(v)) for k, v in atk_metrics.items()
                    }
                    val_attacks[attack_name] = self._reduce_metrics(val_attacks[attack_name])

                # rollout 曲线（val）
                val_infer_curves = {}
                if self.curve_enabled:
                    curve_max = self.curve_max_batches if self.curve_max_batches > 0 else val_max
                    for attack_name in attack_names:
                        infer_sum = np.zeros(infer_steps + 1, dtype=np.float64)
                        n_curve = 0
                        val_curve_start = time.time()
                        for i, batch in enumerate(val_loader):
                            if curve_max and i >= curve_max:
                                break
                            messages = torch.randint(
                                0, 2,
                                (batch['image'].shape[0], self.wm_adapter.message_length),
                                dtype=torch.float32, device=self.device,
                            )
                            infer_curve = self._rollout_curve_step(
                                batch, attack_name, messages, infer_steps=infer_steps
                            )
                            infer_sum += infer_curve
                            n_curve += 1
                            self._log_progress(
                                epoch=epoch,
                                phase=f'val-infer-curve:{attack_name}',
                                step_idx=i,
                                total_steps=curve_max,
                                start_ts=val_curve_start,
                            )
                        if n_curve > 0:
                            infer_mean = (infer_sum / n_curve).astype(np.float32)
                            val_infer_curves[attack_name] = self._reduce_vector(infer_mean)

                # rollout 曲线（train probe）
                train_probe_infer_curves = {}
                if self.train_probe_enabled:
                    probe_max = self.train_probe_curve_max_batches
                    if probe_max <= 0:
                        probe_max = self.train_probe_max_batches if self.train_probe_max_batches > 0 else self.curve_max_batches
                    probe_total = len(train_loader)
                    if probe_max > 0:
                        probe_total = min(probe_total, probe_max)
                    for attack_name in attack_names:
                        infer_sum = np.zeros(infer_steps + 1, dtype=np.float64)
                        n_curve = 0
                        probe_curve_start = time.time()
                        for i, batch in enumerate(train_loader):
                            if probe_max > 0 and i >= probe_max:
                                break
                            messages = torch.randint(
                                0, 2,
                                (batch['image'].shape[0], self.wm_adapter.message_length),
                                dtype=torch.float32, device=self.device,
                            )
                            infer_curve = self._rollout_curve_step(
                                batch, attack_name, messages, infer_steps=infer_steps
                            )
                            infer_sum += infer_curve
                            n_curve += 1
                            self._log_progress(
                                epoch=epoch,
                                phase=f'train-rollout-curve:{attack_name}',
                                step_idx=i,
                                total_steps=probe_total,
                                start_ts=probe_curve_start,
                            )
                        if n_curve > 0:
                            infer_mean = (infer_sum / n_curve).astype(np.float32)
                            train_probe_infer_curves[attack_name] = self._reduce_vector(infer_mean)

                if self.use_three_phase_schedule and runtime['phase'] == 'phase1':
                    if val_infer_curves:
                        step_idx = self.phase1_target_step
                        max_valid_step = min(int(c.shape[0] - 1) for c in val_infer_curves.values())
                        step_idx = min(step_idx, max_valid_step)
                        step_uplifts = [float(c[step_idx] - c[0]) for c in val_infer_curves.values()]
                        avg_step_uplift = float(np.mean(step_uplifts))
                        if self.main:
                            self.logger.info(
                                f'phase1_gate_check: step{step_idx}_uplift={avg_step_uplift:+.4f}  '
                                f'target={self.phase1_target_uplift:+.4f}  '
                                f'met={avg_step_uplift >= self.phase1_target_uplift}'
                            )
                        if self.phase1_gate_enabled and avg_step_uplift >= self.phase1_target_uplift:
                            self.phase1_gate_met = True
                    elif self.main and self.phase1_gate_enabled:
                        self.logger.warning(
                            'phase1 gate enabled but val rollout curve is empty; '
                            'will fallback to phase1_max_epochs.'
                        )

                if self.main:
                    self.logger.log_epoch(
                        epoch=epoch,
                        train=train_summary,
                        val_clean=clean_summary,
                        val_attacks=val_attacks,
                    )
                    self.logger.info(
                        'rollout_dynamics[train] '
                        f'step_move={train_summary.get("step_move", float("nan")):.4f}  '
                        f'target_step_move={train_summary.get("target_step_move", float("nan")):.4f}  '
                        f'step_move_ratio={train_summary.get("step_move_ratio", float("nan")):.4f}  '
                        f'toward={train_summary.get("toward", float("nan")):.4f}'
                    )
                    for attack_name, metrics in sorted(val_attacks.items()):
                        self.logger.info(
                            f'val_breakdown[{attack_name}] '
                            f'attack_raw_acc={metrics.get("attack_raw_acc", float("nan")):.4f}  '
                            f'stage1_acc={metrics.get("stage1_acc", float("nan")):.4f}  '
                            f'stage2_acc={metrics.get("stage2_acc", float("nan")):.4f}  '
                            f'acc_ref_wm={metrics.get("acc_ref_wm", float("nan")):.4f}  '
                            f'acc_ref_fake={metrics.get("acc_ref_fake", float("nan")):.4f}'
                        )
                    if self.log_rollout_curve:
                        for attack_name, curve in train_probe_infer_curves.items():
                            self._log_rollout_curve('train', attack_name, curve)
                        for attack_name, curve in val_infer_curves.items():
                            self._log_rollout_curve('val', attack_name, curve)

                    def _curve_summary(curve_map):
                        if not curve_map:
                            return float('nan'), float('nan')
                        uplifts = [float(c[-1] - c[0]) for c in curve_map.values()]
                        step_last = [float(c[-1]) for c in curve_map.values()]
                        return float(np.mean(uplifts)), float(np.mean(step_last))

                    train_uplift, train_step_last = _curve_summary(train_probe_infer_curves)
                    val_uplift, val_step_last = _curve_summary(val_infer_curves)
                    if not np.isnan(train_uplift):
                        self.logger.info(
                            f'epoch_uplift[train] avg_uplift={train_uplift:+.4f}  '
                            f'avg_step_last={train_step_last:.4f}'
                        )
                    if not np.isnan(val_uplift):
                        self.logger.info(
                            f'epoch_uplift[val] avg_uplift={val_uplift:+.4f}  '
                            f'avg_step_last={val_step_last:.4f}'
                        )

                    # 样例图
                    sample_batch = next(iter(val_loader))
                    sample_msg   = torch.randint(
                        0, 2,
                        (sample_batch['image'].shape[0],
                         self.wm_adapter.message_length),
                        dtype=torch.float32, device=self.device,
                    )
                    self._save_sample(
                        sample_batch, attack_names[0], sample_msg, epoch,
                        infer_steps=infer_steps
                    )

                    # Best checkpoint（优先 val_uplift，再看 val_step_last）
                    if val_infer_curves:
                        avg_val_uplift = float(np.mean(
                            [float(c[-1] - c[0]) for c in val_infer_curves.values()]
                        ))
                        avg_val_step_last = float(np.mean(
                            [float(c[-1]) for c in val_infer_curves.values()]
                        ))
                    else:
                        avg_val_uplift = float(np.mean(
                            [v['stage2_acc'] - v['stage1_acc'] for v in val_attacks.values()]
                        ))
                        avg_val_step_last = float(np.mean(
                            [v['stage2_acc'] for v in val_attacks.values()]
                        ))

                    eps = 1e-8
                    better_uplift = avg_val_uplift > (self.best_uplift + eps)
                    tie_uplift = abs(avg_val_uplift - self.best_uplift) <= eps
                    better_step_last = avg_val_step_last > (self.best_step10_acc + eps)
                    if better_uplift or (tie_uplift and better_step_last):
                        self.best_uplift = avg_val_uplift
                        self.best_step10_acc = avg_val_step_last
                        self._save_checkpoint(epoch, 'best')
                        self.logger.info(
                            f'  >> New best: val_uplift={self.best_uplift:+.4f}  '
                            f'val_step_last={self.best_step10_acc:.4f}  '
                            f'→ checkpoints/unet/best.pth'
                        )

            # ── 定期保存 ────────────────────────────────────────────────────
            if self.main:
                self._save_checkpoint(epoch, 'last')
                save_freq = getattr(cfg.training, 'save_freq', 10)
                if (epoch + 1) % save_freq == 0:
                    self._save_checkpoint(epoch, f'epoch_{epoch:03d}')

            if self.world_size > 1:
                dist.barrier()

            if self.scheduler is not None:
                self.scheduler.step()

        if self.main:
            self.logger.info('Training complete.')
            self.logger.close()


# ── Stage 1 config 推断 ───────────────────────────────────────────────────────

def _infer_stage1_cfg(stage1_ckpt_path: str, cfg_s2):
    """
    尝试从以下位置按优先级加载 Stage 1 配置：
      1. <ckpt_stem>.config.yaml（同名 config 快照，导出 checkpoint 时附带）
      2. runs/<run>/config.yaml（标准 run 目录结构）
      3. stage2_unet.yaml 中的 stage1_model 字段（手动 fallback）
    """
    import yaml
    from types import SimpleNamespace

    def _to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [_to_ns(i) for i in d]
        return d

    def _load_yaml_ns(path):
        with open(path) as f:
            return _to_ns(yaml.safe_load(f))

    # 优先级 1：同名 .config.yaml（如 best_20260310.pth → best_20260310.config.yaml）
    stem = os.path.splitext(stage1_ckpt_path)[0]
    sibling_cfg = stem + '.config.yaml'
    if os.path.isfile(sibling_cfg):
        return _load_yaml_ns(sibling_cfg)

    # 优先级 2：runs/<run>/config.yaml 快照
    ckpt_dir  = os.path.dirname(stage1_ckpt_path)           # .../checkpoints/vae
    run_dir   = os.path.dirname(os.path.dirname(ckpt_dir))  # .../runs/<run>
    snap_path = os.path.join(run_dir, 'config.yaml')

    if os.path.isfile(snap_path):
        return _load_yaml_ns(snap_path)

    # Fallback: 从 stage2_unet.yaml 读 stage1_model 字段
    s1_model_cfg = getattr(cfg_s2, 'stage1_model', None)
    if s1_model_cfg is not None:
        from types import SimpleNamespace
        return SimpleNamespace(model=s1_model_cfg)

    raise RuntimeError(
        f'无法确定 Stage 1 模型配置。\n'
        f'  尝试路径：{snap_path}（不存在）\n'
        f'  请在 stage2_unet.yaml 中添加 stage1_model 字段，\n'
        f'  或者确保 stage1_ckpt_path 在标准 runs/ 目录结构下。'
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='ReMark Stage 2: Latent U-Net Training')
    p.add_argument('--config',      required=True,
                   help='Stage 2 YAML 配置文件，如 configs/stage2_unet.yaml')
    p.add_argument('--override',    default=None,
                   help='实验 override 文件（可选）')
    p.add_argument('--stage1-ckpt', default=None,
                   help='Stage 1 VAE checkpoint 路径（覆盖 config 中的 paths.stage1_checkpoint）')
    p.add_argument('--resume',      default=None,
                   help='续训：已有 run 目录名（如 stage2_20260311_104200）')
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


def main():
    rank, local_rank, world_size = setup_ddp()
    args = parse_args()
    cfg  = load_config(args.config, args.override)
    set_seed(args.seed, rank)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available()
                          else 'cpu')

    # Stage 1 checkpoint 路径
    stage1_ckpt = (
        args.stage1_ckpt
        or getattr(getattr(cfg, 'paths', None), 'stage1_checkpoint', None)
    )
    if not stage1_ckpt or not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(
            f'Stage 1 checkpoint 不存在：{stage1_ckpt}\n'
            f'请通过 --stage1-ckpt 指定，或在 configs/stage2_unet.yaml 的 '
            f'paths.stage1_checkpoint 中配置。'
        )

    # 推断 Stage 1 模型配置
    cfg_s1 = _infer_stage1_cfg(stage1_ckpt, cfg)

    logger = None
    if is_main(rank):
        logger = RunLogger(
            cfg,
            config_path=args.config,
            override_path=args.override,
            stage='stage2',
            resume_run=args.resume,
        )
        logger.info(f'Run dir    : {logger.run_dir}')
        logger.info(f'Device     : {device}  |  world_size: {world_size}')
        logger.info(f'Stage1 ckpt: {stage1_ckpt}')

    train_loader, val_loader, train_sampler = build_loaders(cfg, rank, world_size)

    if is_main(rank):
        logger.info(f'Train: {len(train_loader.dataset)} samples  '
                    f'Val: {len(val_loader.dataset)} samples')

    resume_ckpt = None
    if args.resume:
        resume_ckpt = os.path.join(
            getattr(getattr(cfg, 'paths', None), 'runs_dir', 'runs'),
            args.resume, 'checkpoints', 'unet', 'last.pth',
        )

    trainer = Trainer(
        cfg, cfg_s1, stage1_ckpt,
        logger, device, rank, world_size,
        resume_path=resume_ckpt,
    )

    if is_main(rank):
        n_params = sum(p.numel() for p in trainer._unet_module.parameters())
        logger.info(f'U-Net: {n_params/1e6:.2f}M params')
        logger.info(f'SLERP n_steps={trainer.n_steps}  '
                    f'max_infer_steps={trainer.max_infer}  '
                    f'latent_noise_std={trainer.latent_noise_std:.4f}')
        logger.info(f'Timestep conditioning: enabled={trainer.time_cond}  '
                    f'time_min={trainer.time_min:.1f}  '
                    f'time_max={trainer.time_max:.1f}')
        logger.info(f'Autoregressive training: enabled=true  '
                    f'per_step_update={trainer.per_step_update}  '
                    f'separate_bce_step={trainer.separate_bce_step}  '
                    f'bce_only_step_scale={trainer.bce_only_step_scale:.2f}  '
                    f'bptt_rollout={trainer.bptt_rollout}  '
                    f'bptt_horizon={trainer.bptt_horizon}  '
                    f'phase_schedule={trainer.enable_phase_schedule}  '
                    f'phase_a_epochs={trainer.phase_a_epochs}  '
                    f'tf_warmup_epochs={trainer.tf_warmup_epochs}  '
                    f'tf_start={trainer.tf_start_prob:.2f}  '
                    f'tf_end={trainer.tf_end_prob:.2f}  '
                    f'phase_b_tf_end={trainer.phase_b_tf_end_prob:.2f}')
        logger.info(
            f'Supervision: mode={trainer.supervision_mode}  '
            f'image_l1_w={trainer.image_l1_weight:.3f}  '
            f'latent_aux_w={trainer.latent_aux_weight:.3f}'
        )
        if trainer.use_three_phase_schedule:
            logger.info(
                'Three-phase schedule: '
                f'phase1[min,max]=[{trainer.phase1_min_epochs},{trainer.phase1_max_epochs}]  '
                f'phase2_epochs={trainer.phase2_epochs}  '
                f'gate_step={trainer.phase1_target_step}  '
                f'gate_uplift_target={trainer.phase1_target_uplift:+.4f}  '
                f'phase_n_steps=({trainer.phase1_n_steps},{trainer.phase2_n_steps},{trainer.phase3_n_steps})  '
                f'phase2_tf_end={trainer.phase2_tf_end_prob:.2f}  '
                f'phase3_lr_scale={trainer.phase3_lr_scale:.2f}'
            )
        logger.info('Train/Infer decode reference: fixed fake_images (aligned)')
        logger.info(f'Tail weighting: enabled={trainer.tail_enabled}  '
                    f'last_k={trainer.tail_last_k}  '
                    f'weight={trainer.tail_weight:.3f}')
        logger.info(
            f'Step-weight normalization: normalize={trainer.normalize_step_weights}  '
            f'sum={trainer.step_weights.sum().item():.3f}  '
            f'min={trainer.step_weights.min().item():.3f}  '
            f'max={trainer.step_weights.max().item():.3f}'
        )
        logger.info(f'Final-step weighting: l1={trainer.final_step_l1_weight:.3f}  '
                    f'bce={trainer.final_step_bce_weight:.3f}')
        logger.info(
            f'Step-wise BCE weighting: intermediate={trainer.bce_intermediate_weight:.3f}  '
            f'final={trainer.bce_final_weight:.3f}'
        )
        logger.info(
            f'Direction losses: direction_w={trainer.direction_loss_weight:.3f}  '
            f'progress_w={trainer.progress_loss_weight:.3f}  '
            f'progress_margin={trainer.progress_margin:.4f}  '
            f'step_size_w={trainer.step_size_loss_weight:.3f}'
        )
        logger.info(
            f'Forward mode: {trainer.unet_prediction_mode}  delta_scale={trainer.delta_scale:.3f}  '
            f'unet_output_blend={trainer.unet_output_blend:.3f}'
        )
        logger.info(
            f'Extra losses: move_floor_w={trainer.move_floor_weight:.3f} '
            f'(min_step={trainer.move_floor_min_step:.4f})  '
            f'bce_progress_w={trainer.bce_progress_weight:.3f} '
            f'(margin={trainer.bce_progress_margin:.4f})  '
            f'terminal_anchor_w={trainer.terminal_anchor_weight:.3f}'
        )
        logger.info(f'Optimizer: lr={cfg.training.lr:.2e}  wd={trainer.weight_decay:.2e}  '
                    f'scheduler={"cosine" if trainer.use_lr_scheduler else "none"}')
        logger.info(
            'Point-curve logging: '
            f'rollout={trainer.log_rollout_curve}'
        )
        logger.info(
            'Train probe: '
            f'enabled={trainer.train_probe_enabled}  '
            f'max_batches={trainer.train_probe_max_batches}  '
            f'curve_max_batches={trainer.train_probe_curve_max_batches}'
        )

    trainer.train(train_loader, val_loader, train_sampler)
    cleanup_ddp()


if __name__ == '__main__':
    main()
