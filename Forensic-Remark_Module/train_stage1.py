"""
ReMark Stage 1 训练脚本 — VAE 图像修复（支持单卡 / DDP 多卡）

启动方式：
    # 单卡
    python train_stage1.py --config configs/stage1_vae.yaml

    # DDP 多卡（推荐）
    torchrun --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml

    # 续训
    torchrun --nproc_per_node=2 train_stage1.py \\
        --config configs/stage1_vae.yaml \\
        --resume stage1_20260308_2131
"""

import argparse
import csv
import os
import random
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
from tqdm import tqdm

from utils.config import load_config
from utils.message_bits import (
    deterministic_messages_from_paths,
    landmark_messages_from_paths,
)
from utils.logger import RunLogger
from data.dataset import ReMark_Dataset, build_dataloader
from network.vae import build_vae
from network.losses import LossComputer
from wm_adapters.registry import build_wm_adapter
from attacks.registry import build_attack, ATTACK_REGISTRY


# ── DDP 工具函数 ───────────────────────────────────────────────────────────────

def setup_ddp():
    """初始化进程组，返回 (rank, local_rank, world_size)。单卡时均返回 0/1。"""
    if 'RANK' not in os.environ:
        return 0, 0, 1   # 单卡直接运行

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


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def set_seed(seed: int, rank: int = 0):
    """每个 rank 使用不同种子，保证数据多样性"""
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _atomic_save_image(tensor: torch.Tensor, final_path: str, nrow: int):
    """
    原子写图，避免 VSCode 预览在写入中途读取导致“图片加载失败”。
    """
    out_dir = os.path.dirname(final_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_img_", suffix=".png", dir=out_dir)
    os.close(fd)
    try:
        save_image(tensor, tmp_path, nrow=nrow)
        os.replace(tmp_path, final_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_loaders(cfg, rank, world_size):
    """构建 DataLoader，DDP 时使用 DistributedSampler。"""
    def _make(csv_path, mode):
        dataset = ReMark_Dataset(
            csv_path=csv_path,
            image_size=cfg.data.image_size,
            mode=mode,
            use_wm_cache=getattr(cfg.efficiency, 'cache_wm_images', False),
            center_crop=getattr(cfg.data, 'center_crop', 0),
        )

        # Optional hard guard: enforce exact sample count for each epoch source CSV.
        required = int(getattr(cfg.data, 'train_size_must_be', 0) or 0)
        if mode == 'train' and required > 0 and len(dataset) != required:
            raise ValueError(
                f"train_size_must_be={required}, but got {len(dataset)} rows in {csv_path}"
            )

        if world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank,
                shuffle=(mode == 'train'), drop_last=(mode == 'train'),
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.training.batch_size,
                sampler=sampler,
                num_workers=getattr(cfg.efficiency, 'attack_num_workers', 2),
                pin_memory=True,
            ), sampler
        else:
            return build_dataloader(csv_path, cfg, mode=mode), None

    train_loader, train_sampler = _make(cfg.data.train_csv, 'train')
    val_loader,   _             = _make(cfg.data.val_csv,   'val')
    return train_loader, val_loader, train_sampler


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, cfg, logger, device, rank, world_size, resume_path=None):
        self.cfg        = cfg
        self.logger     = logger
        self.device     = device
        self.rank       = rank
        self.world_size = world_size
        self.main       = is_main(rank)

        # VAE：只有参数需要梯度同步，用 DDP 包装
        vae_raw = build_vae(cfg).to(device)
        if world_size > 1:
            self.vae = DDP(vae_raw, device_ids=[device.index])
        else:
            self.vae = vae_raw

        # 获取原始模块（保存/推理用）
        self._vae_module = vae_raw

        # 水印模型（冻结，每张卡独立持有，不需要 DDP）
        self.wm_adapter = build_wm_adapter(cfg.wm_model, cfg)

        # 攻击模型（冻结推理，每张卡独立，不需要 DDP）
        # CUDA_VISIBLE_DEVICES 已由 torchrun 设置，StarGAN 的 DataParallel
        # 只会看到当前进程的那张卡，不再与 DDP 冲突
        self.online_attacks = {}
        for name in getattr(cfg.attacks, 'online', []):
            if name not in ATTACK_REGISTRY:
                if self.main:
                    logger.warning(f'Attack "{name}" not registered, skipping.')
                continue
            self.online_attacks[name] = build_attack(name, cfg)
            if self.main:
                logger.info(f'Online attack loaded: {name}')

        # Validation attack set:
        # - default: same as online attacks
        # - if validation.attacks is provided, include those entries (except "clean")
        self.val_attacks = dict(self.online_attacks)
        self._pending_val_attack_names = []
        val_start_epoch_cfg = int(getattr(getattr(cfg, 'training', None), 'val_start_epoch', 0))
        preflight_enabled = bool(getattr(getattr(cfg, 'preflight_eval', None), 'enabled', True))
        self._defer_val_attack_load = (val_start_epoch_cfg > 0) and (not preflight_enabled)
        val_attack_names = list(getattr(getattr(cfg, 'validation', None), 'attacks', []) or [])
        if val_attack_names:
            for name in val_attack_names:
                name = str(name)
                if name == 'clean':
                    continue
                if name in self.val_attacks:
                    continue
                if name not in ATTACK_REGISTRY:
                    if self.main:
                        logger.warning(f'Validation attack "{name}" not registered, skipping.')
                    continue
                if self._defer_val_attack_load:
                    self._pending_val_attack_names.append(name)
                else:
                    self.val_attacks[name] = build_attack(name, cfg)
                    if self.main:
                        logger.info(f'Validation-only attack loaded: {name}')
        if self.main and self._pending_val_attack_names:
            logger.info(
                'Deferred validation-only attacks until first validation: '
                + ', '.join(sorted(self._pending_val_attack_names))
            )
        if self.main:
            all_val_names = ['clean'] + sorted(set(list(self.val_attacks.keys()) + list(self._pending_val_attack_names)))
            self.logger.info('Validation attacks: ' + ', '.join(all_val_names))
        # 训练阶段攻击采样权重（支持 identity）
        # 语义：
        #   attacks.sample_weights: {simswap:1, diffswap:1, arc2face:1, identity:1}
        # 若未提供，则对 online attacks 做均匀采样，不额外引入 identity。
        sample_weights_cfg = getattr(getattr(cfg, 'attacks', None), 'sample_weights', None)
        raw_weights = {}
        if sample_weights_cfg is not None:
            if isinstance(sample_weights_cfg, dict):
                raw_weights = {str(k): float(v) for k, v in sample_weights_cfg.items()}
            else:
                raw_weights = {
                    str(k): float(v)
                    for k, v in vars(sample_weights_cfg).items()
                    if not str(k).startswith('_')
                }

        self.train_attack_choices = []
        self.train_attack_choice_names = []
        self.train_attack_choice_weights = []
        if raw_weights:
            for attack_name in self.online_attacks.keys():
                w = max(float(raw_weights.get(attack_name, 0.0)), 0.0)
                if w > 0:
                    self.train_attack_choices.append((attack_name, self.online_attacks[attack_name]))
                    self.train_attack_choice_names.append(attack_name)
                    self.train_attack_choice_weights.append(w)
            w_identity = max(float(raw_weights.get('identity', 0.0)), 0.0)
            if w_identity > 0:
                self.train_attack_choices.append(('identity', None))
                self.train_attack_choice_names.append('identity')
                self.train_attack_choice_weights.append(w_identity)
        else:
            # 兼容旧行为：只在 online attacks 里均匀采样
            for attack_name, attack in self.online_attacks.items():
                self.train_attack_choices.append((attack_name, attack))
                self.train_attack_choice_names.append(attack_name)
                self.train_attack_choice_weights.append(1.0)

        if self.main and self.train_attack_choices:
            self.logger.info(
                'Train attack sampler: ' +
                ', '.join(
                    f'{n}={w:.3f}'
                    for n, w in zip(self.train_attack_choice_names, self.train_attack_choice_weights)
                )
            )
        self._last_train_attack_name = 'identity'
        self._epoch_attack_counts = {}

        # 损失
        self.loss_computer = LossComputer(
            cfg.losses,
            current_epoch=0,
            warmup_epochs=getattr(cfg.training, 'kl_warmup_epochs', 20),
        )

        # 优化器：Adam 不需要随卡数线性缩放 lr（线性缩放是 SGD 的规则）
        base_lr = cfg.training.lr
        self.optimizer = torch.optim.Adam(
            self._vae_module.parameters(),
            lr=base_lr,
            betas=getattr(cfg.training, 'betas', (0.9, 0.999)),
            weight_decay=getattr(cfg.training, 'weight_decay', 1e-5),
        )
        # Cosine Annealing：lr 从 base_lr 衰减到 1% 量级
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=base_lr * 0.01,
        )
        if self.main:
            logger.info(f'lr={base_lr:.2e}  cosine → {base_lr*0.01:.2e}  over {cfg.training.epochs} epochs')

        self.use_amp      = getattr(cfg.efficiency, 'use_amp', False)
        self.scaler       = GradScaler(enabled=self.use_amp)
        self.wm_loss_freq = getattr(cfg.efficiency, 'wm_loss_freq', 4)
        self.cache_wm     = getattr(cfg.efficiency, 'cache_wm_images', False)
        self.message_len  = self.wm_adapter.message_length
        # Optional: generate deterministic per-image messages based on img_path.
        # This is required when replay fakes are built to align with fixed message seeds.
        self.message_mode = str(getattr(cfg.training, 'message_mode', 'random')).lower()
        self.deterministic_messages = bool(getattr(cfg.training, 'deterministic_messages', False))
        self.message_seed_salt = str(getattr(cfg.training, 'message_seed_salt', 'remark_v1'))
        self.landmark_predictor_path = str(getattr(cfg.training, 'landmark_predictor_path', ''))
        self.landmark_bits_cache_dir = str(getattr(cfg.training, 'landmark_bits_cache_dir', ''))
        self.landmark_canonical_bits = int(getattr(cfg.training, 'landmark_canonical_bits', 128))
        self.landmark_bits_per_value = int(getattr(cfg.training, 'landmark_bits_per_value', 4))
        self.preflight_enabled = getattr(
            getattr(cfg, 'preflight_eval', None), 'enabled', True
        )
        self.preflight_max_batches = int(getattr(
            getattr(cfg, 'preflight_eval', None), 'max_batches', 0
        ))
        self.val_max_batches = int(getattr(
            getattr(cfg, 'validation', None), 'max_batches', 0
        ))
        self.progress_enabled = bool(getattr(
            getattr(cfg, 'progress', None), 'enabled', True
        ))
        self.progress_log_interval = max(
            int(getattr(getattr(cfg, 'progress', None), 'log_interval_steps', 10)),
            1,
        )
        self.best_select = str(getattr(cfg.training, 'best_select', 'auto')).lower()
        self.best_score = -float('inf')
        self.best_metric_name = 'none'
        self.val_log_mode = str(getattr(cfg.training, 'val_log_mode', 'aggregate')).strip().lower()
        self.val_per_sample_target = int(getattr(cfg.training, 'val_per_sample_target', 0) or 0)
        self.val_path_order = self._read_img_paths_from_csv(getattr(cfg.data, 'val_csv', ''))
        self.val_expected_samples = int(len(self.val_path_order))
        if self.val_per_sample_target <= 0:
            self.val_per_sample_target = self.val_expected_samples

        # 交替训练：以 clean 为主，按概率掺入 attack 分支
        alt_cfg = getattr(cfg, 'alternating_train', None)
        self.alt_enabled = bool(getattr(alt_cfg, 'enabled', True))
        self.alt_warmup_epochs = int(getattr(alt_cfg, 'warmup_epochs', 10))
        self.alt_mid_start_epoch = int(getattr(alt_cfg, 'mid_start_epoch', 30))
        self.alt_late_start_epoch = int(getattr(alt_cfg, 'late_start_epoch', 80))
        self.alt_warmup_prob = float(getattr(alt_cfg, 'warmup_prob', 0.0))
        self.alt_mid_prob = float(getattr(alt_cfg, 'mid_prob', 0.20))
        self.alt_main_prob = float(getattr(alt_cfg, 'main_prob', 0.35))
        self.alt_late_prob = float(getattr(alt_cfg, 'late_prob', 0.30))
        self.current_attack_prob = 0.0
        self.current_epoch = 0
        self.dual_branch_train = bool(getattr(cfg.training, 'dual_branch_train', False))
        self.dual_clean_weight = float(getattr(cfg.training, 'dual_clean_weight', 1.0))
        self.dual_fake_weight = float(getattr(cfg.training, 'dual_fake_weight', 1.0))
        self._forced_attack_warned = set()

        # Optional: strict epoch-wise attack curriculum, e.g.
        #   0-5 clean(identity), 6-10 simswap, 11-15 arc2face.
        self.epoch_attack_schedule = []
        raw_sched = getattr(cfg.training, 'epoch_attack_schedule', None)
        if raw_sched:
            for item in raw_sched:
                if isinstance(item, dict):
                    start = int(item.get('start', 0))
                    end = int(item.get('end', start))
                    attack = str(item.get('attack', 'identity')).strip().lower()
                else:
                    start = int(getattr(item, 'start', 0))
                    end = int(getattr(item, 'end', start))
                    attack = str(getattr(item, 'attack', 'identity')).strip().lower()
                if end < start:
                    start, end = end, start
                self.epoch_attack_schedule.append({
                    'start': start,
                    'end': end,
                    'attack': attack,
                })
        if self.main and self.epoch_attack_schedule:
            msg = ', '.join(
                f"{x['start']}-{x['end']}:{x['attack']}" for x in self.epoch_attack_schedule
            )
            self.logger.info(f'Epoch attack schedule enabled: {msg}')

        # KL 调度：
        # - 若 kl.weight > 0：沿用 LossComputer 的 warmup（受 training.kl_warmup_epochs 控制）
        # - 若 kl.weight <= 0：启用自动线性调度，0 -> kl.target_weight（默认 0.01），
        #   且同样受 training.kl_warmup_epochs 控制
        self.kl_cfg = getattr(cfg.losses, 'kl', None)
        self.kl_schedule_enabled = False
        self.current_kl_weight = float(getattr(self.kl_cfg, 'weight', 0.0)) if self.kl_cfg else 0.0
        self.kl_target_weight = self.current_kl_weight
        self.kl_auto_warmup_epochs = int(getattr(cfg.training, 'kl_warmup_epochs', 0))
        if self.kl_cfg is not None and float(getattr(self.kl_cfg, 'weight', 0.0)) <= 0.0:
            self.kl_schedule_enabled = True
            self.kl_target_weight = float(getattr(self.kl_cfg, 'target_weight', 0.01))
            # 使用自定义分段调度，关闭现有 warmup，避免重复衰减
            if hasattr(self.kl_cfg, 'warmup'):
                self.kl_cfg.warmup = False
            self.loss_computer.warmup_epochs = 0
            if self.main:
                logger.info(
                    'KL auto schedule enabled (weight<=0): '
                    f'0 -> {self.kl_target_weight:.5f} over '
                    f'{max(self.kl_auto_warmup_epochs, 1)} epochs.'
                )

        self.start_epoch  = 0
        self.best_val_acc = 0.0

        if resume_path and os.path.isfile(resume_path):
            self._load_checkpoint(resume_path)
        if self.main:
            logger.info(f'best checkpoint metric: {self.best_select}')
            if self.val_log_mode == 'per_sample':
                logger.info(
                    'Validation logging mode: per_sample '
                    f'(target={self.val_per_sample_target})'
                )

    def _ensure_val_attacks_loaded(self):
        if not self._pending_val_attack_names:
            return
        for name in list(self._pending_val_attack_names):
            if name in self.val_attacks:
                continue
            if name not in ATTACK_REGISTRY:
                if self.main:
                    self.logger.warning(
                        f'Validation attack "{name}" not registered at deferred-load stage, skipping.'
                    )
                continue
            self.val_attacks[name] = build_attack(name, self.cfg)
            if self.main:
                self.logger.info(f'Validation-only attack loaded (deferred): {name}')
        self._pending_val_attack_names = []

    @staticmethod
    def _read_img_paths_from_csv(csv_path: str):
        csv_path = str(csv_path or '').strip()
        if not csv_path or (not os.path.isfile(csv_path)):
            return []
        paths = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = str(row.get('img_path', '')).strip()
                if p:
                    paths.append(p)
        return paths

    def _gather_and_order_val_records(self, local_records):
        # NOTE:
        # We intentionally avoid dist.all_gather_object here.
        # In multi-GPU validation, each rank already iterates the full val loader
        # (no distributed sampler), so rank0's local records are sufficient for
        # per-sample logging. Using object collectives here can hang on some setups.
        records = list(local_records or [])

        dedup = {}
        for rec in records:
            p = str(rec.get('img_path', '')).strip()
            if p and p not in dedup:
                dedup[p] = rec

        ordered = []
        if self.val_path_order:
            for p in self.val_path_order:
                if p in dedup:
                    ordered.append(dedup.pop(p))
        if dedup:
            ordered.extend(dedup.values())

        target = int(self.val_per_sample_target or 0)
        if target > 0:
            ordered = ordered[:target]
        return ordered

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, tag: str):
        if not self.main:
            return
        path = self.logger.checkpoint_path(tag, model='vae')
        torch.save({
            'epoch':        epoch,
            'vae':          self._vae_module.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'scheduler':    self.scheduler.state_dict(),
            'scaler':       self.scaler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_score':   self.best_score,
            'best_metric_name': self.best_metric_name,
            'best_select':  self.best_select,
        }, path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self._vae_module.load_state_dict(ckpt['vae'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch  = ckpt['epoch'] + 1
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        if 'best_score' in ckpt:
            self.best_score = float(ckpt.get('best_score', -float('inf')))
        else:
            # backward compatibility: old ckpt only tracked clean acc
            self.best_score = (
                float(self.best_val_acc)
                if self.best_select in ('val_clean', 'val_clean_acc')
                else -float('inf')
            )
        self.best_metric_name = str(ckpt.get('best_metric_name', self.best_metric_name))
        if self.main:
            self.logger.info(f'Resumed from epoch {ckpt["epoch"]}')

    @staticmethod
    def _mean_valid(values):
        valid = [float(v) for v in values if np.isfinite(v)]
        if not valid:
            return float('nan')
        return float(np.mean(valid))

    def _select_best_score(self, val_clean_metrics: dict, val_attack_metrics: dict):
        """
        选择 best checkpoint 的打分指标：
          - val_clean_acc
          - val_attack_avg_acc
          - val_attack_avg_delta = mean(acc - raw_acc)
          - auto: 若存在 attack 指标则优先 val_attack_avg_acc，否则 val_clean_acc
        """
        select = self.best_select
        clean_acc = float(val_clean_metrics.get('acc', float('nan'))) if val_clean_metrics else float('nan')
        attack_accs = [m.get('acc', float('nan')) for m in val_attack_metrics.values()]
        attack_raws = [m.get('raw_acc', float('nan')) for m in val_attack_metrics.values()]
        attack_avg_acc = self._mean_valid(attack_accs)
        attack_avg_delta = self._mean_valid([a - r for a, r in zip(attack_accs, attack_raws)])

        if select in ('val_clean', 'val_clean_acc'):
            return clean_acc, 'val_clean_acc'
        if select in ('val_attack_avg_delta', 'attack_delta'):
            return attack_avg_delta, 'val_attack_avg_delta'
        if select in ('val_attack_avg_acc', 'attack_acc'):
            return attack_avg_acc, 'val_attack_avg_acc'

        # auto
        if np.isfinite(attack_avg_acc):
            return attack_avg_acc, 'val_attack_avg_acc'
        return clean_acc, 'val_clean_acc'

    # ── 单步 ──────────────────────────────────────────────────────────────────

    def _build_messages(self, images, batch):
        B = images.shape[0]
        if hasattr(self.wm_adapter, 'build_messages'):
            m = self.wm_adapter.build_messages(batch_size=B, device=self.device, batch=batch)
            if m is not None:
                return m.float().to(self.device)
        if self.message_mode == 'landmark_bits' and batch is not None and ('img_path' in batch):
            paths = batch['img_path']
            if isinstance(paths, (list, tuple)) and len(paths) == B:
                return landmark_messages_from_paths(
                    paths=paths,
                    message_len=self.message_len,
                    device=self.device,
                    predictor_path=self.landmark_predictor_path,
                    cache_dir=self.landmark_bits_cache_dir,
                    canonical_bits=self.landmark_canonical_bits,
                    bits_per_value=self.landmark_bits_per_value,
                )
        if self.deterministic_messages and batch is not None and ('img_path' in batch):
            paths = batch['img_path']
            if isinstance(paths, (list, tuple)) and len(paths) == B:
                return deterministic_messages_from_paths(
                    paths=paths,
                    message_len=self.message_len,
                    device=self.device,
                    salt=self.message_seed_salt,
                )
        return torch.randint(0, 2, (B, self.message_len)).float().to(self.device)

    def _get_wm_image(self, images, batch, force_online=False):
        if (not force_online) and self.cache_wm and 'wm_image' in batch:
            cached_messages = batch.get('wm_message', None)
            if cached_messages is not None:
                cached_messages = cached_messages.to(self.device).float()
            return batch['wm_image'].to(self.device), cached_messages

        messages = self._build_messages(images, batch)
        if hasattr(self.wm_adapter, 'encode_with_batch'):
            return self.wm_adapter.encode_with_batch(images, messages, batch=batch), messages
        return self.wm_adapter.encode(images, messages), messages

    def _get_fake_image(self, wm_images, batch, cover_images=None, apply_attack=True):
        if 'fake_image' in batch and apply_attack:
            self._last_train_attack_name = 'offline_fake'
            return batch['fake_image'].to(self.device)
        if (not apply_attack) or (not self.online_attacks):
            self._last_train_attack_name = 'identity'
            return wm_images
        forced_name = self._get_forced_attack_name(self.current_epoch)
        if forced_name is not None:
            forced_name = forced_name.lower()
            if forced_name in ('identity', 'clean', 'none'):
                self._last_train_attack_name = 'identity'
                return wm_images
            if forced_name in self.online_attacks:
                name = forced_name
                attack = self.online_attacks[name]
            else:
                if self.main and forced_name not in self._forced_attack_warned:
                    self.logger.warning(
                        f'Forced attack "{forced_name}" not available in online attacks; '
                        'falling back to identity.'
                    )
                    self._forced_attack_warned.add(forced_name)
                self._last_train_attack_name = 'identity'
                return wm_images
        elif self.train_attack_choices:
            names = [x[0] for x in self.train_attack_choices]
            weights = self.train_attack_choice_weights
            name = random.choices(names, weights=weights, k=1)[0]
            attack = dict(self.train_attack_choices).get(name, None)
        else:
            name = random.choice(list(self.online_attacks.keys()))
            attack = self.online_attacks[name]
        self._last_train_attack_name = name
        if name == 'identity' or attack is None:
            return wm_images
        with torch.no_grad():
            if cover_images is not None and hasattr(attack, 'attack_with_cover'):
                try:
                    return attack.attack_with_cover(wm_images, cover_images, batch=batch)
                except TypeError:
                    return attack.attack_with_cover(wm_images, cover_images)
            return attack(wm_images)

    def _apply_attack(self, attack_name, attack, wm_images, images, batch):
        if attack_name == 'clean' or attack is None:
            return wm_images
        if hasattr(attack, 'attack_with_cover'):
            try:
                return attack.attack_with_cover(wm_images, images, batch=batch)
            except TypeError:
                return attack.attack_with_cover(wm_images, images)
        return attack(wm_images)

    @staticmethod
    def _linear_interp(epoch: int, start_epoch: int, end_epoch: int, start_v: float, end_v: float) -> float:
        if end_epoch <= start_epoch:
            return end_v
        if epoch <= start_epoch:
            return start_v
        if epoch >= end_epoch:
            return end_v
        ratio = (epoch - start_epoch) / float(end_epoch - start_epoch)
        return start_v + (end_v - start_v) * ratio

    def _attack_probability(self, epoch: int) -> float:
        # dual-branch 模式下，每步都显式包含 fake 分支，不再依赖 attack_prob 抽样
        if self.dual_branch_train and self.online_attacks:
            return 1.0
        if self._get_forced_attack_name(epoch) is not None:
            return 1.0
        if (not self.alt_enabled) or (not self.online_attacks):
            return 0.0
        if epoch < self.alt_warmup_epochs:
            p = self.alt_warmup_prob
        elif epoch < self.alt_mid_start_epoch:
            p = self.alt_mid_prob
        elif epoch < self.alt_late_start_epoch:
            p = self.alt_main_prob
        else:
            p = self.alt_late_prob
        return max(0.0, min(1.0, float(p)))

    def _scheduled_kl_weight(self, epoch: int) -> float:
        if not self.kl_schedule_enabled:
            return float(getattr(self.kl_cfg, 'weight', 0.0)) if self.kl_cfg is not None else 0.0
        if self.kl_auto_warmup_epochs <= 0:
            return self.kl_target_weight
        return self._linear_interp(
            epoch=epoch,
            start_epoch=0,
            end_epoch=self.kl_auto_warmup_epochs,
            start_v=0.0,
            end_v=self.kl_target_weight,
        )

    def _get_forced_attack_name(self, epoch: int):
        if not self.epoch_attack_schedule:
            return None
        for item in self.epoch_attack_schedule:
            if item['start'] <= epoch <= item['end']:
                return item['attack']
        return None

    def _train_step(self, batch, step):
        images = batch['image'].to(self.device)
        wm_images, messages = self._get_wm_image(images, batch)
        use_dual = self.dual_branch_train and bool(self.online_attacks)
        apply_attack = bool(self.online_attacks) and (random.random() < self.current_attack_prob)
        x_fake = self._get_fake_image(
            wm_images,
            batch,
            cover_images=images,
            apply_attack=(True if use_dual else apply_attack),
        )
        if use_dual or apply_attack:
            choice_name = getattr(self, '_last_train_attack_name', 'identity')
        else:
            choice_name = 'identity'
        self._epoch_attack_counts[choice_name] = self._epoch_attack_counts.get(choice_name, 0) + int(images.shape[0])

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            if use_dual:
                # fake 分支（重建 attacked）
                x_hat_fake, mu_fake, logvar_fake = self.vae(x_fake)
                fake_inputs = {'l1': (x_hat_fake, x_fake), 'kl': (mu_fake, logvar_fake)}
                if getattr(self.cfg.losses, 'lpips', None) and \
                        getattr(self.cfg.losses.lpips, 'enabled', True):
                    fake_inputs['lpips'] = (x_hat_fake, x_fake)
                if messages is not None and step % self.wm_loss_freq == 0:
                    # Keep WM decode/BCE path in fp32 to avoid fp16 overflow -> NaN.
                    with autocast(enabled=False):
                        logits_fake = self.wm_adapter.decode(x_hat_fake.float())
                    logits_fake = torch.nan_to_num(logits_fake, nan=0.0, posinf=30.0, neginf=-30.0)
                    fake_inputs['bce'] = (logits_fake, messages)
                total_fake, breakdown_fake = self.loss_computer.compute(**fake_inputs)

                # clean 分支（重建 wm）
                x_hat_clean, mu_clean, logvar_clean = self.vae(wm_images)
                clean_inputs = {'l1': (x_hat_clean, wm_images), 'kl': (mu_clean, logvar_clean)}
                if getattr(self.cfg.losses, 'lpips', None) and \
                        getattr(self.cfg.losses.lpips, 'enabled', True):
                    clean_inputs['lpips'] = (x_hat_clean, wm_images)
                if messages is not None and step % self.wm_loss_freq == 0:
                    with autocast(enabled=False):
                        logits_clean = self.wm_adapter.decode(x_hat_clean.float())
                    logits_clean = torch.nan_to_num(logits_clean, nan=0.0, posinf=30.0, neginf=-30.0)
                    clean_inputs['bce'] = (logits_clean, messages)
                total_clean, breakdown_clean = self.loss_computer.compute(**clean_inputs)

                w_clean = max(self.dual_clean_weight, 0.0)
                w_fake = max(self.dual_fake_weight, 0.0)
                denom = max(w_clean + w_fake, 1e-8)
                total = (w_clean * total_clean + w_fake * total_fake) / denom

                breakdown = {}
                for key in set(breakdown_clean.keys()) | set(breakdown_fake.keys()):
                    v_clean = breakdown_clean.get(key, 0.0)
                    v_fake = breakdown_fake.get(key, 0.0)
                    breakdown[key] = (w_clean * v_clean + w_fake * v_fake) / denom
                breakdown['loss_clean'] = float(total_clean.detach().item())
                breakdown['loss_fake'] = float(total_fake.detach().item())
            else:
                # 单分支：按 attack_prob 抽样 clean/fake
                recon_target = x_fake if apply_attack else wm_images
                x_hat, mu, logvar = self.vae(x_fake)
                loss_inputs = {'l1': (x_hat, recon_target), 'kl': (mu, logvar)}
                if getattr(self.cfg.losses, 'lpips', None) and \
                        getattr(self.cfg.losses.lpips, 'enabled', True):
                    loss_inputs['lpips'] = (x_hat, recon_target)
                if messages is not None and step % self.wm_loss_freq == 0:
                    with autocast(enabled=False):
                        logits = self.wm_adapter.decode(x_hat.float())
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
                    loss_inputs['bce'] = (logits, messages)
                total, breakdown = self.loss_computer.compute(**loss_inputs)

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self._vae_module.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {'loss': total.item(), **breakdown}

    @torch.no_grad()
    def _val_clean_step(self, batch):
        images = batch['image'].to(self.device)
        wm_images, messages = self._get_wm_image(images, batch)
        # 验证固定 clean 恒等映射，避免随机 attack 使验证抖动
        x_fake = self._get_fake_image(wm_images, batch, cover_images=images, apply_attack=False)
        x_hat, mu, logvar = self._vae_module(x_fake)   # 验证用原始模块

        loss_inputs = {'l1': (x_hat, wm_images), 'kl': (mu, logvar)}
        raw_acc = float('nan')
        acc = float('nan')
        per_sample = []
        if messages is not None:
            # raw_acc: 不经过任何攻击和修复，直接从 wm 图解码。
            logits_raw = self.wm_adapter.decode(wm_images)
            raw_pred = (torch.sigmoid(logits_raw) > 0.5).float()
            raw_acc = (raw_pred == messages).float().mean().item()

            logits = self.wm_adapter.decode(x_hat)
            loss_inputs['bce'] = (logits, messages)
            pred = (torch.sigmoid(logits) > 0.5).float()
            bit_eq = (pred == messages).float()
            acc  = bit_eq.mean().item()
            raw_bit_eq = (raw_pred == messages).float()
            sample_raw = raw_bit_eq.mean(dim=1)
            sample_rec = bit_eq.mean(dim=1)
            paths = batch.get('img_path', None)
            for i in range(int(messages.shape[0])):
                p = str(paths[i]) if isinstance(paths, (list, tuple)) and i < len(paths) else f'idx_{i}'
                per_sample.append({
                    'img_path': p,
                    'raw_acc': float(sample_raw[i].item()),
                    'acc': float(sample_rec[i].item()),
                })

        total, breakdown = self.loss_computer.compute(**loss_inputs)
        return ({'loss': total.item(), 'raw_acc': raw_acc, 'acc': acc, **breakdown}, per_sample)

    @torch.no_grad()
    def _val_attack_step(self, batch, attack_name, attack):
        images = batch['image'].to(self.device)
        wm_images, messages = self._get_wm_image(images, batch)

        attacked = self._apply_attack(attack_name, attack, wm_images, images, batch)
        x_hat, mu, logvar = self._vae_module(attacked)

        loss_inputs = {'l1': (x_hat, attacked), 'kl': (mu, logvar)}
        raw_acc = float('nan')
        rec_acc = float('nan')
        per_sample = []
        if messages is not None:
            logits_raw = self.wm_adapter.decode(attacked)
            raw_pred = (torch.sigmoid(logits_raw) > 0.5).float()
            raw_acc = (raw_pred == messages).float().mean().item()

            logits_rec = self.wm_adapter.decode(x_hat)
            rec_pred = (torch.sigmoid(logits_rec) > 0.5).float()
            bit_eq_rec = (rec_pred == messages).float()
            rec_acc = bit_eq_rec.mean().item()
            loss_inputs['bce'] = (logits_rec, messages)
            bit_eq_raw = (raw_pred == messages).float()
            sample_raw = bit_eq_raw.mean(dim=1)
            sample_rec = bit_eq_rec.mean(dim=1)
            paths = batch.get('img_path', None)
            for i in range(int(messages.shape[0])):
                p = str(paths[i]) if isinstance(paths, (list, tuple)) and i < len(paths) else f'idx_{i}'
                per_sample.append({
                    'img_path': p,
                    'raw_acc': float(sample_raw[i].item()),
                    'acc': float(sample_rec[i].item()),
                })

        total, breakdown = self.loss_computer.compute(**loss_inputs)
        return ({'loss': total.item(), 'raw_acc': raw_acc, 'acc': rec_acc, **breakdown}, per_sample)

    def _reduce_metrics(self, metrics: dict) -> dict:
        """跨卡平均指标（仅 DDP 时需要）"""
        if self.world_size <= 1:
            return metrics
        result = {}
        for k, v in metrics.items():
            t = torch.tensor(v, device=self.device)
            # 兼容旧版 PyTorch：ReduceOp 可能不支持 AVG。
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            result[k] = (t / float(self.world_size)).item()
        return result

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

    @torch.no_grad()
    def _run_preflight_baseline(self, val_loader):
        """
        训练前基线评估：
          不经过 ReMark（VAE/U-Net），直接评估 WM 在 clean 和各 deepfake attack 下的提取 ACC/BER。
        结果仅写入 train.log 开头。
        """
        if not self.preflight_enabled:
            if self.main:
                self.logger.info('Preflight baseline eval: disabled by config.')
            return

        # clean + configured preflight attacks (fallback: clean + validation attacks)
        preflight_names = list(getattr(getattr(self.cfg, 'preflight_eval', None), 'attacks', []) or [])
        if not preflight_names:
            preflight_names = ['clean'] + sorted(self.val_attacks.keys())
        attack_items = []
        for attack_name in preflight_names:
            attack_name = str(attack_name)
            if attack_name == 'clean':
                attack_items.append((attack_name, None))
                continue
            attack = self.val_attacks.get(attack_name, None)
            if attack is None:
                if self.main:
                    self.logger.warning(f'Preflight attack "{attack_name}" not loaded, skipping.')
                continue
            attack_items.append((attack_name, attack))

        if self.main:
            self.logger.info('=' * 72)
            self.logger.info('Preflight baseline (NO ReMark) started')
            self.logger.info(
                f'wm_model={self.cfg.wm_model}  '
                f'cache_wm_images={self.cache_wm}  '
                f'val_max_batches={self.preflight_max_batches if self.preflight_max_batches > 0 else "all"}'
            )
            if self.cache_wm:
                self.logger.info(
                    'cache_wm_images=True: preflight will re-encode online to keep message ground-truth.'
                )
            if self.message_mode == 'landmark_bits':
                self.logger.info(
                    'message_mode=landmark_bits: '
                    f'predictor={self.landmark_predictor_path or "default"}  '
                    f'cache_dir={self.landmark_bits_cache_dir or "(none)"}  '
                    f'canonical_bits={self.landmark_canonical_bits}  '
                    f'bits_per_value={self.landmark_bits_per_value}'
                )
            elif self.deterministic_messages:
                self.logger.info(
                    f'deterministic_messages=True: message_seed_salt={self.message_seed_salt}'
                )

        for attack_name, attack in attack_items:
            # 统计 bit-level 指标
            local = torch.zeros(5, device=self.device)
            # [0]=correct_bits, [1]=total_bits, [2]=sample_acc_sum, [3]=num_samples, [4]=num_batches
            saved_sample = False

            iterator = val_loader
            if self.main:
                iterator = tqdm(
                    val_loader,
                    desc=f'Preflight [{attack_name}]',
                    dynamic_ncols=True,
                    disable=not self.main
                )

            for bidx, batch in enumerate(iterator):
                if self.preflight_max_batches > 0 and bidx >= self.preflight_max_batches:
                    break

                images = batch['image'].to(self.device)
                wm_images, messages = self._get_wm_image(images, batch, force_online=True)
                attacked = self._apply_attack(attack_name, attack, wm_images, images, batch)
                logits = self.wm_adapter.decode(attacked)

                if self.main and not saved_sample:
                    self._save_preflight_sample(attack_name, images, wm_images, attacked, attack=attack)
                    saved_sample = True

                pred = (torch.sigmoid(logits) > 0.5).float()
                bit_eq = (pred == messages).float()

                local[0] += bit_eq.sum()                  # correct bits
                local[1] += float(messages.numel())       # total bits
                local[2] += bit_eq.mean(dim=1).sum()      # sample-wise acc sum
                local[3] += float(messages.shape[0])      # num samples
                local[4] += 1.0                           # num batches

            if self.world_size > 1:
                dist.all_reduce(local, op=dist.ReduceOp.SUM)

            correct_bits = float(local[0].item())
            total_bits   = max(float(local[1].item()), 1.0)
            sample_acc_sum = float(local[2].item())
            num_samples = max(float(local[3].item()), 1.0)
            num_batches = int(local[4].item())

            bit_acc = correct_bits / total_bits
            ber = 1.0 - bit_acc
            sample_acc = sample_acc_sum / num_samples

            if self.main:
                self.logger.info(
                    f'[Preflight] attack={attack_name:>12s}  '
                    f'bit_acc={bit_acc:.4f}  ber={ber:.4f}  '
                    f'sample_acc={sample_acc:.4f}  '
                    f'samples={int(num_samples)}  bits={int(total_bits)}'
                )

        if self.main:
            self.logger.info('=' * 72)

    def _save_preflight_sample(self, attack_name, images, wm_images, attacked, attack=None):
        """保存 preflight 对比图：原图 / 含水印 / 攻击后。"""
        cfg_pf = getattr(self.cfg, 'preflight_eval', None)
        target_nrow = int(getattr(cfg_pf, 'sample_nrow', 8))
        target_nrow = max(target_nrow, 1)
        n_take = min(target_nrow, images.shape[0])

        def _to_disp(t: torch.Tensor) -> torch.Tensor:
            return ((t[:n_take].clamp(-1, 1) + 1) / 2).cpu()

        def _pad_to_nrow(t: torch.Tensor) -> torch.Tensor:
            if t.shape[0] >= target_nrow:
                return t[:target_nrow]
            pad_n = target_nrow - t.shape[0]
            pad = torch.zeros(
                pad_n, t.shape[1], t.shape[2], t.shape[3], dtype=t.dtype
            )
            return torch.cat([t, pad], dim=0)

        row_clean = _pad_to_nrow(_to_disp(images))
        row_wm = _pad_to_nrow(_to_disp(wm_images))
        row_attack = _pad_to_nrow(_to_disp(attacked))

        grid = torch.cat([row_clean, row_wm, row_attack], dim=0)
        safe_name = attack_name.replace('/', '_').replace(' ', '_')
        out = os.path.join(self.logger.sample_dir, f'preflight_{safe_name}.png')
        _atomic_save_image(grid, out, nrow=target_nrow)
        self.logger.info(f'Preflight sample[{attack_name}] -> {out}')

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader, train_sampler=None):
        cfg       = self.cfg
        save_freq = getattr(cfg.training, 'save_freq', 10)
        val_freq  = getattr(cfg.training, 'val_freq',  1)
        val_start_epoch = int(getattr(cfg.training, 'val_start_epoch', 0))

        # 训练前先评估“无 ReMark”基线，便于后续与训练结果对比
        self._run_preflight_baseline(val_loader)
        if self.world_size > 1:
            dist.barrier()

        for epoch in range(self.start_epoch, cfg.training.epochs):
            self.current_epoch = epoch
            self._epoch_attack_counts = {}
            self.loss_computer.current_epoch = epoch
            self.current_attack_prob = self._attack_probability(epoch)
            self.current_kl_weight = self._scheduled_kl_weight(epoch)
            if self.kl_cfg is not None:
                self.kl_cfg.weight = self.current_kl_weight
            l1_weight_now = self.loss_computer.get_effective_weight('l1')
            bce_weight_now = self.loss_computer.get_effective_weight('bce')
            if self.main:
                if self.dual_branch_train and self.online_attacks:
                    self.logger.info(
                        f'Epoch {epoch:03d} schedule: '
                        f'attack_mode=dual(1:1 clean+fake)  '
                        f'kl_weight={self.current_kl_weight:.5f}  '
                        f'l1_weight={l1_weight_now:.4f}  '
                        f'bce_weight={bce_weight_now:.4f}'
                    )
                else:
                    self.logger.info(
                        f'Epoch {epoch:03d} schedule: '
                        f'attack_prob={self.current_attack_prob:.2f}  '
                        f'kl_weight={self.current_kl_weight:.5f}  '
                        f'l1_weight={l1_weight_now:.4f}  '
                        f'bce_weight={bce_weight_now:.4f}'
                    )

            # DDP：每个 epoch 重置 sampler 使各卡看到不同数据
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # ── 训练 ──────────────────────────────────────────────────────────
            self.vae.train()
            accum, n_steps = {}, 0
            train_start = time.time()
            train_total_steps = len(train_loader)
            pbar = tqdm(train_loader,
                        desc=f'Epoch {epoch:03d} [train]',
                        dynamic_ncols=True,
                        disable=not self.main)   # 只有 rank 0 显示进度条
            for step, batch in enumerate(pbar):
                result = self._train_step(batch, step)
                for k, v in result.items():
                    accum[k] = accum.get(k, 0.0) + v
                n_steps += 1
                if self.main:
                    pbar.set_postfix(loss=f'{result["loss"]:.4f}',
                                     l1=f'{result.get("l1",0):.4f}',
                                     bce=f'{result.get("bce",0):.4f}')
                self._log_progress(
                    epoch=epoch,
                    phase='train',
                    step_idx=step,
                    total_steps=train_total_steps,
                    start_ts=train_start,
                )
            train_metrics = self._reduce_metrics(
                {k: v / n_steps for k, v in accum.items()})
            # 统计本 epoch 实际使用到的攻击样本量，避免出现某攻击只用了极少样本
            if self.train_attack_choice_names:
                name_list = list(self.train_attack_choice_names)
                for extra in sorted(self._epoch_attack_counts.keys()):
                    if extra not in name_list:
                        name_list.append(extra)
                cnt = torch.tensor(
                    [float(self._epoch_attack_counts.get(n, 0)) for n in name_list],
                    device=self.device,
                    dtype=torch.float32,
                )
                if self.world_size > 1:
                    dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
                if self.main:
                    total_cnt = float(cnt.sum().item())
                    if total_cnt > 0:
                        msg = ' | '.join(
                            f'{n}:{int(cnt[i].item())}({cnt[i].item()/total_cnt:.2%})'
                            for i, n in enumerate(name_list)
                        )
                        self.logger.info(f'Epoch {epoch:03d} attack_usage: {msg}')

            # ── 验证（所有卡都跑，再平均）────────────────────────────────────
            val_clean_metrics, val_attack_metrics, first_batch = {}, {}, None
            val_attack_samples = {}
            should_validate = (epoch >= val_start_epoch) and (epoch % val_freq == 0)
            if self.main and (not should_validate) and epoch < val_start_epoch:
                self.logger.info(
                    f'Epoch {epoch:03d}: skip validation '
                    f'(val_start_epoch={val_start_epoch})'
                )
            if should_validate:
                self._ensure_val_attacks_loaded()
                self._vae_module.eval()
                val_total_steps = len(val_loader)
                if self.val_max_batches > 0:
                    val_total_steps = min(val_total_steps, self.val_max_batches)
                val_accum, n_val = {}, 0
                val_clean_start = time.time()
                for batch in tqdm(val_loader,
                                  desc=f'Epoch {epoch:03d} [val-clean]',
                                  dynamic_ncols=True,
                                  disable=not self.main):
                    if self.val_max_batches > 0 and n_val >= self.val_max_batches:
                        break
                    result, _ = self._val_clean_step(batch)
                    for k, v in result.items():
                        val_accum[k] = val_accum.get(k, 0.0) + v
                    n_val += 1
                    if first_batch is None:
                        first_batch = batch
                    self._log_progress(
                        epoch=epoch,
                        phase='val-clean',
                        step_idx=n_val - 1,
                        total_steps=val_total_steps,
                        start_ts=val_clean_start,
                    )
                if n_val > 0:
                    val_clean_metrics = self._reduce_metrics(
                        {k: v / n_val for k, v in val_accum.items()})

                for attack_name, attack in sorted(self.val_attacks.items(), key=lambda x: x[0]):
                    atk_accum, n_atk = {}, 0
                    atk_records_local = []
                    atk_start = time.time()
                    for batch in tqdm(
                            val_loader,
                            desc=f'Epoch {epoch:03d} [val-fake:{attack_name}]',
                            dynamic_ncols=True,
                            disable=not self.main):
                        if self.val_max_batches > 0 and n_atk >= self.val_max_batches:
                            break
                        result, atk_records = self._val_attack_step(batch, attack_name, attack)
                        for k, v in result.items():
                            atk_accum[k] = atk_accum.get(k, 0.0) + v
                        atk_records_local.extend(atk_records)
                        n_atk += 1
                        self._log_progress(
                            epoch=epoch,
                            phase=f'val-fake:{attack_name}',
                            step_idx=n_atk - 1,
                            total_steps=val_total_steps,
                            start_ts=atk_start,
                        )
                    if n_atk > 0:
                        val_attack_metrics[attack_name] = self._reduce_metrics(
                            {k: v / n_atk for k, v in atk_accum.items()})
                        if self.val_log_mode == 'per_sample':
                            val_attack_samples[attack_name] = self._gather_and_order_val_records(
                                atk_records_local
                            )

            # ── 日志 / 保存（只在 rank 0）────────────────────────────────────
            if self.main:
                if self.val_log_mode == 'per_sample':
                    train_str = '  '.join(f'{k}={v:.4f}' for k, v in train_metrics.items())
                    self.logger.info(f'Epoch {epoch:03d} | train: {train_str}')
                    for attack_name in sorted(val_attack_samples.keys()):
                        rows = val_attack_samples[attack_name]
                        self.logger.info(
                            f'Epoch {epoch:03d} val_per_sample[{attack_name}] total={len(rows)}'
                        )
                        for i, rec in enumerate(rows):
                            self.logger.info(
                                f'val_sample[{attack_name}] idx={i:03d}  '
                                f'raw_acc={float(rec["raw_acc"]):.4f}  '
                                f'acc={float(rec["acc"]):.4f}  '
                                f'path={rec["img_path"]}'
                            )
                else:
                    self.logger.log_epoch(
                        epoch,
                        train=train_metrics,
                        val_clean=val_clean_metrics,
                        val_attacks=val_attack_metrics,
                    )
                if first_batch is not None:
                    self._save_sample(first_batch, epoch)
                self._save_checkpoint(epoch, 'last')
                if val_clean_metrics and np.isfinite(val_clean_metrics.get('acc', float('nan'))):
                    self.best_val_acc = max(self.best_val_acc, float(val_clean_metrics['acc']))
                best_score, best_metric = self._select_best_score(
                    val_clean_metrics=val_clean_metrics,
                    val_attack_metrics=val_attack_metrics,
                )
                if np.isfinite(best_score) and best_score > self.best_score:
                    self.best_score = float(best_score)
                    self.best_metric_name = best_metric
                    self._save_checkpoint(epoch, 'best')
                    self.logger.info(
                        f'  >> New best ({best_metric})={best_score:.4f} '
                        f'→ checkpoints/vae/best.pth'
                    )
                if epoch % save_freq == 0:
                    self._save_checkpoint(epoch, f'epoch_{epoch:03d}')

            self.scheduler.step()

            # DDP：等 rank 0 保存完再进入下一 epoch
            if self.world_size > 1:
                dist.barrier()

        if self.main:
            self.logger.info('Training complete.')
            self.logger.close()

    @torch.no_grad()
    def _save_sample(self, batch, epoch):
        images = batch['image'].to(self.device)
        wm_images, _ = self._get_wm_image(images, batch)
        use_attack = bool(self.online_attacks) and (self.current_attack_prob > 0.0)
        x_fake = self._get_fake_image(wm_images, batch, cover_images=images, apply_attack=use_attack)
        x_hat, _, _ = self._vae_module(x_fake)
        self.logger.save_sample(
            {'orig': images, 'wm': wm_images, 'fake': x_fake, 'hat': x_hat},
            epoch)


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',   default='configs/stage1_vae.yaml')
    p.add_argument('--override', default=None)
    p.add_argument('--resume',   default=None,
                   help='已有 run 目录名，用于续训（如 stage1_20260308_2131）')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    rank, local_rank, world_size = setup_ddp()
    args = parse_args()
    cfg  = load_config(args.config, args.override)
    set_seed(args.seed, rank)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available()
                          else 'cpu')

    resume_ckpt = None
    if args.resume:
        resume_ckpt = os.path.join(
            getattr(getattr(cfg, 'paths', None), 'runs_dir', 'runs'),
            args.resume, 'checkpoints', 'vae', 'last.pth',
        )

    # RunLogger 只在 rank 0 创建（其余 rank 传 None，训练时用 self.main 守卫）
    logger = None
    if is_main(rank):
        logger = RunLogger(cfg,
                           config_path=args.config,
                           override_path=args.override,
                           stage='stage1',
                           resume_run=args.resume)
        logger.info(f'Run dir   : {logger.run_dir}')
        logger.info(f'Device    : {device}  |  world_size: {world_size}')

    train_loader, val_loader, train_sampler = build_loaders(cfg, rank, world_size)

    if is_main(rank):
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        logger.info(f'Train: {n_train} samples  Val: {n_val} samples  '
                    f'({n_train // (cfg.training.batch_size * world_size)} steps/epoch/GPU)')

    trainer = Trainer(cfg, logger, device, rank, world_size,
                      resume_path=resume_ckpt)

    if is_main(rank):
        n_params = sum(p.numel() for p in trainer._vae_module.parameters())
        logger.info(f'VAE: {n_params/1e6:.1f}M params')

    trainer.train(train_loader, val_loader, train_sampler)
    cleanup_ddp()


if __name__ == '__main__':
    main()
