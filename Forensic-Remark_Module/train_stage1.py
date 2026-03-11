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
import os
import random
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


def build_loaders(cfg, rank, world_size):
    """构建 DataLoader，DDP 时使用 DistributedSampler。"""
    def _make(csv_path, mode):
        dataset = ReMark_Dataset(
            csv_path=csv_path,
            image_size=cfg.data.image_size,
            mode=mode,
            use_wm_cache=getattr(cfg.efficiency, 'cache_wm_images', False),
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
        if self.main:
            self.logger.info(f'Resumed from epoch {ckpt["epoch"]}')

    # ── 单步 ──────────────────────────────────────────────────────────────────

    def _get_wm_image(self, images, batch, force_online=False):
        B = images.shape[0]
        if (not force_online) and self.cache_wm and 'wm_image' in batch:
            return batch['wm_image'].to(self.device), None
        messages = torch.randint(0, 2, (B, self.message_len)).float().to(self.device)
        return self.wm_adapter.encode(images, messages), messages

    def _get_fake_image(self, wm_images, batch, cover_images=None, apply_attack=True):
        if 'fake_image' in batch and apply_attack:
            return batch['fake_image'].to(self.device)
        if (not apply_attack) or (not self.online_attacks):
            return wm_images
        name = random.choice(list(self.online_attacks.keys()))
        with torch.no_grad():
            attack = self.online_attacks[name]
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

    def _train_step(self, batch, step):
        images = batch['image'].to(self.device)
        wm_images, messages = self._get_wm_image(images, batch)
        apply_attack = bool(self.online_attacks) and (random.random() < self.current_attack_prob)
        x_fake = self._get_fake_image(wm_images, batch, cover_images=images, apply_attack=apply_attack)
        # clean 分支重建 wm_images；attack 分支重建 attacked（即 x_fake）
        recon_target = x_fake if apply_attack else wm_images

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            x_hat, mu, logvar = self.vae(x_fake)
            loss_inputs = {'l1': (x_hat, recon_target), 'kl': (mu, logvar)}
            if getattr(self.cfg.losses, 'lpips', None) and \
                    getattr(self.cfg.losses.lpips, 'enabled', True):
                loss_inputs['lpips'] = (x_hat, recon_target)
            if messages is not None and step % self.wm_loss_freq == 0:
                logits = self.wm_adapter.decode(x_hat)
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
        acc = float('nan')
        if messages is not None:
            logits = self.wm_adapter.decode(x_hat)
            loss_inputs['bce'] = (logits, messages)
            pred = (torch.sigmoid(logits) > 0.5).float()
            acc  = (pred == messages).float().mean().item()

        total, breakdown = self.loss_computer.compute(**loss_inputs)
        return {'loss': total.item(), 'acc': acc, **breakdown}

    @torch.no_grad()
    def _val_attack_step(self, batch, attack_name, attack):
        images = batch['image'].to(self.device)
        wm_images, messages = self._get_wm_image(images, batch)

        attacked = self._apply_attack(attack_name, attack, wm_images, images, batch)
        x_hat, mu, logvar = self._vae_module(attacked)

        loss_inputs = {'l1': (x_hat, attacked), 'kl': (mu, logvar)}
        raw_acc = float('nan')
        rec_acc = float('nan')
        if messages is not None:
            logits_raw = self.wm_adapter.decode(attacked)
            raw_pred = (torch.sigmoid(logits_raw) > 0.5).float()
            raw_acc = (raw_pred == messages).float().mean().item()

            logits_rec = self.wm_adapter.decode(x_hat)
            rec_pred = (torch.sigmoid(logits_rec) > 0.5).float()
            rec_acc = (rec_pred == messages).float().mean().item()
            loss_inputs['bce'] = (logits_rec, messages)

        total, breakdown = self.loss_computer.compute(**loss_inputs)
        return {'loss': total.item(), 'raw_acc': raw_acc, 'acc': rec_acc, **breakdown}

    def _reduce_metrics(self, metrics: dict) -> dict:
        """跨卡平均指标（仅 DDP 时需要）"""
        if self.world_size <= 1:
            return metrics
        result = {}
        for k, v in metrics.items():
            t = torch.tensor(v, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            result[k] = t.item()
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

        # clean + 每个 online attack 各跑一遍
        attack_items = [('clean', None)] + sorted(self.online_attacks.items(), key=lambda x: x[0])

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
        nrow = min(8, images.shape[0])
        rows = []
        for t in (images, wm_images):
            rows.append(((t[:nrow].clamp(-1, 1) + 1) / 2).cpu())

        rows.append(((attacked[:nrow].clamp(-1, 1) + 1) / 2).cpu())

        grid = torch.cat(rows, dim=0)
        safe_name = attack_name.replace('/', '_').replace(' ', '_')
        out = os.path.join(self.logger.sample_dir, f'preflight_{safe_name}.png')
        save_image(grid, out, nrow=nrow)
        self.logger.info(f'Preflight sample[{attack_name}] -> {out}')

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader, train_sampler=None):
        cfg       = self.cfg
        save_freq = getattr(cfg.training, 'save_freq', 10)
        val_freq  = getattr(cfg.training, 'val_freq',  1)

        # 训练前先评估“无 ReMark”基线，便于后续与训练结果对比
        self._run_preflight_baseline(val_loader)
        if self.world_size > 1:
            dist.barrier()

        for epoch in range(self.start_epoch, cfg.training.epochs):
            self.loss_computer.current_epoch = epoch
            self.current_attack_prob = self._attack_probability(epoch)
            self.current_kl_weight = self._scheduled_kl_weight(epoch)
            if self.kl_cfg is not None:
                self.kl_cfg.weight = self.current_kl_weight
            if self.main:
                self.logger.info(
                    f'Epoch {epoch:03d} schedule: '
                    f'attack_prob={self.current_attack_prob:.2f}  '
                    f'kl_weight={self.current_kl_weight:.5f}'
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

            # ── 验证（所有卡都跑，再平均）────────────────────────────────────
            val_clean_metrics, val_attack_metrics, first_batch = {}, {}, None
            if epoch % val_freq == 0:
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
                    result = self._val_clean_step(batch)
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

                for attack_name, attack in sorted(self.online_attacks.items(), key=lambda x: x[0]):
                    atk_accum, n_atk = {}, 0
                    atk_start = time.time()
                    for batch in tqdm(
                            val_loader,
                            desc=f'Epoch {epoch:03d} [val-fake:{attack_name}]',
                            dynamic_ncols=True,
                            disable=not self.main):
                        if self.val_max_batches > 0 and n_atk >= self.val_max_batches:
                            break
                        result = self._val_attack_step(batch, attack_name, attack)
                        for k, v in result.items():
                            atk_accum[k] = atk_accum.get(k, 0.0) + v
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

            # ── 日志 / 保存（只在 rank 0）────────────────────────────────────
            if self.main:
                self.logger.log_epoch(
                    epoch,
                    train=train_metrics,
                    val_clean=val_clean_metrics,
                    val_attacks=val_attack_metrics,
                )
                if first_batch is not None:
                    self._save_sample(first_batch, epoch)
                self._save_checkpoint(epoch, 'last')
                if val_clean_metrics and val_clean_metrics.get('acc', 0) > self.best_val_acc:
                    self.best_val_acc = val_clean_metrics['acc']
                    self._save_checkpoint(epoch, 'best')
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
