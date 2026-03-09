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

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
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

    def _get_wm_image(self, images, batch):
        B = images.shape[0]
        if self.cache_wm and 'wm_image' in batch:
            return batch['wm_image'].to(self.device), None
        messages = torch.randint(0, 2, (B, self.message_len)).float().to(self.device)
        return self.wm_adapter.encode(images, messages), messages

    def _get_fake_image(self, wm_images, batch):
        if 'fake_image' in batch:
            return batch['fake_image'].to(self.device)
        if not self.online_attacks:
            return wm_images
        name = random.choice(list(self.online_attacks.keys()))
        with torch.no_grad():
            return self.online_attacks[name](wm_images)

    def _train_step(self, batch, step):
        images = batch['image'].to(self.device)
        wm_images, messages = self._get_wm_image(images, batch)
        x_fake = self._get_fake_image(wm_images, batch)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            x_hat, mu, logvar = self.vae(x_fake)
            loss_inputs = {'l1': (x_hat, wm_images), 'kl': (mu, logvar)}
            if getattr(self.cfg.losses, 'lpips', None) and \
                    getattr(self.cfg.losses.lpips, 'enabled', True):
                loss_inputs['lpips'] = (x_hat, wm_images)
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
    def _val_step(self, batch):
        images = batch['image'].to(self.device)
        wm_images, messages = self._get_wm_image(images, batch)
        x_fake = self._get_fake_image(wm_images, batch)
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

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader, train_sampler=None):
        cfg       = self.cfg
        save_freq = getattr(cfg.training, 'save_freq', 10)
        val_freq  = getattr(cfg.training, 'val_freq',  1)

        for epoch in range(self.start_epoch, cfg.training.epochs):
            self.loss_computer.current_epoch = epoch

            # DDP：每个 epoch 重置 sampler 使各卡看到不同数据
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # ── 训练 ──────────────────────────────────────────────────────────
            self.vae.train()
            accum, n_steps = {}, 0
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
            train_metrics = self._reduce_metrics(
                {k: v / n_steps for k, v in accum.items()})

            # ── 验证（所有卡都跑，再平均）────────────────────────────────────
            val_metrics, first_batch = {}, None
            if epoch % val_freq == 0:
                self._vae_module.eval()
                val_accum, n_val = {}, 0
                for batch in tqdm(val_loader,
                                  desc=f'Epoch {epoch:03d} [val]',
                                  dynamic_ncols=True,
                                  disable=not self.main):
                    result = self._val_step(batch)
                    for k, v in result.items():
                        val_accum[k] = val_accum.get(k, 0.0) + v
                    n_val += 1
                    if first_batch is None:
                        first_batch = batch
                val_metrics = self._reduce_metrics(
                    {k: v / n_val for k, v in val_accum.items()})

            # ── 日志 / 保存（只在 rank 0）────────────────────────────────────
            if self.main:
                self.logger.log_epoch(epoch, train_metrics, val_metrics)
                if first_batch is not None:
                    self._save_sample(first_batch, epoch)
                self._save_checkpoint(epoch, 'last')
                if val_metrics and val_metrics.get('acc', 0) > self.best_val_acc:
                    self.best_val_acc = val_metrics['acc']
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
        x_fake = self._get_fake_image(wm_images, batch)
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
