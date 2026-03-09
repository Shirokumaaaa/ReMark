"""
RunLogger — 训练日志管理器

每次训练运行都在 runs/<run_name>/ 下创建独立目录，
避免 logging/ checkpoints/ samples/ 三个目录各自堆积大量文件。

目录结构：
  runs/
    stage1_20260308_2113/          ← 一个 run = 一条记录，一眼看清
      config.yaml                  ← 配置快照，便于复现
      train.log                    ← 文本日志（只记录每 epoch 摘要）
      metrics.csv                  ← 机器可读的 per-epoch 指标
      samples/
        epoch_000.png
        epoch_005.png
      checkpoints/
        last.pth  best.pth  epoch_050.pth

用法：
    logger = RunLogger(cfg, config_path='configs/stage1_vae.yaml',
                       override_path='configs/experiments/quick_test.yaml',
                       stage='stage1')
    logger.info('Training started')
    logger.log_epoch(epoch=0, train={'loss': 1.2}, val={'loss': 1.1, 'acc': 0.9})
    logger.save_sample({'orig': t1, 'wm': t2, 'fake': t3, 'hat': t4}, epoch=0)
    ckpt_path = logger.checkpoint_path('last')
"""

import csv
import logging
import os
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace

import torch
import yaml
from torchvision.utils import save_image


class RunLogger:
    """
    统一管理单次训练 run 的所有输出：文本日志、指标 CSV、样例图、checkpoint。

    Args:
        cfg:           load_config() 返回的 SimpleNamespace
        config_path:   基础配置文件路径（用于快照）
        override_path: override 配置路径（可选）
        stage:         run 名称前缀，如 'stage1' / 'stage2'
        resume_run:    若指定已有 run 目录名，在该目录下续写，不创建新目录
    """

    def __init__(self, cfg, config_path: str, override_path: str = None,
                 stage: str = 'stage1', resume_run: str = None):
        runs_root = getattr(getattr(cfg, 'paths', None), 'runs_dir',
                            os.path.join(os.path.dirname(os.path.dirname(
                                os.path.abspath(__file__))), 'runs'))

        if resume_run:
            self.run_dir = os.path.join(runs_root, resume_run)
            if not os.path.isdir(self.run_dir):
                raise FileNotFoundError(
                    f"续训目录不存在: {self.run_dir}"
                )
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M')
            self.run_dir = os.path.join(runs_root, f'{stage}_{ts}')

        self.sample_dir = os.path.join(self.run_dir, 'samples')
        self.ckpt_dir   = os.path.join(self.run_dir, 'checkpoints')
        self._metrics_path = os.path.join(self.run_dir, 'metrics.csv')

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir,   exist_ok=True)

        self._logger = self._setup_text_logger()
        self._csv_writer = None
        self._csv_file   = None
        self._csv_cols   = None

        if not resume_run:
            self._save_config_snapshot(cfg, config_path, override_path)

    # ── 文本日志 ──────────────────────────────────────────────────────────────

    def _setup_text_logger(self) -> logging.Logger:
        name = f'remark.{os.path.basename(self.run_dir)}'
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')

        # stdout
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # 文件：只记录 epoch 级别摘要（INFO），不记录 step 级别噪音
        fh = logging.FileHandler(os.path.join(self.run_dir, 'train.log'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        return logger

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    # ── 指标 CSV ─────────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int,
                  train: dict = None, val: dict = None):
        """
        记录一个 epoch 的指标到文本日志和 metrics.csv。

        Args:
            epoch: 当前 epoch 编号
            train: {'loss': 1.2, 'l1': 0.5, ...}（训练集指标）
            val:   {'loss': 1.1, 'acc': 0.9, ...}（验证集指标，可含不同 key）
        """
        train = train or {}
        val   = val   or {}

        # 文本日志：一行摘要
        train_str = '  '.join(f'{k}={v:.4f}' for k, v in train.items())
        val_str   = '  '.join(f'{k}={v:.4f}' for k, v in val.items())
        self.info(f'Epoch {epoch:03d} | train: {train_str} | val: {val_str}')

        # CSV
        row = {'epoch': epoch}
        row.update({f'train_{k}': v for k, v in train.items()})
        row.update({f'val_{k}':   v for k, v in val.items()})
        self._write_csv_row(row)

    def _write_csv_row(self, row: dict):
        cols = list(row.keys())

        # 首次写入：建立列头（可能随 epoch 增加新 key，如 lpips）
        if self._csv_cols is None:
            self._csv_cols = cols
            self._csv_file = open(self._metrics_path, 'a', newline='')
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_cols,
                extrasaction='ignore')
            # 只在文件为空时写 header
            if os.path.getsize(self._metrics_path) == 0:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        if self._csv_file:
            self._csv_file.close()

    # ── 样例图 ────────────────────────────────────────────────────────────────

    def save_sample(self, tensors: dict, epoch: int, nrow: int = 8):
        """
        拼接多组图像并保存。

        Args:
            tensors: {'orig': B×C×H×W, 'wm': ..., 'fake': ..., 'hat': ...}
                     所有 tensor 均为 canonical [-1,1]
            epoch:   当前 epoch（用于文件名）
            nrow:    每行显示几张图
        """
        rows = []
        for t in tensors.values():
            rows.append(((t[:nrow].clamp(-1, 1) + 1) / 2).cpu())
        grid = torch.cat(rows, dim=0)

        path = os.path.join(self.sample_dir, f'epoch_{epoch:03d}.png')
        save_image(grid, path, nrow=nrow)
        self.info(f'Sample → {os.path.relpath(path, self.run_dir)}')

    # ── Checkpoint 路径 ───────────────────────────────────────────────────────

    def checkpoint_path(self, tag: str, model: str = 'model') -> str:
        """
        返回 checkpoint 完整路径。

        目录结构：checkpoints/<model>/<tag>.pth
          例：checkpoints/vae/best.pth
              checkpoints/unet/epoch_050.pth

        Args:
            tag:   'last' / 'best' / 'epoch_050' 等
            model: 模型名，用作子目录名（默认 'model'）
        """
        model_dir = os.path.join(self.ckpt_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, f'{tag}.pth')

    # ── 配置快照 ──────────────────────────────────────────────────────────────

    def _save_config_snapshot(self, cfg: SimpleNamespace,
                              config_path: str, override_path: str):
        """
        将实际生效的配置写入 run 目录，便于日后复现实验。
        同时记录原始文件路径方便溯源。
        """
        snap = _ns_to_dict(cfg)
        snap['_meta'] = {
            'config_path':   config_path,
            'override_path': override_path,
            'run_dir':       self.run_dir,
            'created_at':    datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(self.run_dir, 'config.yaml'), 'w') as f:
            yaml.dump(snap, f, allow_unicode=True, default_flow_style=False)

    # ── __repr__ ──────────────────────────────────────────────────────────────

    def __repr__(self):
        return f'RunLogger(run_dir={self.run_dir})'


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _ns_to_dict(obj):
    """SimpleNamespace → 可序列化的普通 dict（递归）"""
    if isinstance(obj, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_dict(i) for i in obj]
    return obj
