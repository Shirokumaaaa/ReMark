import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttack
from .registry import register_attack


def _unwrap_data_parallel(module: nn.Module):
    """递归解包 nn.DataParallel，使冻结模型在 DDP 进程中只占用当前进程的那张卡。"""
    for name, child in module.named_children():
        if isinstance(child, nn.DataParallel):
            setattr(module, name, child.module)
        else:
            _unwrap_data_parallel(child)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LAMPMARK_ROOT = os.path.join(_PROJECT_ROOT, 'Forensic-LampMark')


@register_attack('stargan')
class StarGANAttack(BaseAttack):
    """
    StarGAN v2 攻击适配器。

    复用 Forensic-LampMark/model/deepfake_manipulations.py 中的 StarGanModel，
    不重新实现模型逻辑，只做格式转换和接口统一。

    格式说明：
      StarGAN 期望：[-1, 1]（与 canonical 一致，preprocess 只处理尺寸）
      StarGAN 输出：[-1, 1]，256×256（postprocess 恢复原始尺寸）

    失败检测：
      检测输出是否与输入几乎相同（人脸检测失败时 StarGAN 返回原图），
      发现静默失败时跳过该样本并记录警告。
    """

    _model_instance = None  # 类级缓存，避免每次实例化都重新加载权重

    def __init__(self, cfg):
        super().__init__(cfg)
        self._load_model()

    def _load_model(self):
        if StarGANAttack._model_instance is not None:
            self._model = StarGANAttack._model_instance
            return

        if _LAMPMARK_ROOT not in sys.path:
            sys.path.insert(0, _LAMPMARK_ROOT)

        try:
            from model.deepfake_manipulations import StarGanModel
        except ImportError as e:
            raise ImportError(
                f"StarGAN 依赖加载失败，请检查路径 {_LAMPMARK_ROOT}\n{e}"
            )

        img_size = getattr(getattr(self.cfg, 'data', None), 'image_size', 128)
        # StarGanModel 内部：
        #   1. 调用 argparse.parse_args() 读取 sys.argv（会与训练脚本参数冲突）
        #   2. 使用相对路径加载 wing.ckpt 等权重（需要 cwd=_LAMPMARK_ROOT）
        # 两个问题统一在此处理：临时替换 sys.argv 并切换工作目录。
        _saved_argv = sys.argv
        _saved_cwd  = os.getcwd()
        sys.argv = sys.argv[:1]
        os.chdir(_LAMPMARK_ROOT)
        try:
            self._model = StarGanModel(img_size=img_size, mode='test')
        finally:
            sys.argv = _saved_argv
            os.chdir(_saved_cwd)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        _unwrap_data_parallel(self._model)   # 防止 DDP 多进程争抢同一张卡

        StarGANAttack._model_instance = self._model

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        canonical [-1,1] BCHW → StarGAN 输入。
        StarGAN 期望 256×256，图像值域与 canonical 一致无需转换。
        """
        return F.interpolate(images, size=(256, 256),
                             mode='bilinear', align_corners=False)

    def generate(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        StarGAN 推理。使用 batch 内 roll 作为 target（与 LampMark 实现一致）。
        """
        device = preprocessed.device
        self._model.to(device)

        img_source = torch.roll(preprocessed, 1, 0)
        y_trg = torch.randint(2, size=(preprocessed.shape[0],),
                              dtype=torch.long, device=device)
        s_ref = self._model.style_encoder(preprocessed, y_trg)
        masks = self._model.fan.get_heatmap(preprocessed) \
            if self._model.args.w_hpf > 0 else None

        with torch.no_grad():
            fake = self._model.generator(img_source, s_ref, masks=masks)
        return fake

    def postprocess(self, output: torch.Tensor, original_size: tuple) -> torch.Tensor:
        """
        StarGAN 输出（256×256）→ canonical [-1,1] BCHW，恢复原始尺寸。
        同时检测静默失败（输出与输入高度相似时说明 StarGAN 未真正变换）。
        """
        restored = F.interpolate(output, size=original_size,
                                 mode='bilinear', align_corners=False)
        restored = restored.clamp(-1, 1)
        return restored

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        覆盖基类 __call__，添加静默失败检测。
        """
        out = super().__call__(images)
        self._check_silent_failure(images, out)
        return out

    def _check_silent_failure(self, original: torch.Tensor, fake: torch.Tensor):
        """
        检测 StarGAN 是否发生静默失败（输出与输入几乎相同）。
        失败时记录警告，但不中断训练（跳过由上层逻辑处理）。
        """
        with torch.no_grad():
            diff = (original - fake).abs().mean().item()
        if diff < 0.01:
            import warnings
            warnings.warn(
                f"[StarGANAttack] 检测到可能的静默失败：输入输出平均差值 {diff:.4f} < 0.01，"
                f"StarGAN 可能未真正变换图像（人脸检测失败？）"
            )
