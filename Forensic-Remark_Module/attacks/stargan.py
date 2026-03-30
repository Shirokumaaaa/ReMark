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
        self.random_domain = bool(getattr(getattr(cfg, 'attack_options', None),
                                          'stargan_random_domain', True))
        self.fixed_domain = int(getattr(getattr(cfg, 'attack_options', None),
                                        'stargan_fixed_domain', 0))
        self.mode = str(getattr(getattr(cfg, 'attack_options', None),
                                'stargan_mode', 'latent')).lower()
        self.source_mode = str(getattr(getattr(cfg, 'attack_options', None),
                                       'stargan_source_mode', 'self')).lower()
        self.blend_alpha = float(getattr(getattr(cfg, 'attack_options', None),
                                         'stargan_blend_alpha', 1.0))
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
        self._force_load_checkpoint(self._model)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        _unwrap_data_parallel(self._model)   # 防止 DDP 多进程争抢同一张卡
        self._mapping_network = getattr(
            self._model.solver, "mapping_network_ema",
            getattr(self._model.solver, "mapping_network", None)
        )

        StarGANAttack._model_instance = self._model

    def _force_load_checkpoint(self, model):
        """
        强制正确加载 StarGAN checkpoint。
        deepfake_manipulations.py 内部的手动加载会把 state_dict 喂给 DataParallel 外壳，
        key 缺少 `module.` 前缀时可能静默失败，这里显式加载到 `.module`。
        """
        ckpt_path = os.path.join(
            _LAMPMARK_ROOT,
            "model", "stargan", "expr", "checkpoints", "celeba_hq",
            f"{model.args.resume_iter:06d}_nets_ema.ckpt"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"StarGAN checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu')
        loaded = []

        def _load_group(group):
            for name, net in group.items():
                if name not in ckpt:
                    continue
                target = net.module if isinstance(net, nn.DataParallel) else net
                msg = target.load_state_dict(ckpt[name], strict=False)
                loaded.append((name, len(msg.missing_keys), len(msg.unexpected_keys)))

        _load_group(model.solver.nets)
        _load_group(model.solver.nets_ema)

        # 重新绑定 inference 使用对象（确保用到刚加载的 EMA 权重）
        model.generator = getattr(model.solver, "generator_ema", model.solver.generator)
        model.style_encoder = getattr(model.solver, "style_encoder_ema", model.solver.style_encoder)
        model.fan = getattr(model.solver, "fan_ema", model.solver.fan)

        if not loaded:
            raise RuntimeError(
                f"StarGAN checkpoint loaded 0 modules from {ckpt_path}. "
                "Please check checkpoint structure."
            )

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
        img_source = preprocessed if self.source_mode == 'self' else torch.roll(preprocessed, 1, 0)
        if self.random_domain:
            y_trg = torch.randint(2, size=(preprocessed.shape[0],),
                                  dtype=torch.long, device=device)
        else:
            y_trg = torch.full((preprocessed.shape[0],), self.fixed_domain,
                               dtype=torch.long, device=device)
        s_ref = self._get_style_code(preprocessed, y_trg)
        masks = self._model.fan.get_heatmap(img_source) \
            if self._model.args.w_hpf > 0 else None

        with torch.no_grad():
            fake = self._model.generator(img_source, s_ref, masks=masks)
            fake = self._blend_with_source(fake, img_source)
        return fake

    def attack_with_cover(self, wm_images: torch.Tensor,
                          cover_images: torch.Tensor) -> torch.Tensor:
        """
        与 LampMark 的 StarGanModel.forward 对齐：
          source <- roll(clean cover)
          target/style <- wm image
        这通常比仅用 wm_images 自身构造 source 更稳定。
        """
        original_size = (wm_images.shape[-2], wm_images.shape[-1])
        wm_pre = self.preprocess(wm_images)
        cover_pre = self.preprocess(cover_images)

        device = wm_pre.device
        self._model.to(device)

        if self.source_mode == 'self':
            img_source = wm_pre
        else:
            img_source = torch.roll(cover_pre, 1, 0)
        if self.random_domain:
            y_trg = torch.randint(2, size=(wm_pre.shape[0],),
                                  dtype=torch.long, device=device)
        else:
            y_trg = torch.full((wm_pre.shape[0],), self.fixed_domain,
                               dtype=torch.long, device=device)
        s_ref = self._get_style_code(wm_pre, y_trg)
        masks = self._model.fan.get_heatmap(img_source) \
            if self._model.args.w_hpf > 0 else None
        with torch.no_grad():
            fake = self._model.generator(img_source, s_ref, masks=masks)
            fake = self._blend_with_source(fake, wm_pre)

        out = self.postprocess(fake, original_size)
        self._validate_output(out, original_size)
        self._check_silent_failure(wm_images, out)
        return out

    def _blend_with_source(self, fake: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        通过 source-fake 融合降低“换人感”。
        alpha 越小越接近原图，越大越接近 deepfake 输出。
        """
        a = min(max(self.blend_alpha, 0.0), 1.0)
        if a <= 0.0:
            return source
        if a >= 1.0:
            return fake
        return source * (1.0 - a) + fake * a

    def _get_style_code(self, ref_or_target: torch.Tensor, y_trg: torch.Tensor) -> torch.Tensor:
        """
        StarGAN style 生成模式：
          - latent（默认）：mapping_network(z, y)，更稳定，不依赖 ref 域标签一致性
          - reference：style_encoder(ref, y)
        """
        if self.mode == 'reference':
            return self._model.style_encoder(ref_or_target, y_trg)

        if self._mapping_network is None:
            # 回退到 reference，避免因模型结构差异直接报错
            return self._model.style_encoder(ref_or_target, y_trg)

        z = torch.randn(ref_or_target.shape[0], self._model.args.latent_dim,
                        device=ref_or_target.device)
        return self._mapping_network(z, y_trg)

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


@register_attack('stargan_fixed')
class StarGANFixedDomainAttack(StarGANAttack):
    """
    固定目标域版本：
      - 不随机采样目标域
      - 始终使用 attack_options.stargan_fixed_domain（默认 1）
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.random_domain = False
        self.fixed_domain = int(getattr(getattr(cfg, 'attack_options', None),
                                        'stargan_fixed_domain', 1))
