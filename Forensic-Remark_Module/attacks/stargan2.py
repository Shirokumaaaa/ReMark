import os
import importlib.util
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F

from .base import BaseAttack
from .registry import register_attack


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STARGAN2_ROOT = os.path.join(_PROJECT_ROOT, "Attack-StarGAN", "network", "noise_layers", "stargan")


@register_attack("stargan2")
class StarGAN2Attack(BaseAttack):
    """
    Attack-StarGAN（StarGAN v1/CelebA 属性翻转）适配器。

    - 代码来源：ReMark/Attack-StarGAN/network/noise_layers/stargan
    - 模型：Generator(conv_dim=64, c_dim=len(selected_attrs), repeat_num=6)
    - 需要 batch['attrs']（来自 CelebA 属性 CSV / dataset），用于构造 c_org / c_trg。
    """

    _MODEL_CACHE: Dict[Tuple[str, int], torch.nn.Module] = {}

    def __init__(self, cfg):
        super().__init__(cfg)
        opt = getattr(cfg, "attack_options", None)

        self.selected_attrs: List[str] = list(getattr(
            opt,
            "stargan2_selected_attrs",
            ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
        ))
        self.attr_to_idx = {name: i for i, name in enumerate(self.selected_attrs)}

        self.image_size = int(getattr(opt, "stargan2_image_size", 256))
        self.flip_attr_name = str(getattr(opt, "stargan2_flip_attr", "Male"))
        self.flip_attr_idx = int(getattr(
            opt,
            "stargan2_flip_attr_idx",
            self.attr_to_idx.get(self.flip_attr_name, 3),
        ))
        # 与 Attack-StarGAN/network/noise_layers/StarGAN.py 对齐：
        # 仅对单个维度做 1-x 翻转，不做额外的 hair_exclusive 约束。
        # （保留 cfg 字段 stargan2_hair_exclusive 也不影响接口兼容）


        self.ckpt_path = self._resolve_ckpt_path(str(getattr(opt, "stargan2_ckpt", "")).strip())
        self._load_model()

    def _resolve_ckpt_path(self, configured: str) -> str:
        candidates = []
        if configured:
            candidates.append(configured)
        candidates.extend([
            os.path.join(_PROJECT_ROOT, "Attack-StarGAN", "stargan", "models", "200000-G.ckpt"),
            os.path.join(_PROJECT_ROOT, "Attack-StarGAN", "network", "noise_layers", "stargan", "stargan", "models", "200000-G.ckpt"),
        ])
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Attack-StarGAN checkpoint not found. "
            f"checked: {candidates}"
        )

    def _load_model(self):
        cache_key = (self.ckpt_path, len(self.selected_attrs))
        if cache_key in StarGAN2Attack._MODEL_CACHE:
            self._model = StarGAN2Attack._MODEL_CACHE[cache_key]
            return

        model_py = os.path.join(_STARGAN2_ROOT, "model.py")
        if not os.path.exists(model_py):
            raise FileNotFoundError(f"Attack-StarGAN model.py not found: {model_py}")
        try:
            spec = importlib.util.spec_from_file_location("attack_stargan2_model", model_py)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            Generator = module.Generator
        except Exception as e:
            raise ImportError(f"无法加载 Attack-StarGAN Generator（file={model_py}）\n{e}")

        model = Generator(64, len(self.selected_attrs), 6)
        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        msg = model.load_state_dict(state, strict=False)
        if len(msg.missing_keys) > 0:
            raise RuntimeError(
                f"Attack-StarGAN checkpoint 与当前结构不匹配，missing_keys={len(msg.missing_keys)}"
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        StarGAN2Attack._MODEL_CACHE[cache_key] = model

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        # Generator 期望 canonical [-1,1] 输入；这里只做尺寸对齐。
        return F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def generate(self, preprocessed: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("stargan2 必须使用 attack_with_cover(..., batch=...) 以读取属性标签。")

    def postprocess(self, output: torch.Tensor, original_size: tuple) -> torch.Tensor:
        restored = F.interpolate(output, size=original_size, mode="bilinear", align_corners=False)
        return restored.clamp(-1, 1)

    def _build_target_labels(self, batch, bsz: int, device: torch.device) -> torch.Tensor:
        if not isinstance(batch, dict) or "attrs" not in batch:
            raise RuntimeError(
                "stargan2 需要 batch['attrs']。"
                "请使用包含 CelebA 属性列（如 Black_Hair/...）的 CSV。"
            )
        attrs = batch["attrs"]
        if attrs.ndim != 2 or attrs.shape[1] < len(self.selected_attrs):
            raise RuntimeError(
                f"batch['attrs'] 维度不匹配：got {tuple(attrs.shape)}, "
                f"expect [B, {len(self.selected_attrs)}]"
            )
        if attrs.shape[0] != bsz:
            raise RuntimeError(
                f"batch size mismatch: attrs={attrs.shape[0]} vs images={bsz}"
            )
        c_org = attrs[:, :len(self.selected_attrs)].to(device=device, dtype=torch.float32)
        c_org = (c_org > 0.5).float()
        return c_org

    def attack_with_cover(self, wm_images: torch.Tensor, cover_images: torch.Tensor, batch=None) -> torch.Tensor:
        del cover_images  # StarGAN v1 仅依赖输入与属性标签。
        original_size = (wm_images.shape[-2], wm_images.shape[-1])
        pre = self.preprocess(wm_images)
        device = pre.device
        self._model.to(device)
        # 按 Attack-StarGAN/train_denoise_step.py 使用的 StarGAN_multi_step：
        #   - label 初始来自 batch['attrs']
        #   - 连续 3 次生成
        #   - 每一步随机选一个维度 k，对该维度做 1-x 翻转（对整个 batch 同一 k）
        modified_label = self._build_target_labels(batch, pre.shape[0], device)
        current = pre
        c_dim = int(modified_label.size(1))
        # StarGAN_multi_step 内部是 torch.randint(0, 5)，这里保持一致：c_dim==5 才精确复现。
        randint_upper = 5 if c_dim == 5 else c_dim
        with torch.no_grad():
            for _ in range(3):
                k = torch.randint(0, randint_upper, (1,), device=device).item()
                if k < 0 or k >= c_dim:
                    # 理论上不会发生（k 来自 randint_upper），但做一次防御式检查更稳。
                    raise RuntimeError(f"StarGAN multi-step k 越界: k={k}, c_dim={c_dim}")
                modified_label[:, k] = 1.0 - modified_label[:, k]
                current = self._model(current, modified_label)
        out = self.postprocess(current, original_size)

        self._validate_output(out, original_size)
        return out

