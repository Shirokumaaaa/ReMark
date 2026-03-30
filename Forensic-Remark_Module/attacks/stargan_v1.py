import os
import importlib.util
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .base import BaseAttack
from .registry import register_attack


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SEPMARK_ROOT = os.path.join(_PROJECT_ROOT, "Forensic-SepMark")


@register_attack("stargan_v1_fixed")
class StarGANV1FixedAttack(BaseAttack):
    """
    StarGAN v1（CelebA 属性翻转）攻击适配器。

    说明：
      - 该版本依赖 batch 中的 CelebA 属性标签（sample['attrs']）。
      - 固定翻转一个属性（由 stargan_v1_fixed_attr / stargan_v1_fixed_attr_idx 指定）。
    """

    _MODEL_CACHE: Dict[Tuple[str, int], torch.nn.Module] = {}

    def __init__(self, cfg):
        super().__init__(cfg)
        opt = getattr(cfg, "attack_options", None)
        self.selected_attrs = list(getattr(
            opt,
            "stargan_v1_selected_attrs",
            ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
        ))
        self.attr_to_idx = {name: i for i, name in enumerate(self.selected_attrs)}
        self.image_size = int(getattr(opt, "stargan_v1_image_size", 256))
        self.fixed_attr_name = str(getattr(opt, "stargan_v1_fixed_attr", "Blond_Hair"))
        self.fixed_attr_idx = int(getattr(
            opt,
            "stargan_v1_fixed_attr_idx",
            self.attr_to_idx.get(self.fixed_attr_name, 1),
        ))
        self.hair_exclusive = bool(getattr(opt, "stargan_v1_hair_exclusive", False))
        if self.fixed_attr_idx < 0 or self.fixed_attr_idx >= len(self.selected_attrs):
            raise ValueError(
                f"stargan_v1_fixed_attr_idx={self.fixed_attr_idx} 越界，"
                f"attrs={self.selected_attrs}"
            )
        self.hair_color_indices = [
            self.attr_to_idx[a]
            for a in ("Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair")
            if a in self.attr_to_idx
        ]
        self.ckpt_path = self._resolve_ckpt_path(str(getattr(opt, "stargan_v1_ckpt", "")).strip())
        self._load_model()

    def _resolve_ckpt_path(self, configured: str) -> str:
        candidates = []
        if configured:
            candidates.append(configured)
        candidates.extend([
            os.path.join(_SEPMARK_ROOT, "network", "noise_layers", "stargan", "256", "200000-G.ckpt"),
            "/home/ldy/..workspace/zhou/repair/modelckpt/200000-G.ckpt",
            "/home/ldy/..workspace/zhou/repair/network/noise_layers/stargan/stargan/models/200000-G.ckpt",
        ])
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "StarGAN v1 checkpoint not found. "
            f"checked: {candidates}"
        )

    def _load_model(self):
        cache_key = (self.ckpt_path, len(self.selected_attrs))
        if cache_key in StarGANV1FixedAttack._MODEL_CACHE:
            self._model = StarGANV1FixedAttack._MODEL_CACHE[cache_key]
            return

        model_py = os.path.join(_SEPMARK_ROOT, "network", "noise_layers", "stargan", "model.py")
        if not os.path.exists(model_py):
            raise FileNotFoundError(f"StarGAN v1 model.py not found: {model_py}")
        try:
            spec = importlib.util.spec_from_file_location("stargan_v1_model", model_py)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            Generator = module.Generator
        except Exception as e:
            raise ImportError(f"无法加载 StarGAN v1 Generator（file={model_py}）\n{e}")

        model = Generator(64, len(self.selected_attrs), 6)
        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        msg = model.load_state_dict(state, strict=False)
        if len(msg.missing_keys) > 0:
            raise RuntimeError(
                f"StarGAN v1 checkpoint 与当前结构不匹配，missing_keys={len(msg.missing_keys)}"
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        StarGANV1FixedAttack._MODEL_CACHE[cache_key] = model

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def generate(self, preprocessed: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("stargan_v1_fixed 必须使用 attack_with_cover(..., batch=...) 以读取属性标签。")

    def postprocess(self, output: torch.Tensor, original_size: tuple) -> torch.Tensor:
        restored = F.interpolate(output, size=original_size, mode="bilinear", align_corners=False)
        return restored.clamp(-1, 1)

    def _build_target_labels(self, batch, bsz: int, device: torch.device) -> torch.Tensor:
        if not isinstance(batch, dict) or "attrs" not in batch:
            raise RuntimeError(
                "stargan_v1_fixed 需要 batch['attrs']。"
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
        c_trg = c_org.clone()
        idx = self.fixed_attr_idx
        if self.hair_exclusive and idx in self.hair_color_indices:
            c_trg[:, self.hair_color_indices] = 0.0
            c_trg[:, idx] = 1.0
        else:
            c_trg[:, idx] = 1.0 - c_org[:, idx]
        return c_trg

    def attack_with_cover(self, wm_images: torch.Tensor, cover_images: torch.Tensor, batch=None) -> torch.Tensor:
        del cover_images  # StarGAN v1 仅用 wm 图和属性标签，不需要 cover。
        original_size = (wm_images.shape[-2], wm_images.shape[-1])
        pre = self.preprocess(wm_images)
        device = pre.device
        self._model.to(device)
        c_trg = self._build_target_labels(batch, pre.shape[0], device)
        with torch.no_grad():
            fake = self._model(pre, c_trg)
        out = self.postprocess(fake, original_size)
        self._validate_output(out, original_size)
        return out
