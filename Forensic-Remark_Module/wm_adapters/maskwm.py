import os
import sys

import torch
from omegaconf import OmegaConf

from .base import BaseWMAdapter
from .registry import register_wm


_MASKWM_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Forensic-MaskWM",
)


def _get_nested_attr(obj, name: str, default=None):
    if obj is None:
        return default
    return getattr(obj, name, default)


@register_wm("maskwm")
@register_wm("MaskWM")
@register_wm("mask_wm")
class MaskWMAdapter(BaseWMAdapter):
    """
    MaskWM 水印适配器。

    默认以 D_32bits 做全图水印（message_length=32, image_size=256）。
    也可通过 config.wm_adapter_maskwm.model_name 切到 ED_* 版本。
    """

    _model_cache = None
    _cache_key = None

    def __init__(self, cfg):
        mcfg = _get_nested_attr(cfg, "wm_adapter_maskwm", None)
        paths_cfg = _get_nested_attr(cfg, "wm_adapter_paths", None)
        ckpt_cfg = _get_nested_attr(cfg, "wm_adapter_ckpts", None)

        self._root = _get_nested_attr(
            mcfg, "root",
            _get_nested_attr(paths_cfg, "maskwm", _MASKWM_ROOT),
        )
        self._model_name = str(_get_nested_attr(mcfg, "model_name", "D_128bits"))
        self._blue = bool(_get_nested_attr(mcfg, "blue", True))
        self._use_jnd = bool(_get_nested_attr(mcfg, "use_jnd", True))
        self._jnd_factor = float(_get_nested_attr(
            mcfg, "jnd_factor", 1.75 if self._model_name.startswith("ED_") else 1.3
        ))

        self._model_cfg_path = _get_nested_attr(
            mcfg, "model_config",
            os.path.join(self._root, "configs", "model", f"{self._model_name}.yaml"),
        )
        self._ckpt_path = _get_nested_attr(
            mcfg, "ckpt",
            _get_nested_attr(
                ckpt_cfg, "maskwm",
                os.path.join(self._root, "checkpoints", f"{self._model_name}.pth"),
            ),
        )

        super().__init__(cfg)

    @property
    def image_size(self) -> int:
        return int(self._model_cfg["wm_enc_config"]["image_size"])

    @property
    def message_length(self) -> int:
        return int(self._model_cfg["wm_enc_config"]["message_length"])

    def _load_models(self):
        if not os.path.exists(self._model_cfg_path):
            raise FileNotFoundError(f"MaskWM model config not found: {self._model_cfg_path}")
        if not os.path.exists(self._ckpt_path):
            raise FileNotFoundError(f"MaskWM checkpoint not found: {self._ckpt_path}")

        self._model_cfg = OmegaConf.load(self._model_cfg_path)

        cache_key = (self._root, self._model_cfg_path, self._ckpt_path)
        if MaskWMAdapter._cache_key == cache_key and MaskWMAdapter._model_cache is not None:
            self._encoder = MaskWMAdapter._model_cache
            self._decoder = self._encoder
            return

        if self._root not in sys.path:
            sys.path.insert(0, self._root)

        from models.Mask_Model import WatermarkModel

        wm = WatermarkModel(
            wm_enc_config=self._model_cfg["wm_enc_config"],
            wm_dec_config=self._model_cfg["wm_dec_config"],
            noise_layers="Identity()",
        )
        state = torch.load(self._ckpt_path, map_location="cpu")
        wm.load_state_dict(state, strict=True)
        wm.eval()

        self._encoder = wm
        self._decoder = wm
        MaskWMAdapter._model_cache = wm
        MaskWMAdapter._cache_key = cache_key

    def _build_embed_mask(self, images: torch.Tensor):
        # ED_* 需要 mask；D_* 传 None 即全图逻辑。
        if self._model_name.startswith("ED_"):
            b, _, h, w = images.shape
            return torch.ones((b, 1, h, w), device=images.device, dtype=images.dtype)
        return None

    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._encoder.to(device)
        mask = self._build_embed_mask(images)
        wm = self._encoder.encoder(
            images, messages.float(), mask=mask,
            use_jnd=self._use_jnd, jnd_factor=self._jnd_factor, blue=self._blue
        )
        return wm.clamp(-1, 1)

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._decoder.to(device)
        decoded, _ = self._decoder.decoder(images, mask=None)
        # MaskWM decoder 输出以 [0,1] 为主，转 logits 供 BCE 使用。
        eps = 1e-6
        p = decoded.clamp(eps, 1.0 - eps)
        return torch.logit(p)
