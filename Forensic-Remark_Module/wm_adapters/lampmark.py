import os
import sys

import torch

from .base import BaseWMAdapter
from .registry import register_wm


_LAMPMARK_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Forensic-LampMark",
)


def _get_nested_attr(obj, name: str, default=None):
    if obj is None:
        return default
    return getattr(obj, name, default)


@register_wm("lampmark")
class LampMarkAdapter(BaseWMAdapter):
    """
    LampMark 适配器（Encoder/Decoder 双模型）。

    约定：
      - 输入图像：canonical [-1, 1]
      - 输入消息：canonical {0, 1}，默认 64 bit
      - 编码输出：[-1, 1]
      - 解码输出：默认转为 BCE 友好的 logits（linear-centered）
    """

    _encoder_cache = None
    _decoder_cache = None
    _cache_key = None

    def __init__(self, cfg):
        lcfg = _get_nested_attr(cfg, "wm_adapter_lampmark", None)
        paths_cfg = _get_nested_attr(cfg, "wm_adapter_paths", None)
        ckpt_cfg = _get_nested_attr(cfg, "wm_adapter_ckpts", None)

        self._root = str(
            _get_nested_attr(lcfg, "root", _get_nested_attr(paths_cfg, "lampmark", _LAMPMARK_ROOT))
        )
        self._image_size = int(_get_nested_attr(lcfg, "image_size", 128))
        self._message_length = int(_get_nested_attr(lcfg, "message_length", 128))
        self._encoder_channels = int(_get_nested_attr(lcfg, "encoder_channels", 64))
        self._encoder_blocks = int(_get_nested_attr(lcfg, "encoder_blocks", 3))
        self._decoder_channels = int(_get_nested_attr(lcfg, "decoder_channels", 64))
        self._decoder_blocks = int(_get_nested_attr(lcfg, "decoder_blocks", 1))
        self._diffusion_length = int(_get_nested_attr(lcfg, "diffusion_length", 256))

        default_enc_ckpt = os.path.join(
            self._root, "weights", f"{self._image_size}_{self._message_length}",
            "deepfake", "encoder_epoch_30.pth"
        )
        default_dec_ckpt = os.path.join(
            self._root, "weights", f"{self._image_size}_{self._message_length}",
            "deepfake", "decoder_epoch_30.pth"
        )
        self._encoder_ckpt = str(
            _get_nested_attr(lcfg, "encoder_ckpt", _get_nested_attr(ckpt_cfg, "lampmark_encoder", default_enc_ckpt))
        )
        self._decoder_ckpt = str(
            _get_nested_attr(lcfg, "decoder_ckpt", _get_nested_attr(ckpt_cfg, "lampmark_decoder", default_dec_ckpt))
        )

        self._decoder_logits_mode = str(_get_nested_attr(lcfg, "decoder_logits_mode", "linear_centered")).lower()
        self._decoder_logits_scale = float(_get_nested_attr(lcfg, "decoder_logits_scale", 10.0))
        super().__init__(cfg)

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def message_length(self) -> int:
        return self._message_length

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self._root, p)

    def _load_models(self):
        enc_ckpt = self._resolve_path(self._encoder_ckpt)
        dec_ckpt = self._resolve_path(self._decoder_ckpt)
        if not os.path.exists(enc_ckpt):
            raise FileNotFoundError(f"LampMark encoder checkpoint not found: {enc_ckpt}")
        if not os.path.exists(dec_ckpt):
            raise FileNotFoundError(f"LampMark decoder checkpoint not found: {dec_ckpt}")

        cache_key = (
            self._root,
            enc_ckpt,
            dec_ckpt,
            self._image_size,
            self._message_length,
            self._encoder_channels,
            self._encoder_blocks,
            self._decoder_channels,
            self._decoder_blocks,
            self._diffusion_length,
        )
        if (
            LampMarkAdapter._cache_key == cache_key
            and LampMarkAdapter._encoder_cache is not None
            and LampMarkAdapter._decoder_cache is not None
        ):
            self._encoder = LampMarkAdapter._encoder_cache
            self._decoder = LampMarkAdapter._decoder_cache
            return

        if self._root not in sys.path:
            sys.path.insert(0, self._root)

        try:
            from model.encoder_decoder import Encoder, Decoder
        except ImportError as e:
            raise ImportError(f"LampMark 依赖加载失败，请检查路径 {self._root}\n{e}")

        encoder = Encoder(
            self._image_size,
            self._encoder_channels,
            self._encoder_blocks,
            self._message_length,
            diffusion_length=self._diffusion_length,
        )
        decoder = Decoder(
            self._image_size,
            self._decoder_channels,
            self._decoder_blocks,
            self._message_length,
            diffusion_length=self._diffusion_length,
        )

        encoder.load_state_dict(torch.load(enc_ckpt, map_location="cpu"), strict=True)
        decoder.load_state_dict(torch.load(dec_ckpt, map_location="cpu"), strict=True)

        self._encoder = encoder
        self._decoder = decoder
        LampMarkAdapter._encoder_cache = encoder
        LampMarkAdapter._decoder_cache = decoder
        LampMarkAdapter._cache_key = cache_key

    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._encoder.to(device)
        encoded = self._encoder(images, messages.float())
        return encoded.clamp(-1, 1)

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._decoder.to(device)
        pred = self._decoder(images)
        mode = self._decoder_logits_mode
        if mode in ("raw", "identity"):
            return pred
        if mode in ("clamped_logit", "logit"):
            eps = 1e-6
            p = pred.clamp(eps, 1.0 - eps)
            return torch.logit(p)

        # default: linear_centered
        # LampMark decoder通常以0.5附近为bit分界，映射到logits空间供BCE优化。
        scale = max(self._decoder_logits_scale, 1e-6)
        return (pred - 0.5) * scale
