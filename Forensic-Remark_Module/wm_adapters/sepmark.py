import importlib
import importlib.util
import os
import sys
from typing import Optional

import torch

from .base import BaseWMAdapter
from .registry import register_wm


_SEPMARK_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Forensic-SepMark"
)


def _get_nested_attr(obj, name: str, default=None):
    if obj is None:
        return default
    return getattr(obj, name, default)


@register_wm("sepmark")
class SepMarkAdapter(BaseWMAdapter):
    """
    SepMark 水印模型适配器。

    SepMark 内部格式：
      - 图像：[-1, 1]（与 canonical 一致）
      - 消息：[-message_range, +message_range]（canonical {0,1} 需转换）
      - 编码：encoder(image, message) -> encoded_image
      - 解码：decoder(image) -> recovered_message (同样约在 ±message_range 附近)

    默认使用 SepMark baseline 的 EC_90 checkpoint。
    """

    _encoder_cache = None
    _decoder_c_cache = None
    _decoder_rf_cache = None
    _cache_key = None

    def __init__(self, cfg):
        sep_cfg = _get_nested_attr(cfg, "wm_adapter_sepmark", None)
        paths_cfg = _get_nested_attr(cfg, "wm_adapter_paths", None)
        ckpt_cfg = _get_nested_attr(cfg, "wm_adapter_ckpts", None)

        self._sep_root = _get_nested_attr(
            sep_cfg, "root",
            _get_nested_attr(paths_cfg, "sepmark", _SEPMARK_ROOT)
        )
        self._message_length = int(_get_nested_attr(sep_cfg, "message_length", 128))
        self._message_range = float(_get_nested_attr(sep_cfg, "message_range", 0.1))
        self._image_size = int(_get_nested_attr(sep_cfg, "image_size", 256))
        self._attention_encoder = _get_nested_attr(sep_cfg, "attention_encoder", "se")
        self._attention_decoder = _get_nested_attr(sep_cfg, "attention_decoder", "se")
        self._decoder_head = str(_get_nested_attr(sep_cfg, "decoder_head", "c")).lower()
        self._decode_logits_mode = str(
            _get_nested_attr(sep_cfg, "decode_logits_mode", "legacy_clamped_logit")
        ).lower()
        self._decode_logits_scale = float(
            _get_nested_attr(sep_cfg, "decode_logits_scale", 1.0)
        )
        self._ckpt_path = _get_nested_attr(
            sep_cfg, "ckpt",
            _get_nested_attr(
                ckpt_cfg, "sepmark",
                os.path.join(
                    self._sep_root,
                    "results",
                    "baseline",
                    "Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54",
                    "models",
                    "EC_90.pth",
                )
            )
        )

        super().__init__(cfg)

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def message_length(self) -> int:
        return self._message_length

    def _load_sepmark_package(self) -> str:
        """
        将 Forensic-SepMark/network 作为独立包加载，避免与 ReMark 自身 network 包重名冲突。
        """
        pkg_name = "_sepmark_network_pkg"
        if pkg_name in sys.modules:
            return pkg_name

        pkg_dir = os.path.join(self._sep_root, "network")
        init_py = os.path.join(pkg_dir, "__init__.py")
        if not os.path.exists(init_py):
            raise FileNotFoundError(f"SepMark network 包不存在：{init_py}")

        spec = importlib.util.spec_from_file_location(
            pkg_name,
            init_py,
            submodule_search_locations=[pkg_dir],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载 SepMark 包：{init_py}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = module
        spec.loader.exec_module(module)
        return pkg_name

    def _load_models(self):
        cache_key = (
            self._sep_root,
            self._ckpt_path,
            self._message_length,
            self._attention_encoder,
            self._attention_decoder,
        )
        if SepMarkAdapter._cache_key == cache_key and \
                SepMarkAdapter._encoder_cache is not None and \
                SepMarkAdapter._decoder_c_cache is not None and \
                SepMarkAdapter._decoder_rf_cache is not None:
            self._encoder = SepMarkAdapter._encoder_cache
            self._decoder_c = SepMarkAdapter._decoder_c_cache
            self._decoder_rf = SepMarkAdapter._decoder_rf_cache
            self._decoder = self._decoder_c if self._decoder_head == "c" else self._decoder_rf
            return

        pkg_name = self._load_sepmark_package()
        enc_mod = importlib.import_module(f"{pkg_name}.Encoder_U")
        dec_mod = importlib.import_module(f"{pkg_name}.Decoder_U")

        DW_Encoder = enc_mod.DW_Encoder
        DW_Decoder = dec_mod.DW_Decoder

        encoder = DW_Encoder(self._message_length, attention=self._attention_encoder)
        decoder_c = DW_Decoder(self._message_length, attention=self._attention_decoder)
        decoder_rf = DW_Decoder(self._message_length, attention=self._attention_decoder)

        if not os.path.exists(self._ckpt_path):
            raise FileNotFoundError(
                f"SepMark checkpoint 不存在：{self._ckpt_path}\n"
                "请设置 config.wm_adapter_ckpts.sepmark 或 wm_adapter_sepmark.ckpt。"
            )

        state = torch.load(self._ckpt_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"SepMark checkpoint 格式异常：{self._ckpt_path}")

        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        dec_c_state = {k[len("decoder_C."):]: v for k, v in state.items() if k.startswith("decoder_C.")}
        dec_rf_state = {k[len("decoder_RF."):]: v for k, v in state.items() if k.startswith("decoder_RF.")}

        if not enc_state or not dec_c_state:
            raise ValueError(
                f"SepMark checkpoint 中未找到 encoder/decoder_C 权重：{self._ckpt_path}"
            )

        encoder.load_state_dict(enc_state, strict=True)
        decoder_c.load_state_dict(dec_c_state, strict=True)
        if dec_rf_state:
            decoder_rf.load_state_dict(dec_rf_state, strict=True)
        else:
            decoder_rf.load_state_dict(dec_c_state, strict=False)

        self._encoder = encoder
        self._decoder_c = decoder_c
        self._decoder_rf = decoder_rf
        self._decoder = decoder_c if self._decoder_head == "c" else decoder_rf

        SepMarkAdapter._encoder_cache = encoder
        SepMarkAdapter._decoder_c_cache = decoder_c
        SepMarkAdapter._decoder_rf_cache = decoder_rf
        SepMarkAdapter._cache_key = cache_key

    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._encoder.to(device)
        sep_messages = (messages.float() * 2.0 - 1.0) * self._message_range
        encoded = self._encoder(images, sep_messages)
        return encoded.clamp(-1, 1)

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._decoder.to(device)
        pred = self._decoder(images)

        # SepMark 输出是连续实值消息（阈值 0 判 bit），转换为 BCE 可用 logits。
        # legacy_clamped_logit: 兼容历史映射（含 clamp，可能产生饱和区）
        # smooth_linear:        直接线性映射为 logits，避免硬裁剪死区
        eps = 1e-6
        r = max(self._message_range, eps)
        mode = self._decode_logits_mode
        if mode in ("legacy", "clamped_logit", "legacy_clamped_logit"):
            p = ((pred / r) + 1.0) * 0.5
            p = p.clamp(eps, 1.0 - eps)
            return torch.logit(p)

        # smooth_linear / linear / direct
        scale = max(self._decode_logits_scale, eps)
        return (pred / r) * scale
