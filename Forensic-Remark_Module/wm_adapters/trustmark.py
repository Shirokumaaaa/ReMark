import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import BaseWMAdapter
from .registry import register_wm


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRUSTMARK_PY_ROOT = os.path.join(_PROJECT_ROOT, "Forensic-TrustMark", "python")
_TRUSTMARK_MODELS_ROOT = os.path.join(_TRUSTMARK_PY_ROOT, "trustmark", "models")


def _get_nested_attr(obj, name: str, default=None):
    if obj is None:
        return default
    return getattr(obj, name, default)


@register_wm("trustmark")
class TrustMarkAdapter(BaseWMAdapter):
    """
    TrustMark(Q) 适配器。

    说明：
      - 仅支持 Q 模式（按需求不支持 C）。
      - 训练使用 raw 100-bit 消息（use_ECC=False），以保证 decode 输出可直接用于 BCE。
      - 图像 canonical 格式保持 [-1, 1] BCHW。
    """

    _cache_key = None
    _cache_models = None

    def __init__(self, cfg):
        tm_cfg = _get_nested_attr(cfg, "wm_adapter_trustmark", None)
        paths_cfg = _get_nested_attr(cfg, "wm_adapter_paths", None)

        self._mode = str(_get_nested_attr(tm_cfg, "mode", "Q")).upper()
        if self._mode != "Q":
            raise ValueError(
                f"TrustMarkAdapter 仅支持 Q 模式（当前 mode={self._mode}）。"
            )

        self._models_root = str(
            _get_nested_attr(
                tm_cfg,
                "models_root",
                _get_nested_attr(paths_cfg, "trustmark", _TRUSTMARK_MODELS_ROOT),
            )
        )
        py_root_default = _TRUSTMARK_PY_ROOT
        self._python_root = str(_get_nested_attr(tm_cfg, "python_root", py_root_default))
        self._message_length = 100
        self._wm_strength = float(_get_nested_attr(tm_cfg, "wm_strength", 1.0))
        super().__init__(cfg)

    @property
    def image_size(self) -> int:
        # TrustMark encoder 在 256x256 上工作，基类会自动做尺寸适配。
        return 256

    @property
    def message_length(self) -> int:
        return self._message_length

    def _import_local_trustmark(self):
        wanted_root = os.path.abspath(self._python_root)
        if wanted_root not in sys.path:
            sys.path.insert(0, wanted_root)

        existing = sys.modules.get("trustmark")
        if existing is not None:
            mod_file = os.path.abspath(getattr(existing, "__file__", ""))
            if mod_file and not mod_file.startswith(wanted_root):
                for k in list(sys.modules.keys()):
                    if k == "trustmark" or k.startswith("trustmark."):
                        del sys.modules[k]

        return importlib.import_module("trustmark")

    def _check_required_files(self):
        required = [
            "trustmark_Q.yaml",
            "encoder_Q.ckpt",
            "decoder_Q.ckpt",
        ]
        missing = []
        root = Path(self._models_root)
        for name in required:
            p = root / name
            if not p.is_file():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                "TrustMark Q 模型文件缺失，请确认以下文件存在：\n" + "\n".join(missing)
            )

    def _load_models(self):
        self._check_required_files()

        cache_key = (self._python_root, self._models_root, self._mode)
        if TrustMarkAdapter._cache_key == cache_key and TrustMarkAdapter._cache_models is not None:
            self._tm, self._encoder, self._decoder, self._dec_resolution = TrustMarkAdapter._cache_models
            return

        trustmark_pkg = self._import_local_trustmark()
        TrustMark = trustmark_pkg.TrustMark

        # 训练阶段需要 decoder logits 参与 BCE，因此使用 raw bit 流（use_ECC=False）。
        tm = TrustMark(
            use_ECC=False,
            verbose=False,
            model_type="Q",
            encoding_type=TrustMark.Encoding.BCH_5,
            loadRemover=False,
            loadBBoxDetector=False,
        )

        if tm.encoder is None or tm.decoder is None:
            raise RuntimeError("TrustMark Q 模型加载失败，请检查模型文件和路径配置。")

        self._tm = tm
        self._encoder = tm.encoder
        self._decoder = tm.decoder
        self._dec_resolution = int(tm.model_resolution_dec)

        TrustMarkAdapter._cache_key = cache_key
        TrustMarkAdapter._cache_models = (
            self._tm,
            self._encoder,
            self._decoder,
            self._dec_resolution,
        )

    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        # images: BCHW, already resized to 256 by BaseWMAdapter
        if messages.shape[1] != self._message_length:
            raise ValueError(
                f"TrustMark message length must be {self._message_length}, got {messages.shape[1]}"
            )

        device = images.device
        self._encoder.to(device)
        msg = messages.float().to(device)

        stego, _ = self._encoder(images, msg)
        residual = stego.clamp(-1, 1) - images
        residual = residual - residual.mean(dim=(2, 3), keepdim=True)  # align with official encode logic
        out = (images + self._wm_strength * residual).clamp(-1, 1)
        return out

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        # BaseWMAdapter 已将输入 resize 到 256，这里再按官方解码逻辑 resize 到 245。
        device = images.device
        self._decoder.to(device)
        dec_in = F.interpolate(
            images,
            size=(self._dec_resolution, self._dec_resolution),
            mode="bilinear",
            align_corners=False,
        )
        logits = self._decoder.decoder(dec_in)
        return logits
