import copy
import importlib
import os
import sys
from typing import Any, Dict

import torch
import yaml

from .base import BaseWMAdapter
from .registry import register_wm


_LAWA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Forensic-LaWa",
)

_PRESET_SPECS: Dict[str, Dict[str, Any]] = {
    "official_sd14_48": {
        "config_path": "configs/SD14_LaWa_inference.yaml",
        "ckpt": "weights/LaWa/last.ckpt",
        "first_stage_ckpt": "weights/first_stage_models/first_stage_KL-f8.ckpt",
        "message_length": 128,
        "image_size": 128,
        "description": "Official SD1.4 + KL-f8 48-bit LaWa checkpoint.",
    },
    "official_sd14_48_traincfg": {
        "config_path": "configs/SD14_LaWa.yaml",
        "ckpt": "weights/LaWa/last.ckpt",
        "first_stage_ckpt": "weights/first_stage_models/first_stage_KL-f8.ckpt",
        "message_length": 128,
        "image_size": 128,
        "description": "Official 48-bit checkpoint with the training-time model config.",
    },
    "prompt_finetuned_sd14_48": {
        "config_path": "configs/SD14_LaWa_prompt_dataset.yaml",
        "ckpt": "outputs/train_result/checkpoints/last.ckpt",
        "first_stage_ckpt": "weights/first_stage_models/first_stage_KL-f8.ckpt",
        "message_length": 128,
        "image_size": 128,
        "description": "Local 48-bit checkpoint fine-tuned on Stable-Diffusion-Prompts.",
    },
}

_DEFAULT_PRESET_BY_WM_NAME = {
    "lawa": "official_sd14_48",
    "LaWa": "official_sd14_48",
    "lawa_official": "official_sd14_48",
    "lawa_sd14": "official_sd14_48",
    "lawa_sd14_48": "official_sd14_48",
    "lawa_klf8_48": "official_sd14_48",
    "lawa_traincfg": "official_sd14_48_traincfg",
    "lawa_prompt": "prompt_finetuned_sd14_48",
    "lawa_prompt_finetuned": "prompt_finetuned_sd14_48",
    "lawa_prompt_dataset": "prompt_finetuned_sd14_48",
}


def _get_nested_attr(obj, name: str, default=None):
    if obj is None:
        return default
    return getattr(obj, name, default)


def _module_belongs_to_root(module, root: str) -> bool:
    root = os.path.abspath(root)
    mod_file = getattr(module, "__file__", None)
    if mod_file:
        return os.path.abspath(mod_file).startswith(root)
    mod_paths = getattr(module, "__path__", None)
    if mod_paths:
        return any(os.path.abspath(path).startswith(root) for path in mod_paths)
    return False


def _purge_foreign_modules(prefix: str, root: str):
    for key, module in list(sys.modules.items()):
        if key != prefix and not key.startswith(prefix + "."):
            continue
        if module is None:
            continue
        if _module_belongs_to_root(module, root):
            continue
        del sys.modules[key]


class _AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr_dict(value):
    if isinstance(value, dict):
        return _AttrDict({k: _to_attr_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(v) for v in value]
    return value


@register_wm("lawa_prompt_dataset")
@register_wm("lawa_prompt_finetuned")
@register_wm("lawa_prompt")
@register_wm("lawa_traincfg")
@register_wm("lawa_klf8_48")
@register_wm("lawa_sd14_48")
@register_wm("lawa_sd14")
@register_wm("lawa_official")
@register_wm("LaWa")
@register_wm("lawa")
class LaWaAdapter(BaseWMAdapter):
    """
    LaWa watermark adapter.

    LaWa embeds the watermark inside the KL-f8 VAE decoder. For ReMark we expose
    it through the standard image-in / image-out adapter API:

      canonical image [-1,1]
        -> LaWa VAE encode (deterministic posterior mode)
        -> modified decoder inserts the {-1,+1} message
        -> watermarked image [-1,1]

    The decoder head is LaWa's ResNet50 message extractor, whose raw outputs are
    already BCE-ready logits.
    """

    _model_cache = None
    _cache_key = None

    def __init__(self, cfg):
        lcfg = _get_nested_attr(cfg, "wm_adapter_lawa", None)
        paths_cfg = _get_nested_attr(cfg, "wm_adapter_paths", None)
        ckpt_cfg = _get_nested_attr(cfg, "wm_adapter_ckpts", None)

        self._wm_name = str(getattr(cfg, "wm_model", "lawa"))
        self._root = str(
            _get_nested_attr(lcfg, "root", _get_nested_attr(paths_cfg, "lawa", _LAWA_ROOT))
        )

        default_preset = _DEFAULT_PRESET_BY_WM_NAME.get(self._wm_name, "official_sd14_48")
        self._preset = str(_get_nested_attr(lcfg, "preset", default_preset)).lower()
        if self._preset not in _PRESET_SPECS:
            raise ValueError(
                f"Unknown LaWa preset '{self._preset}'. Available presets: {sorted(_PRESET_SPECS)}"
            )
        preset = _PRESET_SPECS[self._preset]

        self._config_path = self._resolve_path(
            str(_get_nested_attr(lcfg, "config_path", preset["config_path"]))
        )
        self._model_ckpt = self._resolve_path(
            str(
                _get_nested_attr(
                    lcfg,
                    "ckpt",
                    _get_nested_attr(
                        ckpt_cfg,
                        "lawa",
                        _get_nested_attr(ckpt_cfg, "lawa_modified_decoder", preset["ckpt"]),
                    ),
                )
            )
        )
        self._first_stage_ckpt = self._resolve_path(
            str(
                _get_nested_attr(
                    lcfg,
                    "first_stage_ckpt",
                    _get_nested_attr(
                        ckpt_cfg,
                        "lawa_first_stage",
                        preset["first_stage_ckpt"],
                    ),
                )
            )
        )
        self._image_size = _get_nested_attr(lcfg, "image_size", preset.get("image_size", 128))
        self._message_length_override = _get_nested_attr(
            lcfg, "message_length", preset.get("message_length", 128)
        )
        self._strict_checkpoint = bool(_get_nested_attr(lcfg, "strict_checkpoint", False))
        self._disable_noise = bool(_get_nested_attr(lcfg, "disable_noise", True))
        self._disable_perceptual_loss = bool(
            _get_nested_attr(lcfg, "disable_perceptual_loss", True)
        )

        self._message_length = int(self._message_length_override)
        self._model = None
        super().__init__(cfg)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self._root, path)

    def _ensure_import_paths(self):
        wanted_root = os.path.abspath(self._root)
        if wanted_root not in sys.path:
            sys.path.insert(0, wanted_root)

        # FIN / other projects also use a top-level "models" package. Since
        # ReMark binds exactly one wm_model per run, we can safely remap these
        # names to the LaWa tree before importing.
        for prefix in ("models", "ldm", "stable-diffusion"):
            existing = sys.modules.get(prefix)
            if existing is not None and _module_belongs_to_root(existing, wanted_root):
                continue
            _purge_foreign_modules(prefix, wanted_root)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LaWa config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "model" not in data:
            raise ValueError(f"LaWa config must contain a top-level 'model' field: {path}")
        return data

    def _build_model_params(self) -> Dict[str, Any]:
        cfg_dict = self._load_yaml(self._config_path)
        params = copy.deepcopy(cfg_dict["model"]["params"])

        first_stage_cfg = params["first_stage_config"]
        first_stage_cfg["params"]["ckpt_path"] = self._first_stage_ckpt

        decoder_cfg = params["decoder_config"]
        decoder_cfg.setdefault("params", {})
        decoder_cfg["params"]["message_len"] = int(self._message_length_override)

        params["ckpt_path"] = "__none__"
        params["use_ema"] = False
        if self._disable_noise:
            params["noise_config"] = "__none__"
        if self._disable_perceptual_loss:
            params["perceptual_loss_weight"] = 0.0
            params["lpips_loss_weights_path"] = None

        for key in (
            "first_stage_config",
            "decoder_config",
            "discriminator_config",
            "noise_config",
            "addition_network_config",
        ):
            if key in params and isinstance(params[key], dict):
                params[key] = _to_attr_dict(params[key])

        return params

    def _load_local_lawa(self):
        self._ensure_import_paths()
        try:
            return importlib.import_module("models.modifiedAEDecoder")
        except ModuleNotFoundError as e:
            raise ImportError(
                "LaWa dependencies are not available in the current environment. "
                "Please install the packages from Forensic-LaWa/environment.yml "
                f"and make sure '{self._root}' is intact.\nOriginal error: {e}"
            ) from e

    def _load_models(self):
        if not os.path.isdir(self._root):
            raise FileNotFoundError(f"LaWa root directory not found: {self._root}")
        if not os.path.isfile(self._model_ckpt):
            raise FileNotFoundError(
                f"LaWa modified decoder checkpoint not found: {self._model_ckpt}"
            )
        if not os.path.isfile(self._first_stage_ckpt):
            raise FileNotFoundError(
                f"LaWa KL-f8 checkpoint not found: {self._first_stage_ckpt}"
            )

        cache_key = (
            self._root,
            self._preset,
            self._config_path,
            self._model_ckpt,
            self._first_stage_ckpt,
            int(self._message_length_override),
            self._disable_noise,
            self._disable_perceptual_loss,
        )
        if LaWaAdapter._cache_key == cache_key and LaWaAdapter._model_cache is not None:
            self._model = LaWaAdapter._model_cache
            self._encoder = self._model
            self._decoder = self._model.decoder
            self._message_length = int(self._model.message_len)
            return

        lawa_mod = self._load_local_lawa()
        model_params = self._build_model_params()
        model = lawa_mod.LaWa(**model_params)

        state = torch.load(self._model_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        try:
            missing, unexpected = model.load_state_dict(state, strict=self._strict_checkpoint)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to load LaWa checkpoint. This usually means the selected "
                "config/message_length does not match the checkpoint family.\n"
                f"Config: {self._config_path}\nCheckpoint: {self._model_ckpt}\n"
                f"Preset: {self._preset}\nOriginal error: {e}"
            ) from e

        if self._strict_checkpoint and (missing or unexpected):
            raise RuntimeError(
                f"LaWa strict checkpoint loading failed. Missing={missing}, unexpected={unexpected}"
            )

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self._model = model
        self._encoder = model
        self._decoder = model.decoder
        self._message_length = int(model.message_len)

        LaWaAdapter._model_cache = model
        LaWaAdapter._cache_key = cache_key

    @property
    def image_size(self):
        if self._image_size is None:
            return None
        return int(self._image_size)

    @property
    def message_length(self) -> int:
        return int(self._message_length)

    def _encode_first_stage_mode(self, images: torch.Tensor) -> torch.Tensor:
        posterior = self._model.ae.encode(images)
        if isinstance(posterior, torch.Tensor):
            z = posterior
        elif hasattr(posterior, "mode"):
            # Use the posterior mode instead of random sampling so that the
            # adapter is deterministic across epochs/caches.
            z = posterior.mode()
        else:
            raise TypeError(f"Unsupported LaWa posterior type: {type(posterior)}")
        return self._model.scale_factor * z

    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._model.to(device)

        latent = self._encode_first_stage_mode(images)
        latent_pp = self._model.ae.post_quant_conv((1.0 / self._model.scale_factor) * latent)
        image_rec = self._model.ae.decoder(latent_pp).clamp(-1, 1)

        lawa_messages = messages.float() * 2.0 - 1.0
        _, watermarked = self._model(latent_pp, image_rec, lawa_messages)
        return watermarked.clamp(-1, 1)

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._model.decoder.to(device)
        return self._model.decoder(images)
