import hashlib
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .base import BaseWMAdapter
from .registry import register_wm


_SLEEPERMARK_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Forensic-SleeperMark",
)
_HF_SD14_CACHE_ROOT = "/home/ldy/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots"


def _get_nested_attr(obj, name: str, default=None):
    if obj is None:
        return default
    return getattr(obj, name, default)


def _resolve_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _find_local_sd14_snapshot() -> Optional[str]:
    if not os.path.isdir(_HF_SD14_CACHE_ROOT):
        return None
    snaps = sorted(os.listdir(_HF_SD14_CACHE_ROOT))
    for snap in snaps:
        path = os.path.join(_HF_SD14_CACHE_ROOT, snap)
        if os.path.isfile(os.path.join(path, "model_index.json")):
            return path
    return None


class _View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class _Repeat(nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.repeat(1, *self.sizes)


def _zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def _conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class _Linear(nn.Module):
    def __init__(self, in_features, out_features, activation="relu"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs


class _Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation="relu", strides=1, init=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        if init == "kaiming_normal":
            nn.init.kaiming_normal_(self.conv.weight)
        if init == "zero":
            nn.init.constant_(self.conv.weight, 0)
            nn.init.constant_(self.conv.bias, 0)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.act is not None:
            outputs = self.act(outputs)
        return outputs


class _Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class _SleeperSecretEncoder(nn.Module):
    def __init__(self, secret_size, base_res=32, resolution=64):
        super().__init__()
        log_resolution = int(torch.log2(torch.tensor(resolution)).item())
        log_base = int(torch.log2(torch.tensor(base_res)).item())
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_size, base_res * base_res),
            nn.SiLU(),
            nn.Linear(base_res * base_res, base_res * base_res),
            nn.SiLU(),
            _View(-1, 1, base_res, base_res),
            _Repeat(4, 1, 1),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            _zero_module(_conv_nd(2, 4, 4, 3, padding=1)),
        )

    def forward(self, sec):
        return self.secret_scaler(sec)


class _SleeperExtractor(nn.Module):
    def __init__(self, secret_size=48):
        super().__init__()
        self.decoder = nn.Sequential(
            _Conv2D(4, 64, 3, strides=2, activation="selu"),
            _Conv2D(64, 64, 3, activation="selu"),
            _Conv2D(64, 128, 3, strides=2, activation="selu"),
            _Conv2D(128, 128, 3, activation="selu"),
            _Conv2D(128, 256, 3, strides=2, activation="selu"),
            _Conv2D(256, 256, 3, activation="selu"),
            _Conv2D(256, 512, 3, strides=2, activation="selu"),
            _Conv2D(512, 512, 3, activation="selu"),
            _Flatten(),
        )
        self.mlps = nn.Sequential(
            _Linear(8192, 2048, activation="selu"),
            _Linear(2048, 2048, activation="selu"),
            _Linear(2048, 2048, activation="selu"),
            nn.Dropout(p=0.1),
            _Linear(2048, secret_size, activation=None),
        )

    def forward(self, latent):
        return self.mlps(self.decoder(latent))


@register_wm("sleepermark")
@register_wm("SleeperMark")
@register_wm("sleeper_mark")
class SleeperMarkAdapter(BaseWMAdapter):
    """
    SleeperMark Stage1 adapter.

    接口对齐 ReMark 的 image-in/image-out 流程：
      canonical image [-1,1] -> SD VAE latent
      + secret encoder residual latent
      -> SD VAE decode 得到水印图像
      -> extractor 输出 message logits
    """

    _cache_key = None
    _cache_models: Optional[Tuple[nn.Module, nn.Module, nn.Module]] = None
    _native_pipe_cache_key = None
    _native_pipe = None

    def __init__(self, cfg):
        scfg = _get_nested_attr(cfg, "wm_adapter_sleepermark", None)
        paths_cfg = _get_nested_attr(cfg, "wm_adapter_paths", None)
        ckpt_cfg = _get_nested_attr(cfg, "wm_adapter_ckpts", None)

        self._root = str(
            _get_nested_attr(scfg, "root", _get_nested_attr(paths_cfg, "sleepermark", _SLEEPERMARK_ROOT))
        )
        self._image_size = int(_get_nested_attr(scfg, "image_size", 512))
        self._message_length = int(_get_nested_attr(scfg, "message_length", 128))
        self._base_res = int(_get_nested_attr(scfg, "base_res", 32))
        self._latent_resolution = int(_get_nested_attr(scfg, "latent_resolution", self._image_size // 8))
        self._deterministic_posterior = bool(_get_nested_attr(scfg, "deterministic_posterior", True))
        self._decode_logits_scale = float(_get_nested_attr(scfg, "decode_logits_scale", 1.0))
        self._encode_mode = str(_get_nested_attr(scfg, "encode_mode", "residual")).strip().lower()
        if self._encode_mode not in ("residual", "native_trigger"):
            raise ValueError(
                f"invalid wm_adapter_sleepermark.encode_mode={self._encode_mode}, "
                "expected 'residual' or 'native_trigger'"
            )
        self._prompt_column = str(_get_nested_attr(scfg, "prompt_column", "prompt"))
        self._trigger = str(_get_nested_attr(scfg, "trigger", "*[Z]& "))
        self._native_seed = int(_get_nested_attr(scfg, "native_seed", 0))
        self._native_num_inference_steps = int(_get_nested_attr(scfg, "native_num_inference_steps", 50))
        self._native_guidance_scale = float(_get_nested_attr(scfg, "native_guidance_scale", 7.5))

        default_encoder_ckpt = _resolve_existing_path(
            [
                os.path.join(self._root, "Stage2", "pretrainedWM", "encoder.pth"),
                os.path.join(self._root, "Stage1", "output_dir", "encoder.pth"),
            ]
        )
        default_decoder_ckpt = _resolve_existing_path(
            [
                os.path.join(self._root, "Stage2", "pretrainedWM", "decoder.pth"),
                os.path.join(self._root, "Stage1", "output_dir", "decoder.pth"),
            ]
        )

        self._encoder_ckpt = str(
            _get_nested_attr(scfg, "encoder_ckpt", _get_nested_attr(ckpt_cfg, "sleepermark_encoder", default_encoder_ckpt))
        )
        self._decoder_ckpt = str(
            _get_nested_attr(scfg, "decoder_ckpt", _get_nested_attr(ckpt_cfg, "sleepermark_decoder", default_decoder_ckpt))
        )

        local_sd14 = _find_local_sd14_snapshot()
        self._vae_model_path = str(
            _get_nested_attr(
                scfg,
                "vae_model_path",
                _get_nested_attr(
                    paths_cfg,
                    "stable_diffusion_v1_4",
                    local_sd14 if local_sd14 is not None else "CompVis/stable-diffusion-v1-4",
                ),
            )
        )
        self._native_unet_dir = str(
            _get_nested_attr(
                scfg,
                "native_unet_dir",
                os.path.join(self._root, "Stage2", "Output"),
            )
        )
        self._secret_pt_path = str(
            _get_nested_attr(
                scfg,
                "secret_pt_path",
                os.path.join(self._root, "Stage2", "pretrainedWM", "secret.pt"),
            )
        )
        self._fixed_secret = None
        self._to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        super().__init__(cfg)

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def message_length(self) -> int:
        return self._message_length

    def _load_vae(self):
        from diffusers import AutoencoderKL

        path = self._vae_model_path
        if os.path.isdir(path):
            if os.path.isdir(os.path.join(path, "vae")):
                return AutoencoderKL.from_pretrained(path, subfolder="vae")
            return AutoencoderKL.from_pretrained(path)
        return AutoencoderKL.from_pretrained(path, subfolder="vae")

    def _load_models(self):
        if self._encode_mode == "residual" and not os.path.isfile(self._encoder_ckpt):
            raise FileNotFoundError(f"SleeperMark encoder checkpoint not found: {self._encoder_ckpt}")
        if not os.path.isfile(self._decoder_ckpt):
            raise FileNotFoundError(f"SleeperMark decoder checkpoint not found: {self._decoder_ckpt}")

        cache_key = (
            self._root,
            self._encode_mode,
            self._encoder_ckpt,
            self._decoder_ckpt,
            self._vae_model_path,
            self._message_length,
            self._base_res,
            self._latent_resolution,
            self._deterministic_posterior,
        )
        if SleeperMarkAdapter._cache_key == cache_key and SleeperMarkAdapter._cache_models is not None:
            self._encoder, self._decoder, self._vae = SleeperMarkAdapter._cache_models
            return

        encoder = None
        if self._encode_mode == "residual":
            encoder = _SleeperSecretEncoder(
                secret_size=self._message_length,
                base_res=self._base_res,
                resolution=self._latent_resolution,
            )
        decoder = _SleeperExtractor(secret_size=self._message_length)

        if encoder is not None:
            encoder.load_state_dict(torch.load(self._encoder_ckpt, map_location="cpu"), strict=True)
        decoder.load_state_dict(torch.load(self._decoder_ckpt, map_location="cpu"), strict=True)

        vae = self._load_vae()
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        self._encoder = encoder
        self._decoder = decoder
        self._vae = vae

        SleeperMarkAdapter._cache_key = cache_key
        SleeperMarkAdapter._cache_models = (encoder, decoder, vae)

    def _ensure_native_pipe(self, device: torch.device):
        if self._encode_mode != "native_trigger":
            return
        if not os.path.isdir(self._native_unet_dir):
            raise FileNotFoundError(f"SleeperMark native UNet dir not found: {self._native_unet_dir}")
        cache_key = (self._native_unet_dir, self._vae_model_path, str(device))
        if SleeperMarkAdapter._native_pipe_cache_key == cache_key and SleeperMarkAdapter._native_pipe is not None:
            return
        from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler

        pipe = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            unet=UNet2DConditionModel.from_pretrained(self._native_unet_dir),
            safety_checker=None,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        SleeperMarkAdapter._native_pipe_cache_key = cache_key
        SleeperMarkAdapter._native_pipe = pipe

    def _seed_from_path(self, path: str, idx: int) -> int:
        key = f"{path}|{idx}|{self._native_seed}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def build_messages(self, batch_size: int, device: torch.device, batch=None):
        if self._encode_mode != "native_trigger":
            return None
        if not os.path.isfile(self._secret_pt_path):
            raise FileNotFoundError(f"SleeperMark secret not found: {self._secret_pt_path}")
        if self._fixed_secret is None:
            self._fixed_secret = torch.load(self._secret_pt_path, map_location="cpu").float().view(1, -1)
        return self._fixed_secret.to(device).repeat(batch_size, 1)

    @torch.no_grad()
    def encode_with_batch(self, images: torch.Tensor, messages: torch.Tensor, batch=None) -> torch.Tensor:
        if self._encode_mode != "native_trigger":
            return self.encode(images, messages)
        if batch is None or self._prompt_column not in batch:
            raise KeyError(
                f"SleeperMark native_trigger mode requires batch['{self._prompt_column}'] prompts."
            )
        prompts = batch[self._prompt_column]
        if not isinstance(prompts, (list, tuple)) or len(prompts) != images.shape[0]:
            raise ValueError(
                f"SleeperMark native_trigger expects '{self._prompt_column}' as list with batch size length."
            )
        img_paths = batch.get("img_path", None)
        if img_paths is None or not isinstance(img_paths, (list, tuple)) or len(img_paths) != images.shape[0]:
            img_paths = [f"idx_{i}" for i in range(images.shape[0])]

        device = images.device
        self._ensure_native_pipe(device)
        pipe = SleeperMarkAdapter._native_pipe
        generated = []
        for i, prompt in enumerate(prompts):
            gen = torch.Generator(device=device).manual_seed(self._seed_from_path(str(img_paths[i]), i))
            out = pipe(
                prompt=self._trigger + str(prompt),
                generator=gen,
                num_inference_steps=self._native_num_inference_steps,
                guidance_scale=self._native_guidance_scale,
            )
            img = out.images[0] if hasattr(out, "images") else out[0][0]
            generated.append(self._to_tensor(img.resize((self._image_size, self._image_size))).to(device))
        wm = torch.stack(generated, dim=0)
        return F.interpolate(wm, size=images.shape[-2:], mode="bilinear", align_corners=False).clamp(-1.0, 1.0)

    def _images_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        # canonical [-1,1] -> [0,1] -> SD-VAE latent
        images_01 = ((images + 1.0) * 0.5).clamp(0.0, 1.0)
        vae_in = images_01 * 2.0 - 1.0
        posterior = self._vae.encode(vae_in).latent_dist
        latent = posterior.mode() if self._deterministic_posterior else posterior.sample()
        return latent * self._vae.config.scaling_factor

    def _latent_to_images(self, latent: torch.Tensor) -> torch.Tensor:
        vae_latent = latent / self._vae.config.scaling_factor
        decoded = self._vae.decode(vae_latent).sample
        images_01 = ((decoded + 1.0) * 0.5).clamp(0.0, 1.0)
        return images_01 * 2.0 - 1.0

    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        if self._encode_mode == "native_trigger":
            raise RuntimeError(
                "native_trigger mode requires batch prompts; call encode_with_batch(...) from training loop."
            )
        device = images.device
        self._vae.to(device)
        if self._encoder is None:
            raise RuntimeError("SleeperMark residual encoder is not loaded.")
        self._encoder.to(device)

        latent = self._images_to_latent(images)
        residual = self._encoder(messages.float())
        wm_latent = latent + residual
        return self._latent_to_images(wm_latent).clamp(-1.0, 1.0)

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._vae.to(device)
        self._decoder.to(device)

        latent = self._images_to_latent(images)
        logits = self._decoder(latent)
        return logits * self._decode_logits_scale
