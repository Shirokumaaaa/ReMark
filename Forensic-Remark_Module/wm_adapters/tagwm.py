import hashlib
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T

from .base import BaseWMAdapter
from .registry import register_wm


_TAGWM_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Forensic-TAG-WM",
)


def _get_nested_attr(obj, name: str, default=None):
    if obj is None:
        return default
    return getattr(obj, name, default)


def _safe_logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    q = p.clamp(eps, 1.0 - eps)
    return torch.log(q / (1.0 - q))


@register_wm("tagwm")
@register_wm("TAGWM")
@register_wm("tag_wm")
@register_wm("TAG_WM")
@register_wm("tag-wm")
@register_wm("TAG-WM")
class TAGWMAdapter(BaseWMAdapter):
    """
    TAG-WM adapter aligned to the native SD2 workflow:
      prompt + watermark latent -> watermarked image
      image -> SD inversion -> watermark decode
    """

    _cache_key = None
    _cache_models: Optional[Tuple[object, object, object]] = None

    def __init__(self, cfg):
        tcfg = _get_nested_attr(cfg, "wm_adapter_tagwm", None)
        paths_cfg = _get_nested_attr(cfg, "wm_adapter_paths", None)
        ckpt_cfg = _get_nested_attr(cfg, "wm_adapter_ckpts", None)

        self._root = str(
            _get_nested_attr(tcfg, "root", _get_nested_attr(paths_cfg, "tagwm", _TAGWM_ROOT))
        )
        self._image_size = int(_get_nested_attr(tcfg, "image_size", 512))
        self._message_length = int(_get_nested_attr(tcfg, "message_length", 128))
        self._prompt_column = str(_get_nested_attr(tcfg, "prompt_column", "prompt"))
        self._native_num_inference_steps = int(_get_nested_attr(tcfg, "native_num_inference_steps", 50))
        self._native_guidance_scale = float(_get_nested_attr(tcfg, "native_guidance_scale", 7.5))
        self._native_num_inversion_steps = int(
            _get_nested_attr(tcfg, "native_num_inversion_steps", self._native_num_inference_steps)
        )
        self._native_scheduler = str(_get_nested_attr(tcfg, "native_scheduler", "DDIM")).upper()
        self._tester_prompt = str(_get_nested_attr(tcfg, "tester_prompt", ""))
        self._native_seed = int(_get_nested_attr(tcfg, "native_seed", 0))
        self._use_tamper_loc = bool(_get_nested_attr(tcfg, "use_tamper_loc", False))
        self._center_interval_ratio = float(_get_nested_attr(tcfg, "center_interval_ratio", 0.5))
        self._shuffle_random_seed = int(_get_nested_attr(tcfg, "shuffle_random_seed", 133563))
        self._encrypt_random_seed = int(_get_nested_attr(tcfg, "encrypt_random_seed", 133563))
        self._tlt_intervals_num = int(_get_nested_attr(tcfg, "tlt_intervals_num", 3))
        self._fpr = float(_get_nested_attr(tcfg, "fpr", 1e-6))
        self._user_number = int(_get_nested_attr(tcfg, "user_number", 1_000_000))

        default_model_path = os.path.join(self._root, "models", "stable-diffusion-2-1-base")
        self._model_path = str(_get_nested_attr(tcfg, "model_path", default_model_path))

        default_dvrd_ckpt = os.path.join(
            self._root, "DVRD", "checkpoints", "trainsize-512_epochnum-100_totalstep-33400.pt"
        )
        self._dvrd_ckpt = str(
            _get_nested_attr(tcfg, "dvrd_ckpt", _get_nested_attr(ckpt_cfg, "tagwm_dvrd", default_dvrd_ckpt))
        )

        if torch.cuda.is_available():
            self._runtime_device = f"cuda:{torch.cuda.current_device()}"
        else:
            self._runtime_device = "cpu"
        self._to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        super().__init__(cfg)

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def message_length(self) -> int:
        return self._message_length

    def _load_models(self):
        if not os.path.isdir(self._model_path):
            raise FileNotFoundError(f"TAG-WM model_path not found: {self._model_path}")
        if not os.path.isfile(os.path.join(self._model_path, "unet", "config.json")):
            raise FileNotFoundError(f"TAG-WM UNet config missing under: {self._model_path}/unet")
        if not os.path.isfile(os.path.join(self._model_path, "vae", "config.json")):
            raise FileNotFoundError(f"TAG-WM VAE config missing under: {self._model_path}/vae")
        if not os.path.isfile(self._dvrd_ckpt):
            raise FileNotFoundError(f"TAG-WM DVRD checkpoint not found: {self._dvrd_ckpt}")

        cache_key = (
            self._root,
            self._model_path,
            self._dvrd_ckpt,
            self._message_length,
            self._center_interval_ratio,
            self._shuffle_random_seed,
            self._encrypt_random_seed,
            self._tlt_intervals_num,
            self._fpr,
            self._user_number,
            self._runtime_device,
            self._native_scheduler,
        )
        if TAGWMAdapter._cache_key == cache_key and TAGWMAdapter._cache_models is not None:
            self._pipe, self._embedder, self._text_embeddings = TAGWMAdapter._cache_models
            self._encoder = self._pipe.vae
            self._decoder = self._pipe.vae
            return

        if self._root not in sys.path:
            sys.path.insert(0, self._root)

        from diffusers import DDIMScheduler, DEISMultistepScheduler, DPMSolverMultistepScheduler, PNDMScheduler, StableDiffusionPipeline, UniPCMultistepScheduler
        from applied_to_sd2.watermark_embedder import WatermarkEmbedder

        schedulers = {
            "DDIM": DDIMScheduler,
            "UNIPC": UniPCMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DEIS": DEISMultistepScheduler,
            "DPMSOLVER": DPMSolverMultistepScheduler,
        }
        if self._native_scheduler not in schedulers:
            raise ValueError(f"Unsupported TAG-WM scheduler: {self._native_scheduler}")

        scheduler = schedulers[self._native_scheduler].from_pretrained(self._model_path, subfolder="scheduler")
        pipe_dtype = torch.float16 if self._runtime_device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            self._model_path,
            scheduler=scheduler,
            torch_dtype=pipe_dtype,
            use_safetensors=True,
            safety_checker=None,
        )
        pipe = pipe.to(self._runtime_device)
        # Ensure all major modules share the same dtype/device under DDP.
        pipe.unet = pipe.unet.to(device=self._runtime_device, dtype=pipe_dtype)
        pipe.vae = pipe.vae.to(device=self._runtime_device, dtype=pipe_dtype)
        pipe.text_encoder = pipe.text_encoder.to(device=self._runtime_device, dtype=pipe_dtype)

        embedder = WatermarkEmbedder(
            wm_len=self._message_length,
            center_interval_ratio=self._center_interval_ratio,
            shuffle_random_seed=self._shuffle_random_seed,
            encrypt_random_seed=self._encrypt_random_seed,
            tlt_intervals_num=self._tlt_intervals_num,
            fpr=self._fpr,
            user_number=self._user_number,
            optimize_tamper_loc_method=None,
            DVRD_checkpoint_path=self._dvrd_ckpt,
            DVRD_train_size=self._image_size,
            device=self._runtime_device,
        )
        text_embeddings = self._get_text_embedding(pipe, self._tester_prompt).to(
            device=self._runtime_device,
            dtype=pipe.unet.dtype,
        )

        self._pipe = pipe
        self._embedder = embedder
        self._text_embeddings = text_embeddings
        self._encoder = pipe.vae
        self._decoder = pipe.vae

        TAGWMAdapter._cache_key = cache_key
        TAGWMAdapter._cache_models = (pipe, embedder, text_embeddings)

    def _ensure_runtime_device(self, target_device: torch.device):
        if target_device.type == "cuda":
            dev_index = target_device.index
            if dev_index is None:
                dev_index = torch.cuda.current_device()
            wanted = f"cuda:{int(dev_index)}"
        else:
            wanted = "cpu"
        if wanted == self._runtime_device:
            return
        self._runtime_device = wanted
        TAGWMAdapter._cache_key = None
        TAGWMAdapter._cache_models = None
        self._load_models()

    def _build_tlt(self, latent_len: int) -> np.ndarray:
        return np.arange(latent_len) % 2

    @torch.inference_mode()
    def _get_text_embedding(self, pipe, prompt: str) -> torch.Tensor:
        text_input_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return pipe.text_encoder(text_input_ids.to(self._runtime_device))[0]

    @torch.inference_mode()
    def _get_image_latents(self, image: torch.Tensor, sample: bool = False) -> torch.Tensor:
        image = image.to(device=self._runtime_device, dtype=self._pipe.vae.dtype)
        encoding_dist = self._pipe.vae.encode(image).latent_dist
        encoding = encoding_dist.sample() if sample else encoding_dist.mode()
        return encoding * 0.18215

    @torch.inference_mode()
    def _forward_diffusion(self, latents: torch.Tensor) -> torch.Tensor:
        scheduler = self._pipe.scheduler
        scheduler.set_timesteps(self._native_num_inversion_steps)
        timesteps_tensor = scheduler.timesteps.to(self._runtime_device)
        latents = latents * scheduler.init_noise_sigma

        for t in reversed(timesteps_tensor):
            latent_model_input = scheduler.scale_model_input(latents, t)
            noise_pred = self._pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=self._text_embeddings,
            ).sample
            prev_timestep = (
                t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            )
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else scheduler.final_alpha_cumprod
            )
            alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = (
                alpha_prod_t_prev ** 0.5
                * (
                    (alpha_prod_t ** -0.5 - alpha_prod_t_prev ** -0.5) * latents
                    + (
                        (1 / alpha_prod_t_prev - 1) ** 0.5
                        - (1 / alpha_prod_t - 1) ** 0.5
                    )
                    * noise_pred
                )
                + latents
            )
        return latents

    def _seed_from_path(self, path: str, idx: int) -> int:
        key = f"{path}|{idx}|{self._native_seed}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    @torch.no_grad()
    def encode_with_batch(self, images: torch.Tensor, messages: torch.Tensor, batch=None) -> torch.Tensor:
        if batch is None or self._prompt_column not in batch:
            raise KeyError(f"TAG-WM native mode requires batch['{self._prompt_column}'] prompts.")

        prompts = batch[self._prompt_column]
        if not isinstance(prompts, (list, tuple)) or len(prompts) != images.shape[0]:
            raise ValueError(f"TAG-WM expects '{self._prompt_column}' as list with batch size length.")

        img_paths = batch.get("img_path", None)
        if img_paths is None or not isinstance(img_paths, (list, tuple)) or len(img_paths) != images.shape[0]:
            img_paths = [f"idx_{i}" for i in range(images.shape[0])]

        device = images.device
        self._ensure_runtime_device(device)
        generated = []
        for i, prompt in enumerate(prompts):
            wm_bits = messages[i].detach().round().clamp(0.0, 1.0).to(torch.int64)
            latent_shape = (
                self._pipe.unet.in_channels,
                self._image_size // 8,
                self._image_size // 8,
            )
            tlt = self._build_tlt(int(np.prod(latent_shape)))
            init_latents_w, _ = self._embedder.embedding_wm_tlt(
                wm_bits.to(self._runtime_device),
                tlt,
                latent_size=latent_shape,
            )
            init_latents_w = init_latents_w.to(
                device=self._runtime_device,
                dtype=self._pipe.unet.dtype,
            )
            generator = torch.Generator(device=self._runtime_device).manual_seed(
                self._seed_from_path(str(img_paths[i]), i)
            )
            out = self._pipe(
                str(prompt),
                num_images_per_prompt=1,
                guidance_scale=self._native_guidance_scale,
                num_inference_steps=self._native_num_inference_steps,
                height=self._image_size,
                width=self._image_size,
                latents=init_latents_w,
                generator=generator,
                output_type="pil",
            )
            img = out.images[0] if hasattr(out, "images") else out[0][0]
            generated.append(self._to_tensor(img.resize((self._image_size, self._image_size))).to(device))

        wm = torch.stack(generated, dim=0)
        return self._resize(wm, (images.shape[-2], images.shape[-1])).clamp(-1.0, 1.0)

    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("TAG-WM native mode requires prompts; call encode_with_batch(...) from training loop.")

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        self._ensure_runtime_device(device)

        img_in = self._resize(images, self._image_size).to(self._runtime_device)
        logits = []
        for i in range(img_in.shape[0]):
            image_latents = self._get_image_latents(img_in[i : i + 1], sample=False)
            reversed_latents = self._forward_diffusion(image_latents)
            reversed_wm_repeat, reversed_tlt = self._embedder.deembedding_wm_tlt(reversed_latents)
            pred_tamper_loc_latent_quantized = None
            if self._use_tamper_loc:
                latent_len = int(reversed_wm_repeat.numel())
                latent_hw = self._image_size // 8
                tlt = self._build_tlt(latent_len)
                pred_tamper_loc_latent = self._embedder.get_tamper_loc_latent(
                    tlt=tlt,
                    reversed_tlt=reversed_tlt,
                    latent_size=(4, latent_hw, latent_hw),
                    tamper_confidence=0.5,
                    optimize=False,
                )
                pred_tamper_loc_latent_quantized = (
                    (torch.mean(pred_tamper_loc_latent, dim=0, keepdim=True) >= 0).int().repeat(4, 1, 1)
                )
            reversed_wm = self._embedder.calc_watermark(
                wm_len=self._message_length,
                wm_repeat=reversed_wm_repeat,
                pred_tamper_loc_latent=pred_tamper_loc_latent_quantized,
                with_tamper_loc=self._use_tamper_loc,
            )
            logits.append(_safe_logit(reversed_wm.float().to(device)))
        return torch.stack(logits, dim=0)
