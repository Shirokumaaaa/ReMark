#!/usr/bin/env python
# -*- coding: utf-8 -*-


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import json
import hashlib
from PIL import Image 
import numpy as np
from pathlib import Path
import torch 
from torch.utils.data import Dataset, DataLoader
# from functools import partial
import pytorch_lightning as pl
from ldm.util import instantiate_from_config
import pandas as pd
import os, random
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torchvision import transforms

_PROMPT_GENERATOR_CACHE: Dict[Tuple[str, str, str, str, int, int, float, str], Tuple[torch.nn.Module, Any]] = {}


def worker_init_fn(_):
    """
    This function is from RoSteALS
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModule(pl.LightningDataModule):
    def __init__(self, train, validation, batch_size= 8, num_workers=None , use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if self.use_worker_init_fn:
            self.init_fn = worker_init_fn
        else:
            self.init_fn = None
        
        self.train_config = train
        self.validation_config = validation

    def setup(self, stage= None):
        self.dataset_train = instantiate_from_config(self.train_config)
        self.dataset_validation = instantiate_from_config(self.validation_config)
        self.train_num_workers = self._resolve_num_workers(self.dataset_train, split="train")
        self.validation_num_workers = self._resolve_num_workers(self.dataset_validation, split="validation")

    def _resolve_num_workers(self, dataset, split):
        if getattr(dataset, "force_num_workers_zero", False) and self.num_workers != 0:
            print(f"Using num_workers=0 for {split} because the dataset generates images on the fly.")
            return 0
        return self.num_workers
       

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          num_workers=self.train_num_workers, shuffle=True,
                          worker_init_fn=self.init_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_validation, batch_size=self.batch_size,
                          num_workers=self.validation_num_workers, shuffle=False,
                          worker_init_fn=self.init_fn, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_validation, batch_size=self.batch_size,
                          num_workers=self.validation_num_workers, shuffle=False,
                          worker_init_fn=self.init_fn, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset_validation, batch_size=self.batch_size,
                          num_workers=self.validation_num_workers, shuffle=False,
                          worker_init_fn=self.init_fn, drop_last=True)


def build_random_resized_crop_transforms(resize, transform=None):
    if resize != 'all':
        if transform is None:
            return [transforms.RandomResizedCrop((resize, resize), scale=(0.8, 1.0), ratio=(0.75, 1.33))]
        if isinstance(transform, (list, tuple)):
            return list(transform)
        return [transform]
    return [
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    ]


def sample_message(message_len):
    msgs = torch.rand(message_len) > 0.5
    return 2 * msgs.type(torch.float32) - 1.


def load_image_as_training_array(image, transforms_list):
    image = image.convert('RGB')
    transform = random.choice(transforms_list)
    image = transform(image)
    return np.array(image, dtype=np.float32) / 127.5 - 1.


def detect_prompt_column(df, prompt_column=None):
    if prompt_column is not None:
        if prompt_column not in df.columns:
            raise ValueError(f"Prompt column '{prompt_column}' was not found. Available columns: {list(df.columns)}")
        return prompt_column

    candidate_columns = ["prompt", "prompts", "text", "caption", "captions", "Prompt", "Prompts", "TEXT"]
    for column in candidate_columns:
        if column in df.columns:
            return column

    for column in df.columns:
        lowered = column.lower()
        if any(keyword in lowered for keyword in ("prompt", "caption", "text")):
            return column

    string_columns = [column for column in df.columns if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object]
    if len(string_columns) == 1:
        return string_columns[0]

    raise ValueError(
        "Unable to infer the prompt column from the parquet file. "
        f"Available columns: {list(df.columns)}. Please set 'prompt_column' explicitly."
    )


def read_parquet_with_hint(path):
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Reading parquet prompt datasets requires 'pyarrow' or 'fastparquet'. "
            "The provided Forensic-LaWa environment already includes 'fastparquet'."
        ) from exc


class StableDiffusionPromptRenderer:
    def __init__(
        self,
        sd_config,
        sd_ckpt,
        sampler="ddim",
        ddim_steps=50,
        ddim_eta=0.0,
        scale=7.5,
        H=512,
        W=512,
        C=4,
        f=8,
        precision="autocast",
        device=None,
    ):
        self.sd_config = sd_config
        self.sd_ckpt = sd_ckpt
        self.sampler = sampler
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.scale = scale
        self.H = H
        self.W = W
        self.C = C
        self.f = f
        self.precision = precision
        self.device = self.resolve_device(device)

    @staticmethod
    def resolve_device(device):
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device("cuda", local_rank)
        return torch.device("cpu")

    @staticmethod
    def load_model_from_config(config, ckpt, verbose=False):
        print(f"Loading Stable Diffusion model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        state_dict = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        model = instantiate_from_config(config.model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if verbose and missing:
            print(f"Missing keys: {missing}")
        if verbose and unexpected:
            print(f"Unexpected keys: {unexpected}")
        model.eval()
        return model

    def get_components(self):
        if not os.path.exists(self.sd_config):
            raise FileNotFoundError(f"Stable Diffusion config was not found: {self.sd_config}")
        if not os.path.exists(self.sd_ckpt):
            raise FileNotFoundError(
                f"Stable Diffusion checkpoint was not found: {self.sd_ckpt}. "
                "Download the SD v1.4 checkpoint first or update 'sd_ckpt' in the config."
            )

        cache_key = (
            os.path.abspath(self.sd_config),
            os.path.abspath(self.sd_ckpt),
            str(self.device),
            self.sampler,
            self.ddim_steps,
            self.ddim_eta,
            self.scale,
            self.precision,
        )
        if cache_key in _PROMPT_GENERATOR_CACHE:
            return _PROMPT_GENERATOR_CACHE[cache_key]

        from omegaconf import OmegaConf
        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.models.diffusion.plms import PLMSSampler
        from ldm.models.diffusion.dpm_solver import DPMSolverSampler
        from utils_img import no_ssl_verification

        config = OmegaConf.load(self.sd_config)
        with no_ssl_verification():
            model = self.load_model_from_config(config, self.sd_ckpt)
        model = model.to(self.device)

        if self.sampler == "plms":
            sampler = PLMSSampler(model)
        elif self.sampler == "dpm_solver":
            sampler = DPMSolverSampler(model)
        else:
            sampler = DDIMSampler(model)

        _PROMPT_GENERATOR_CACHE[cache_key] = (model, sampler)
        return model, sampler

    @torch.no_grad()
    def render(self, prompt, seed):
        model, sampler = self.get_components()
        device_ids = [self.device.index] if self.device.type == "cuda" and self.device.index is not None else []
        autocast_context = torch.autocast("cuda") if self.precision == "autocast" and self.device.type == "cuda" else nullcontext()

        with torch.random.fork_rng(devices=device_ids):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            with autocast_context:
                with model.ema_scope() if hasattr(model, "ema_scope") else nullcontext():
                    uc = None
                    if self.scale != 1.0:
                        uc = model.get_learned_conditioning([""])
                    conditioning = model.get_learned_conditioning([prompt])
                    shape = [self.C, self.H // self.f, self.W // self.f]
                    samples, _ = sampler.sample(
                        S=self.ddim_steps,
                        conditioning=conditioning,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=self.scale,
                        unconditional_conditioning=uc,
                        eta=self.ddim_eta,
                        x_T=None,
                    )
                    decoded = model.decode_first_stage(samples)
                    decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                    image = decoded[0].detach().cpu().permute(1, 2, 0).numpy()
                    return Image.fromarray((255.0 * image).astype(np.uint8))

class dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_list, message_len=48, resize=256, transform=None, **kwargs):
        super().__init__()
        self.transform = build_random_resized_crop_transforms(resize, transform)
        self.data_dir = data_dir
        self.data_list = pd.read_csv(data_list)['path'].tolist()
        self.N = len(self.data_list)
        self.kwargs = kwargs
        self.message_len = message_len
    
    def __getitem__(self, index):
        path = self.data_list[index]
        img = Image.open(os.path.join(self.data_dir, path))
        img = load_image_as_training_array(img, self.transform)
        msgs = sample_message(self.message_len)
        return {'image': img, 'message': msgs}

    def __len__(self) -> int:
        return self.N 


class PromptParquetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        message_len=48,
        resize=256,
        transform=None,
        prompt_column=None,
        cache_dir=None,
        generate_images=True,
        sd_config="stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        sd_ckpt="weights/stable-diffusion-v1/model.ckpt",
        sampler="ddim",
        ddim_steps=50,
        ddim_eta=0.0,
        scale=7.5,
        H=512,
        W=512,
        C=4,
        f=8,
        precision="autocast",
        seed=42,
        device=None,
        max_samples=None,
        **kwargs,
    ):
        super().__init__()
        self.transform = build_random_resized_crop_transforms(resize, transform)
        self.message_len = message_len
        self.cache_dir = cache_dir
        self.generate_images = generate_images
        self.seed = seed
        self.force_num_workers_zero = bool(generate_images)
        self.kwargs = kwargs

        prompt_df = read_parquet_with_hint(data_path)
        prompt_column = detect_prompt_column(prompt_df, prompt_column=prompt_column)
        prompts = [prompt.strip() for prompt in prompt_df[prompt_column].dropna().astype(str).tolist() if prompt.strip()]
        if max_samples is not None:
            prompts = prompts[:max_samples]
        if len(prompts) == 0:
            raise ValueError(f"No usable prompts were found in {data_path}")

        self.prompts = prompts
        self.N = len(self.prompts)
        self.renderer = StableDiffusionPromptRenderer(
            sd_config=sd_config,
            sd_ckpt=sd_ckpt,
            sampler=sampler,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            scale=scale,
            H=H,
            W=W,
            C=C,
            f=f,
            precision=precision,
            device=device,
        )

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, index, prompt):
        if self.cache_dir is None:
            return None
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{index:06d}_{prompt_hash}.png")

    def save_generated_image(self, image, cache_path):
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        image.save(tmp_path, format="PNG")
        os.replace(tmp_path, cache_path)

    def load_or_generate_image(self, index):
        prompt = self.prompts[index]
        cache_path = self.get_cache_path(index, prompt)
        if cache_path is not None and os.path.exists(cache_path):
            with Image.open(cache_path) as cached_image:
                return cached_image.convert("RGB")

        if not self.generate_images:
            raise FileNotFoundError(
                f"Cached image not found for prompt index {index}. "
                f"Expected cache file: {cache_path}"
            )

        image = self.renderer.render(prompt, seed=self.seed + index)
        if cache_path is not None and not os.path.exists(cache_path):
            self.save_generated_image(image, cache_path)
        return image

    def __getitem__(self, index):
        img = self.load_or_generate_image(index)
        img = load_image_as_training_array(img, self.transform)
        msgs = sample_message(self.message_len)
        return {"image": img, "message": msgs}

    def __len__(self) -> int:
        return self.N

# class WrappedDataset(Dataset):
#     """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

#     def __init__(self, dataset):
#         self.data = dataset

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# class DataModuleFromConfig(pl.LightningDataModule):
#     """
#     This code is from RoSteALS paper
#     """
#     def __init__(self, batch_size, train=None, validation=None, test=None, predict=None, wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
#                  shuffle_val_dataloader=False):
#         super().__init__()
#         self.batch_size = batch_size
#         self.dataset_configs = dict()
#         self.num_workers = num_workers if num_workers is not None else batch_size * 2
#         self.use_worker_init_fn = use_worker_init_fn
#         if train is not None:
#             self.dataset_configs["train"] = train
#             self.train_dataloader = self._train_dataloader
#         if validation is not None:
#             self.dataset_configs["validation"] = validation
#             self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
#         if test is not None:
#             self.dataset_configs["test"] = test
#             self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
#         if predict is not None:
#             self.dataset_configs["predict"] = predict
#             self.predict_dataloader = self._predict_dataloader
#         self.wrap = wrap

#     def prepare_data(self):
#         for data_cfg in self.dataset_configs.values():
#             print(data_cfg)
#             instantiate_from_config(data_cfg)

#     def setup(self, stage=None):
#         self.datasets = dict(
#             (k, instantiate_from_config(self.dataset_configs[k]))
#             for k in self.dataset_configs)
#         if self.wrap:
#             for k in self.datasets:
#                 self.datasets[k] = WrappedDataset(self.datasets[k])

#     def _train_dataloader(self):
#         if self.use_worker_init_fn:
#             init_fn = worker_init_fn
#         else:
#             init_fn = None
#         return DataLoader(self.datasets["train"], batch_size=self.batch_size,
#                           num_workers=self.num_workers, shuffle=True,
#                           worker_init_fn=init_fn, drop_last=True)

#     def _val_dataloader(self, shuffle=False):
#         if self.use_worker_init_fn:
#             init_fn = worker_init_fn
#         else:
#             init_fn = None
#         return DataLoader(self.datasets["validation"],
#                           batch_size=self.batch_size,
#                           num_workers=self.num_workers,
#                           worker_init_fn=init_fn,
#                           shuffle=shuffle, drop_last=True)

#     def _test_dataloader(self, shuffle=False):
#         if self.use_worker_init_fn:
#             init_fn = worker_init_fn
#         else:
#             init_fn = None

#         return DataLoader(self.datasets["test"], batch_size=self.batch_size,
#                           num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

#     def _predict_dataloader(self, shuffle=False):
#         if self.use_worker_init_fn:
#             init_fn = worker_init_fn
#         else:
#             init_fn = None
#         return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
#                           num_workers=self.num_workers, worker_init_fn=init_fn)
