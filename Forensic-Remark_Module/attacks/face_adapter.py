import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import BaseAttack
from .registry import register_attack


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_RESULTS_JSONL = os.path.join(
    _PROJECT_ROOT, 'Attack-Face-Adapter', 'outputs', 'generation_results.jsonl'
)
_DEFAULT_OUTPUTS_BASE = os.path.join(_PROJECT_ROOT, 'Attack-Face-Adapter')


def _norm_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


@register_attack('face_adapter')
@register_attack('Face_Adapter')
@register_attack('FaceAdapter')
class FaceAdapterAttack(BaseAttack):
    """
    Face-Adapter attack adapter.

    Supports two modes:
      - online: call Attack-Face-Adapter infer.py and cache outputs
      - replay: JSONL replay lookup
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        opts = getattr(cfg, 'attack_options', None)

        self.mode = str(getattr(opts, 'face_adapter_mode', 'online')).strip().lower()
        if self.mode not in ('online', 'replay'):
            raise ValueError(f"[FaceAdapterAttack] invalid face_adapter_mode={self.mode}")

        self.allow_missing = bool(getattr(opts, 'face_adapter_allow_missing', False))
        self._swap_check_enabled = bool(getattr(opts, 'enforce_nontrivial_swap', True))
        self._swap_check_eps = float(getattr(opts, 'nontrivial_swap_eps', 1e-4))

        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # replay settings
        self.results_jsonl = _norm_path(getattr(opts, 'face_adapter_results_jsonl', _DEFAULT_RESULTS_JSONL))
        self.outputs_base = _norm_path(getattr(opts, 'face_adapter_outputs_base', _DEFAULT_OUTPUTS_BASE))
        self.replay_key = str(getattr(opts, 'face_adapter_replay_key', 'source_image'))
        self.replay_map = {}

        # online settings
        self.repo_root = _norm_path(getattr(opts, 'face_adapter_repo_root', os.path.join(_PROJECT_ROOT, 'Attack-Face-Adapter')))
        self.python_bin = str(getattr(opts, 'face_adapter_python_bin', sys.executable)).strip()
        self.online_checkpoint = _norm_path(getattr(
            opts,
            'face_adapter_online_checkpoint',
            os.path.join(self.repo_root, 'checkpoints'),
        ))
        self.online_base_model = str(getattr(opts, 'face_adapter_online_base_model', 'runwayml/stable-diffusion-v1-5')).strip()
        user_hf_cache = str(getattr(opts, 'face_adapter_online_hf_cache', '')).strip()
        if user_hf_cache:
            self.online_hf_cache = _norm_path(user_hf_cache)
        else:
            # Prefer global HF cache when available (project local `hub` can be empty).
            global_hf_cache = _norm_path(os.path.join('~', '.cache', 'huggingface', 'hub'))
            repo_hf_cache = _norm_path(os.path.join(self.repo_root, 'hub'))
            self.online_hf_cache = global_hf_cache if os.path.exists(global_hf_cache) else repo_hf_cache
        self.online_local_files_only = bool(getattr(opts, 'face_adapter_online_local_files_only', True))
        self.online_timeout_sec = int(getattr(opts, 'face_adapter_online_timeout_sec', 3600))
        self.online_crop_ratio = float(getattr(opts, 'face_adapter_online_crop_ratio', 0.81))
        self.online_num_inference_steps = int(getattr(opts, 'face_adapter_online_num_inference_steps', 25))
        self.online_guidance_scale = float(getattr(opts, 'face_adapter_online_guidance_scale', 5.0))
        self.online_compose_to_original = bool(getattr(opts, 'face_adapter_online_compose_to_original', True))
        self.blend_alpha = float(getattr(opts, 'face_adapter_blend_alpha', 0.92))
        self.cache_key_salt = str(getattr(opts, 'face_adapter_cache_key_salt', 'wm_target_v1')).strip()
        self.online_cache_dir = _norm_path(getattr(
            opts,
            'face_adapter_online_cache_dir',
            '/tmp/remark_face_adapter_online_cache',
        ))
        os.makedirs(self.online_cache_dir, exist_ok=True)

        source_csv = str(getattr(opts, 'face_adapter_source_csv', getattr(getattr(cfg, 'data', None), 'train_csv', ''))).strip()
        self.source_pool = self._load_source_pool_from_csv(source_csv)
        fixed_source_opt = str(getattr(opts, 'face_adapter_fixed_source_path', '')).strip()
        self.fixed_source_path = _norm_path(fixed_source_opt) if fixed_source_opt else ''
        if self.fixed_source_path and (not os.path.exists(self.fixed_source_path)):
            self.fixed_source_path = ''

        self.online_fallback_replay = bool(getattr(opts, 'face_adapter_online_fallback_replay', True))

        if self.mode == 'replay' or self.online_fallback_replay:
            self.replay_map = self._build_replay_map(self.results_jsonl)
            if self.mode == 'replay' and len(self.replay_map) == 0:
                raise RuntimeError(
                    f"[FaceAdapterAttack] No valid replay entries loaded from {self.results_jsonl}"
                )

        if self.mode == 'online':
            if not os.path.exists(self.repo_root):
                raise RuntimeError(f"[FaceAdapterAttack] repo_root not found: {self.repo_root}")
            if len(self.source_pool) == 0 and (not self.fixed_source_path):
                raise RuntimeError("[FaceAdapterAttack] source pool is empty for online mode.")

    def _resolve_output_path(self, p: str, jsonl_path: str) -> str:
        p = str(p)
        if os.path.isabs(p) and os.path.exists(p):
            return p
        cand1 = os.path.abspath(os.path.join(os.path.dirname(jsonl_path), p))
        if os.path.exists(cand1):
            return cand1
        cand2 = os.path.abspath(os.path.join(self.outputs_base, p))
        if os.path.exists(cand2):
            return cand2
        return cand1

    def _build_replay_map(self, jsonl_path: str):
        if not os.path.exists(jsonl_path):
            return {}
        replay = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get('ok', False):
                    continue
                key_val = rec.get(self.replay_key, None)
                outs = rec.get('outputs', [])
                if key_val is None or not outs:
                    continue
                out_path = self._resolve_output_path(outs[0], jsonl_path)
                if not os.path.exists(out_path):
                    continue
                replay[_norm_path(key_val)] = out_path
        return replay

    def _load_source_pool_from_csv(self, csv_path: str):
        csv_path = str(csv_path).strip()
        if not csv_path:
            return []
        p = Path(csv_path)
        if not p.exists():
            return []
        out = []
        with p.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ip = row.get('img_path', '')
                if not ip:
                    continue
                ip = _norm_path(ip)
                if os.path.exists(ip):
                    out.append(ip)
        return out

    def _pick_source_path(self) -> str:
        if self.fixed_source_path:
            return self.fixed_source_path
        if len(self.source_pool) == 0:
            return ''
        return self.source_pool[0]

    def _cache_path(self, target_path: str) -> str:
        raw = f'{self.cache_key_salt}::{_norm_path(target_path)}'
        key = hashlib.sha1(raw.encode('utf-8')).hexdigest()
        return os.path.join(self.online_cache_dir, f'{key}.png')

    def _tensor_to_pil(self, image_t: torch.Tensor) -> Image.Image:
        image_t = image_t.detach().cpu().clamp(-1.0, 1.0)
        image_t = (image_t + 1.0) * 0.5
        return transforms.ToPILImage()(image_t)

    def _generate_online_for_targets(self, target_paths, target_images=None):
        normalized_paths = [_norm_path(p) for p in target_paths]
        if target_images is not None and len(target_images) != len(normalized_paths):
            raise ValueError('[FaceAdapterAttack] target_images and target_paths length mismatch.')
        missing = []
        for i, target_path in enumerate(normalized_paths):
            if os.path.exists(self._cache_path(target_path)):
                continue
            target_image = None if target_images is None else target_images[i]
            missing.append((target_path, target_image))
        if len(missing) == 0:
            return

        source_path = self._pick_source_path()
        if not source_path:
            raise RuntimeError('[FaceAdapterAttack] no source image available for online generation.')

        tmp_root = tempfile.mkdtemp(prefix='remark_face_adapter_online_')
        try:
            src_dir = os.path.join(tmp_root, 'source')
            tgt_dir = os.path.join(tmp_root, 'target')
            out_dir = os.path.join(tmp_root, 'out')
            os.makedirs(src_dir, exist_ok=True)
            os.makedirs(tgt_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            src_name = 'src.png'
            shutil.copy2(source_path, os.path.join(src_dir, src_name))

            name_to_target = {}
            for i, (target_path, target_image) in enumerate(missing):
                ext = os.path.splitext(target_path)[1].lower() or '.png'
                tname = f'{i:06d}{ext}'
                target_out = os.path.join(tgt_dir, tname)
                if target_image is None:
                    shutil.copy2(target_path, target_out)
                else:
                    self._tensor_to_pil(target_image).save(target_out)
                name_to_target[os.path.splitext(tname)[0]] = target_path

            cmd = [
                self.python_bin,
                'infer.py',
                '-ckpt', self.online_checkpoint,
                '-o', out_dir,
                '-s', src_dir,
                '-t', tgt_dir,
                '-d', self.online_hf_cache,
                '-b', self.online_base_model,
                '-r', str(self.online_crop_ratio),
                '--num_inference_steps', str(self.online_num_inference_steps),
                '--guidance_scale', str(self.online_guidance_scale),
            ]
            cmd.append('--compose_to_original' if self.online_compose_to_original else '--no_compose_to_original')
            if self.online_local_files_only:
                cmd.append('-c')
            env = os.environ.copy()
            # Ensure CUDA/cuDNN runtime libs from the selected python env are visible
            # when launching without explicit `conda activate`.
            py_bin_dir = os.path.dirname(self.python_bin)
            env_root = os.path.dirname(py_bin_dir) if py_bin_dir else ''
            env_lib = os.path.join(env_root, 'lib') if env_root else ''
            torch_lib = os.path.join(env_root, 'lib', 'python3.10', 'site-packages', 'torch', 'lib') if env_root else ''
            ld_parts = []
            if env_lib and os.path.isdir(env_lib):
                ld_parts.append(env_lib)
            if torch_lib and os.path.isdir(torch_lib):
                ld_parts.append(torch_lib)
            if env.get('LD_LIBRARY_PATH'):
                ld_parts.append(env['LD_LIBRARY_PATH'])
            if ld_parts:
                env['LD_LIBRARY_PATH'] = ':'.join(ld_parts)
            if py_bin_dir:
                env['PATH'] = py_bin_dir + (':' + env['PATH'] if env.get('PATH') else '')
            env['PYTHONPATH'] = self.repo_root + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            subprocess.run(
                cmd,
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=self.online_timeout_sec,
            )

            swap_dir = os.path.join(out_dir, 'swap')
            generated = [x for x in os.listdir(swap_dir) if x.lower().endswith('.png')] if os.path.isdir(swap_dir) else []
            for g in generated:
                # expected format after patch: src_<targetstem>.png
                stem = os.path.splitext(g)[0]
                if '_' not in stem:
                    continue
                target_stem = stem.split('_', 1)[1]
                tp = name_to_target.get(target_stem, None)
                if tp is None:
                    continue
                shutil.copy2(os.path.join(swap_dir, g), self._cache_path(tp))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return images.clamp(-1, 1)

    def generate(self, preprocessed: torch.Tensor) -> torch.Tensor:
        return preprocessed

    def postprocess(self, output: torch.Tensor, original_size: tuple) -> torch.Tensor:
        if output.shape[-2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        return output.clamp(-1, 1)

    def _load_fake_tensor(self, path: str, h: int, w: int, device: torch.device) -> torch.Tensor:
        img = Image.open(path).convert('RGB').resize((w, h), Image.BICUBIC)
        return self._to_tensor(img).to(device)

    def _resolve_fake_path(self, target_path: str):
        key = _norm_path(target_path)
        if self.mode == 'online':
            cp = self._cache_path(key)
            if os.path.exists(cp):
                return cp
            if self.online_fallback_replay:
                return self.replay_map.get(key, None)
            return None
        return self.replay_map.get(key, None)

    def attack_with_cover(self, wm_images: torch.Tensor, cover_images: torch.Tensor, batch=None) -> torch.Tensor:
        if batch is None or 'img_path' not in batch:
            if self.allow_missing:
                return wm_images
            raise KeyError("[FaceAdapterAttack] batch['img_path'] is required.")

        paths = batch['img_path']
        if not isinstance(paths, (list, tuple)):
            if self.allow_missing:
                return wm_images
            raise TypeError("[FaceAdapterAttack] batch['img_path'] must be list/tuple of paths.")

        if self.mode == 'online':
            try:
                self._generate_online_for_targets(paths, wm_images)
            except Exception:
                if not (self.online_fallback_replay or self.allow_missing):
                    raise

        b, _, h, w = wm_images.shape
        out = []
        swapped_ok = []
        for i in range(b):
            key = _norm_path(paths[i])
            fake_path = self._resolve_fake_path(key)
            if fake_path is None or (not os.path.exists(fake_path)):
                if self.allow_missing:
                    out.append(wm_images[i].detach())
                    swapped_ok.append(False)
                    continue
                raise KeyError(
                    f"[FaceAdapterAttack] missing fake for: {key}\n"
                    f"mode={self.mode}"
                )
            out.append(self._load_fake_tensor(fake_path, h, w, wm_images.device))
            swapped_ok.append(True)

        attacked = torch.stack(out, dim=0).clamp(-1, 1)
        a = min(max(float(self.blend_alpha), 0.0), 1.0)
        if a < 1.0:
            attacked = attacked * a + wm_images * (1.0 - a)
        valid_idx = [i for i, ok in enumerate(swapped_ok) if ok]
        if len(valid_idx) > 0:
            idx_t = torch.as_tensor(valid_idx, device=wm_images.device, dtype=torch.long)
            self._assert_nontrivial_swap(wm_images.index_select(0, idx_t), attacked.index_select(0, idx_t))
        return attacked

    def _assert_nontrivial_swap(self, original: torch.Tensor, swapped: torch.Tensor):
        if not self._swap_check_enabled:
            return
        with torch.no_grad():
            per_sample_diff = (original - swapped).abs().mean(dim=(1, 2, 3))
            bad = per_sample_diff < self._swap_check_eps
        if bool(bad.any()):
            bad_n = int(bad.sum().item())
            raise RuntimeError(
                f"[FaceAdapterAttack] nontrivial swap check failed: "
                f"{bad_n}/{original.shape[0]} samples have mean_abs_diff < "
                f"{self._swap_check_eps:.1e}."
            )
