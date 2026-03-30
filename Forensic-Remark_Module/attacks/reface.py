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
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

from .base import BaseAttack
from .registry import register_attack


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_RESULTS_JSONL = os.path.join(
    _PROJECT_ROOT, 'Attack-REFace', 'outputs', 'generation_results.jsonl'
)
_DEFAULT_OUTPUTS_BASE = os.path.join(_PROJECT_ROOT, 'Attack-REFace')


def _norm_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


def _parse_label_list(raw, default):
    if raw is None:
        return list(default)
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return list(default)
    return [int(x.strip()) for x in text.split(',') if str(x).strip()]


@register_attack('reface')
@register_attack('ReFace')
class ReFaceAttack(BaseAttack):
    """
    REFace attack adapter.

    Supports two modes:
      - online: call Attack-REFace inference script and cache outputs
      - replay: JSONL replay lookup
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        opts = getattr(cfg, 'attack_options', None)

        self.mode = str(getattr(opts, 'reface_mode', 'online')).strip().lower()
        if self.mode not in ('online', 'replay'):
            raise ValueError(f"[ReFaceAttack] invalid reface_mode={self.mode}")

        self.allow_missing = bool(getattr(opts, 'reface_allow_missing', False))
        self._swap_check_enabled = bool(getattr(opts, 'enforce_nontrivial_swap', True))
        self._swap_check_eps = float(getattr(opts, 'nontrivial_swap_eps', 1e-4))

        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # replay settings
        self.results_jsonl = _norm_path(getattr(opts, 'reface_results_jsonl', _DEFAULT_RESULTS_JSONL))
        self.outputs_base = _norm_path(getattr(opts, 'reface_outputs_base', _DEFAULT_OUTPUTS_BASE))
        self.replay_key = str(getattr(opts, 'reface_replay_key', 'source_image'))
        self.replay_map = {}

        # online settings
        self.repo_root = _norm_path(getattr(opts, 'reface_repo_root', os.path.join(_PROJECT_ROOT, 'Attack-REFace')))
        self.python_bin = str(getattr(opts, 'reface_python_bin', sys.executable)).strip()
        self.online_config = _norm_path(getattr(
            opts,
            'reface_online_config',
            os.path.join(self.repo_root, 'models', 'REFace', 'configs', 'project_ffhq.yaml'),
        ))
        self.online_ckpt = _norm_path(getattr(
            opts,
            'reface_online_ckpt',
            os.path.join(self.repo_root, 'models', 'REFace', 'checkpoints', 'last.ckpt'),
        ))
        self.online_ddim_steps = int(getattr(opts, 'reface_online_ddim_steps', 40))
        self.online_scale = float(getattr(opts, 'reface_online_scale', 3.5))
        self.online_timeout_sec = int(getattr(opts, 'reface_online_timeout_sec', 3600))
        self.online_batch_size = max(int(getattr(opts, 'reface_online_batch_size', 4)), 1)
        self.online_start_from_target = bool(getattr(opts, 'reface_online_start_from_target', True))
        self.online_target_start_noise_t = int(getattr(opts, 'reface_online_target_start_noise_t', 200))
        self.cache_key_salt = str(getattr(opts, 'reface_cache_key_salt', 'wm_target_softblend_v1')).strip()
        self.online_cache_dir = _norm_path(getattr(
            opts,
            'reface_online_cache_dir',
            '/tmp/remark_reface_online_cache',
        ))
        os.makedirs(self.online_cache_dir, exist_ok=True)

        source_csv = str(getattr(opts, 'reface_source_csv', getattr(getattr(cfg, 'data', None), 'train_csv', ''))).strip()
        self.source_pool = self._load_source_pool_from_csv(source_csv)
        fixed_source_opt = str(getattr(opts, 'reface_fixed_source_path', '')).strip()
        self.fixed_source_path = _norm_path(fixed_source_opt) if fixed_source_opt else ''
        if self.fixed_source_path and (not os.path.exists(self.fixed_source_path)):
            self.fixed_source_path = ''

        self.online_fallback_replay = bool(getattr(opts, 'reface_online_fallback_replay', True))
        self.blend_alpha = float(getattr(opts, 'reface_blend_alpha', 0.95))
        self.face_region_labels = _parse_label_list(
            getattr(opts, 'reface_face_region_labels', None),
            default=[1, 2, 3, 5, 6, 7, 9],
        )
        self.mask_expand_px = int(getattr(opts, 'reface_mask_expand_px', 6))
        self.mask_blur_radius = float(getattr(opts, 'reface_mask_blur_radius', 5.0))

        if self.mode == 'replay' or self.online_fallback_replay:
            self.replay_map = self._build_replay_map(self.results_jsonl)
            if self.mode == 'replay' and len(self.replay_map) == 0:
                raise RuntimeError(
                    f"[ReFaceAttack] No valid replay entries loaded from {self.results_jsonl}"
                )

        if self.mode == 'online':
            if not os.path.exists(self.repo_root):
                raise RuntimeError(f"[ReFaceAttack] repo_root not found: {self.repo_root}")
            if len(self.source_pool) == 0 and (not self.fixed_source_path):
                raise RuntimeError("[ReFaceAttack] source pool is empty for online mode.")

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

    def _pick_source_path(self, target_path: str = '') -> str:
        if self.fixed_source_path:
            return self.fixed_source_path
        if len(self.source_pool) == 0:
            return ''
        if target_path:
            key = hashlib.sha1(_norm_path(target_path).encode('utf-8')).hexdigest()
            idx = int(key, 16) % len(self.source_pool)
            return self.source_pool[idx]
        return self.source_pool[0]

    def _cache_path(self, target_path: str) -> str:
        key = hashlib.sha1(f'{self.cache_key_salt}::{_norm_path(target_path)}'.encode('utf-8')).hexdigest()
        return os.path.join(self.online_cache_dir, f'{key}.png')

    def _mask_cache_path(self, target_path: str) -> str:
        raw = f'{self.cache_key_salt}::{_norm_path(target_path)}::mask'
        key = hashlib.sha1(raw.encode('utf-8')).hexdigest()
        return os.path.join(self.online_cache_dir, f'{key}.png')

    def _tensor_to_pil(self, image_t: torch.Tensor) -> Image.Image:
        image_t = image_t.detach().cpu().clamp(-1.0, 1.0)
        image_t = (image_t + 1.0) * 0.5
        return transforms.ToPILImage()(image_t)

    def _generate_online_for_targets(self, target_paths, target_images=None):
        normalized_paths = [_norm_path(p) for p in target_paths]
        if target_images is not None and len(target_images) != len(normalized_paths):
            raise ValueError('[ReFaceAttack] target_images and target_paths length mismatch.')

        missing_records = []
        for i, target_path in enumerate(normalized_paths):
            if os.path.exists(self._cache_path(target_path)):
                continue
            target_image = None if target_images is None else target_images[i]
            missing_records.append((target_path, target_image))
        if len(missing_records) == 0:
            return

        grouped_records = {}
        for target_path, target_image in missing_records:
            source_path = self._pick_source_path(target_path)
            if not source_path:
                raise RuntimeError('[ReFaceAttack] no source image available for online generation.')
            grouped_records.setdefault(source_path, []).append((target_path, target_image))

        for source_path, records in grouped_records.items():
            chunk_size = self.online_batch_size
            for s in range(0, len(records), chunk_size):
                chunk = records[s:s + chunk_size]
                image_by_path = {tp: ti for tp, ti in chunk}
                try:
                    missing_paths = self._run_online_chunk(source_path, chunk)
                except subprocess.TimeoutExpired:
                    # One slow sample should not kill the whole chunk:
                    # retry one-by-one to isolate pathological targets.
                    if len(chunk) <= 1:
                        raise
                    for single in chunk:
                        self._run_online_chunk(source_path, [single])
                    missing_paths = [
                        tp for tp, _ in chunk if not os.path.exists(self._cache_path(tp))
                    ]

                if not missing_paths:
                    continue

                unresolved = []
                for target_path in missing_paths:
                    single_missing = self._run_online_chunk(
                        source_path,
                        [(target_path, image_by_path.get(target_path, None))],
                    )
                    if single_missing:
                        unresolved.extend(single_missing)

                if not unresolved:
                    continue

                # If face detection fails on wm target, retry using clean target path.
                still_missing = []
                for target_path in unresolved:
                    if image_by_path.get(target_path, None) is None:
                        still_missing.append(target_path)
                        continue
                    single_missing = self._run_online_chunk(source_path, [(target_path, None)])
                    if single_missing:
                        still_missing.extend(single_missing)

                if still_missing:
                    uniq_missing = sorted(set(still_missing))
                    raise RuntimeError(
                        "[ReFaceAttack] online generation missing outputs after retries. "
                        f"source={source_path} missing={uniq_missing}"
                    )

        unresolved_all = [
            tp for tp in normalized_paths if not os.path.exists(self._cache_path(tp))
        ]
        if unresolved_all:
            raise RuntimeError(
                "[ReFaceAttack] online generation finished with unresolved targets: "
                f"{sorted(set(unresolved_all))}"
            )

    def _run_online_chunk(self, source_path: str, records):
        tmp_root = tempfile.mkdtemp(prefix='remark_reface_online_')
        try:
            src_dir = os.path.join(tmp_root, 'source')
            tgt_dir = os.path.join(tmp_root, 'target')
            out_dir = os.path.join(tmp_root, 'out')
            base_dir = os.path.join(tmp_root, 'base')
            os.makedirs(src_dir, exist_ok=True)
            os.makedirs(tgt_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(base_dir, exist_ok=True)

            src_name = os.path.basename(source_path)
            shutil.copy2(source_path, os.path.join(src_dir, src_name))

            for i, (target_path, target_image) in enumerate(records):
                ext = os.path.splitext(target_path)[1].lower() or '.png'
                target_out = os.path.join(tgt_dir, f'{i:06d}{ext}')
                if target_image is None:
                    shutil.copy2(target_path, target_out)
                else:
                    self._tensor_to_pil(target_image).save(target_out)

            cmd = [
                self.python_bin,
                'scripts/inference_swap_selected.py',
                '--outdir', out_dir,
                '--target_folder', tgt_dir,
                '--src_folder', src_dir,
                '--Base_dir', base_dir,
                '--config', self.online_config,
                '--ckpt', self.online_ckpt,
                '--n_samples', '1',
                '--scale', str(self.online_scale),
                '--ddim_steps', str(self.online_ddim_steps),
            ]
            if self.online_start_from_target:
                cmd.extend([
                    '--Start_from_target',
                    '--target_start_noise_t', str(self.online_target_start_noise_t),
                ])
            env = os.environ.copy()
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

            result_dir = os.path.join(out_dir, 'results', '0')
            mask_dir = os.path.join(base_dir, 'mask_frames')
            generated = sorted([x for x in os.listdir(result_dir) if x.lower().endswith('.png')]) if os.path.isdir(result_dir) else []
            generated_by_idx = {}
            for fn in generated:
                stem = os.path.splitext(fn)[0]
                if stem.isdigit():
                    generated_by_idx[int(stem)] = os.path.join(result_dir, fn)
            missing_paths = []
            for i, (target_path, _) in enumerate(records):
                out_img = generated_by_idx.get(i, None)
                cand = os.path.join(result_dir, f'{i}.png')
                if os.path.exists(cand):
                    out_img = cand
                if out_img is None and i < len(generated):
                    out_img = os.path.join(result_dir, generated[i])
                if out_img and os.path.exists(out_img):
                    shutil.copy2(out_img, self._cache_path(target_path))
                else:
                    missing_paths.append(target_path)
                mask_img = os.path.join(mask_dir, f'{i}.png')
                if os.path.exists(mask_img):
                    shutil.copy2(mask_img, self._mask_cache_path(target_path))
            return missing_paths
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

    def _load_face_mask_tensor(self, path: str, h: int, w: int, device: torch.device):
        if not path or (not os.path.exists(path)):
            return None
        mask_img = Image.open(path).convert('L').resize((w, h), Image.NEAREST)
        mask_np = np.array(mask_img, dtype=np.uint8)
        face_mask = np.isin(mask_np, np.array(self.face_region_labels, dtype=np.uint8)).astype(np.uint8) * 255
        mask_pil = Image.fromarray(face_mask, mode='L')
        if self.mask_expand_px > 0:
            for _ in range(self.mask_expand_px):
                mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
        if self.mask_blur_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=self.mask_blur_radius))
        mask_t = transforms.ToTensor()(mask_pil).to(device)
        return mask_t.clamp(0.0, 1.0)

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
            raise KeyError("[ReFaceAttack] batch['img_path'] is required.")

        paths = batch['img_path']
        if not isinstance(paths, (list, tuple)):
            if self.allow_missing:
                return wm_images
            raise TypeError("[ReFaceAttack] batch['img_path'] must be list/tuple of paths.")

        if self.mode == 'online':
            try:
                self._generate_online_for_targets(paths, wm_images)
            except Exception:
                if not (self.online_fallback_replay or self.allow_missing):
                    raise

        b, _, h, w = wm_images.shape
        out = []
        masks = []
        swapped_ok = []
        for i in range(b):
            key = _norm_path(paths[i])
            fake_path = self._resolve_fake_path(key)
            if fake_path is None or (not os.path.exists(fake_path)):
                if self.allow_missing:
                    out.append(wm_images[i].detach())
                    masks.append(None)
                    swapped_ok.append(False)
                    continue
                raise KeyError(
                    f"[ReFaceAttack] missing fake for: {key}\n"
                    f"mode={self.mode}"
                )
            out.append(self._load_fake_tensor(fake_path, h, w, wm_images.device))
            masks.append(self._load_face_mask_tensor(self._mask_cache_path(key), h, w, wm_images.device))
            swapped_ok.append(True)

        attacked_raw = torch.stack(out, dim=0).clamp(-1, 1)
        a = min(max(float(self.blend_alpha), 0.0), 1.0)
        attacked = []
        for i in range(b):
            base = wm_images[i]
            fake = attacked_raw[i]
            face_mask = masks[i]
            if face_mask is None:
                composed = fake * a + base * (1.0 - a) if a < 1.0 else fake
            else:
                face_fake = fake * a + base * (1.0 - a) if a < 1.0 else fake
                composed = face_fake * face_mask + base * (1.0 - face_mask)
            attacked.append(composed.clamp(-1, 1))
        attacked = torch.stack(attacked, dim=0)
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
                f"[ReFaceAttack] nontrivial swap check failed: "
                f"{bad_n}/{original.shape[0]} samples have mean_abs_diff < "
                f"{self._swap_check_eps:.1e}."
            )
