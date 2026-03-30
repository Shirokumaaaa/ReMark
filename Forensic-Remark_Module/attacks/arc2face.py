import csv
import glob
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import BaseAttack
from .registry import register_attack


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_RESULTS_JSONL = os.path.join(
    _PROJECT_ROOT, 'Attack-arc2face_wrapper', 'outputs_replay_large', 'generation_results.jsonl'
)
_DEFAULT_OUTPUTS_BASE = os.path.join(_PROJECT_ROOT, 'Attack-arc2face_wrapper')


def _norm_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


@register_attack('arc2face')
@register_attack('Arc2Face')
class Arc2FaceAttack(BaseAttack):
    """
    Arc2Face attack adapter.

    Supports two modes:
      - online: in-process Arc2Face inference
      - replay: pre-generated JSONL replay lookup
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        opts = getattr(cfg, 'attack_options', None)
        self.mode = str(getattr(opts, 'arc2face_mode', 'online')).strip().lower()
        if self.mode not in ('online', 'replay'):
            raise ValueError(f"[Arc2FaceAttack] invalid arc2face_mode={self.mode}")

        self.allow_missing = bool(getattr(opts, 'arc2face_allow_missing', False))
        self._swap_check_enabled = bool(getattr(opts, 'enforce_nontrivial_swap', True))
        self._swap_check_eps = float(getattr(opts, 'nontrivial_swap_eps', 1e-4))
        self.blend_alpha = float(getattr(opts, 'arc2face_blend_alpha', 0.89))

        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # replay settings
        self.results_jsonl = _norm_path(getattr(opts, 'arc2face_results_jsonl', _DEFAULT_RESULTS_JSONL))
        self.outputs_base = _norm_path(getattr(opts, 'arc2face_outputs_base', _DEFAULT_OUTPUTS_BASE))
        user_key = str(getattr(opts, 'arc2face_replay_key', 'target_image')).strip()
        fallback = ['target_image', 'source_image', 'expression_image']
        self.replay_keys = [user_key] + [k for k in fallback if k != user_key]
        self._replay_min_unique_ratio = float(getattr(opts, 'arc2face_min_unique_output_ratio', 0.0))
        self.replay_map = {}
        self._replay_stats = {'total': 0, 'ok': 0, 'used_keys': {}, 'unique_outputs': 0}

        # online settings
        self.wrapper_root = _norm_path(getattr(
            opts,
            'arc2face_wrapper_root',
            os.path.join(_PROJECT_ROOT, 'Attack-arc2face_wrapper'),
        ))
        self.models_dir = _norm_path(getattr(
            opts,
            'arc2face_models_dir',
            '/mnt/personal_workspace/chenkeyu/Arc2Face/models',
        ))
        self.strict_cuda_provider = bool(getattr(opts, 'arc2face_strict_cuda_provider', True))
        self.use_ref_adapter = bool(getattr(opts, 'arc2face_use_ref_adapter', True))
        self.online_num_steps = int(getattr(opts, 'arc2face_online_num_steps', 35))
        self.online_guidance_scale = float(getattr(opts, 'arc2face_online_guidance_scale', 3.0))
        self.online_num_images = int(getattr(opts, 'arc2face_online_num_images', 1))
        self.online_exp_adapter_scale = float(getattr(opts, 'arc2face_online_exp_adapter_scale', 1.0))
        self.online_lora_ref_scale = float(getattr(opts, 'arc2face_online_lora_ref_scale', 1.0))
        self.online_seed = int(getattr(opts, 'arc2face_online_seed', 42))

        # Stabilization knobs for low-resolution training pipeline:
        # generate at >=256 then resize back to training resolution.
        self.online_min_output_size = int(getattr(opts, 'arc2face_online_min_output_size', 512))
        self.online_output_size = int(getattr(opts, 'arc2face_online_output_size', 0))

        # expression image source: wm (default) or clean cover image.
        self.online_expression_from = str(getattr(opts, 'arc2face_online_expression_from', 'wm')).strip().lower()
        if self.online_expression_from not in ('wm', 'cover', 'path'):
            raise ValueError(
                f"[Arc2FaceAttack] invalid arc2face_online_expression_from={self.online_expression_from}"
            )

        # reference image source used by ref-adapter.
        self.online_reference_mode = str(getattr(opts, 'arc2face_online_reference_mode', 'expression')).strip().lower()
        if self.online_reference_mode not in ('source', 'expression', 'none'):
            raise ValueError(
                f"[Arc2FaceAttack] invalid arc2face_online_reference_mode={self.online_reference_mode}"
            )

        self.online_source_mode = str(getattr(opts, 'arc2face_online_source_mode', 'roll_batch')).strip().lower()
        if self.online_source_mode not in ('roll_batch', 'dataset_pool'):
            raise ValueError(
                f"[Arc2FaceAttack] invalid arc2face_online_source_mode={self.online_source_mode}"
            )
        self.source_pool = []
        if self.online_source_mode == 'dataset_pool':
            source_csv = str(getattr(opts, 'arc2face_source_csv', getattr(getattr(cfg, 'data', None), 'train_csv', ''))).strip()
            self.source_pool = self._load_source_pool_from_csv(source_csv)
            if len(self.source_pool) == 0:
                raise RuntimeError(
                    "[Arc2FaceAttack] online_source_mode=dataset_pool but source pool is empty."
                )

        self._generator = None
        self._generator_lock = threading.Lock()
        self._online_cfg_cls = None

        if self.mode == 'replay':
            self.replay_map, self._replay_stats = self._build_replay_map(self.results_jsonl)
            if len(self.replay_map) == 0:
                raise RuntimeError(
                    f"[Arc2FaceAttack] No valid replay entries loaded from {self.results_jsonl}"
                )
            total_ok = max(int(self._replay_stats.get('ok', 0)), 1)
            unique_outputs = int(self._replay_stats.get('unique_outputs', 0))
            unique_ratio = unique_outputs / float(total_ok)
            if unique_ratio < self._replay_min_unique_ratio:
                raise RuntimeError(
                    "[Arc2FaceAttack] Replay mapping looks corrupted/misaligned: "
                    f"ok_records={total_ok}, unique_outputs={unique_outputs}, "
                    f"ratio={unique_ratio:.4f} < min_ratio={self._replay_min_unique_ratio:.4f}.\n"
                    "Use a replay jsonl where each target maps to meaningful outputs."
                )

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

    def _pick_key(self, rec: dict):
        for k in self.replay_keys:
            v = rec.get(k, None)
            if v:
                return k, v
        return None, None

    def _build_replay_map(self, jsonl_path: str):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"[Arc2FaceAttack] results jsonl not found: {jsonl_path}"
            )
        replay = {}
        stats = {'total': 0, 'ok': 0, 'used_keys': {}, 'unique_outputs': 0}
        output_set = set()

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats['total'] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get('ok', False):
                    continue
                stats['ok'] += 1

                key_name, key_val = self._pick_key(rec)
                outs = rec.get('outputs', [])
                if key_val is None or not outs:
                    continue
                out_path = self._resolve_output_path(outs[0], jsonl_path)
                if not os.path.exists(out_path):
                    continue

                replay[_norm_path(key_val)] = out_path
                output_set.add(out_path)
                stats['used_keys'][key_name] = int(stats['used_keys'].get(key_name, 0)) + 1

        stats['unique_outputs'] = len(output_set)
        return replay, stats

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

    def _ensure_online_generator(self):
        if self._generator is not None:
            return
        with self._generator_lock:
            if self._generator is not None:
                return
            self._inject_cuda_runtime_libs()
            if self.wrapper_root not in sys.path:
                sys.path.insert(0, self.wrapper_root)
            # exp_utils reads ARC2FACE_MODELS_DIR at import time
            os.environ["ARC2FACE_MODELS_DIR"] = self.models_dir
            from arc2face.expression_generator import Arc2FaceExpressionGenerator, ExpressionGenerationConfig
            self._online_cfg_cls = ExpressionGenerationConfig
            self._generator = Arc2FaceExpressionGenerator(
                models_dir=self.models_dir,
                strict_cuda_provider=self.strict_cuda_provider,
            )

    @staticmethod
    def _inject_cuda_runtime_libs():
        """
        Ensure CUDA11 runtime libs are discoverable for onnxruntime CUDA EP.
        This runs before importing arc2face.expression_generator (which imports ort).
        """
        if not torch.cuda.is_available():
            return
        home = os.path.expanduser('~')
        pkgs_root = os.path.join(home, 'miniconda3', 'pkgs')
        cand = []
        cand.extend(sorted(glob.glob(os.path.join(pkgs_root, 'libcublas-11.*', 'lib')), reverse=True))
        cand.extend(sorted(glob.glob(os.path.join(pkgs_root, 'cudnn-8.*', 'lib')), reverse=True))
        cand.extend(sorted(glob.glob(os.path.join(pkgs_root, 'cudatoolkit-11.*', 'lib')), reverse=True))
        # Always include current conda env lib first (where we may link required CUDA libs).
        exe_dir = os.path.dirname(sys.executable)
        env_root = os.path.dirname(exe_dir)
        cand.insert(0, os.path.join(env_root, 'lib'))
        # Fallback from REFace env torch runtime (contains libcudnn*.so.8).
        cand.append(os.path.join(home, 'miniconda3', 'envs', 'REFace', 'lib', 'python3.10', 'site-packages', 'torch', 'lib'))

        existing = [p for p in cand if os.path.isdir(p)]
        if not existing:
            return
        old = os.environ.get('LD_LIBRARY_PATH', '')
        parts = [p for p in old.split(':') if p] if old else []
        for p in existing:
            if p not in parts:
                parts.insert(0, p)
        os.environ['LD_LIBRARY_PATH'] = ':'.join(parts)

    @staticmethod
    def _tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
        x = img_t.detach().float().clamp(-1, 1)
        x = ((x + 1.0) * 0.5 * 255.0).round().to(torch.uint8)
        x = x.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(x)

    def _pick_source_path(self, target_path: str) -> str:
        if len(self.source_pool) == 0:
            return ''
        key = _norm_path(target_path)
        idx = int(hashlib.sha1(key.encode('utf-8')).hexdigest(), 16) % len(self.source_pool)
        src = self.source_pool[idx]
        if _norm_path(src) == key and len(self.source_pool) > 1:
            src = self.source_pool[(idx + 1) % len(self.source_pool)]
        return src

    def _attack_online(self, wm_images: torch.Tensor, cover_images: torch.Tensor, batch=None) -> torch.Tensor:
        self._ensure_online_generator()
        b, _, h, w = wm_images.shape
        out = []

        batch_paths = None
        if batch is not None and ('img_path' in batch):
            paths = batch['img_path']
            if isinstance(paths, (list, tuple)) and len(paths) == b:
                batch_paths = [str(p) for p in paths]

        for i in range(b):
            try:
                if self.online_source_mode == 'dataset_pool':
                    if batch_paths is None:
                        raise KeyError("[Arc2FaceAttack] batch['img_path'] is required for dataset_pool source mode.")
                    target_path = batch_paths[i]
                    source_path = self._pick_source_path(target_path)
                    if not source_path:
                        raise RuntimeError("[Arc2FaceAttack] source pool is empty.")
                    source_img = source_path
                else:
                    if batch_paths is not None:
                        source_img = batch_paths[(i + 1) % b]
                    else:
                        source_img = self._tensor_to_pil(cover_images[(i + 1) % b])

                # Primary expression source follows config; fallback list is built below
                # for rare face-detector misses on WM images.
                if self.online_expression_from == 'path' and batch_paths is not None:
                    expr_img_primary = batch_paths[i]
                elif self.online_expression_from == 'cover':
                    expr_img_primary = self._tensor_to_pil(cover_images[i])
                else:
                    expr_img_primary = self._tensor_to_pil(wm_images[i])

                if self.online_output_size > 0:
                    out_size = self.online_output_size
                else:
                    out_size = max(self.online_min_output_size, max(8, int(round(h / 8.0)) * 8))
                out_size = max(8, int(round(float(out_size) / 8.0)) * 8)

                if self.online_reference_mode == 'source':
                    ref_img = source_img
                elif self.online_reference_mode == 'none':
                    ref_img = None
                else:
                    ref_img = expr_img_primary

                cfg = self._online_cfg_cls(
                    use_ref_adapter=self.use_ref_adapter,
                    lora_ref_scale=self.online_lora_ref_scale,
                    num_steps=self.online_num_steps,
                    guidance_scale=self.online_guidance_scale,
                    num_images=self.online_num_images,
                    exp_adapter_scale=self.online_exp_adapter_scale,
                    output_size=out_size,
                    seed=self.online_seed + i,
                )
                expr_candidates = [expr_img_primary]
                # Keep Arc2Face online robust: when expression detector misses on WM,
                # retry with clean cover image, then original path (if available).
                expr_cover = self._tensor_to_pil(cover_images[i])
                if not (self.online_expression_from == 'cover'):
                    expr_candidates.append(expr_cover)
                if batch_paths is not None:
                    expr_path = batch_paths[i]
                    if not (self.online_expression_from == 'path'):
                        expr_candidates.append(expr_path)

                imgs = None
                last_exc = None
                for expr_img in expr_candidates:
                    try:
                        imgs = self._generator.generate(
                            source_image=source_img,
                            expression_image=expr_img,
                            reference_image=ref_img,
                            config=cfg,
                        )
                        break
                    except ValueError as e:
                        msg = str(e)
                        if "Face detection failed on expression image." in msg:
                            last_exc = e
                            continue
                        raise
                if imgs is None:
                    # Preserve original behavior when all expression candidates fail.
                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError("[Arc2FaceAttack] failed to generate outputs.")
                fake = imgs[0].resize((w, h), Image.BICUBIC)
                fake_t = self._to_tensor(fake).to(wm_images.device)
                out.append(fake_t)
            except Exception:
                if self.allow_missing:
                    out.append(wm_images[i].detach())
                    continue
                raise
        attacked = torch.stack(out, dim=0).clamp(-1, 1)
        a = min(max(float(self.blend_alpha), 0.0), 1.0)
        if a < 1.0:
            attacked = attacked * a + wm_images * (1.0 - a)
        self._assert_nontrivial_swap(wm_images, attacked)
        return attacked

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

    def attack_with_cover(self, wm_images: torch.Tensor, cover_images: torch.Tensor, batch=None) -> torch.Tensor:
        if self.mode == 'online':
            return self._attack_online(wm_images, cover_images, batch=batch)

        if batch is None or 'img_path' not in batch:
            if self.allow_missing:
                return wm_images
            raise KeyError(
                "[Arc2FaceAttack] batch['img_path'] is required for replay lookup."
            )
        paths = batch['img_path']
        if not isinstance(paths, (list, tuple)):
            if self.allow_missing:
                return wm_images
            raise TypeError("[Arc2FaceAttack] batch['img_path'] must be list/tuple of paths.")

        b, _, h, w = wm_images.shape
        out = []
        for i in range(b):
            key = _norm_path(paths[i])
            fake_path = self.replay_map.get(key, None)
            if fake_path is None:
                if self.allow_missing:
                    out.append(wm_images[i].detach())
                    continue
                raise KeyError(
                    f"[Arc2FaceAttack] missing replay for target: {key}\n"
                    f"results_jsonl={self.results_jsonl}\n"
                    f"replay_keys={self.replay_keys}"
                )
            out.append(self._load_fake_tensor(fake_path, h, w, wm_images.device))
        attacked = torch.stack(out, dim=0)
        attacked = attacked.clamp(-1, 1)
        a = min(max(float(self.blend_alpha), 0.0), 1.0)
        if a < 1.0:
            attacked = attacked * a + wm_images * (1.0 - a)
        self._assert_nontrivial_swap(wm_images, attacked)
        return attacked

    def _assert_nontrivial_swap(self, original: torch.Tensor, swapped: torch.Tensor):
        if not self._swap_check_enabled:
            return
        with torch.no_grad():
            per_sample_diff = (original - swapped).abs().mean(dim=(1, 2, 3))
            bad = (per_sample_diff < self._swap_check_eps)
        if bool(bad.any()):
            bad_n = int(bad.sum().item())
            raise RuntimeError(
                f"[Arc2FaceAttack] nontrivial swap check failed: "
                f"{bad_n}/{original.shape[0]} samples have mean_abs_diff < "
                f"{self._swap_check_eps:.1e}."
            )
