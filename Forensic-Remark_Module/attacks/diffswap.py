import json
import os
import csv
import hashlib
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .base import BaseAttack
from .registry import register_attack


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_RESULTS_JSONL = os.path.join(
    _PROJECT_ROOT, 'Attack-DiffSwap', 'outputs', 'generation_results.jsonl'
)
_DEFAULT_OUTPUTS_BASE = os.path.join(_PROJECT_ROOT, 'Attack-DiffSwap', 'outputs')


def _norm_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p)))


@register_attack('diffswap')
@register_attack('DiffSwap')
class DiffSwapAttack(BaseAttack):
    """
    DiffSwap deepfake adapter.

    Supports two modes:
      - replay: use pre-generated outputs (fast, default)
      - online: run Attack-DiffSwap pipeline per target and cache outputs

    Replay JSONL expected format (one record per line):
        {"ok": true, "source_image": "/abs/path/to/wm_img.png", "outputs": ["/abs/path/to/swap_result.png"]}

    The default lookup key is `source_image`, so each watermarked-image path maps
    to a pre-generated DiffSwap output that carries the same target identity with
    a different source face.

    Config options (under attack_options):
        diffswap_mode           : "replay" | "online" (default: replay)
        diffswap_results_jsonl  : path to the JSONL file  (default: Attack-DiffSwap/outputs/generation_results.jsonl)
        diffswap_outputs_base   : base dir for relative path resolution  (default: Attack-DiffSwap/outputs)
        diffswap_allow_missing  : if True, fall back to wm_image when key not found  (default: False)
        diffswap_replay_key     : JSONL field used as lookup key  (default: 'source_image')
        diffswap_online_python_bin      : python for online subprocess
        diffswap_online_repo_root       : Attack-DiffSwap repo root
        diffswap_online_pipeline_script : pipeline script path (default: <repo>/pipeline.py)
        diffswap_online_cache_dir       : cache dir for online outputs
        diffswap_online_timeout_sec     : timeout per target
        diffswap_online_fixed_source_path : optional fixed source image path
        diffswap_source_csv             : optional source pool csv (img_path col)
        enforce_nontrivial_swap : enable MAD check to catch silent identity leakage  (default: True)
        nontrivial_swap_eps     : MAD threshold below which swap is flagged as trivial  (default: 1e-4)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        opts = getattr(cfg, 'attack_options', None)
        self.mode = str(getattr(opts, 'diffswap_mode', 'replay')).strip().lower()
        if self.mode not in ('replay', 'online'):
            raise ValueError(f"[DiffSwapAttack] invalid diffswap_mode={self.mode}")
        self.results_jsonl = _norm_path(getattr(opts, 'diffswap_results_jsonl', _DEFAULT_RESULTS_JSONL))
        self.outputs_base  = _norm_path(getattr(opts, 'diffswap_outputs_base',  _DEFAULT_OUTPUTS_BASE))
        self.allow_missing = bool(getattr(opts, 'diffswap_allow_missing', False))
        self.replay_key    = str(getattr(opts,  'diffswap_replay_key',    'source_image'))
        self.online_repo_root = _norm_path(getattr(
            opts, 'diffswap_online_repo_root', os.path.join(_PROJECT_ROOT, 'Attack-DiffSwap')
        ))
        self.online_python_bin = str(getattr(opts, 'diffswap_online_python_bin', '/home/ldy/miniconda3/envs/DiffSwap/bin/python')).strip()
        self.online_pipeline_script = _norm_path(getattr(
            opts, 'diffswap_online_pipeline_script', os.path.join(self.online_repo_root, 'pipeline.py')
        ))
        self.online_cache_dir = _norm_path(getattr(opts, 'diffswap_online_cache_dir', '/tmp/remark_diffswap_online_cache'))
        os.makedirs(self.online_cache_dir, exist_ok=True)
        self.online_timeout_sec = int(getattr(opts, 'diffswap_online_timeout_sec', 1800))
        self.online_max_retries = int(getattr(opts, 'diffswap_online_max_retries', 4))
        self.online_tgt_scale = float(getattr(opts, 'diffswap_online_tgt_scale', 0.01))
        self.online_source_csv = str(getattr(opts, 'diffswap_source_csv', getattr(getattr(cfg, 'data', None), 'train_csv', ''))).strip()
        self.source_pool = self._load_source_pool_from_csv(self.online_source_csv)
        self.valid_source_pool = self._load_valid_source_pool_from_repo()
        fixed_source_opt = str(getattr(opts, 'diffswap_online_fixed_source_path', '')).strip()
        self.fixed_source_path = _norm_path(fixed_source_opt) if fixed_source_opt else ''
        if self.fixed_source_path and (not os.path.exists(self.fixed_source_path)):
            self.fixed_source_path = ''
        self.online_fallback_replay = bool(getattr(opts, 'diffswap_online_fallback_replay', True))
        self.online_fallback_to_input = bool(getattr(opts, 'diffswap_online_fallback_to_input', True))
        self.blend_alpha = float(getattr(opts, 'diffswap_blend_alpha', 0.85))
        self._swap_check_enabled = bool(getattr(opts, 'enforce_nontrivial_swap', True))
        self._swap_check_eps     = float(getattr(opts, 'nontrivial_swap_eps', 1e-4))
        self._warned_online_fail = False
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.replay_map = {}
        if self.mode == 'replay' or self.online_fallback_replay:
            self.replay_map = self._build_replay_map(self.results_jsonl)
            if self.mode == 'replay' and len(self.replay_map) == 0:
                raise RuntimeError(
                    f"[DiffSwapAttack] No valid replay entries loaded from {self.results_jsonl}"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_output_path(self, p: str, jsonl_path: str) -> str:
        """Resolve a (possibly relative) output path from the JSONL record."""
        p = str(p)
        if os.path.isabs(p) and os.path.exists(p):
            return p
        # Try relative to JSONL directory first
        cand1 = os.path.abspath(os.path.join(os.path.dirname(jsonl_path), p))
        if os.path.exists(cand1):
            return cand1
        # Then relative to outputs base
        cand2 = os.path.abspath(os.path.join(self.outputs_base, p))
        if os.path.exists(cand2):
            return cand2
        return cand1  # return best guess so error messages are informative

    def _build_replay_map(self, jsonl_path: str) -> dict:
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"[DiffSwapAttack] results jsonl not found: {jsonl_path}\n"
                f"Run Attack-DiffSwap/tests/faceswap_portrait.py first to generate pre-swapped images, "
                f"then point diffswap_results_jsonl at the produced JSONL."
            )
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
                outs    = rec.get('outputs', [])
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

    def _load_valid_source_pool_from_repo(self):
        src_dir = os.path.join(self.online_repo_root, 'data', 'portrait', 'source')
        if not os.path.isdir(src_dir):
            return []
        out = []
        for name in sorted(os.listdir(src_dir)):
            p = os.path.join(src_dir, name)
            if os.path.isfile(p):
                out.append(_norm_path(p))
        return out

    def _pick_source_path(self, target_path: str, attempt: int = 0) -> str:
        if self.fixed_source_path:
            return self.fixed_source_path
        pool = self.valid_source_pool if len(self.valid_source_pool) > 0 else self.source_pool
        if len(pool) == 0:
            return ''
        h = int(hashlib.sha1(_norm_path(target_path).encode('utf-8')).hexdigest(), 16)
        return pool[(h + max(int(attempt), 0)) % len(pool)]

    def _tensor_to_pil(self, image: torch.Tensor) -> Image.Image:
        image = image.detach().float().cpu().clamp(-1, 1)
        image = ((image + 1.0) * 127.5).round().to(torch.uint8)
        image = image.permute(1, 2, 0).contiguous().numpy()
        return Image.fromarray(image)

    def _wm_tensor_hash(self, image: torch.Tensor) -> str:
        arr = image.detach().float().cpu().clamp(-1, 1).mul(32767.0).round().to(torch.int16).numpy()
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def _cache_path(self, target_key: str, wm_hash: str = '') -> str:
        cache_key = _norm_path(target_key)
        if wm_hash:
            cache_key = cache_key + '|' + wm_hash
        cache_key = cache_key + f'|tgt_scale={self.online_tgt_scale:.6f}'
        key = hashlib.sha1(cache_key.encode('utf-8')).hexdigest()
        return os.path.join(self.online_cache_dir, f'{key}.png')

    def _ensure_online_generated(self, target_key: str, wm_image: torch.Tensor):
        wm_hash = self._wm_tensor_hash(wm_image)
        cache_path = self._cache_path(target_key, wm_hash=wm_hash)
        if os.path.exists(cache_path):
            return
        if not os.path.exists(self.online_pipeline_script):
            raise FileNotFoundError(f'[DiffSwapAttack] pipeline script not found: {self.online_pipeline_script}')

        if not self.fixed_source_path and len(self.source_pool) == 0:
            raise RuntimeError('[DiffSwapAttack] no source image available for online mode.')

        last_error = None
        max_try = max(int(self.online_max_retries), 1)
        for attempt in range(max_try):
            source_path = self._pick_source_path(target_key, attempt=attempt)
            if not source_path:
                continue

            tmp_root = tempfile.mkdtemp(prefix='remark_diffswap_online_')
            try:
                # Build a minimal writable workspace with symlinks to heavy assets/code.
                for item in ('checkpoints', 'configs', 'data_preprocessing', 'ldm', 'src', 'tests', 'utils'):
                    src = os.path.join(self.online_repo_root, item)
                    dst = os.path.join(tmp_root, item)
                    if os.path.exists(src) and (not os.path.exists(dst)):
                        os.symlink(src, dst)
                os.makedirs(os.path.join(tmp_root, 'data', 'portrait_jpg', 'source'), exist_ok=True)
                os.makedirs(os.path.join(tmp_root, 'data', 'portrait_jpg', 'target'), exist_ok=True)

                src_name = os.path.basename(source_path)
                tgt_stem = os.path.splitext(os.path.basename(target_key))[0] or wm_hash[:12]
                tgt_name = f'{tgt_stem}_{wm_hash[:12]}.png'
                shutil.copy2(source_path, os.path.join(tmp_root, 'data', 'portrait_jpg', 'source', src_name))
                self._tensor_to_pil(wm_image).save(
                    os.path.join(tmp_root, 'data', 'portrait_jpg', 'target', tgt_name)
                )

                cmd = [
                    self.online_python_bin,
                    self.online_pipeline_script,
                    '--tgt_scale',
                    str(self.online_tgt_scale),
                ]
                env = os.environ.copy()
                py_bin_dir = os.path.dirname(self.online_python_bin)
                if py_bin_dir:
                    env['PATH'] = py_bin_dir + (':' + env['PATH'] if env.get('PATH') else '')
                env['PYTHONPATH'] = tmp_root + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
                cp = subprocess.run(
                    cmd,
                    cwd=tmp_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=self.online_timeout_sec,
                )
                if cp.returncode != 0:
                    out_tail = '\n'.join((cp.stdout or '').splitlines()[-40:])
                    err_tail = '\n'.join((cp.stderr or '').splitlines()[-40:])
                    last_error = RuntimeError(
                        f"[DiffSwapAttack] online pipeline failed (attempt {attempt + 1}/{max_try}, code={cp.returncode}).\n"
                        f"source={source_path}\n"
                        f"target_key={target_key}\n"
                        f"target_file={tgt_name}\n"
                        f"stdout_tail:\n{out_tail}\n"
                        f"stderr_tail:\n{err_tail}"
                    )
                    continue

                tgt_out_stem = os.path.splitext(tgt_name)[0]
                cands = glob(os.path.join(tmp_root, 'data', 'portrait', 'swap_res_ori', '**', f'{tgt_out_stem}.*'), recursive=True)
                cands = [p for p in cands if os.path.isfile(p)]
                if not cands:
                    last_error = RuntimeError(
                        f"[DiffSwapAttack] online pipeline produced no output image "
                        f"(attempt {attempt + 1}/{max_try}). source={source_path} target_key={target_key} target_file={tgt_name}"
                    )
                    continue
                shutil.copy2(cands[0], cache_path)
                return
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)

        if last_error is not None:
            raise last_error
        raise RuntimeError('[DiffSwapAttack] online generation failed with no valid source candidates.')

    def _resolve_fake_path(self, target_path: str, wm_image: torch.Tensor = None):
        key = _norm_path(target_path)
        if self.mode == 'online':
            wm_hash = self._wm_tensor_hash(wm_image) if wm_image is not None else ''
            cp = self._cache_path(key, wm_hash=wm_hash)
            if os.path.exists(cp):
                return cp
            try:
                if wm_image is None:
                    raise RuntimeError('[DiffSwapAttack] wm_image is required in online mode.')
                self._ensure_online_generated(key, wm_image)
            except Exception:
                if self.online_fallback_replay:
                    return self.replay_map.get(key, None)
                raise
            return cp if os.path.exists(cp) else None
        return self.replay_map.get(key, None)

    # ------------------------------------------------------------------
    # BaseAttack interface
    # ------------------------------------------------------------------

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        return images.clamp(-1, 1)

    def generate(self, preprocessed: torch.Tensor) -> torch.Tensor:
        # Path-aware replay is done in attack_with_cover(); bare __call__ is a no-op.
        return preprocessed

    def postprocess(self, output: torch.Tensor, original_size: tuple) -> torch.Tensor:
        if output.shape[-2:] != torch.Size(list(original_size)):
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        return output.clamp(-1, 1)

    # ------------------------------------------------------------------
    # Primary training entry point
    # ------------------------------------------------------------------

    def _load_fake_tensor(self, path: str, h: int, w: int, device: torch.device) -> torch.Tensor:
        img = Image.open(path).convert('RGB').resize((w, h), Image.BICUBIC)
        return self._to_tensor(img).to(device)

    def attack_with_cover(
        self,
        wm_images: torch.Tensor,
        cover_images: torch.Tensor,
        batch=None,
    ) -> torch.Tensor:
        """
        Replay pre-generated DiffSwap outputs keyed by img_path.

        Args:
            wm_images    : watermarked images, canonical [-1,1] BCHW
            cover_images : unused (DiffSwap is offline); kept for interface parity
            batch        : must contain 'img_path' list[str] for replay lookup
        Returns:
            Fake (swapped) images, canonical [-1,1] BCHW
        """
        paths = None
        if batch is not None and 'img_path' in batch:
            paths = batch['img_path']
            if (paths is not None) and (not isinstance(paths, (list, tuple))):
                if self.allow_missing:
                    return wm_images
                raise TypeError("[DiffSwapAttack] batch['img_path'] must be list/tuple of paths.")
        elif self.mode != 'online':
            if self.allow_missing:
                return wm_images
            raise KeyError("[DiffSwapAttack] batch['img_path'] is required for replay lookup.")

        b, _, h, w = wm_images.shape
        out = []
        for i in range(b):
            if paths is not None and len(paths) == b:
                key = _norm_path(paths[i])
            else:
                key = f'online_sample_{i}'
            fake_path = None
            err = None
            try:
                fake_path = self._resolve_fake_path(
                    key,
                    wm_image=(wm_images[i] if self.mode == 'online' else None),
                )
            except Exception as e:
                err = e
            if fake_path is None:
                if (err is not None) and (not self._warned_online_fail):
                    lines = str(err).splitlines() if err is not None else []
                    emsg = ' | '.join(lines[:4]) if lines else 'unknown'
                    print(f"[DiffSwapAttack] warning: online generation failed once. {emsg}")
                    self._warned_online_fail = True
                if self.allow_missing or (self.mode == 'online' and self.online_fallback_to_input):
                    out.append(wm_images[i].detach())
                    continue
                if err is not None:
                    raise RuntimeError(str(err))
                raise RuntimeError(
                    f"[DiffSwapAttack] failed to produce attack output for: {key}\n"
                    f"mode={self.mode}"
                )
            out.append(self._load_fake_tensor(fake_path, h, w, wm_images.device))

        attacked = torch.stack(out, dim=0).clamp(-1, 1)
        a = min(max(float(self.blend_alpha), 0.0), 1.0)
        if a < 1.0:
            attacked = attacked * a + wm_images * (1.0 - a)
        # In online fallback mode some samples may intentionally return wm_images.
        if not (self.mode == 'online' and self.online_fallback_to_input):
            self._assert_nontrivial_swap(wm_images, attacked)
        return attacked

    def _assert_nontrivial_swap(self, original: torch.Tensor, swapped: torch.Tensor):
        """Catch silent fallback where replay output is near-identical to the input."""
        if not self._swap_check_enabled:
            return
        with torch.no_grad():
            per_sample_diff = (original - swapped).abs().mean(dim=(1, 2, 3))
            bad = per_sample_diff < self._swap_check_eps
        if bool(bad.any()):
            bad_n = int(bad.sum().item())
            raise RuntimeError(
                f"[DiffSwapAttack] nontrivial swap check failed: "
                f"{bad_n}/{original.shape[0]} samples have mean_abs_diff < "
                f"{self._swap_check_eps:.1e}. Possible replay/input mismatch or "
                f"DiffSwap generation failure."
            )
