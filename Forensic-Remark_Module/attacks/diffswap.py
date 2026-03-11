import json
import os
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
    DiffSwap deepfake replay adapter.

    DiffSwap (CVPR 2023) is a diffusion-based face-swapping model that is too slow
    for online DataLoader generation.  This adapter replays pre-generated outputs
    produced by Attack-DiffSwap/tests/faceswap_portrait.py (+ pipeline.py blending).

    Expected JSONL format (one record per line):
        {"ok": true, "source_image": "/abs/path/to/wm_img.png", "outputs": ["/abs/path/to/swap_result.png"]}

    The default lookup key is `source_image`, so each watermarked-image path maps
    to a pre-generated DiffSwap output that carries the same target identity with
    a different source face.

    Config options (under attack_options):
        diffswap_results_jsonl  : path to the JSONL file  (default: Attack-DiffSwap/outputs/generation_results.jsonl)
        diffswap_outputs_base   : base dir for relative path resolution  (default: Attack-DiffSwap/outputs)
        diffswap_allow_missing  : if True, fall back to wm_image when key not found  (default: False)
        diffswap_replay_key     : JSONL field used as lookup key  (default: 'source_image')
        enforce_nontrivial_swap : enable MAD check to catch silent identity leakage  (default: True)
        nontrivial_swap_eps     : MAD threshold below which swap is flagged as trivial  (default: 1e-4)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        opts = getattr(cfg, 'attack_options', None)
        self.results_jsonl = _norm_path(getattr(opts, 'diffswap_results_jsonl', _DEFAULT_RESULTS_JSONL))
        self.outputs_base  = _norm_path(getattr(opts, 'diffswap_outputs_base',  _DEFAULT_OUTPUTS_BASE))
        self.allow_missing = bool(getattr(opts, 'diffswap_allow_missing', False))
        self.replay_key    = str(getattr(opts,  'diffswap_replay_key',    'source_image'))
        self._swap_check_enabled = bool(getattr(opts, 'enforce_nontrivial_swap', True))
        self._swap_check_eps     = float(getattr(opts, 'nontrivial_swap_eps', 1e-4))
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.replay_map = self._build_replay_map(self.results_jsonl)
        if len(self.replay_map) == 0:
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
        if batch is None or 'img_path' not in batch:
            if self.allow_missing:
                return wm_images
            raise KeyError(
                "[DiffSwapAttack] batch['img_path'] is required for replay lookup."
            )
        paths = batch['img_path']
        if not isinstance(paths, (list, tuple)):
            if self.allow_missing:
                return wm_images
            raise TypeError("[DiffSwapAttack] batch['img_path'] must be list/tuple of paths.")

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
                    f"[DiffSwapAttack] missing replay entry for: {key}\n"
                    f"results_jsonl={self.results_jsonl}"
                )
            out.append(self._load_fake_tensor(fake_path, h, w, wm_images.device))

        attacked = torch.stack(out, dim=0).clamp(-1, 1)
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
