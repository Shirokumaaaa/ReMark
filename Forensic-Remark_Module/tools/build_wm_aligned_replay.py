#!/usr/bin/env python3
"""
Build replay outputs aligned to watermark messages.

For each record in input JSONL:
  1) Load source_image and fake output image
  2) Re-encode source_image with WM adapter using deterministic message from img_path
  3) Inject fake edits only on face mask region, while keeping WM background
  4) Save aligned attacked image and write new JSONL

This makes replay outputs consistent with message GT used online (when
training.deterministic_messages=true and same message_seed_salt).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter

from utils.config import load_config
from utils.message_bits import deterministic_messages_from_paths
from wm_adapters.registry import build_wm_adapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build WM-aligned replay JSONL")
    p.add_argument("--config", default="configs/stage1_vae.yaml")
    p.add_argument("--override", default=None)
    p.add_argument("--input-jsonl", type=str, required=True)
    p.add_argument("--output-jsonl", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--outputs-base", type=str, default="")
    p.add_argument("--source-root", type=str, default="")
    p.add_argument("--source-key", type=str, default="source_image")
    p.add_argument("--output-index", type=int, default=0)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--mode", type=str, default="hybrid", choices=["ellipse", "diff", "hybrid", "seamless"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--match-stats", action="store_true", default=False)
    p.add_argument("--mask-floor", type=float, default=0.20, help="Blend in a minimum face mask to avoid holes.")
    p.add_argument("--mask-close-ks", type=int, default=9, help="Morph close kernel size (odd).")
    p.add_argument("--mask-open-ks", type=int, default=5, help="Morph open kernel size (odd).")
    p.add_argument("--auto-trim-black-border", action="store_true", default=False)
    p.add_argument("--trim-black-threshold", type=float, default=0.03)

    p.add_argument("--mask-center-x", type=float, default=0.0)
    p.add_argument("--mask-center-y", type=float, default=-0.08)
    p.add_argument("--mask-radius-x", type=float, default=0.55)
    p.add_argument("--mask-radius-y", type=float, default=0.68)
    p.add_argument("--mask-softness", type=float, default=0.10)

    p.add_argument("--diff-threshold", type=float, default=0.06)
    p.add_argument("--diff-softness", type=float, default=0.03)
    p.add_argument("--diff-blur-ks", type=int, default=9)

    p.add_argument("--save-format", type=str, default="png", choices=["png", "jpg", "jpeg"])
    p.add_argument("--jpeg-quality", type=int, default=95)

    p.add_argument("--message-seed-salt", type=str, default="remark_v1")
    p.add_argument("--keep-original-on-fail", action="store_true", default=True)
    p.add_argument("--no-keep-original-on-fail", action="store_true", default=False)
    return p.parse_args()


def _as_abs(path_str: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path_str)))


def _resolve_path(path_str: str, jsonl_path: Path, outputs_base: Optional[Path], source_root: Optional[Path]) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    candidates = [(jsonl_path.parent / p).resolve()]
    if outputs_base is not None:
        candidates.append((outputs_base / p).resolve())
    if source_root is not None:
        candidates.append((source_root / p).resolve())
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _load_rgb_float(path: Path, resize_to: Optional[tuple] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if resize_to is not None and img.size != resize_to:
        img = img.resize(resize_to, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def _auto_trim_black_border(arr: np.ndarray, black_thr: float = 0.03) -> np.ndarray:
    g = arr.mean(axis=2)
    valid = g > float(black_thr)
    if not bool(valid.any()):
        return arr
    ys, xs = np.where(valid)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = arr.shape[:2]
    box_h = max(y1 - y0, 1)
    box_w = max(x1 - x0, 1)
    area_ratio = (box_h * box_w) / float(h * w)
    # Trim only when content is clearly letterboxed/padded.
    if area_ratio > 0.98:
        return arr
    return arr[y0:y1, x0:x1, :]


def _save_rgb_float(path: Path, arr: np.ndarray, fmt: str, jpeg_quality: int) -> None:
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("jpg", "jpeg"):
        img.save(path, quality=int(jpeg_quality))
    else:
        img.save(path)


def _ellipse_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float, softness: float) -> np.ndarray:
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32).reshape(1, w)
    rx = max(rx, 1e-4)
    ry = max(ry, 1e-4)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    k = 1.0 / max(softness, 1e-4)
    return np.clip(1.0 / (1.0 + np.exp(-((1.0 - d) * k))), 0.0, 1.0).astype(np.float32)


def _blur_mask(mask: np.ndarray, blur_ks: int) -> np.ndarray:
    k = max(int(blur_ks), 1)
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return np.clip(mask, 0.0, 1.0)
    radius = max((k - 1) / 2.0, 0.0)
    pil = Image.fromarray(np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    out = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def _diff_mask(fake: np.ndarray, source: np.ndarray, thr: float, soft: float, blur_ks: int) -> np.ndarray:
    diff = np.abs(fake - source).mean(axis=2)
    soft = max(float(soft), 1e-4)
    m = 1.0 / (1.0 + np.exp(-((diff - float(thr)) / soft)))
    return _blur_mask(m, blur_ks)


def _morph_refine(mask: np.ndarray, close_ks: int, open_ks: int) -> np.ndarray:
    m = np.clip(mask, 0.0, 1.0)
    m8 = (m * 255.0).astype(np.uint8)

    cks = max(int(close_ks), 1)
    if cks % 2 == 0:
        cks += 1
    oks = max(int(open_ks), 1)
    if oks % 2 == 0:
        oks += 1

    if cks > 1:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cks, cks))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kc)
    if oks > 1:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (oks, oks))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, ko)
    out = m8.astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def _match_region_stats(fake: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = np.clip(mask, 0.0, 1.0)[..., None]
    w = np.maximum(np.sum(m, axis=(0, 1), keepdims=True), 1e-6)
    fake_mean = np.sum(fake * m, axis=(0, 1), keepdims=True) / w
    ref_mean = np.sum(ref * m, axis=(0, 1), keepdims=True) / w
    fake_var = np.sum(((fake - fake_mean) * m) ** 2, axis=(0, 1), keepdims=True) / w
    ref_var = np.sum(((ref - ref_mean) * m) ** 2, axis=(0, 1), keepdims=True) / w
    fake_std = np.sqrt(np.maximum(fake_var, 1e-8))
    ref_std = np.sqrt(np.maximum(ref_var, 1e-8))
    return np.clip((fake - fake_mean) / fake_std * ref_std + ref_mean, 0.0, 1.0)


def _blend(fake: np.ndarray, source: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    h, w = source.shape[:2]
    face = _ellipse_mask(h, w, args.mask_center_x, args.mask_center_y, args.mask_radius_x, args.mask_radius_y, args.mask_softness)
    dmask = _diff_mask(fake, source, args.diff_threshold, args.diff_softness, args.diff_blur_ks)
    if args.mode == "diff":
        mask = dmask
    elif args.mode == "hybrid":
        mask = np.clip(face * dmask, 0.0, 1.0)
    elif args.mode == "seamless":
        mask = np.clip(face * dmask, 0.0, 1.0)
    else:
        mask = face
    mask = np.maximum(mask, face * np.clip(float(args.mask_floor), 0.0, 1.0))
    mask = _morph_refine(mask, close_ks=args.mask_close_ks, open_ks=args.mask_open_ks)
    mask = _blur_mask(mask, blur_ks=max(int(args.diff_blur_ks), 3))

    fake_used = _match_region_stats(fake, source, mask) if args.match_stats else fake
    if args.mode == "seamless":
        src_u8 = np.clip(fake_used * 255.0, 0.0, 255.0).astype(np.uint8)
        dst_u8 = np.clip(source * 255.0, 0.0, 255.0).astype(np.uint8)
        m_u8 = np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)
        ys, xs = np.where(m_u8 > 8)
        if len(xs) > 16:
            center = (int(xs.mean()), int(ys.mean()))
            blended = cv2.seamlessClone(src_u8, dst_u8, m_u8, center, cv2.NORMAL_CLONE).astype(np.float32) / 255.0
            alpha = np.clip(float(args.alpha), 0.0, 1.0)
            return np.clip(source * (1.0 - alpha) + blended * alpha, 0.0, 1.0)
    alpha = np.clip(float(args.alpha), 0.0, 1.0)
    mixed_face = source + alpha * (fake_used - source)
    m3 = mask[..., None]
    return np.clip(source * (1.0 - m3) + mixed_face * m3, 0.0, 1.0)


def _np_to_torch_img(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    # [H,W,C] [0,1] -> [1,C,H,W] [-1,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return t * 2.0 - 1.0


def _torch_to_np_img(t: torch.Tensor) -> np.ndarray:
    # [1,C,H,W] [-1,1] -> [H,W,C] [0,1]
    x = ((t.detach().float().cpu().squeeze(0).permute(1, 2, 0) + 1.0) * 0.5).numpy()
    return np.clip(x, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    if args.no_keep_original_on_fail:
        args.keep_original_on_fail = False

    cfg = load_config(args.config, args.override)

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    wm_adapter = build_wm_adapter(cfg.wm_model, cfg)
    if hasattr(wm_adapter, "_encoder") and wm_adapter._encoder is not None:
        wm_adapter._encoder.to(dev)

    in_jsonl = _as_abs(args.input_jsonl)
    out_jsonl = _as_abs(args.output_jsonl)
    out_dir = _as_abs(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    outputs_base = _as_abs(args.outputs_base) if args.outputs_base else None
    source_root = _as_abs(args.source_root) if args.source_root else None

    total = ok = fail = 0
    with in_jsonl.open("r", encoding="utf-8") as fi, out_jsonl.open("w", encoding="utf-8") as fo:
        for rec_idx, line in enumerate(fi):
            if args.max_records > 0 and total >= args.max_records:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fail += 1
                continue

            try:
                if not rec.get("ok", False):
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                outputs = rec.get("outputs", [])
                if not outputs:
                    raise ValueError("record has empty outputs")
                if args.output_index < 0 or args.output_index >= len(outputs):
                    raise IndexError("output_index out of range")

                src_raw = rec.get(args.source_key, None)
                if not src_raw:
                    raise KeyError(f"missing source key: {args.source_key}")

                source_path = _resolve_path(str(src_raw), in_jsonl, outputs_base=None, source_root=source_root)
                fake_path = _resolve_path(str(outputs[args.output_index]), in_jsonl, outputs_base=outputs_base, source_root=None)

                src_np = _load_rgb_float(source_path)
                h, w = src_np.shape[:2]
                fake_np = _load_rgb_float(fake_path)
                if args.auto_trim_black_border:
                    fake_np = _auto_trim_black_border(fake_np, black_thr=args.trim_black_threshold)
                if fake_np.shape[0] != h or fake_np.shape[1] != w:
                    fake_np = _load_rgb_float(fake_path, resize_to=(w, h))

                with torch.no_grad():
                    src_t = _np_to_torch_img(src_np, dev)
                    msg_t = deterministic_messages_from_paths(
                        [str(source_path)],
                        message_len=wm_adapter.message_length,
                        device=dev,
                        salt=args.message_seed_salt,
                    )
                    wm_t = wm_adapter.encode(src_t, msg_t)
                    wm_np = _torch_to_np_img(wm_t)

                aligned_np = _blend(fake=fake_np, source=wm_np, args=args)

                ridx = int(rec.get("index", rec_idx))
                stem = Path(fake_path).stem
                ext = "jpg" if args.save_format in ("jpg", "jpeg") else "png"
                out_path = (out_dir / f"{ridx:06d}_{stem}_wm_aligned.{ext}").resolve()
                _save_rgb_float(out_path, aligned_np, args.save_format, args.jpeg_quality)

                new_rec = dict(rec)
                new_rec["outputs"] = [str(out_path)]
                new_rec["wm_aligned"] = {
                    "enabled": True,
                    "wm_model": str(cfg.wm_model),
                    "source_key": args.source_key,
                    "message_seed_salt": args.message_seed_salt,
                    "mode": args.mode,
                    "alpha": float(args.alpha),
                    "match_stats": bool(args.match_stats),
                    "original_output": str(fake_path),
                }
                fo.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                if args.keep_original_on_fail:
                    rec["wm_aligned_error"] = str(exc)
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if args.progress_every > 0:
                    print(f"[WARN] rec={rec_idx} failed: {exc}", flush=True)

            if args.progress_every > 0 and total % args.progress_every == 0:
                print(f"[progress] total={total} ok={ok} fail={fail}", flush=True)

    print(
        f"[done] input={in_jsonl} output_jsonl={out_jsonl} output_dir={out_dir} total={total} ok={ok} fail={fail}",
        flush=True,
    )


if __name__ == "__main__":
    main()
