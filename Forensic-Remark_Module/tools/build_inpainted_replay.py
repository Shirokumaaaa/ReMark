#!/usr/bin/env python3
"""
Build face-focused replay outputs from an existing generation_results.jsonl.

This script post-processes diffusion replay outputs so that only the face region
keeps the fake edits while the background is preserved from source_image.
It produces:
  1) blended images in an output directory
  2) a new JSONL with updated outputs paths

It is model-agnostic and can be used for:
  - Face-Adapter
  - ReFace
  - DiffSwap
  - Arc2Face wrapper
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build face-focused inpainted replay JSONL.")
    p.add_argument("--input-jsonl", type=str, required=True, help="Original generation_results.jsonl")
    p.add_argument(
        "--output-jsonl",
        type=str,
        default="",
        help="Path for the new jsonl (default: sibling generation_results_inpaint.jsonl).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for blended images (default: <jsonl_dir>/inpainted_replay).",
    )
    p.add_argument(
        "--outputs-base",
        type=str,
        default="",
        help="Optional base dir used to resolve relative output image paths from jsonl.",
    )
    p.add_argument(
        "--source-root",
        type=str,
        default="",
        help="Optional base dir used to resolve relative source_image paths from jsonl.",
    )
    p.add_argument("--source-key", type=str, default="source_image")
    p.add_argument("--output-index", type=int, default=0, help="Which entry in outputs[] to use.")
    p.add_argument("--max-records", type=int, default=0, help="0 means all records.")
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--save-format", type=str, default="png", choices=["png", "jpg", "jpeg"])
    p.add_argument("--jpeg-quality", type=int, default=95)

    p.add_argument("--mode", type=str, default="hybrid", choices=["ellipse", "diff", "hybrid"])
    p.add_argument("--alpha", type=float, default=1.0, help="Face region blend strength.")
    p.add_argument("--match-stats", action="store_true", default=False, help="Match fake region color stats to source.")

    p.add_argument("--mask-center-x", type=float, default=0.0)
    p.add_argument("--mask-center-y", type=float, default=-0.08)
    p.add_argument("--mask-radius-x", type=float, default=0.55)
    p.add_argument("--mask-radius-y", type=float, default=0.68)
    p.add_argument("--mask-softness", type=float, default=0.10)

    p.add_argument("--diff-threshold", type=float, default=0.06)
    p.add_argument("--diff-softness", type=float, default=0.03)
    p.add_argument("--diff-blur-ks", type=int, default=9)

    p.add_argument(
        "--keep-original-on-fail",
        action="store_true",
        default=True,
        help="If post-process fails for one record, keep original record unchanged.",
    )
    p.add_argument("--no-keep-original-on-fail", action="store_true", default=False)
    return p.parse_args()


def _as_abs(path_str: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path_str)))


def _resolve_path(path_str: str, jsonl_path: Path, outputs_base: Optional[Path], source_root: Optional[Path]) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p
    candidates = []
    candidates.append((jsonl_path.parent / p).resolve())
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
    m = 1.0 / (1.0 + np.exp(-((1.0 - d) * k)))
    return np.clip(m.astype(np.float32), 0.0, 1.0)


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
    m = _blur_mask(m, blur_ks)
    return np.clip(m, 0.0, 1.0)


def _match_region_stats(fake: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = np.clip(mask, 0.0, 1.0)[..., None]
    w = np.sum(m, axis=(0, 1), keepdims=True)
    w = np.maximum(w, 1e-6)
    fake_mean = np.sum(fake * m, axis=(0, 1), keepdims=True) / w
    src_mean = np.sum(source * m, axis=(0, 1), keepdims=True) / w
    fake_var = np.sum(((fake - fake_mean) * m) ** 2, axis=(0, 1), keepdims=True) / w
    src_var = np.sum(((source - src_mean) * m) ** 2, axis=(0, 1), keepdims=True) / w
    fake_std = np.sqrt(np.maximum(fake_var, 1e-8))
    src_std = np.sqrt(np.maximum(src_var, 1e-8))
    matched = (fake - fake_mean) / fake_std * src_std + src_mean
    return np.clip(matched, 0.0, 1.0)


def _blend_face_region(fake: np.ndarray, source: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    h, w = source.shape[:2]
    face = _ellipse_mask(
        h=h,
        w=w,
        cx=float(args.mask_center_x),
        cy=float(args.mask_center_y),
        rx=float(args.mask_radius_x),
        ry=float(args.mask_radius_y),
        softness=float(args.mask_softness),
    )
    dmask = _diff_mask(
        fake=fake,
        source=source,
        thr=float(args.diff_threshold),
        soft=float(args.diff_softness),
        blur_ks=int(args.diff_blur_ks),
    )
    if args.mode == "diff":
        mask = dmask
    elif args.mode == "hybrid":
        mask = np.clip(face * dmask, 0.0, 1.0)
    else:
        mask = face

    fake_used = _match_region_stats(fake, source, mask) if bool(args.match_stats) else fake
    alpha = np.clip(float(args.alpha), 0.0, 1.0)
    mixed_face = source + alpha * (fake_used - source)
    mask3 = mask[..., None]
    out = source * (1.0 - mask3) + mixed_face * mask3
    return np.clip(out, 0.0, 1.0)


def _output_name(rec: Dict[str, Any], rec_idx: int, output_src: Path, fmt: str) -> str:
    ridx = rec.get("index", rec_idx)
    stem = output_src.stem
    ext = "jpg" if fmt in ("jpg", "jpeg") else "png"
    return f"{int(ridx):06d}_{stem}_inpaint.{ext}"


def main() -> None:
    args = parse_args()
    if args.no_keep_original_on_fail:
        args.keep_original_on_fail = False

    in_jsonl = _as_abs(args.input_jsonl)
    if not in_jsonl.exists():
        raise FileNotFoundError(f"input jsonl not found: {in_jsonl}")

    if args.output_jsonl:
        out_jsonl = _as_abs(args.output_jsonl)
    else:
        out_jsonl = in_jsonl.parent / "generation_results_inpaint.jsonl"

    if args.output_dir:
        out_dir = _as_abs(args.output_dir)
    else:
        out_dir = in_jsonl.parent / "inpainted_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    outputs_base = _as_abs(args.outputs_base) if args.outputs_base else None
    source_root = _as_abs(args.source_root) if args.source_root else None

    total = 0
    ok = 0
    fail = 0

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
                    raise ValueError("record has empty outputs.")
                if args.output_index < 0 or args.output_index >= len(outputs):
                    raise IndexError(
                        f"output_index={args.output_index} out of range for outputs size={len(outputs)}."
                    )
                src_raw = rec.get(args.source_key, None)
                if not src_raw:
                    raise KeyError(f"missing source key: {args.source_key}")

                source_path = _resolve_path(str(src_raw), in_jsonl, outputs_base=None, source_root=source_root)
                fake_path = _resolve_path(str(outputs[args.output_index]), in_jsonl, outputs_base=outputs_base, source_root=None)

                src_img = _load_rgb_float(source_path)
                h, w = src_img.shape[:2]
                fake_img = _load_rgb_float(fake_path, resize_to=(w, h))

                blended = _blend_face_region(fake=fake_img, source=src_img, args=args)
                out_name = _output_name(rec, rec_idx, fake_path, args.save_format)
                out_path = (out_dir / out_name).resolve()
                _save_rgb_float(out_path, blended, args.save_format, args.jpeg_quality)

                new_rec = dict(rec)
                new_rec["outputs"] = [str(out_path)]
                new_rec["inpaint_postprocess"] = {
                    "enabled": True,
                    "mode": args.mode,
                    "alpha": float(args.alpha),
                    "match_stats": bool(args.match_stats),
                    "source_key": args.source_key,
                    "original_output": str(fake_path),
                }
                fo.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                if args.keep_original_on_fail:
                    try:
                        rec["inpaint_postprocess_error"] = str(exc)
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                if args.progress_every > 0:
                    print(f"[WARN] rec={rec_idx} failed: {exc}", flush=True)

            if args.progress_every > 0 and total % args.progress_every == 0:
                print(f"[progress] total={total} ok={ok} fail={fail}", flush=True)

    print(
        f"[done] input={in_jsonl} output_jsonl={out_jsonl} output_dir={out_dir} "
        f"total={total} ok={ok} fail={fail}",
        flush=True,
    )


if __name__ == "__main__":
    main()
