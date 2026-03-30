#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Triplet:
    source: Optional[Union[Path, Image.Image]]
    target: Optional[Union[Path, Image.Image]]
    fake: Optional[Union[Path, Image.Image]]
    method: str
    note: str = ""


def _resolve(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(os.path.abspath(os.path.expanduser(path_str)))
    return p if p.exists() else None


def _load_rgb(path: Path, size: int = 256) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32)


def _is_ok_record(rec: Dict) -> bool:
    if not rec.get("ok", False):
        return False
    outs = rec.get("outputs", [])
    if not outs:
        return False
    return _resolve(outs[0]) is not None


def _first_ok_record(jsonl_path: Path) -> Dict:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _is_ok_record(rec):
                return rec
    raise RuntimeError(f"No valid replay record found in: {jsonl_path}")

def _pick_diffswap_record(jsonl_path: Path, max_scan: int = 400) -> Dict:
    """Pick a non-trivial DiffSwap sample by max(target,fake) MAD."""
    target_root = ROOT / "Attack-DiffSwap" / "data" / "portrait" / "target"
    best = None
    best_mad = -1.0
    scanned = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            if max_scan > 0 and scanned > max_scan:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _is_ok_record(rec):
                continue
            fake = _resolve(rec.get("outputs", [None])[0])
            if fake is None:
                continue
            tgt = _resolve(str(target_root / f"{fake.stem}.png"))
            if tgt is None:
                continue
            try:
                A = _load_rgb(fake, 192)
                B = _load_rgb(tgt, 192)
                mad = float(np.abs(A - B).mean() / 255.0)
            except Exception:
                continue
            if mad > best_mad:
                best_mad = mad
                best = dict(rec)
                best["_diffswap_mad"] = mad
    if best is None:
        return _first_ok_record(jsonl_path)
    return best


def _pick_arc2face_record(jsonl_path: Path) -> Dict:
    """Prefer a non-self (source != expression) record with largest source-target gap."""
    cand = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _is_ok_record(rec):
                continue
            src = _resolve(rec.get("source_image"))
            tgt = _resolve(rec.get("expression_image"))
            if src is None or tgt is None:
                continue
            if str(src) == str(tgt):
                continue
            try:
                d = float(np.abs(_load_rgb(src, 192) - _load_rgb(tgt, 192)).mean())
            except Exception:
                d = 0.0
            cand.append((d, rec))
    if cand:
        cand.sort(key=lambda x: x[0], reverse=True)
        return cand[0][1]
    return _first_ok_record(jsonl_path)


def _to_image(x: Optional[Union[Path, Image.Image]], size: int) -> Image.Image:
    if x is None:
        return Image.new("RGB", (size, size), (40, 40, 40))
    if isinstance(x, Image.Image):
        return x.convert("RGB").resize((size, size), Image.BICUBIC)
    return Image.open(x).convert("RGB").resize((size, size), Image.BICUBIC)


def _maybe_path_variants(base_dir: Path, stem: str) -> List[Path]:
    return [
        base_dir / f"{stem}.png",
        base_dir / f"{stem}.jpg",
        base_dir / f"{stem}.jpeg",
    ]


def _resolve_diffswap(rec: Dict) -> Triplet:
    fake = _resolve(rec["outputs"][0])
    if fake is None:
        raise RuntimeError("DiffSwap fake path missing")

    src_id = fake.parent.name
    tgt_id = fake.stem
    src_candidates = _maybe_path_variants(ROOT / "Attack-DiffSwap" / "data" / "portrait" / "source", src_id)
    tgt_candidates = _maybe_path_variants(ROOT / "Attack-DiffSwap" / "data" / "portrait" / "target", tgt_id)

    source = next((p for p in src_candidates if p.exists()), None)
    target = next((p for p in tgt_candidates if p.exists()), None)

    if target is None:
        target = _resolve(rec.get("source_image"))

    mad = rec.get("_diffswap_mad", None)
    if mad is None:
        note = f"src_id={src_id}, tgt_id={tgt_id}"
    else:
        note = f"src_id={src_id}, tgt_id={tgt_id}, mad={mad:.4f}"
    return Triplet(source=source, target=target, fake=fake, method="DiffSwap", note=note)


def _resolve_arc2face(rec: Dict) -> Triplet:
    # Arc2Face semantics in this repo:
    # - base/target image: source_image (or target_image)
    # - donor/driver image: expression_image
    target = _resolve(rec.get("target_image")) or _resolve(rec.get("source_image"))
    source = _resolve(rec.get("expression_image")) or _resolve(rec.get("source_image"))
    fake = _resolve(rec["outputs"][0])

    note = ""
    if source is not None and target is not None:
        note = f"src={source.stem}, tgt={target.stem}"
    return Triplet(source=source, target=target, fake=fake, method="Arc2Face", note=note)


def _strip_variant(stem: str) -> str:
    for suffix in ("_inpaint", "_ref", "_GT", "_mask"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _resolve_reface(rec: Dict) -> Triplet:
    fake = _resolve(rec["outputs"][0])
    if fake is None:
        raise RuntimeError("ReFace fake path missing")

    # outputs path example: .../results/results/2/000000000000.png
    bucket = rec.get("replay_meta", {}).get("bucket", None)
    if bucket is None and fake.parent.name.isdigit():
        bucket = fake.parent.name

    pair_key = rec.get("replay_meta", {}).get("pair_index", "")
    if not pair_key:
        pair_key = _strip_variant(fake.stem)[:12]

    source = None
    target = None

    if bucket is not None:
        source = _resolve(str(ROOT / "Attack-REFace" / "data" / "faceswap_outputs" / "Outs" / "source_cropped" / f"{bucket}.png"))

    pi = _safe_int(pair_key)
    if pi is not None:
        target = _resolve(str(ROOT / "Attack-REFace" / "data" / "faceswap_outputs" / "Outs" / "target_cropped" / f"{pi}.png"))

    # Fallback to per-pair rendered intermediates
    if source is None:
        source = _resolve(str(fake.parent / f"{pair_key}_ref.png"))
    if target is None:
        target = _resolve(str(fake.parent / f"{pair_key}_GT.png")) or _resolve(rec.get("source_image"))

    note = f"bucket={bucket}, pair={pair_key}"
    return Triplet(source=source, target=target, fake=fake, method="ReFace", note=note)


def _infer_grid_layout(w: int, h: int, nrow: int = 4):
    for pad in (2, 0, 1, 4, 8):
        inner_w = w - pad * (nrow + 1)
        inner_h = h - pad * 2
        if inner_w <= 0 or inner_h <= 0:
            continue
        if inner_w % nrow != 0:
            continue
        cell_w = inner_w // nrow
        cell_h = inner_h
        if abs(cell_w - cell_h) <= 2:
            return pad, cell_w, cell_h
    return 0, w // nrow, h


def _crop_face_adapter_concat(concat_path: Path) -> Tuple[Image.Image, Image.Image]:
    img = Image.open(concat_path).convert("RGB")
    w, h = img.size
    pad, cell_w, cell_h = _infer_grid_layout(w, h, nrow=4)
    if cell_w <= 0 or cell_h <= 0:
        raise RuntimeError(f"Invalid concat image size: {concat_path} size={img.size}")

    # columns: [source, target, reenact, swap]
    src_x0 = pad
    src_y0 = pad
    tgt_x0 = pad + (cell_w + pad)
    tgt_y0 = pad
    source = img.crop((src_x0, src_y0, src_x0 + cell_w, src_y0 + cell_h))
    target = img.crop((tgt_x0, tgt_y0, tgt_x0 + cell_w, tgt_y0 + cell_h))
    return source, target


def _resolve_face_adapter(rec: Dict) -> Triplet:
    fake = _resolve(rec["outputs"][0])
    if fake is None:
        raise RuntimeError("Face-Adapter fake path missing")

    pair = rec.get("replay_meta", {}).get("pair", "")
    if not pair:
        stem = fake.stem
        pair = stem[:-5] if stem.endswith("_swap") else stem

    concat_dir = ROOT / "Attack-Face-Adapter" / "output_celebahq" / "concat"
    concat_path = None
    for ext in ("jpg", "png", "jpeg"):
        p = concat_dir / f"{pair}.{ext}"
        if p.exists():
            concat_path = p

    source = None
    target = None
    note = f"pair={pair}"
    if concat_path is not None:
        src_img, tgt_img = _crop_face_adapter_concat(concat_path)
        source = src_img
        target = tgt_img
        note += " (from concat)"
    else:
        if "_" in pair:
            a, b = pair.split("_", 1)
            test_root = ROOT / "Dataset-CelebA_HQ" / "test"
            source = _resolve(str(test_root / f"{a}.jpg"))
            target = _resolve(str(test_root / f"{b}.jpg"))
            note += " (fallback id parse)"

    return Triplet(source=source, target=target, fake=fake, method="Face-Adapter", note=note)


def _draw_grid(triplets: List[Triplet], output_path: Path, cell_size: int = 320) -> None:
    n = len(triplets)
    left_pad = 180
    top_pad = 90
    right_pad = 30
    bottom_pad = 30
    gap_x = 12
    gap_y = 12

    width = left_pad + n * cell_size + (n - 1) * gap_x + right_pad
    height = top_pad + 3 * cell_size + 2 * gap_y + bottom_pad
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    rows = ["Source", "Target", "Fake"]
    for r, row_name in enumerate(rows):
        y = top_pad + r * (cell_size + gap_y) + cell_size // 2 - 8
        draw.text((20, y), row_name, fill=(235, 235, 235), font=font)

    for c, t in enumerate(triplets):
        x0 = left_pad + c * (cell_size + gap_x)
        draw.text((x0 + 6, 16), t.method, fill=(245, 245, 245), font=font)
        if t.note:
            draw.text((x0 + 6, 34), t.note[:54], fill=(170, 170, 170), font=font)

        imgs = [
            _to_image(t.source, cell_size),
            _to_image(t.target, cell_size),
            _to_image(t.fake, cell_size),
        ]
        for r, img in enumerate(imgs):
            y0 = top_pad + r * (cell_size + gap_y)
            canvas.paste(img, (x0, y0))
            draw.rectangle((x0, y0, x0 + cell_size - 1, y0 + cell_size - 1), outline=(95, 95, 95), width=1)
            missing = (r == 0 and t.source is None) or (r == 1 and t.target is None) or (r == 2 and t.fake is None)
            if missing:
                draw.text((x0 + 8, y0 + 8), "MISSING", fill=(255, 120, 120), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Source/Target/Fake triplets for diffusion attacks.")
    p.add_argument("--output", required=True, help="Output image path")
    p.add_argument("--cell-size", type=int, default=320)
    p.add_argument(
        "--diffswap-jsonl",
        default=str(ROOT / "Attack-DiffSwap" / "outputs_replay_large" / "generation_results.jsonl"),
    )
    p.add_argument(
        "--arc2face-jsonl",
        default=str(ROOT / "Attack-arc2face_wrapper" / "outputs" / "generation_results.jsonl"),
    )
    p.add_argument(
        "--reface-jsonl",
        default=str(ROOT / "Attack-REFace" / "outputs" / "generation_results.jsonl"),
    )
    p.add_argument(
        "--face-adapter-jsonl",
        default=str(ROOT / "Attack-Face-Adapter" / "outputs_replay_fixed" / "generation_results.jsonl"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    method_jsonl = [
        ("DiffSwap", Path(args.diffswap_jsonl)),
        ("Arc2Face", Path(args.arc2face_jsonl)),
        ("ReFace", Path(args.reface_jsonl)),
        ("Face-Adapter", Path(args.face_adapter_jsonl)),
    ]

    for _, jp in method_jsonl:
        if not jp.exists():
            raise FileNotFoundError(f"JSONL not found: {jp}")

    rec_diff = _pick_diffswap_record(Path(args.diffswap_jsonl), max_scan=400)
    rec_arc2 = _pick_arc2face_record(Path(args.arc2face_jsonl))
    rec_refa = _first_ok_record(Path(args.reface_jsonl))
    rec_fada = _first_ok_record(Path(args.face_adapter_jsonl))

    triplets = [
        _resolve_diffswap(rec_diff),
        _resolve_arc2face(rec_arc2),
        _resolve_reface(rec_refa),
        _resolve_face_adapter(rec_fada),
    ]

    out = Path(args.output).resolve()
    _draw_grid(triplets, out, cell_size=int(args.cell_size))
    print(f"[done] saved grid: {out}")
    for t in triplets:
        print(f"- {t.method}: {t.note}")


if __name__ == "__main__":
    main()
