#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw


def _read_paths(csv_path: str, limit: int) -> List[str]:
    out: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = str(row.get("img_path", "")).strip()
            if p and os.path.exists(p):
                out.append(p)
            if len(out) >= limit:
                break
    if len(out) < 2:
        raise RuntimeError(f"Not enough valid image paths from: {csv_path}")
    return out


def _build_pairs(paths: List[str], n_pairs: int) -> List[Tuple[str, str]]:
    n = min(n_pairs, len(paths))
    return [(paths[(i + 1) % n], paths[i]) for i in range(n)]


def _load_arc2face(wrapper_root: str):
    import sys

    if wrapper_root not in sys.path:
        sys.path.insert(0, wrapper_root)
    from arc2face.expression_generator import Arc2FaceExpressionGenerator, ExpressionGenerationConfig

    return Arc2FaceExpressionGenerator, ExpressionGenerationConfig


def _make_grid(
    pairs: List[Tuple[str, str]],
    method_name: str,
    output_path: str,
    generator,
    cfg_cls,
    output_size: int,
    num_steps: int,
    guidance_scale: float,
    reference_mode: str,
):
    cell = 192
    pad = 8
    canvas = Image.new("RGB", ((cell + pad) * len(pairs) + pad, (cell + pad) * 3 + pad + 20), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 2), method_name, fill=(220, 220, 220))

    for i, (src, tgt) in enumerate(pairs):
        x = pad + i * (cell + pad)
        y1 = pad + 20
        y2 = y1 + cell + pad
        y3 = y2 + cell + pad
        src_img = Image.open(src).convert("RGB").resize((cell, cell), Image.BICUBIC)
        tgt_img = Image.open(tgt).convert("RGB").resize((cell, cell), Image.BICUBIC)

        if reference_mode == "source":
            ref = src
        elif reference_mode == "expression":
            ref = tgt
        else:
            ref = None

        cfg = cfg_cls(
            use_ref_adapter=(reference_mode != "none"),
            lora_ref_scale=1.0,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            num_images=1,
            exp_adapter_scale=1.0,
            output_size=output_size,
            seed=1234 + i,
        )

        try:
            fake = generator.generate(source_image=src, expression_image=tgt, reference_image=ref, config=cfg)[0]
            fake = fake.resize((cell, cell), Image.BICUBIC)
        except Exception as e:
            fake = Image.new("RGB", (cell, cell), (90, 10, 10))
            ed = ImageDraw.Draw(fake)
            ed.text((5, 5), "ERR", fill=(255, 255, 255))
            ed.text((5, 24), str(e)[:60], fill=(255, 255, 255))

        canvas.paste(src_img, (x, y1))
        canvas.paste(tgt_img, (x, y2))
        canvas.paste(fake, (x, y3))

    draw.text((2, pad + 20 + cell // 2), "Source", fill=(180, 180, 180))
    draw.text((2, pad + 20 + cell + pad + cell // 2), "Target", fill=(180, 180, 180))
    draw.text((2, pad + 20 + 2 * (cell + pad) + cell // 2), "Fake", fill=(180, 180, 180))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    ap = argparse.ArgumentParser(description="Standalone Arc2Face quality check (no watermark embedding).")
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--wrapper_root", default="/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper")
    ap.add_argument("--models_dir", default="/mnt/personal_workspace/chenkeyu/Arc2Face/models")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--n_pairs", type=int, default=8)
    ap.add_argument("--read_limit", type=int, default=32)
    ap.add_argument("--output_size", type=int, default=512)
    ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--reference_mode", choices=["source", "expression", "none"], default="source")
    ap.add_argument("--strict_cuda_provider", action="store_true")
    args = ap.parse_args()

    os.environ["ARC2FACE_MODELS_DIR"] = args.models_dir
    Arc2FaceExpressionGenerator, ExpressionGenerationConfig = _load_arc2face(args.wrapper_root)

    paths = _read_paths(args.manifest_csv, args.read_limit)
    pairs = _build_pairs(paths, args.n_pairs)

    print("[debug_arc2face_standalone] init generator...")
    gen = Arc2FaceExpressionGenerator(
        models_dir=args.models_dir,
        strict_cuda_provider=args.strict_cuda_provider,
    )
    print("[debug_arc2face_standalone] generator ready")

    method_name = (
        f"Arc2Face standalone | ref={args.reference_mode} | "
        f"output={args.output_size} | steps={args.num_steps} | cfg={args.guidance_scale}"
    )
    _make_grid(
        pairs=pairs,
        method_name=method_name,
        output_path=args.out_png,
        generator=gen,
        cfg_cls=ExpressionGenerationConfig,
        output_size=args.output_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        reference_mode=args.reference_mode,
    )
    print(f"[debug_arc2face_standalone] saved: {args.out_png}")


if __name__ == "__main__":
    main()
