#!/usr/bin/env python
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


DEFAULT_MODELS_DIR = "/mnt/personal_workspace/chenkeyu/Arc2Face/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from a triplet manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "CSV columns: source_image (identity input), expression_image (target expression), "
            "reference_image (for Reference Adapter)."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    parser.add_argument("--limit", type=int, default=16, help="How many rows to process.")
    parser.add_argument("--start_index", type=int, default=0, help="Row start offset in manifest.")
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--exp_adapter_scale", type=float, default=1.0)
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--lora_ref_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_ref_adapter", action="store_true", default=True)
    parser.add_argument("--disable_ref_adapter", action="store_true", default=False)
    parser.add_argument("--strict_cuda_provider", action="store_true", default=True)
    parser.add_argument("--no_strict_cuda_provider", action="store_true", default=False)
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR,
                        help="Arc2Face models directory.")
    parser.add_argument(
        "--force_reference_source",
        action="store_true",
        default=True,
        help="Force reference_image = source_image (paper-aligned default).",
    )
    parser.add_argument(
        "--no_force_reference_source",
        action="store_true",
        default=False,
        help="Use manifest reference_image directly.",
    )
    parser.add_argument("--allow_self_source", action="store_true", default=False,
                        help="Allow source_image == expression_image (default: disallow).")
    return parser.parse_args()


def resolve_reference(row: dict, force_reference_source: bool) -> str:
    if force_reference_source:
        return row["source_image"]
    return row.get("reference_image") or row["source_image"]


def main() -> None:
    args = parse_args()
    if args.no_force_reference_source:
        args.force_reference_source = False
    os.environ["ARC2FACE_MODELS_DIR"] = args.models_dir
    print("[INFO] importing generator module...", flush=True)
    from arc2face.expression_generator import Arc2FaceExpressionGenerator, ExpressionGenerationConfig
    print("[INFO] generator module imported.", flush=True)

    use_ref = args.use_ref_adapter and not args.disable_ref_adapter
    strict_cuda = args.strict_cuda_provider and not args.no_strict_cuda_provider

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generation_results.jsonl"
    print(f"[INFO] output_dir={output_dir}", flush=True)
    print(f"[INFO] manifest={args.manifest}", flush=True)
    print(f"[INFO] models_dir={args.models_dir}", flush=True)

    cfg = ExpressionGenerationConfig(
        use_ref_adapter=use_ref,
        lora_ref_scale=args.lora_ref_scale,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        num_images=args.num_images,
        exp_adapter_scale=args.exp_adapter_scale,
        output_size=args.output_size,
        seed=args.seed,
    )

    print("[INFO] initializing generator...", flush=True)
    generator = Arc2FaceExpressionGenerator(models_dir=args.models_dir, strict_cuda_provider=strict_cuda)
    print("[INFO] generator initialized.", flush=True)

    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    begin = max(0, args.start_index)
    end = min(len(reader), begin + max(0, args.limit))
    rows = reader[begin:end]
    print(f"[INFO] rows_to_process={len(rows)} start={begin} end={end}", flush=True)

    with open(log_path, "a", encoding="utf-8") as log_f:
        for row in rows:
            idx = int(row.get("index", -1))
            print(f"[INFO] processing index={idx}", flush=True)
            source = row["source_image"]
            expression = row["expression_image"]
            reference = resolve_reference(row, args.force_reference_source)
            if (not args.allow_self_source) and (source == expression):
                raise ValueError(
                    f"source_image equals expression_image at index={idx}; "
                    "this run enforces non-self source."
                )
            result = {
                "index": idx,
                "source_image": source,
                "expression_image": expression,
                "reference_image": reference,
                "ok": False,
                "outputs": [],
                "error": None,
            }
            try:
                images = generator.generate(
                    source_image=source,
                    expression_image=expression,
                    reference_image=reference,
                    config=cfg,
                )
                out_paths = []
                for j, img in enumerate(images):
                    out_name = f"{idx:06d}_{j}.png"
                    out_path = output_dir / out_name
                    img.save(out_path)
                    out_paths.append(str(out_path))
                result["ok"] = True
                result["outputs"] = out_paths
                print(f"[OK] index={idx} saved={len(out_paths)}")
            except Exception as exc:  # noqa: BLE001
                result["error"] = str(exc)
                print(f"[FAIL] index={idx} error={exc}")
            log_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            log_f.flush()


if __name__ == "__main__":
    main()
