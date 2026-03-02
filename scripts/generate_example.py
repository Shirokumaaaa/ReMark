#!/usr/bin/env python
import csv
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from arc2face.expression_generator import Arc2FaceExpressionGenerator, ExpressionGenerationConfig


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    cfg_path = ROOT_DIR / "config" / "generate_config.json"
    cfg_all = load_config(cfg_path)

    models_dir = cfg_all["models_dir"]
    manifest = Path(cfg_all["manifest"])
    output_dir = Path(cfg_all["output_dir"])
    limit = int(cfg_all.get("limit", 8))
    start_index = int(cfg_all.get("start_index", 0))
    strict_cuda = bool(cfg_all.get("strict_cuda_provider", True))
    cfg = cfg_all["config"]

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generation_results.jsonl"

    gen_cfg = ExpressionGenerationConfig(
        use_ref_adapter=bool(cfg.get("use_ref_adapter", True)),
        lora_ref_scale=float(cfg.get("lora_ref_scale", 1.0)),
        num_steps=int(cfg.get("num_steps", 10)),
        guidance_scale=float(cfg.get("guidance_scale", 3.0)),
        num_images=int(cfg.get("num_images", 1)),
        exp_adapter_scale=float(cfg.get("exp_adapter_scale", 1.0)),
        output_size=int(cfg.get("output_size", 256)),
        seed=cfg.get("seed", None),
    )

    generator = Arc2FaceExpressionGenerator(models_dir=models_dir, strict_cuda_provider=strict_cuda)

    with manifest.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    begin = max(0, start_index)
    end = min(len(rows), begin + max(0, limit))
    rows = rows[begin:end]

    with log_path.open("a", encoding="utf-8") as log_f:
        for row in rows:
            idx = int(row.get("index", -1))
            source = row["source_image"]
            expression = row["expression_image"]
            reference = row.get("reference_image") or source
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
                    config=gen_cfg,
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
