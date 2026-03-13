#!/usr/bin/env python3
"""
构建与 Arc2Face wrapper 相同格式的 generation_results.jsonl（供 Forensic-Remark_Module replay 使用）。

场景：
  - 你已经用 Face-Adapter 跑完推理，手上有一张 CSV，记录了「输入图片 → 换脸输出图片」的对应关系；
  - 本脚本读取该 CSV，并在指定目录写出 generation_results.jsonl，
    每一行的字段与 Attack-arc2face_wrapper/scripts/generate_from_manifest.py 保持一致：

    {
      "index": <int>,
      "source_image": "<原始/含水印图路径>",
      "expression_image": "<可选，占位>",
      "reference_image": "<可选，占位>",
      "ok": true,
      "outputs": ["<换脸结果路径>"],
      "error": null
    }

用法示例（在 Attack-Face-Adapter 根目录）：

  python tools/build_generation_results_from_csv.py \
    --mapping-csv data/celebahq_eval/face_adapter_pairs.csv \
    --output-dir outputs

要求 mapping-csv 至少包含两列：
  - source_image : 与 Forensic-Remark_Module 中 CSV 的 img_path 一致（用于 replay key）
  - output_image : 对应的 Face-Adapter 换脸结果路径

其余列会被忽略。
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mapping-csv",
        type=Path,
        required=True,
        help="CSV with at least columns: source_image, output_image",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generation_results.jsonl will be written (created if not exists).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping_csv: Path = args.mapping_csv
    out_dir: Path = args.output_dir

    if not mapping_csv.is_file():
        raise FileNotFoundError(f"mapping-csv not found: {mapping_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "generation_results.jsonl"

    with mapping_csv.open("r", encoding="utf-8", newline="") as f_in, \
            jsonl_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        if "source_image" not in reader.fieldnames or "output_image" not in reader.fieldnames:
            raise ValueError(
                f"mapping-csv must contain columns 'source_image' and 'output_image', "
                f"got {reader.fieldnames}"
            )
        for idx, row in enumerate(reader):
            src = row["source_image"]
            out = row["output_image"]
            rec = {
                "index": int(row.get("index", idx)),
                "source_image": src,
                "expression_image": row.get("expression_image", ""),
                "reference_image": row.get("reference_image", ""),
                "ok": bool(row.get("ok", True)),
                "outputs": [out],
                "error": None,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()

