#!/usr/bin/env python
import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build Arc2Face triplets for expression transfer with identity/background preservation: "
            "source = reference = each dataset image, expression = random different image."
        )
    )
    p.add_argument("--input_csv", type=str, required=True, help="CSV with column `img_path`.")
    p.add_argument("--output_csv", type=str, required=True, help="Output triplet CSV.")
    p.add_argument("--seed", type=int, default=1234, help="Random seed.")
    p.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    inp = Path(args.input_csv)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        paths = [str(Path(r["img_path"]).resolve()) for r in rd]

    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images to build non-self expression mapping.")

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    rows = []
    for i, src in enumerate(paths):
        expression = src
        while expression == src:
            expression = rng.choice(paths)
        rows.append(
            {
                "index": i,
                "source_image": src,
                "expression_image": expression,
                "reference_image": src,
            }
        )

    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f, fieldnames=["index", "source_image", "expression_image", "reference_image"]
        )
        wr.writeheader()
        wr.writerows(rows)

    print(f"[OK] wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
