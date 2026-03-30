#!/usr/bin/env python3
"""
Generate "stargan2-required" FFHQ CSV labels.

stargan2 / stargan_v1_fixed require CSV columns that become `batch['attrs']`.
This script takes existing FFHQ manifests (only `img_path`) and outputs new CSVs
with the CelebA-style binary attribute columns:

  img_path,Black_Hair,Blond_Hair,Brown_Hair,Male,Young

IMPORTANT:
This repo does not include an attribute predictor for CelebA attributes on FFHQ.
So by default we generate deterministic *pseudo labels* from `img_path` hash so
the pipeline can run end-to-end.

If you want *real* predicted attributes, ask me for an extension and provide
your attribute classifier (model name/path and preprocessing).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


DEFAULT_ATTRS = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]


def _pseudo_bits_from_img_path(img_path: str, salt: str, n_bits: int) -> list[int]:
    """
    Deterministically map img_path -> {0,1}^n_bits.
    Uses sha256 so results are stable across runs/machines.
    """
    h = hashlib.sha256((salt + "|" + img_path).encode("utf-8")).digest()
    # Use first n_bits bytes, take parity to get 0/1.
    bits: list[int] = []
    for i in range(n_bits):
        b = h[i]  # 0..255
        bits.append(int(b & 1))
    return bits


def label_csv(
    in_csv: Path,
    out_csv: Path,
    attrs: list[str],
    salt: str,
) -> None:
    df = pd.read_csv(in_csv)
    if "img_path" not in df.columns:
        raise ValueError(f"{in_csv} must contain column `img_path`, got {list(df.columns)}")

    bits_cols = {a: [] for a in attrs}
    n_bits = len(attrs)

    for img_path in df["img_path"].astype(str).tolist():
        bits = _pseudo_bits_from_img_path(img_path, salt=salt, n_bits=n_bits)
        for a, bit in zip(attrs, bits):
            bits_cols[a].append(int(bit))

    out_df = pd.DataFrame({"img_path": df["img_path"].astype(str)})
    for a in attrs:
        # stargan2 converts with (attrs > 0.5).float(), so 0/1 is fine.
        out_df[a] = bits_cols[a]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[label_ffhq_for_stargan2] wrote {out_csv} with rows={len(out_df)} cols={list(out_df.columns)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-train-csv", type=Path, required=True)
    p.add_argument("--input-val-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--attrs",
        type=str,
        nargs="+",
        default=DEFAULT_ATTRS,
        help="Attribute column names. Order must match stargan2_selected_attrs first 5 dims.",
    )
    p.add_argument(
        "--salt",
        type=str,
        default="remark_stargan2_pseudo_v1",
        help="Hash salt to make pseudo labels deterministic but configurable.",
    )
    args = p.parse_args()

    out_train = args.output_dir / "ffhq_10k_train_with_attrs.csv"
    out_val = args.output_dir / "ffhq_10k_val_with_attrs.csv"

    label_csv(args.input_train_csv, out_train, attrs=args.attrs, salt=args.salt)
    label_csv(args.input_val_csv, out_val, attrs=args.attrs, salt=args.salt)


if __name__ == "__main__":
    main()

