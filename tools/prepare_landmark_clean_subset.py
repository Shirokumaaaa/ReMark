#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_failures(paths):
    failed = set()
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_path = (row.get("img_path") or "").strip()
                if img_path:
                    failed.add(os.path.abspath(img_path))
    return failed


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def symlink_split(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    kept = 0
    for row in rows:
        src = os.path.abspath(row["img_path"])
        dst = os.path.join(out_dir, os.path.basename(src))
        if not os.path.lexists(dst):
            os.symlink(src, dst)
        kept += 1
    return kept


def build_cache_index(cache_dir):
    bit_dir = os.path.join(os.path.abspath(cache_dir), "bits128")
    if not os.path.isdir(bit_dir):
        return set()
    return {name for name in os.listdir(bit_dir) if name.endswith(".npy")}


def cache_key(image_path, predictor_path, canonical_bits, bits_per_value):
    abs_path = os.path.abspath(image_path)
    key = f"{abs_path}|{os.path.abspath(predictor_path)}|{canonical_bits}|{bits_per_value}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest() + ".npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--failure-csv-glob-dir", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--predictor-path", default="")
    parser.add_argument("--canonical-bits", type=int, default=128)
    parser.add_argument("--bits-per-value", type=int, default=4)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--lampmark-manifest-dir", required=True)
    args = parser.parse_args()

    train_rows = read_rows(args.train_csv)
    val_rows = read_rows(args.val_csv)

    failure_paths = []
    if args.failure_csv_glob_dir and os.path.isdir(args.failure_csv_glob_dir):
        failure_paths = [
            os.path.join(args.failure_csv_glob_dir, x)
            for x in sorted(os.listdir(args.failure_csv_glob_dir))
            if x.endswith("_failures.csv")
        ]
    failed = read_failures(failure_paths)
    cache_index = build_cache_index(args.cache_dir) if args.cache_dir else set()

    def keep_row(row):
        img_path = os.path.abspath(row["img_path"])
        if img_path in failed:
            return False
        if cache_index:
            return cache_key(
                image_path=img_path,
                predictor_path=args.predictor_path,
                canonical_bits=args.canonical_bits,
                bits_per_value=args.bits_per_value,
            ) in cache_index
        return True

    filtered = {
        "train": [row for row in train_rows if keep_row(row)],
        "val": [row for row in val_rows if keep_row(row)],
    }

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    kept_counts = {}
    for split, rows in filtered.items():
        write_csv(os.path.join(out_root, f"{split}.csv"), rows, fieldnames=list(rows[0].keys()) if rows else ["img_path"])
        kept_counts[split] = symlink_split(rows, os.path.join(out_root, split))

    lampmark_dir = os.path.abspath(args.lampmark_manifest_dir)
    os.makedirs(lampmark_dir, exist_ok=True)
    for split, rows in filtered.items():
        write_csv(
            os.path.join(lampmark_dir, f"celeba_hq_128_10k_landmark_clean_{split}.csv"),
            [{"img_path": row["img_path"]} for row in rows],
            fieldnames=["img_path"],
        )

    print(f"failed={len(failed)}")
    print(f"train_kept={kept_counts['train']}")
    print(f"val_kept={kept_counts['val']}")
    if cache_index:
        print(f"cache_hits_mode=1")
    print(f"out_root={out_root}")
    print(f"lampmark_manifest_dir={lampmark_dir}")


if __name__ == "__main__":
    main()
