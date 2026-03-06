#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def numeric_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def write_csv(paths, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path"])
        for p in paths:
            writer.writerow([str(p)])


def materialize_links(paths, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old in dst_dir.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()
    for src in paths:
        dst = dst_dir / src.name
        dst.symlink_to(src)


def collect_split(dataset_root: Path, split: str, n: int):
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    images = sorted(split_dir.glob("*.jpg"), key=numeric_sort_key)
    if n > 0:
        images = images[:n]
    return images


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset_root = repo_root.parent / "Dataset-CelebA_HQ"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--target-split", default="test")
    parser.add_argument("--num-source", type=int, default=8)
    parser.add_argument("--num-target", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=repo_root / "data" / "celebahq_eval")
    parser.add_argument("--portrait-jpg-dir", type=Path, default=repo_root / "data" / "portrait_jpg")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    source_paths = collect_split(dataset_root, args.source_split, args.num_source)
    target_paths = collect_split(dataset_root, args.target_split, args.num_target)

    source_csv = args.out_dir / "source.csv"
    target_csv = args.out_dir / "target.csv"
    write_csv(source_paths, source_csv)
    write_csv(target_paths, target_csv)

    materialize_links(source_paths, args.portrait_jpg_dir / "source")
    materialize_links(target_paths, args.portrait_jpg_dir / "target")

    print(f"dataset_root={dataset_root}")
    print(f"source_count={len(source_paths)} csv={source_csv}")
    print(f"target_count={len(target_paths)} csv={target_csv}")
    print(f"source_links_dir={args.portrait_jpg_dir / 'source'}")
    print(f"target_links_dir={args.portrait_jpg_dir / 'target'}")


if __name__ == "__main__":
    main()
