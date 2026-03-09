import csv
import os
import random
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _read_csv_paths(csv_path):
    paths = []
    if not os.path.isfile(csv_path):
        return paths

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get("img_path", "").strip()
            if img_path and os.path.isfile(img_path):
                paths.append(img_path)
    return paths


def resolve_target_images(default_target_dir):
    # Priority: explicit env -> sibling Dataset-CelebA_HQ csv -> original hardcoded directory.
    candidates = []

    env_csv = os.environ.get("SEPMARK_TARGET_CSV", "").strip()
    if env_csv:
        candidates.extend(_read_csv_paths(env_csv))

    remark_csv = _repo_root().parent / "Dataset-CelebA_HQ" / "val.csv"
    candidates.extend(_read_csv_paths(str(remark_csv)))

    if candidates:
        return candidates

    if default_target_dir and os.path.isdir(default_target_dir):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [
            str(p) for p in Path(default_target_dir).iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
        if files:
            return files

    return []


def random_target(target_images):
    if not target_images:
        return None
    return random.choice(target_images)
