# Copyright 2025 Adobe
# All Rights Reserved.
#
# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from trustmark import TrustMark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrustMark on a CSV manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data_manifests/celeba_hq_128_100_val.csv",
        help="CSV file with an img_path column.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root directory where logs and visualizations are stored.",
    )
    parser.add_argument("--run-name", type=str, default="", help="Optional run name.")
    parser.add_argument("--mode", type=str, default="Q", choices=["B", "C", "P", "Q"])
    parser.add_argument(
        "--encoding",
        type=str,
        default="BCH_5",
        choices=["BCH_SUPER", "BCH_5", "BCH_4", "BCH_3"],
    )
    parser.add_argument("--detectfirst", action="store_true", help="Enable detection-first decoding.")
    parser.add_argument("--rotation", action="store_true", help="Enable rotation-aware decoding.")
    parser.add_argument("--wm-strength", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all rows in manifest.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-watermarked", action="store_true")
    parser.add_argument("--save-compare", action="store_true", default=True)
    parser.add_argument("--save-compare-max", type=int, default=0, help="0 means save all.")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def read_manifest_csv(manifest_csv: str) -> List[Path]:
    csv_path = Path(manifest_csv).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV has no header: {csv_path}")

        if "img_path" not in reader.fieldnames:
            raise ValueError(f"Manifest CSV must contain 'img_path' column: {csv_path}")

        paths: List[Path] = []
        for row in reader:
            raw = (row.get("img_path") or "").strip()
            if not raw:
                continue
            p = Path(raw).expanduser()
            if p.is_file():
                paths.append(p)
    return paths


def deterministic_bitstring(key: str, n_bits: int, seed: int = 1234) -> str:
    bits = ""
    counter = 0
    while len(bits) < n_bits:
        raw = f"{seed}|{key}|{counter}".encode("utf-8")
        digest = hashlib.sha256(raw).digest()
        bits += "".join(f"{b:08b}" for b in digest)
        counter += 1
    return bits[:n_bits]


def psnr_uint8(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.int16)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.int16)
    mse = np.mean(np.square(arr_a - arr_b))
    if mse <= 0:
        return float("inf")
    return 20 * math.log10(255.0) - 10 * math.log10(float(mse))


def bit_accuracy(gt: str, pred: str) -> float:
    if not gt:
        return 0.0
    if not pred:
        return 0.0
    n = len(gt)
    m = min(n, len(pred))
    hit = sum(1 for i in range(m) if gt[i] == pred[i])
    return hit / float(n)


def make_comparison_image(
    cover: Image.Image, stego: Image.Image, title_text: str, amplify: float = 16.0
) -> Image.Image:
    cover_rgb = cover.convert("RGB")
    stego_rgb = stego.convert("RGB")
    if stego_rgb.size != cover_rgb.size:
        stego_rgb = stego_rgb.resize(cover_rgb.size, Image.Resampling.BILINEAR)

    arr_cover = np.asarray(cover_rgb, dtype=np.int16)
    arr_stego = np.asarray(stego_rgb, dtype=np.int16)
    arr_diff = np.abs(arr_stego - arr_cover).astype(np.float32)
    arr_diff = np.clip(arr_diff * amplify, 0, 255).astype(np.uint8)
    diff_vis = Image.fromarray(arr_diff, mode="RGB")

    w, h = cover_rgb.size
    header_h = 30
    canvas = Image.new("RGB", (w * 3, h + header_h), color=(20, 20, 20))
    canvas.paste(cover_rgb, (0, header_h))
    canvas.paste(stego_rgb, (w, header_h))
    canvas.paste(diff_vis, (w * 2, header_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title_text, fill=(240, 240, 240))
    draw.text((8, header_h + 8), "Cover", fill=(255, 255, 255))
    draw.text((w + 8, header_h + 8), "Watermarked", fill=(255, 255, 255))
    draw.text((2 * w + 8, header_h + 8), "Abs Diff x16", fill=(255, 255, 255))
    return canvas


def save_logs_csv(rows: List[Dict], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    encoding_map = {
        "BCH_SUPER": TrustMark.Encoding.BCH_SUPER,
        "BCH_5": TrustMark.Encoding.BCH_5,
        "BCH_4": TrustMark.Encoding.BCH_4,
        "BCH_3": TrustMark.Encoding.BCH_3,
    }
    encoding_enum = encoding_map[args.encoding]

    manifest_path = Path(args.manifest).expanduser()
    image_paths = read_manifest_csv(str(manifest_path))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No valid image found in {manifest_path}")

    run_name = args.run_name.strip()
    if not run_name:
        run_name = datetime.now().strftime("val_eval_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).expanduser() / run_name
    compare_dir = run_dir / "comparisons"
    watermarked_dir = run_dir / "watermarked"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.save_compare:
        compare_dir.mkdir(parents=True, exist_ok=True)
    if args.save_watermarked:
        watermarked_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    metrics_csv_path = run_dir / "metrics.csv"
    summary_json_path = run_dir / "summary.json"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"Manifest: {manifest_path}")
    log(f"Images to process: {len(image_paths)}")
    log(f"Output dir: {run_dir}")
    log(
        f"Model={args.mode} Encoding={args.encoding} "
        f"DetectFirst={args.detectfirst} Rotation={args.rotation} WM_STRENGTH={args.wm_strength}"
    )

    tm = TrustMark(
        verbose=True,
        model_type=args.mode,
        encoding_type=encoding_enum,
        loadBBoxDetector=args.detectfirst,
    )
    capacity = tm.schemaCapacity()
    log(f"Schema capacity: {capacity} bits")

    rows: List[Dict] = []
    stats = {
        "n_total": 0,
        "n_success": 0,
        "n_wm_present": 0,
        "n_clean_false_positive": 0,
        "sum_psnr": 0.0,
        "sum_bit_acc": 0.0,
        "sum_encode_ms": 0.0,
        "sum_decode_ms": 0.0,
    }

    t0 = time.time()
    for idx, img_path in enumerate(image_paths):
        row: Dict = {
            "index": idx,
            "img_path": str(img_path),
            "status": "ok",
        }
        try:
            cover = Image.open(img_path).convert("RGB")
            secret_gt = deterministic_bitstring(str(img_path), capacity, seed=args.seed)

            t_enc = time.time()
            stego = tm.encode(
                cover,
                secret_gt,
                MODE="binary",
                WM_STRENGTH=args.wm_strength,
            )
            encode_ms = (time.time() - t_enc) * 1000.0

            t_dec = time.time()
            secret_pred, wm_present, wm_schema = tm.decode(
                stego.convert("RGB"),
                MODE="binary",
                DETECTFIRST=args.detectfirst,
                ROTATION=args.rotation,
            )
            decode_ms = (time.time() - t_dec) * 1000.0

            clean_pred, clean_present, clean_schema = tm.decode(
                cover,
                MODE="binary",
                DETECTFIRST=args.detectfirst,
                ROTATION=args.rotation,
            )

            exact_match = bool(secret_pred == secret_gt)
            acc = bit_accuracy(secret_gt, secret_pred)
            psnr = psnr_uint8(stego, cover)

            if args.save_watermarked:
                wm_out = watermarked_dir / f"{idx:04d}_{img_path.stem}_{args.mode}.png"
                stego.save(wm_out)

            if args.save_compare and (args.save_compare_max == 0 or idx < args.save_compare_max):
                title = (
                    f"{img_path.name} | present={int(wm_present)} "
                    f"exact={int(exact_match)} bit_acc={acc:.3f} psnr={psnr:.2f}dB"
                )
                comp = make_comparison_image(cover, stego, title_text=title, amplify=16.0)
                comp.save(compare_dir / f"{idx:04d}_{img_path.stem}_compare.jpg", quality=95)

            stats["n_total"] += 1
            stats["n_success"] += int(exact_match)
            stats["n_wm_present"] += int(wm_present)
            stats["n_clean_false_positive"] += int(clean_present)
            stats["sum_psnr"] += float(psnr)
            stats["sum_bit_acc"] += float(acc)
            stats["sum_encode_ms"] += float(encode_ms)
            stats["sum_decode_ms"] += float(decode_ms)

            row.update(
                {
                    "wm_present": int(wm_present),
                    "wm_schema": int(wm_schema),
                    "clean_present": int(clean_present),
                    "clean_schema": int(clean_schema),
                    "exact_match": int(exact_match),
                    "bit_acc": f"{acc:.6f}",
                    "psnr_db": f"{psnr:.6f}",
                    "encode_ms": f"{encode_ms:.3f}",
                    "decode_ms": f"{decode_ms:.3f}",
                    "gt_secret_len": len(secret_gt),
                    "pred_secret_len": len(secret_pred),
                    "clean_pred_len": len(clean_pred),
                    "secret_gt": secret_gt,
                    "secret_pred": secret_pred,
                }
            )
        except Exception as e:
            row["status"] = f"error: {type(e).__name__}: {e}"
            log(f"[Error][{idx}] {img_path}: {e}")

        rows.append(row)
        if (idx + 1) % max(1, args.progress_every) == 0 or (idx + 1) == len(image_paths):
            log(f"Progress: {idx + 1}/{len(image_paths)}")

    elapsed = time.time() - t0
    save_logs_csv(rows, metrics_csv_path)

    n = max(1, stats["n_total"])
    summary = {
        "manifest": str(manifest_path),
        "n_total": stats["n_total"],
        "n_exact_match": stats["n_success"],
        "n_wm_present": stats["n_wm_present"],
        "n_clean_false_positive": stats["n_clean_false_positive"],
        "exact_match_rate": stats["n_success"] / n,
        "wm_present_rate": stats["n_wm_present"] / n,
        "clean_false_positive_rate": stats["n_clean_false_positive"] / n,
        "avg_bit_acc": stats["sum_bit_acc"] / n,
        "avg_psnr_db": stats["sum_psnr"] / n,
        "avg_encode_ms": stats["sum_encode_ms"] / n,
        "avg_decode_ms": stats["sum_decode_ms"] / n,
        "elapsed_sec": elapsed,
        "mode": args.mode,
        "encoding": args.encoding,
        "capacity_bits": capacity,
        "detectfirst": args.detectfirst,
        "rotation": args.rotation,
        "wm_strength": args.wm_strength,
    }

    with summary_json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    log("Done.")
    log(f"metrics.csv: {metrics_csv_path}")
    log(f"summary.json: {summary_json_path}")
    log("Summary:")
    for k, v in summary.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
