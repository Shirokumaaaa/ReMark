#!/usr/bin/env python3
"""
Run SimSwap fake2fake sweep on BCE and KL(target), then summarize metrics.

Run inside sepmark env:
  conda run --no-capture-output -n sepmark python tools/sweep_simswap_kl_bce.py
"""

import csv
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_BASE = "configs/stage1_vae.yaml"
OVERRIDE_DIR = ROOT / "configs" / "experiments" / "sweeps" / "simswap_kl_bce"


EXPERIMENTS = [
    {"bce": 0.5, "kl_target": 0.001},
    {"bce": 1.0, "kl_target": 0.001},
    {"bce": 2.0, "kl_target": 0.001},
    {"bce": 3.0, "kl_target": 0.001},
    {"bce": 0.5, "kl_target": 0.003},
    {"bce": 1.0, "kl_target": 0.003},
    {"bce": 2.0, "kl_target": 0.003},
    {"bce": 3.0, "kl_target": 0.003},
]


def slug(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def write_override(path: Path, bce: float, kl_target: float):
    cfg = {
        "model": {
            "deterministic_latent": True,
            "residual_output": True,
            "residual_scale": 0.5,
        },
        "training": {
            "epochs": 30,
            "batch_size": 8,
            "lr": 3.0e-4,
            "kl_warmup_epochs": 20,
            "save_freq": 10,
            "val_freq": 1,
        },
        "losses": {
            "l1": {"weight": 1.0},
            "lpips": {"weight": 0.0, "enabled": False},
            "bce": {"weight": float(bce)},
            "kl": {"weight": 0, "target_weight": float(kl_target), "warmup": False},
        },
        "attacks": {"online": ["simswap"], "offline": []},
        "attack_options": {
            "enforce_nontrivial_swap": True,
            "nontrivial_swap_eps": 1.0e-4,
        },
        "alternating_train": {
            "enabled": True,
            "warmup_epochs": 0,
            "warmup_prob": 1.0,
            "mid_start_epoch": 1,
            "mid_prob": 1.0,
            "late_start_epoch": 2,
            "main_prob": 1.0,
            "late_prob": 1.0,
        },
        "data": {
            "train_csv": "data_manifests/local_celeba_hq_128_tiny_train.csv",
            "val_csv": "data_manifests/local_celeba_hq_128_tiny_val.csv",
        },
        "preflight_eval": {"enabled": True, "max_batches": 4, "save_csv": True},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def run_cmd(cmd):
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    out_lines = []
    for line in proc.stdout:
        print(line, end="")
        out_lines.append(line)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")
    return "".join(out_lines)


def parse_run_dir(train_output: str) -> str:
    m = re.search(r"Run dir\s*:\s*(.+)", train_output)
    if not m:
        raise RuntimeError("Unable to parse run dir from train output.")
    return m.group(1).strip()


def read_last_val_l1(run_dir: Path):
    metrics_csv = run_dir / "metrics.csv"
    with open(metrics_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["val_l1"])


def read_eval_rows(run_dir: Path):
    eval_csv = run_dir / "attacked_recon_eval_val.csv"
    with open(eval_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    raw = next(r for r in rows if r["path"] == "raw_attacked")
    recon = next(r for r in rows if r["path"] == "vae_recon")
    return raw, recon


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = ROOT / "runs" / f"sweep_simswap_kl_bce_summary_{stamp}.csv"

    out_rows = []
    for idx, exp in enumerate(EXPERIMENTS, start=1):
        bce = exp["bce"]
        kl_target = exp["kl_target"]
        name = f"sweep_simswap_bce{slug(bce)}_kl{slug(kl_target)}"
        override_rel = f"configs/experiments/sweeps/simswap_kl_bce/{name}.yaml"
        override_abs = ROOT / override_rel
        write_override(override_abs, bce=bce, kl_target=kl_target)

        print("\n" + "=" * 88)
        print(f"[{idx}/{len(EXPERIMENTS)}] {name}")
        print("=" * 88)

        train_output = run_cmd(
            [
                "python",
                "train_stage1.py",
                "--config",
                CONFIG_BASE,
                "--override",
                override_rel,
            ]
        )
        run_dir_str = parse_run_dir(train_output)
        run_dir = Path(run_dir_str)

        run_cmd(
            [
                "python",
                "tools/eval_attacked_recon_acc.py",
                "--run-dir",
                str(run_dir),
                "--checkpoint",
                "best",
                "--split",
                "val",
                "--max-batches",
                "8",
                "--num-workers",
                "0",
            ]
        )

        raw, recon = read_eval_rows(run_dir)
        val_l1 = read_last_val_l1(run_dir)
        out_rows.append(
            {
                "run_dir": str(run_dir),
                "override": override_rel,
                "bce_weight": bce,
                "kl_target": kl_target,
                "raw_bit_acc": float(raw["bit_acc"]),
                "recon_bit_acc": float(recon["bit_acc"]),
                "delta_bit_acc": float(recon["bit_acc"]) - float(raw["bit_acc"]),
                "recon_l1": float(recon["recon_l1"]),
                "latent_kl": float(recon["latent_kl"]),
                "mu_abs_mean": float(recon["mu_abs_mean"]),
                "logvar_mean": float(recon["logvar_mean"]),
                "val_l1_clean": val_l1,
            }
        )

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "override",
                "bce_weight",
                "kl_target",
                "raw_bit_acc",
                "recon_bit_acc",
                "delta_bit_acc",
                "recon_l1",
                "latent_kl",
                "mu_abs_mean",
                "logvar_mean",
                "val_l1_clean",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print("\nSweep done.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
