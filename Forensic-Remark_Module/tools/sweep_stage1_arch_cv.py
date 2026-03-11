#!/usr/bin/env python3
"""
Stage1 architecture sweep with cross-attack validation.

Phase A: search candidate architectures on SimSwap.
Phase B: validate top-K architectures on StarGAN + Arc2Face.
"""

import csv
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_BASE = "configs/stage1_vae.yaml"
OVERRIDE_DIR = ROOT / "configs" / "experiments" / "sweeps" / "arch_stage1_cv"

SEARCH_ATTACK = "simswap"
CV_ATTACKS = ["stargan", "Arc2Face"]
TOP_K = 3


ARCH_CANDIDATES = [
    {
        "name": "baseline",
        "model": {"base_channels": 32, "latent_channels": 64, "downsample_factor": 8, "n_res": 2,
                  "deterministic_latent": True, "residual_output": True, "residual_scale": 0.5},
        "batch_size": 8,
    },
    {
        "name": "deep_res3",
        "model": {"base_channels": 32, "latent_channels": 64, "downsample_factor": 8, "n_res": 3,
                  "deterministic_latent": True, "residual_output": True, "residual_scale": 0.5},
        "batch_size": 6,
    },
    {
        "name": "wide_48_96",
        "model": {"base_channels": 48, "latent_channels": 96, "downsample_factor": 8, "n_res": 2,
                  "deterministic_latent": True, "residual_output": True, "residual_scale": 0.5},
        "batch_size": 4,
    },
    {
        "name": "detail_ds4",
        "model": {"base_channels": 32, "latent_channels": 64, "downsample_factor": 4, "n_res": 2,
                  "deterministic_latent": True, "residual_output": True, "residual_scale": 0.5},
        "batch_size": 4,
    },
    {
        "name": "compact_24_48",
        "model": {"base_channels": 24, "latent_channels": 48, "downsample_factor": 8, "n_res": 2,
                  "deterministic_latent": True, "residual_output": True, "residual_scale": 0.5},
        "batch_size": 8,
    },
    {
        "name": "stochastic_latent",
        "model": {"base_channels": 32, "latent_channels": 64, "downsample_factor": 8, "n_res": 2,
                  "deterministic_latent": False, "residual_output": True, "residual_scale": 0.5},
        "batch_size": 8,
    },
]


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
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")
    return "".join(lines)


def parse_run_dir(train_output: str) -> Path:
    m = re.search(r"Run dir\s*:\s*(.+)", train_output)
    if not m:
        raise RuntimeError("Unable to parse run dir.")
    return Path(m.group(1).strip())


def load_metrics_last_val_l1(run_dir: Path):
    with open(run_dir / "metrics.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["val_l1"]) if rows else None


def load_eval(run_dir: Path):
    with open(run_dir / "attacked_recon_eval_val.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    raw = next(r for r in rows if r["path"] == "raw_attacked")
    recon = next(r for r in rows if r["path"] == "vae_recon")
    return raw, recon


def data_for_attack(attack_name: str):
    if attack_name.lower() == "arc2face":
        # Arc2Face replay key depends on original manifest paths.
        return {
            "train_csv": "data_manifests/celeba_hq_128_tiny_val.csv",
            "val_csv": "data_manifests/celeba_hq_128_tiny_val.csv",
            "batch_cap": 4,
        }
    return {
        "train_csv": "data_manifests/local_celeba_hq_128_tiny_train.csv",
        "val_csv": "data_manifests/local_celeba_hq_128_tiny_val.csv",
        "batch_cap": 8,
    }


def build_override(arch: dict, attack_name: str, out_path: Path):
    data_cfg = data_for_attack(attack_name)
    batch_size = min(int(arch["batch_size"]), int(data_cfg["batch_cap"]))
    cfg = {
        "model": arch["model"],
        "training": {
            "epochs": 20,
            "batch_size": batch_size,
            "lr": 3.0e-4,
            "kl_warmup_epochs": 20,
            "save_freq": 10,
            "val_freq": 1,
        },
        "losses": {
            "l1": {"weight": 1.0},
            "lpips": {"weight": 0.0, "enabled": False},
            "bce": {"weight": 1.0},
            "kl": {"weight": 0, "target_weight": 0.001, "warmup": False},
        },
        "attacks": {"online": [attack_name], "offline": []},
        "attack_options": {
            "arc2face_results_jsonl": "/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper/outputs/generation_results.jsonl",
            "arc2face_outputs_base": "/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper",
            "arc2face_replay_key": "source_image",
            "arc2face_allow_missing": False,
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
        "data": {"train_csv": data_cfg["train_csv"], "val_csv": data_cfg["val_csv"]},
        "preflight_eval": {"enabled": True, "max_batches": 4, "save_csv": True},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def run_one(arch: dict, attack_name: str, phase: str):
    ov_name = f"{phase}_{arch['name']}_{attack_name}.yaml".replace("/", "_")
    ov_rel = f"configs/experiments/sweeps/arch_stage1_cv/{ov_name}"
    ov_abs = ROOT / ov_rel
    build_override(arch, attack_name, ov_abs)

    print("\n" + "=" * 96)
    print(f"[{phase}] arch={arch['name']} attack={attack_name}")
    print("=" * 96)

    train_out = run_cmd(["python", "train_stage1.py", "--config", CONFIG_BASE, "--override", ov_rel])
    run_dir = parse_run_dir(train_out)

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

    raw, recon = load_eval(run_dir)
    out = {
        "phase": phase,
        "arch": arch["name"],
        "attack": attack_name,
        "run_dir": str(run_dir),
        "override": ov_rel,
        "raw_bit_acc": float(raw["bit_acc"]),
        "recon_bit_acc": float(recon["bit_acc"]),
        "delta_bit_acc": float(recon["bit_acc"]) - float(raw["bit_acc"]),
        "recon_l1": float(recon["recon_l1"]),
        "latent_kl": float(recon["latent_kl"]),
        "mu_abs_mean": float(recon["mu_abs_mean"]),
        "logvar_mean": float(recon["logvar_mean"]),
        "val_l1_clean": load_metrics_last_val_l1(run_dir),
    }
    return out


def arch_rank_key(r):
    # prioritize attacked-path acc gain, then reconstruction quality.
    return (r["delta_bit_acc"], -r["recon_l1"])


def write_csv(path: Path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = ROOT / "runs" / f"arch_cv_summary_{stamp}.csv"
    leaderboard_csv = ROOT / "runs" / f"arch_cv_leaderboard_{stamp}.csv"

    all_rows = []

    # Phase A: SimSwap search.
    search_rows = []
    for arch in ARCH_CANDIDATES:
        row = run_one(arch, SEARCH_ATTACK, phase="search")
        all_rows.append(row)
        search_rows.append(row)

    top = sorted(search_rows, key=arch_rank_key, reverse=True)[:TOP_K]
    top_names = [r["arch"] for r in top]
    print(f"\nTop-{TOP_K} on {SEARCH_ATTACK}: {top_names}")

    # Phase B: cross-attack validation for top-k.
    top_arch_map = {a["name"]: a for a in ARCH_CANDIDATES if a["name"] in top_names}
    for arch_name in top_names:
        for atk in CV_ATTACKS:
            row = run_one(top_arch_map[arch_name], atk, phase="cv")
            all_rows.append(row)

    write_csv(
        summary_csv,
        all_rows,
        [
            "phase",
            "arch",
            "attack",
            "run_dir",
            "override",
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

    # Build a small cross-attack leaderboard on CV rows only.
    cv_rows = [r for r in all_rows if r["phase"] == "cv"]
    arch_scores = {}
    for r in cv_rows:
        name = r["arch"]
        if name not in arch_scores:
            arch_scores[name] = {"arch": name, "n": 0, "avg_delta": 0.0, "avg_recon_l1": 0.0, "avg_latent_kl": 0.0}
        arch_scores[name]["n"] += 1
        arch_scores[name]["avg_delta"] += r["delta_bit_acc"]
        arch_scores[name]["avg_recon_l1"] += r["recon_l1"]
        arch_scores[name]["avg_latent_kl"] += r["latent_kl"]

    leaderboard = []
    for v in arch_scores.values():
        n = max(v["n"], 1)
        v["avg_delta"] /= n
        v["avg_recon_l1"] /= n
        v["avg_latent_kl"] /= n
        v["score"] = v["avg_delta"] - 0.3 * v["avg_recon_l1"]
        leaderboard.append(v)
    leaderboard.sort(key=lambda x: x["score"], reverse=True)

    write_csv(
        leaderboard_csv,
        leaderboard,
        ["arch", "n", "avg_delta", "avg_recon_l1", "avg_latent_kl", "score"],
    )

    print("\nArchitecture CV sweep complete.")
    print(f"Summary: {summary_csv}")
    print(f"Leaderboard: {leaderboard_csv}")


if __name__ == "__main__":
    main()
