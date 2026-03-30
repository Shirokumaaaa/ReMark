import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.registry import build_attack
from utils.config import load_config
from utils.message_bits import deterministic_messages_from_paths
from wm_adapters.registry import build_wm_adapter


def load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img)
    tensor = tensor * 2.0 - 1.0
    return tensor


def save_triplet_grid(items, out_path: str):
    if not items:
        return
    cell = 256
    canvas = Image.new("RGB", (cell * 3, len(items) * (cell + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, item in enumerate(items):
        y = idx * (cell + 24)
        for col, key in enumerate(("cover", "wm", "fake")):
            img = item[key].resize((cell, cell), Image.BICUBIC)
            canvas.paste(img, (col * cell, y))
        draw.text((4, y + cell + 2), item["name"], fill="black")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_vae.yaml")
    parser.add_argument("--override", default="configs/experiments/preflight_sepmark_diffswap_online_bs1_mb20.yaml")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-dir", default="runs/diffswap_stability_eval")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wm_adapter = build_wm_adapter(cfg.wm_model, cfg)
    attack = build_attack("diffswap", cfg)

    rows = []
    with open(cfg.data.val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows = rows[: args.num_samples]
    total_samples = len(rows)
    total_success = 0
    total_bits = 0
    correct_bits = 0
    failures = []
    triplets = []

    for idx, row in enumerate(rows):
        img_path = row["img_path"]
        name = os.path.basename(img_path)
        try:
            cover = load_image_tensor(img_path).unsqueeze(0).to(device)
            messages = deterministic_messages_from_paths(
                [img_path],
                message_len=wm_adapter.message_length,
                device=device,
                salt=getattr(cfg.training, "message_seed_salt", "remark_v1"),
            )
            wm_image = wm_adapter.encode(cover, messages)
            fake = attack.attack_with_cover(wm_image, cover, batch={"img_path": [img_path]})
            logits = wm_adapter.decode(fake)
            bits = (logits > 0).float()

            correct = (bits == messages).sum().item()
            total = messages.numel()
            correct_bits += int(correct)
            total_bits += int(total)
            total_success += 1

            if len(triplets) < 8:
                to_pil = transforms.ToPILImage()
                triplets.append({
                    "name": f"{idx:02d} {name}",
                    "cover": to_pil(((cover[0].detach().cpu().clamp(-1, 1) + 1.0) * 0.5)),
                    "wm": to_pil(((wm_image[0].detach().cpu().clamp(-1, 1) + 1.0) * 0.5)),
                    "fake": to_pil(((fake[0].detach().cpu().clamp(-1, 1) + 1.0) * 0.5)),
                })

            sample_acc = float(correct) / float(total)
            print(f"[{idx + 1:02d}/{total_samples:02d}] ok  {name}  sample_acc={sample_acc:.4f}", flush=True)
        except Exception as exc:
            failures.append((img_path, str(exc)))
            print(f"[{idx + 1:02d}/{total_samples:02d}] fail {name}  {exc}", flush=True)

    success_acc = (correct_bits / total_bits) if total_bits > 0 else float("nan")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_triplet_grid(triplets, str(save_dir / "triplets.png"))

    with open(save_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"requested={total_samples}\n")
        f.write(f"success={total_success}\n")
        f.write(f"failed={len(failures)}\n")
        f.write(f"success_bit_acc={success_acc:.6f}\n")
        for path, err in failures:
            f.write(f"FAIL\t{path}\t{err}\n")

    print("================================================================", flush=True)
    print(f"requested={total_samples}", flush=True)
    print(f"success={total_success}", flush=True)
    print(f"failed={len(failures)}", flush=True)
    print(f"success_bit_acc={success_acc:.6f}", flush=True)
    print(f"triplets={save_dir / 'triplets.png'}", flush=True)
    print(f"summary={save_dir / 'summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
