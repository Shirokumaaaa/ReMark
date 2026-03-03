#!/usr/bin/env python3
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import JsonConfig, ImageDataset
from model.encoder_decoder import Encoder
from model.deepfake_manipulations import SimSwapModel, StarGanModel, StyleMaskModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./configurations/pretrain_small.json")
    p.add_argument("--out_dir", default="./results/deepfake_samples")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def to_vis_auto(x):
    # If already in [0,1], keep. If in [-1,1], map to [0,1].
    if x.min().item() >= 0.0 and x.max().item() <= 1.0:
        return x.clamp(0, 1)
    if x.min().item() >= -1.0 and x.max().item() <= 1.0:
        return (x * 0.5 + 0.5).clamp(0, 1)
    return x.clamp(0, 1)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    configs = JsonConfig()
    configs.load_json_file(args.config)

    dataset = ImageDataset(
        path_img=f"{configs.img_path}/test",
        path_wm=f"{configs.wm_path}/{configs.img_size}/test",
        img_size=configs.img_size,
        wm_len=configs.watermark_length,
        mode="test",  # avoid random crop to keep face alignment
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)

    imgs, wms = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    imgs = imgs.to(device)
    wms = wms.to(device)

    # Load encoder to get watermarked images.
    encoder = Encoder(configs.img_size, configs.encoder_channels, configs.encoder_blocks, configs.watermark_length).to(device)
    encoder_path = os.path.join(configs.weight_path, "encoder_decoder.pt")
    state = torch.load(encoder_path, map_location="cpu")
    enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state)
    encoder.eval()

    with torch.no_grad():
        imgs_wm = encoder(imgs, wms)

    def save_triplet(name, fake):
        if fake.shape[-1] != imgs.shape[-1]:
            fake = torch.nn.functional.interpolate(fake, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
        orig = to_vis_auto(imgs.detach().cpu())
        wm = to_vis_auto(imgs_wm.detach().cpu())
        fake = to_vis_auto(fake.detach().cpu())
        # per-sample rows: [orig | wm | fake]
        stacked = torch.stack([orig, wm, fake], dim=1)
        grid = make_grid(stacked.view(-1, 3, orig.shape[2], orig.shape[3]), nrow=3)
        out_path = os.path.join(args.out_dir, f"{name}.png")
        save_image(grid, out_path)
        print("saved", out_path)

    # Base conversions
    imgs_01 = (imgs * 0.5 + 0.5).clamp(0, 1)
    imgs_wm_01 = (imgs_wm * 0.5 + 0.5).clamp(0, 1)

    mean05 = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std05 = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    # SimSwap
    try:
        simswap = SimSwapModel(configs.img_size, mode="test").to(device)
        simswap.eval()
        # SimSwap expects [0,1] images (it applies its own ImageNet-style normalization internally)
        fake = simswap([imgs_wm_01, imgs_01, device])
        save_triplet("simswap", fake)
    except Exception as e:
        print("SimSwap failed:", e)

    # StyleMask
    try:
        stylemask = StyleMaskModel(configs.img_size, device, mode="test").to(device)
        stylemask.eval()
        # StyleMask uses inputs normalized to [-1,1]
        imgs_norm = (imgs_01 - mean05) / std05
        imgs_wm_norm = (imgs_wm_01 - mean05) / std05
        # StyleMask encoder expects 256x256 inputs; upsample when using 128.
        if imgs_norm.shape[-1] != 256:
            imgs_norm = torch.nn.functional.interpolate(imgs_norm, size=(256, 256), mode="bilinear", align_corners=False)
            imgs_wm_norm = torch.nn.functional.interpolate(imgs_wm_norm, size=(256, 256), mode="bilinear", align_corners=False)
        fake = stylemask([imgs_wm_norm, imgs_norm, device])
        save_triplet("stylemask", fake)
    except Exception as e:
        print("StyleMask failed:", e)

    # StarGAN
    try:
        stargan = StarGanModel(configs.img_size, mode="test").to(device)
        stargan.eval()
        # StarGAN v2 expects inputs in [-1,1]
        fake = stargan([imgs_wm, imgs, device])
        save_triplet("stargan", fake)
    except Exception as e:
        print("StarGAN failed:", e)


if __name__ == "__main__":
    main()
