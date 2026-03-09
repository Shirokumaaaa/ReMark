import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf


def _to_tensor_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            if t.dtype == torch.bool:
                t = t.float()
            out[k] = t.unsqueeze(0).to(device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0).to(device)
        else:
            out[k] = [v]
    return out


@torch.no_grad()
def run_swap(model, ddim_sampler, batch, ddim_steps=200, ddim_eta=0.0):
    z, c, x, xrec, xc = model.get_input(
        batch,
        model.first_stage_key,
        return_first_stage_outputs=True,
        force_c_encode=True,
        return_original_cond=True,
        swap=True,
    )
    n = x.size(0)
    h, w = z.shape[2], z.shape[3]
    mask = (1 - batch["mask"].float())[:, None]
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
    mask[mask > 0] = 1
    mask[mask <= 0] = 0

    shape = (model.channels, model.image_size, model.image_size)
    with model.ema_scope("Plotting Inpaint"):
        samples, _ = ddim_sampler.sample(
            ddim_steps, n, shape, c, eta=ddim_eta, x0=z[:n], mask=mask, verbose=False
        )
    x_samples = model.decode_first_stage(samples.to(model.device))
    gen = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    gen = (gen[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded", required=True, help="Path to encoded target image.")
    parser.add_argument("--output", required=True, help="Path to save DiffSwap output image.")
    parser.add_argument(
        "--attack-root",
        default="../Attack-DiffSwap",
        help="Path to Attack-DiffSwap repo root.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/diffswap.pth",
        help="Checkpoint path relative to attack-root or absolute.",
    )
    parser.add_argument(
        "--config",
        default="configs/diffswap/default-project.yaml",
        help="Config path relative to attack-root or absolute.",
    )
    parser.add_argument("--source-name", default="", help="Optional source filename in portrait/source.")
    parser.add_argument("--target-name", default="", help="Optional target filename in portrait/target for metadata.")
    parser.add_argument("--tgt-scale", type=float, default=0.01, help="Target preserve scale for DDIM sampler.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    attack_root = Path(args.attack_root)
    if not attack_root.is_absolute():
        attack_root = (repo_root / attack_root).resolve()
    else:
        attack_root = attack_root.resolve()
    sys.path.insert(0, str(attack_root))
    os.chdir(str(attack_root))

    from ldm.util import instantiate_from_config  # noqa: E402
    from ldm.models.diffusion.ddim import DDIMSampler  # noqa: E402
    from ldm.data.portrait import Portrait  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = attack_root / config_path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = attack_root / ckpt_path

    config = OmegaConf.load(str(config_path))
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(str(ckpt_path))
    model = model.to(device).eval()
    model.cond_stage_model.affine_crop = True
    model.cond_stage_model.swap = True

    portrait_root = attack_root / "data" / "portrait"
    dataset = Portrait(str(portrait_root))
    src_list = dataset.src_list
    tgt_list = dataset.tgt_list

    if args.source_name and args.source_name in src_list:
        src_idx = src_list.index(args.source_name)
    else:
        src_idx = random.randrange(len(src_list))
    if args.target_name and args.target_name in tgt_list:
        tgt_idx = tgt_list.index(args.target_name)
    else:
        tgt_idx = random.randrange(len(tgt_list))

    idx = src_idx * len(tgt_list) + tgt_idx
    sample = dataset[idx]

    # Replace target image with provided encoded image (real DiffSwap inference path).
    enc = Image.open(args.encoded).convert("RGB").resize((dataset.size, dataset.size), Image.BICUBIC)
    enc = (np.array(enc).astype(np.float32) / 127.5 - 1.0).astype(np.float32)
    sample["image"] = enc

    batch = _to_tensor_batch(sample, device)
    sampler = DDIMSampler(model, target_preserve_scale=args.tgt_scale)
    out = run_swap(model, sampler, batch)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)

    print(f"source={sample['src']}")
    print(f"target_meta={sample['target']}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
