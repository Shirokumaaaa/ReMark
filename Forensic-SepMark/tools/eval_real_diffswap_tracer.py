import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from PIL import Image


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_attack_batch(sample: dict, device: torch.device) -> dict:
    out = {}
    for k, v in sample.items():
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
def run_real_diffswap(model, ddim_sampler, batch, ddim_steps: int, ddim_eta: float = 0.0):
    z, c, x, _, _ = model.get_input(
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
    out = torch.clamp(x_samples, -1.0, 1.0)
    return out


def tensor_to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    arr = ((x.detach().cpu().clamp(-1, 1).permute(1, 2, 0) + 1.0) * 127.5).numpy()
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_sample_grid(cover, encoded, attacked, save_path: Path) -> None:
    c = tensor_to_uint8_hwc(cover[0])
    e = tensor_to_uint8_hwc(encoded[0])
    a = tensor_to_uint8_hwc(attacked[0])
    r = np.clip(np.abs(e.astype(np.int16) - a.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    canvas = Image.new("RGB", (256 * 4, 256 + 24), (245, 245, 245))
    canvas.paste(Image.fromarray(c), (0, 24))
    canvas.paste(Image.fromarray(e), (256, 24))
    canvas.paste(Image.fromarray(a), (512, 24))
    canvas.paste(Image.fromarray(r), (768, 24))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def decoded_message_error_rate(message: torch.Tensor, decoded_message: torch.Tensor) -> float:
    length = message.shape[0]
    message = message.gt(0)
    decoded_message = decoded_message.gt(0)
    return float(torch.sum(message != decoded_message).item()) / float(length)


def decoded_message_error_rate_batch(messages: torch.Tensor, decoded_messages: torch.Tensor) -> float:
    batch_size = int(messages.shape[0])
    total = 0.0
    for i in range(batch_size):
        total += decoded_message_error_rate(messages[i], decoded_messages[i])
    return total / float(batch_size)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-root", default="../Attack-DiffSwap")
    parser.add_argument("--dataset-path", default="../Dataset-CelebA_HQ")
    parser.add_argument(
        "--sepmark-result-folder",
        default="results/baseline/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/",
    )
    parser.add_argument("--model-epoch", type=int, default=90)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--tgt-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--message-range", type=float, default=0.1)
    parser.add_argument("--save-sample", default="1", choices=["0", "1"])
    args = parser.parse_args()
    attack_root_arg = Path(args.attack_root)
    dataset_path_arg = Path(args.dataset_path)
    if not attack_root_arg.is_absolute():
        attack_root_arg = (repo_root / attack_root_arg).resolve()
    if not dataset_path_arg.is_absolute():
        dataset_path_arg = (repo_root / dataset_path_arg).resolve()
    args.attack_root = str(attack_root_arg)
    args.dataset_path = str(dataset_path_arg)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(str(repo_root))

    with open(repo_root / "cfg" / "test_DualMark.yaml", "r", encoding="utf-8") as f:
        test_cfg = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    result_folder = repo_root / args.sepmark_result_folder
    with open(result_folder / "train_DualMark.yaml", "r", encoding="utf-8") as f:
        train_cfg = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    # SepMark model setup (encoder/decoder only, no training-only dependencies).
    from network.DW_EncoderDecoder import DW_EncoderDecoder
    from utils import maskImgDataset

    encoder_decoder = DW_EncoderDecoder(
        message_length=train_cfg.message_length,
        noise_layers_R=[],
        noise_layers_F=[],
        attention_encoder=train_cfg.attention_encoder,
        attention_decoder=train_cfg.attention_decoder,
    ).to(device).eval()
    ec_path = result_folder / "models" / f"EC_{args.model_epoch}.pth"
    encoder_decoder.load_state_dict(torch.load(str(ec_path), map_location=device), strict=False)

    ds = maskImgDataset(
        os.path.join(args.dataset_path, "test"),
        int(train_cfg.image_size),
        csv_path=os.path.join(args.dataset_path, "test.csv"),
    )
    total = min(args.num_samples, len(ds))

    # Real DiffSwap setup.
    attack_root = Path(args.attack_root).resolve()
    sys.path.insert(0, str(attack_root))
    os.chdir(str(attack_root))
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.data.portrait import Portrait

    cfg = OmegaConf.load(str(attack_root / "configs/diffswap/default-project.yaml"))
    diffswap_model = instantiate_from_config(cfg.model)
    diffswap_model.init_from_ckpt(str(attack_root / "checkpoints/diffswap.pth"))
    diffswap_model = diffswap_model.to(device).eval()
    diffswap_model.cond_stage_model.affine_crop = True
    diffswap_model.cond_stage_model.swap = True
    sampler = DDIMSampler(diffswap_model, target_preserve_scale=args.tgt_scale)
    portrait = Portrait(str(attack_root / "data" / "portrait"))
    src_n, tgt_n = len(portrait.src_list), len(portrait.tgt_list)

    timestamp = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    out_dir = repo_root / "results" / "eval" / f"real_diffswap_tracer_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "real_diffswap_eval.csv"

    rows = []
    tracer_bers, detector_bers = [], []
    t0 = time.time()

    with torch.no_grad():
        for i in range(total):
            image, mask = ds[i]
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            msg = torch.tensor(
                np.random.choice(
                    [-args.message_range, args.message_range],
                    (1, int(train_cfg.message_length)),
                ),
                dtype=torch.float32,
                device=device,
            )

            encoded = encoder_decoder.encoder(image, msg)
            encoded = image + (encoded - image)

            src_idx = random.randrange(src_n)
            tgt_idx = random.randrange(tgt_n)
            sample = portrait[src_idx * tgt_n + tgt_idx]
            sample["image"] = encoded[0].detach().permute(1, 2, 0).cpu().numpy().astype(np.float32)
            attack_batch = to_attack_batch(sample, device)
            attacked = run_real_diffswap(
                diffswap_model, sampler, attack_batch, ddim_steps=args.ddim_steps
            )

            decoded_c = encoder_decoder.decoder_C(attacked)
            decoded_rf = encoder_decoder.decoder_RF(attacked)
            tracer_ber = float(decoded_message_error_rate_batch(msg, decoded_c))
            detector_ber = float(decoded_message_error_rate_batch(msg, decoded_rf))
            tracer_bers.append(tracer_ber)
            detector_bers.append(detector_ber)

            row = {
                "idx": i,
                "src_meta": sample["src"],
                "tgt_meta": sample["target"],
                "tracer_ber": tracer_ber,
                "tracer_acc": 1.0 - tracer_ber,
                "detector_ber": detector_ber,
                "detector_acc": 1.0 - detector_ber,
            }
            rows.append(row)
            print(
                f"[{i+1}/{total}] tracer_ber={tracer_ber:.6f} "
                f"detector_ber={detector_ber:.6f} src={sample['src']} tgt={sample['target']}"
            )

            if i == 0 and args.save_sample == "1":
                make_sample_grid(
                    cover=image,
                    encoded=encoded,
                    attacked=attacked,
                    save_path=out_dir / "sample_real_diffswap.png",
                )

    tracer_ber_mean = float(np.mean(tracer_bers)) if tracer_bers else float("nan")
    detector_ber_mean = float(np.mean(detector_bers)) if detector_bers else float("nan")
    tracer_acc_mean = 1.0 - tracer_ber_mean
    detector_acc_mean = 1.0 - detector_ber_mean
    elapsed = time.time() - t0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "idx", "src_meta", "tgt_meta",
            "tracer_ber", "tracer_acc",
            "detector_ber", "detector_acc",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow(
            {
                "idx": "AVG",
                "src_meta": "",
                "tgt_meta": "",
                "tracer_ber": tracer_ber_mean,
                "tracer_acc": tracer_acc_mean,
                "detector_ber": detector_ber_mean,
                "detector_acc": detector_acc_mean,
            }
        )

    print("\n=== Real DiffSwap Small Eval ===")
    print(f"num_samples={total}")
    print(f"ddim_steps={args.ddim_steps}")
    print(f"tracer_ber={tracer_ber_mean:.6f} tracer_acc={tracer_acc_mean:.6f}")
    print(f"detector_ber={detector_ber_mean:.6f} detector_acc={detector_acc_mean:.6f}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"csv={csv_path}")
    if args.save_sample == "1":
        print(f"sample={out_dir / 'sample_real_diffswap.png'}")


if __name__ == "__main__":
    main()
