import argparse
import glob
import logging
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as c
from models.encoder_decoder import FED
from utils.datasets import INN_Dataset, transform, transform_val
from utils.jpeg import JpegSS, JpegTest
from utils.metric import decoded_message_error_rate_batch, psnr


CKPT_PATTERN = re.compile(r"fed_([0-9.]+)_([0-9]{5})\.pt$")


@dataclass
class Phase:
    epochs: int
    lr: float
    message_weight: float
    stego_weight: float


def stego_loss_fn(stego, cover, device):
    loss_fn = torch.nn.MSELoss(reduce=True)
    return loss_fn(stego, cover).to(device)


def message_loss_fn(recover_message, message, device):
    loss_fn = torch.nn.MSELoss(reduce=True)
    return loss_fn(recover_message, message).to(device)


def move_opt_state_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def parse_ckpt(path):
    m = CKPT_PATTERN.search(os.path.basename(path))
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


def pick_init_ckpt(model_dir, min_epoch):
    cands = []
    for path in glob.glob(os.path.join(model_dir, "fed_*.pt")):
        parsed = parse_ckpt(path)
        if parsed is None:
            continue
        val_psnr, epoch = parsed
        cands.append((val_psnr, epoch, path))
    if not cands:
        raise FileNotFoundError(f"No checkpoint matched fed_*.pt in {model_dir}")

    mature = [x for x in cands if x[1] >= min_epoch]
    pool = mature if mature else cands
    return max(pool, key=lambda x: (x[0], x[1]))


def build_phases(args):
    return [
        Phase(epochs=args.phase1_epochs, lr=args.phase1_lr, message_weight=args.phase1_mw, stego_weight=args.phase1_sw),
        Phase(epochs=args.phase2_epochs, lr=args.phase2_lr, message_weight=args.phase2_mw, stego_weight=args.phase2_sw),
        Phase(epochs=args.phase3_epochs, lr=args.phase3_lr, message_weight=args.phase3_mw, stego_weight=args.phase3_sw),
    ]


def phase_for_step(step, phases):
    cursor = 0
    for p in phases:
        if step < cursor + p.epochs:
            return p
        cursor += p.epochs
    return phases[-1]


def setup_logger(log_path):
    logger = logging.getLogger("auto_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Automated FIN fine-tuning from best existing checkpoint")
    parser.add_argument("--model-dir", default=c.MODEL_PATH, type=str)
    parser.add_argument("--init-ckpt", default="", type=str, help="Optional explicit checkpoint path")
    parser.add_argument("--min-epoch", default=50, type=int, help="Ignore too-early checkpoints when auto-selecting")
    parser.add_argument("--save-freq", default=5, type=int)
    parser.add_argument("--patience", default=10, type=int, help="No-improve epochs before auto weight shift")

    parser.add_argument("--phase1-epochs", default=20, type=int)
    parser.add_argument("--phase1-lr", default=1e-4, type=float)
    parser.add_argument("--phase1-mw", default=80.0, type=float)
    parser.add_argument("--phase1-sw", default=2.0, type=float)

    parser.add_argument("--phase2-epochs", default=30, type=int)
    parser.add_argument("--phase2-lr", default=5e-5, type=float)
    parser.add_argument("--phase2-mw", default=40.0, type=float)
    parser.add_argument("--phase2-sw", default=4.0, type=float)

    parser.add_argument("--phase3-epochs", default=20, type=int)
    parser.add_argument("--phase3-lr", default=2e-5, type=float)
    parser.add_argument("--phase3-mw", default=20.0, type=float)
    parser.add_argument("--phase3-sw", default=6.0, type=float)
    parser.add_argument("--train-workers", default=0, type=int)
    parser.add_argument("--val-workers", default=0, type=int)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs("logging", exist_ok=True)

    log_name = f"auto_train_{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}_{c.message_length}bit.log"
    log_name = log_name.replace(" ", "_").replace("/", "_")
    logger = setup_logger(os.path.join("logging", log_name))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.init_ckpt:
        init_ckpt = args.init_ckpt
        parsed = parse_ckpt(init_ckpt)
        if parsed is None:
            raise ValueError(f"Checkpoint name must match fed_<psnr>_<epoch>.pt: {init_ckpt}")
        base_psnr, base_epoch = parsed
    else:
        base_psnr, base_epoch, init_ckpt = pick_init_ckpt(args.model_dir, args.min_epoch)
    logger.info(f"Init checkpoint: {init_ckpt}")
    logger.info(f"Init checkpoint parsed: val_psnr={base_psnr:.6f}, epoch={base_epoch}")

    phases = build_phases(args)
    total_steps = sum(p.epochs for p in phases)
    logger.info(
        "Schedule: "
        + " | ".join(
            [
                f"P{i+1}(epochs={p.epochs}, lr={p.lr}, mw={p.message_weight}, sw={p.stego_weight})"
                for i, p in enumerate(phases)
            ]
        )
    )

    fed = FED(c.diff, c.message_length).to(device)
    params_trainable = list(filter(lambda p: p.requires_grad, fed.parameters()))
    optim = torch.optim.Adam(params_trainable, lr=phases[0].lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

    ckpt = torch.load(init_ckpt, map_location=device)
    net_state = {k: v for k, v in ckpt["net"].items() if "tmp_var" not in k}
    fed.load_state_dict(net_state)
    if "opt" in ckpt:
        try:
            optim.load_state_dict(ckpt["opt"])
            move_opt_state_to_device(optim, device)
            logger.info("Optimizer state restored from checkpoint")
        except Exception as exc:
            logger.warning(f"Failed to restore optimizer state, continuing with fresh optimizer: {exc}")

    noise_layer = JpegSS(50)
    test_noise_layer = JpegTest(50)

    trainloader = DataLoader(
        INN_Dataset(transforms=transform, mode="train"),
        batch_size=c.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.train_workers,
        drop_last=True,
    )
    testloader = DataLoader(
        INN_Dataset(transforms=transform_val, mode="val"),
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=args.val_workers,
        drop_last=True,
    )

    best_val_psnr = float("-inf")
    no_improve = 0
    start_epoch = base_epoch + 1

    for step in range(total_steps):
        epoch = start_epoch + step
        phase = phase_for_step(step, phases)
        lr = phase.lr
        mw = phase.message_weight
        sw = phase.stego_weight

        for g in optim.param_groups:
            g["lr"] = lr

        fed.train()
        loss_history, stego_loss_history, message_loss_history = [], [], []
        stego_psnr_history, acc_history = [], []

        for cover_img in trainloader:
            cover_img = cover_img.to(device)
            message = torch.Tensor(np.random.choice([-0.5, 0.5], (cover_img.shape[0], c.message_length))).to(device)

            stego_img, left_noise = fed([cover_img, message])
            stego_noise_img = noise_layer(stego_img.clone())

            gauss_noise = torch.zeros(left_noise.shape, device=device)
            _, re_message = fed([stego_noise_img, gauss_noise], rev=True)

            stego_loss = stego_loss_fn(stego_img, cover_img, device)
            message_loss = message_loss_fn(re_message, message, device)
            total_loss = mw * message_loss + sw * stego_loss

            total_loss.backward()
            optim.step()
            optim.zero_grad()

            stego_psnr_history.append(psnr(cover_img, stego_img, 255))
            acc_history.append(1 - decoded_message_error_rate_batch(message, re_message))
            loss_history.append(total_loss.item())
            stego_loss_history.append(stego_loss.item())
            message_loss_history.append(message_loss.item())

        train_loss = float(np.mean(loss_history))
        train_stego_loss = float(np.mean(stego_loss_history))
        train_message_loss = float(np.mean(message_loss_history))
        train_psnr = float(np.mean(stego_psnr_history))
        train_acc = float(np.mean(acc_history))

        fed.eval()
        val_psnr_hist, val_acc_hist = [], []
        with torch.no_grad():
            for test_cover_img in testloader:
                test_cover_img = test_cover_img.to(device)
                test_message = torch.Tensor(np.random.choice([-0.5, 0.5], (test_cover_img.shape[0], c.message_length))).to(device)
                test_stego_img, test_left_noise = fed([test_cover_img, test_message])
                test_stego_noise_img = test_noise_layer(test_stego_img.clone())
                test_zeros = torch.zeros(test_left_noise.shape, device=device)
                _, test_re_message = fed([test_stego_noise_img, test_zeros], rev=True)
                val_psnr_hist.append(psnr(test_cover_img, test_stego_img, 255))
                val_acc_hist.append(1 - decoded_message_error_rate_batch(test_message, test_re_message))

        val_psnr = float(np.mean(val_psnr_hist))
        val_acc = float(np.mean(val_acc_hist))

        logger.info(
            f"Epoch {epoch} | lr={lr:.2e} mw={mw:.2f} sw={sw:.2f} | "
            f"train_loss={train_loss:.4f} stego_loss={train_stego_loss:.4f} msg_loss={train_message_loss:.4f} "
            f"train_psnr={train_psnr:.4f} train_acc={train_acc:.4f} | "
            f"val_psnr={val_psnr:.4f} val_acc={val_acc:.4f}"
        )

        ckpt_name = f"fed_{val_psnr:.6f}_{epoch:05d}.pt"
        if ((step + 1) % args.save_freq) == 0:
            torch.save({"opt": optim.state_dict(), "net": fed.state_dict()}, os.path.join(args.model_dir, ckpt_name))

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            no_improve = 0
            torch.save({"opt": optim.state_dict(), "net": fed.state_dict()}, os.path.join(args.model_dir, "fed_best_auto.pt"))
            logger.info(f"New best val_psnr={best_val_psnr:.4f}, saved fed_best_auto.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                # Shift toward visual quality when plateaued
                phase.message_weight = max(phase.message_weight * 0.8, 5.0)
                phase.stego_weight = min(phase.stego_weight * 1.2, 12.0)
                logger.info(
                    f"Plateau detected ({no_improve} epochs). "
                    f"Adjusted current phase weights -> mw={phase.message_weight:.2f}, sw={phase.stego_weight:.2f}"
                )
                no_improve = 0

    final_path = os.path.join(args.model_dir, "fed_auto_final.pt")
    torch.save({"opt": optim.state_dict(), "net": fed.state_dict()}, final_path)
    logger.info(f"Training finished. Final checkpoint: {final_path}")
    logger.info(f"Best val_psnr observed: {best_val_psnr:.4f}")


if __name__ == "__main__":
    main()
