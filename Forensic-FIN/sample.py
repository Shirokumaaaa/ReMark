"""
Generate sample images using FIN watermarking.

For each image in the test CSV, produces a side-by-side grid:
  cover | stego | stego_after_noise | recovered

Usage:
    python sample.py [--model MODEL_PT] [--num NUM] [--output OUTPUT_DIR]

Default model: experiments/celeba_hq_128/FED.pt  (trained on CelebA-HQ)
Fallback:      experiments/JPEG/FED.pt            (original pretrained)
"""

import os
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config as c
from models.encoder_decoder import FED
from utils.jpeg import JpegTest
from utils.metric import psnr, decoded_message_error_rate_batch


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class CSVDataset(Dataset):
    def __init__(self, csv_path, size=128):
        df = pd.read_csv(csv_path)
        self.files = df['img_path'].tolist()
        self.transform = T.Compose([
            T.Resize([size, size]),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            return self.transform(image)
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)


def denorm(tensor):
    """[-1,1] -> [0,1]"""
    return tensor / 2.0 + 0.5


def main():
    parser = argparse.ArgumentParser(description='FIN Sample Generator')
    parser.add_argument('--model', '-m', default=None, type=str,
                        help='Path to FED model checkpoint (.pt)')
    parser.add_argument('--num', '-n', default=16, type=int,
                        help='Number of sample images to generate')
    parser.add_argument('--output', '-o', default='samples', type=str,
                        help='Output directory for sample images')
    parser.add_argument('--noise-quality', '-q', default=50, type=int,
                        help='JPEG quality for noise layer (default: 50)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve model path
    if args.model is not None:
        model_path = args.model
    elif os.path.exists(os.path.join(c.MODEL_PATH, 'FED.pt')):
        model_path = os.path.join(c.MODEL_PATH, 'FED.pt')
        print(f"Using trained CelebA-HQ model: {model_path}")
    else:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'experiments', 'JPEG', 'FED.pt')
        print(f"CelebA-HQ model not found, falling back to pretrained: {model_path}")

    os.makedirs(args.output, exist_ok=True)

    # Load model
    fed = FED(c.diff, c.message_length).to(device)
    state_dicts = torch.load(model_path, map_location=device)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    fed.load_state_dict(network_state_dict)
    fed.eval()

    noise_layer = JpegTest(args.noise_quality)

    # Load test data
    dataset = CSVDataset(c.TEST_CSV, size=c.cropsize_val)
    num_samples = min(args.num, len(dataset))
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, drop_last=True)

    psnr_list, ber_list = [], []

    with torch.no_grad():
        covers = next(iter(loader)).to(device)
        messages = torch.Tensor(
            np.random.choice([-0.5, 0.5], (covers.shape[0], c.message_length))
        ).to(device)

        # Forward: embed watermark
        stegos, left_noise = fed([covers, messages])

        # Apply JPEG noise
        stegos_noised = noise_layer(stegos.clone())

        # Reverse: recover image and message
        zeros = torch.zeros(left_noise.shape).to(device)
        re_imgs, re_messages = fed([stegos_noised, zeros], rev=True)

        # Metrics
        psnr_val = psnr(covers, stegos, 255)
        ber_val = decoded_message_error_rate_batch(messages, re_messages)
        psnr_list.append(psnr_val)
        ber_list.append(ber_val)

        # Save individual comparison grids (cover | stego | stego+noise | recovered)
        for i in range(covers.shape[0]):
            row = torch.stack([
                denorm(covers[i]),
                denorm(stegos[i]),
                denorm(stegos_noised[i]),
                denorm(re_imgs[i]),
            ])
            torchvision.utils.save_image(
                row,
                os.path.join(args.output, f'sample_{i+1:03d}.png'),
                nrow=4,
                padding=2,
            )

        # Save a combined overview grid (all samples, 4 columns: cover/stego/noised/recovered)
        all_frames = []
        for i in range(covers.shape[0]):
            all_frames.extend([
                denorm(covers[i]),
                denorm(stegos[i]),
                denorm(stegos_noised[i]),
                denorm(re_imgs[i]),
            ])
        torchvision.utils.save_image(
            torch.stack(all_frames),
            os.path.join(args.output, 'overview.png'),
            nrow=4,
            padding=2,
        )

    print(f"Saved {covers.shape[0]} samples to '{args.output}/'")
    print(f"  overview.png  — full grid (cover | stego | stego+JPEG | recovered)")
    print(f"  sample_NNN.png — per-image rows")
    print(f"Avg PSNR (cover vs stego): {np.mean(psnr_list):.2f} dB")
    print(f"Avg BER  (message accuracy): {1 - np.mean(ber_list):.4f}")


if __name__ == '__main__':
    main()
