import csv
import os
import random
import string
import time
from shutil import copyfile

import kornia
import lpips
import numpy as np
import torch
import yaml
from PIL import Image
from easydict import EasyDict
from torch.utils.data import DataLoader
from torchvision import transforms

from network.Dual_Mark import Network
from network.noise_layers import *
from utils import attrsImgDataset, maskImgDataset, save_images, get_random_images, concatenate_images


criterion_LPIPS = lpips.LPIPS().to("cuda" if torch.cuda.is_available() else "cpu")


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_path(path="temp/"):
    return path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".png"


def sanitize_noise_name(noise_layer):
    return ''.join(ch if ch.isalnum() else '_' for ch in noise_layer).strip('_')


def is_stargan_noise(noise_layer):
    return noise_layer.startswith("StarGAN")


def build_test_dataloader(noise_layer, dataset_path, image_size, batch_size):
    test_dir = os.path.join(dataset_path, "test")
    test_csv = os.path.join(dataset_path, "test.csv")
    if is_stargan_noise(noise_layer):
        test_dataset = attrsImgDataset(test_dir, image_size, "celebahq", csv_path=test_csv)
    else:
        test_dataset = maskImgDataset(test_dir, image_size, csv_path=test_csv)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def maybe_quantize_tensor_batch(x, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    y = x.clone()
    for index in range(y.shape[0]):
        single_image = ((y[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(single_image)
        file = get_path()
        while os.path.exists(file):
            file = get_path()
        im.save(file)
        read = np.array(Image.open(file), dtype=np.uint8)
        os.remove(file)
        y[index] = transform(read).unsqueeze(0).to(device)
    return y


def evaluate_noise_layer(network, noise_layer, dataloader, message_length, message_range, result_folder, writer, max_steps, save_images_number):
    noise_name = sanitize_noise_name(noise_layer)
    noise_module = eval(noise_layer)

    test_result = {
        "tracer_decoder_ber": 0.0,
        "detector_decoder_ber": 0.0,
        "detector_zero_ber": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0
    }

    total_steps = len(dataloader) if max_steps <= 0 else min(len(dataloader), max_steps)
    save_count = min(save_images_number, total_steps)
    saved_iterations = np.random.choice(np.arange(1, total_steps + 1), size=save_count, replace=False)
    saved_all = None

    test_log = os.path.join(result_folder, f"test_log_{noise_name}_{time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())}.txt")

    print(f"\nStart Testing: {noise_layer}\n")

    for step, (image, mask) in enumerate(dataloader, 1):
        image = image.to(network.device)
        message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(network.device)

        network.encoder_decoder.eval()
        network.discriminator.eval()

        with torch.no_grad():
            images, messages, masks = image.to(network.device), message.to(network.device), mask.to(network.device)

            encoded_images = network.encoder_decoder.module.encoder(images, messages)
            encoded_images = images + (encoded_images - images)
            encoded_images = maybe_quantize_tensor_batch(encoded_images, image.device)

            psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=11, reduction="mean").item()
            lpips_val = torch.mean(criterion_LPIPS(encoded_images.detach(), images)).item()

            noised_images = noise_module([encoded_images.clone(), images, masks])
            noised_images = maybe_quantize_tensor_batch(noised_images, image.device)

            decoded_messages_C = network.encoder_decoder.module.decoder_C(noised_images)
            decoded_messages_RF = network.encoder_decoder.module.decoder_RF(noised_images)

        tracer_decoder_ber = network.decoded_message_error_rate_batch(messages, decoded_messages_C)
        detector_decoder_ber = network.decoded_message_error_rate_batch(messages, decoded_messages_RF)
        detector_zero_ber = network.decoded_message_error_rate_batch(torch.zeros_like(messages), decoded_messages_RF)

        result = {
            "tracer_decoder_ber": tracer_decoder_ber,
            "detector_decoder_ber": detector_decoder_ber,
            "detector_zero_ber": detector_zero_ber,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_val
        }

        for key in result:
            test_result[key] += float(result[key])
            writer.add_scalar(f"Test/{noise_name}/{key}", float(result[key]), step)

        if step in saved_iterations:
            if saved_all is None:
                saved_all = get_random_images(image, encoded_images, noised_images)
            else:
                saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

        content = f"Noise={noise_layer} Image {step}: " + ",".join([f"{k}={result[k]}" for k in result]) + "\n"
        with open(test_log, "a") as file:
            file.write(content)
        print(content.strip())

        if max_steps > 0 and step >= max_steps:
            break

    avg = {k: test_result[k] / step for k in test_result}
    avg["tracer_decoder_acc"] = 1.0 - avg["tracer_decoder_ber"]
    avg["detector_decoder_acc"] = 1.0 - avg["detector_decoder_ber"]
    avg["detector_zero_acc"] = 1.0 - avg["detector_zero_ber"]

    summary = (
        f"Average Noise={noise_layer}: "
        f"tracer_ber={avg['tracer_decoder_ber']},tracer_acc={avg['tracer_decoder_acc']},"
        f"detector_ber={avg['detector_decoder_ber']},detector_acc={avg['detector_decoder_acc']},"
        f"detector_zero_ber={avg['detector_zero_ber']},detector_zero_acc={avg['detector_zero_acc']},"
        f"psnr={avg['psnr']},ssim={avg['ssim']},lpips={avg['lpips']}\n"
    )
    with open(test_log, "a") as file:
        file.write(summary)
    print(summary.strip())

    if saved_all is not None:
        os.makedirs(os.path.join(result_folder, "images"), exist_ok=True)
        save_images(saved_all, "test_" + noise_name, os.path.join(result_folder, "images"), resize_to=None)

    return {
        "noise": noise_layer,
        "tracer_decoder_ber": avg["tracer_decoder_ber"],
        "tracer_decoder_acc": avg["tracer_decoder_acc"],
        "detector_decoder_ber": avg["detector_decoder_ber"],
        "detector_decoder_acc": avg["detector_decoder_acc"],
        "detector_zero_ber": avg["detector_zero_ber"],
        "detector_zero_acc": avg["detector_zero_acc"],
        "psnr": avg["psnr"],
        "ssim": avg["ssim"],
        "lpips": avg["lpips"],
        "steps": step,
        "status": "ok"
    }


def main():
    seed_torch(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('cfg/test_DualMark.yaml', 'r') as f:
        test_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    result_folder = "results/" + os.environ.get("SEPMARK_TEST_RESULT_FOLDER", test_args.result_folder)
    model_epoch = int(os.environ.get("SEPMARK_TEST_MODEL_EPOCH", str(test_args.model_epoch)))
    batch_size = int(os.environ.get("SEPMARK_TEST_BATCH_SIZE", str(test_args.batch_size)))
    noise_layer = os.environ.get("SEPMARK_TEST_NOISE_LAYER", str(test_args.noise_layer))

    with open(result_folder + '/train_DualMark.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    lr = args.lr
    beta1 = args.beta1
    image_size = args.image_size
    message_length = args.message_length
    message_range = args.message_range
    attention_encoder = args.attention_encoder
    attention_decoder = args.attention_decoder
    weight = args.weight
    dataset_path = os.environ.get("SEPMARK_DATASET_PATH", args.dataset_path)
    save_images_number = int(os.environ.get("SEPMARK_SAVE_IMAGES_NUMBER", str(args.save_images_number)))
    noise_layers_R = list(args.noise_layers.pool_R)
    noise_layers_F = list(args.noise_layers.pool_F)

    eval_all_noises = os.environ.get("SEPMARK_EVAL_ALL_NOISES", "0") == "1"
    max_steps = int(os.environ.get("SEPMARK_TEST_MAX_STEPS", "0"))

    copyfile("cfg/test_DualMark.yaml", result_folder + "test_DualMark" + time.strftime("_%Y_%m_%d__%H_%M_%S", time.localtime()) + ".yaml")

    writer = _NullWriter()
    if os.environ.get("SEPMARK_ENABLE_TB", "0") == "1":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs/' + result_folder + time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime()))

    # Use empty noise pools in network construction: test applies selected noise explicitly.
    network = Network(message_length, [], [], device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight)
    EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
    network.load_model_ed(EC_path)

    if eval_all_noises:
        eval_noise_layers = []
        seen = set()
        for n in noise_layers_R + noise_layers_F:
            if n not in seen:
                eval_noise_layers.append(n)
                seen.add(n)
    else:
        eval_noise_layers = [noise_layer]

    rows = []
    for n in eval_noise_layers:
        try:
            dataloader = build_test_dataloader(n, dataset_path, image_size, batch_size)
            row = evaluate_noise_layer(
                network=network,
                noise_layer=n,
                dataloader=dataloader,
                message_length=message_length,
                message_range=message_range,
                result_folder=result_folder,
                writer=writer,
                max_steps=max_steps,
                save_images_number=save_images_number,
            )
            rows.append(row)
        except Exception as e:
            rows.append({
                "noise": n,
                "tracer_decoder_ber": "",
                "tracer_decoder_acc": "",
                "detector_decoder_ber": "",
                "detector_decoder_acc": "",
                "detector_zero_ber": "",
                "detector_zero_acc": "",
                "psnr": "",
                "ssim": "",
                "lpips": "",
                "steps": 0,
                "status": f"fail: {e}",
            })
            print(f"[FAIL] noise={n} error={e}")

    csv_path = os.path.join(result_folder, f"noise_eval_{time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "noise",
            "tracer_decoder_ber", "tracer_decoder_acc",
            "detector_decoder_ber", "detector_decoder_acc",
            "detector_zero_ber", "detector_zero_acc",
            "psnr", "ssim", "lpips", "steps", "status"
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for r in rows:
            writer_csv.writerow(r)

    print("\n=== Noise Evaluation Summary ===")
    for r in rows:
        print(r)
    print(f"summary_csv={csv_path}")

    writer.close()


if __name__ == '__main__':
    main()
