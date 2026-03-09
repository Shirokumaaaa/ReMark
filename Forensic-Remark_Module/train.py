from torch.utils.data import DataLoader
from torchvision import transforms as T
from data_loader import CelebA
from utils.load_train_setting import *
from network.Network import *
from network import Encoder_MP
from network import Decoder
import os
from datetime import datetime
from utils import *
from tqdm import tqdm
import torch
import random
import numpy as np
import time
# from network.autoencoder import VQModel  # 或 AutoencoderKL
from network.autoencoder import PlainAE
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity as compare_ssim
from kornia.losses import SSIMLoss

train_image_dir = "/home/ldy/..workspace/kei/MBRS/datasets/CelebA-train/images"
val_image_dir = "/home/ldy/..workspace/kei/MBRS/datasets/CelebA-val/images"
train_attr_path = "/home/ldy/..workspace/kei/MBRS/datasets/CelebA-train/list_attr_celeba5001-15000.txt"
val_attr_path = "/home/ldy/..workspace/kei/MBRS/datasets/CelebA-val/list_attr_celeba5000.txt"

selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

transform = T.Compose([
    T.CenterCrop(178), 
    T.Resize(128),     
    T.ToTensor(),     
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
])

train_dataset = CelebA(
    image_dir=train_image_dir,
    attr_path=train_attr_path,
    selected_attrs=selected_attrs,
    transform=transform
)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

val_dataset = CelebA(
    image_dir=val_image_dir,
    attr_path=val_attr_path,
    selected_attrs=selected_attrs,
    transform=transform
)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
if n_gpus > 1:
    print(f"Using {n_gpus} GPUs for DataParallel.")
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, only_decoder)
if n_gpus > 1:
    autoencoder = torch.nn.DataParallel(PlainAE(lr=1e-4, perceptual_weight=0.1, z_channels=4)).to(device)
else:
    autoencoder = PlainAE(lr=1e-4, perceptual_weight=0.1, z_channels=4).to(device)
optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)

network.load_model(path_encoder_decoder='/home/ldy/..workspace/kei/repair/modelckpt/EC_100.pth', 
                   path_discriminator='/home/ldy/..workspace/kei/repair/modelckpt/D_8.pth')

print('load succeed')

train_ae_losses = []
val_ae_losses = []
ssim_loss_fn = SSIMLoss(11)  # window size 11，默认即可

file_path = '/home/ldy/..workspace/kei/repair/modelckpt/AE.pth'
if not os.path.exists(file_path):
    for epoch in range(200):
        network.encoder_decoder.eval()
        network.discriminator.eval()

        start_time = time.time()

        running_result = {
            "error_rate": 0.0,
            "pnsr": 0.0,
            "ssim": 0.0,
            "g_loss_on_denoise": 0.0,
        }

        autoencoder.train()
        epoch_ae_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Autoencoder Train Epoch {epoch}"):
            images = batch[0].to(device, non_blocking=True)
            batch_size = images.shape[0]
            # 生成随机水印
            message = torch.Tensor(np.random.choice([0,1], (images.shape[0], message_length))).to(device, non_blocking=True)
            # 嵌入水印
            with torch.no_grad():
                images_wm = network.encoder_decoder.module.encode_to_image(images, message)
            optimizer_ae.zero_grad()
            # AE输入为嵌入水印后的图像
            x_hat = autoencoder(images_wm)
            with torch.no_grad():
                perc = autoencoder.module.perc(x_hat, images).mean() if n_gpus > 1 else autoencoder.perc(x_hat, images).mean()
            # MSE损失
            l1 = torch.nn.functional.l1_loss(x_hat, images_wm)
            # 水印提取准确率损失
            decoded_message = network.encoder_decoder.module.decode_from_image(x_hat)
            # 用BCE损失
            bce = torch.nn.functional.binary_cross_entropy_with_logits(decoded_message, message)
            # 总损失（可调权重）
            loss = 0.1*perc + 10*l1 + 10*bce
            loss.backward()
            optimizer_ae.step()
            epoch_ae_train_loss += loss.item()
        epoch_ae_train_loss /= len(train_dataloader)
        train_ae_losses.append(epoch_ae_train_loss)
        # 保存AE模型权重
        ae_save_path = '/home/ldy/..workspace/kei/repair/modelckpt/AE.pth'
        if n_gpus > 1:
            torch.save(autoencoder.module.state_dict(), ae_save_path)
        else:
            torch.save(autoencoder.state_dict(), ae_save_path)
        print(f"Saved AE model to {ae_save_path}")

        # 验证
        autoencoder.eval()
        epoch_ae_val_loss = 0
        psnr_total = 0
        ssim_total = 0
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Autoencoder Val Epoch {epoch}"):
                images = batch[0].to(device, non_blocking=True)
                x_hat = autoencoder(images)
                l1 = torch.nn.functional.l1_loss(x_hat, images)
                perc = autoencoder.module.perc(x_hat, images).mean() if n_gpus > 1 else autoencoder.perc(x_hat, images).mean()
                loss = l1 + 0.1 * perc
                epoch_ae_val_loss += loss.item()
                # 计算PSNR
                imgs = images.detach().cpu()
                recons = x_hat.detach().cpu()
                for i in range(imgs.shape[0]):
                    img = imgs[i].permute(1,2,0).numpy()
                    recon = recons[i].permute(1,2,0).numpy()
                    img = (img + 1) / 2  # [-1,1] -> [0,1]
                    recon = (recon + 1) / 2
                    psnr_total += compare_psnr(img, recon, data_range=1.0)
                    n_val += 1
                # 计算SSIM（kornia，输入需为[0,1]且BCHW）
                imgs_01 = (imgs + 1) / 2
                recons_01 = (recons + 1) / 2
                # kornia的SSIMLoss是1-ssim，取平均后用1-mean得到平均ssim
                ssim_val = 1 - ssim_loss_fn(recons_01, imgs_01)
                ssim_total += ssim_val.item() * imgs_01.shape[0]
        epoch_ae_val_loss /= len(val_dataloader)
        val_ae_losses.append(epoch_ae_val_loss)
        avg_psnr = psnr_total / n_val
        avg_ssim = ssim_total / n_val

        print(f"[Autoencoder] Epoch {epoch}: Train Loss={epoch_ae_train_loss:.4f}, Val Loss={epoch_ae_val_loss:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        

        # ===== 计算验证集水印提取准确率 =====
        watermark_acc_total = 0
        watermark_count = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Watermark Acc Val Epoch {epoch}"):
                images = batch[0].to(device, non_blocking=True)
                batch_size = images.shape[0]
                # 生成随机水印
                message = torch.Tensor(np.random.choice([0,1], (batch_size, message_length))).to(device, non_blocking=True)
                # 嵌入水印
                images_wm = network.encoder_decoder.module.encode_to_image(images, message)
                # AE重建
                x_hat = autoencoder(images_wm)
                # 提取水印
                decoded_message = network.encoder_decoder.module.decode_from_image(x_hat)
                # 计算准确率
                pred = (torch.sigmoid(decoded_message) > 0.5).float()
                acc = (pred == message).float().mean().item()
                watermark_acc_total += acc * batch_size
                watermark_count += batch_size
        watermark_acc = watermark_acc_total / watermark_count if watermark_count > 0 else 0
        print(f"[Autoencoder] Epoch {epoch}: Watermark Extraction Accuracy on Val = {watermark_acc:.4f}")
        # 写入日志文件
        with open("metrics_log.txt", "a") as f:
            f.write(f"Epoch {epoch}: acc={watermark_acc:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}\n")

        # 可视化重建效果
        if epoch % 5 == 0 or epoch == denoise_train_epoch - 1:
            images_vis = images[:8].detach().cpu()
            x_hat_vis = x_hat[:8].detach().cpu()
            # 还原到[0,1]
            images_vis = (images_vis + 1) / 2
            x_hat_vis = (x_hat_vis + 1) / 2
            grid = torch.cat([images_vis, x_hat_vis], dim=0)
            save_image(grid, f"./autoencoder_recon_epoch{epoch}.png", nrow=8)
            print(f"Saved reconstruction visualization at epoch {epoch}")

        # 可视化损失曲线
        if epoch % 5 == 0 or epoch == denoise_train_epoch - 1:
            plt.figure()
            plt.plot(train_ae_losses, label='Train Loss')
            plt.plot(val_ae_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Autoencoder Training/Validation Loss')
            plt.savefig('./autoencoder_loss_curve.png')
            plt.close()
            print("Saved loss curve.")
# import sys
# sys.exit()
'''
train denoise
'''
# num is the batch count
num = 0

for epoch in range(denoise_train_epoch):
    for _, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch{epoch} Training Denoise")):
        images = images.to(device)
        message = torch.Tensor(np.random.choice([0,1], (images.shape[0], message_length))).to(device)

        result = network.train_denoise(images, message, labels)

    # for key in result:
    #     running_result[key] += float(result[key])

'''
train result
'''
# content = "Epoch" + str(epoch) + ":" + str(int(time.time() - start_time)) + '\n'
# for key in running_result:
#     content +=key +"-" + str(running_result[key] / num) + ","
# content += "\n"

# with open(result_folder + "/train_denoise_log.txt", "a") as file:
#     file.write(content)
# print(content)

