from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator
from kornia.losses import SSIMLoss
import torchvision.transforms as T
from network.noise_layers.StarGAN import StarGAN
from network.noise_layers.StarGAN import StarGAN_multi_step
from torchvision.utils import save_image
import os
from torch.utils.data import DataLoader, TensorDataset
from network.autoencoder import PlainAE
from network.unet import UNetModel
# from network.vae import VQModel,AutoencoderKL
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import numpy as np

class Network:

    def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
                 only_decoder=False):
        # device
        self.device = device

        # network
        if not with_diffusion:
            self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers).to(device)
        else:
            self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length, noise_layers).to(device)

        self.discriminator = Discriminator().to(device)

        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        if only_decoder:
            for p in self.encoder_decoder.module.encoder.parameters():
                p.requires_grad = False

        # mark "cover" as 1, "encoded" as 0
        self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
        self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

        # optimizer
        print(lr)
        self.opt_encoder_decoder = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # loss function
        self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
        self.criterion_MSE = nn.MSELoss().to(device)

        # weight of encoder-decoder loss
        self.discriminator_weight = 0.0001
        self.encoder_weight = 1
        self.decoder_weight = 10

    def slerp( self, val, low, high):
        """
        球面插值
        val: 插值比例（0~1）
        low, high: 两个 latent 向量，shape [latent_dim] 或 [batch_size, latent_dim]
        """
        low_norm = low / low.norm(dim=-1, keepdim=True)
        high_norm = high / high.norm(dim=-1, keepdim=True)
        dot = (low_norm * high_norm).sum(dim=-1, keepdim=True)
        omega = torch.acos(torch.clamp(dot, -1, 1))
        so = torch.sin(omega)
        return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    

    def train_denoise_sim(self, 
                       train_dataloader: DataLoader,
                       val_dataloader: DataLoader,
                       num_epochs=20,
                       message_length=64,
                       more_val = False,
                       ):
        for param in self.encoder_decoder.parameters():
            param.requires_grad = False

        for param in self.discriminator.parameters():
            param.requires_grad = False

        self.encoder_decoder.eval()
        self.discriminator.eval()

        # 加载stargan和水印嵌入模型
        stargan_layer = StarGAN(
            g_conv_dim=64, c_dim=5, g_repeat_num=6, device=self.device,
            model_path="/home/ldy/..workspace/zhou/repair/models/200000-G.ckpt"
        )
        cnt=1
        # -------------------- 0.测试StarGAN模型是否已准备好-----------------------------------------
        for images, labels in tqdm(train_dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)          
            B = images.size(0)

            message_gt = torch.randint(low=0, high=2, size=(B, message_length), device=device).float()
            # 需要测两个东西
            # 一个是原图水印准确率
            # 一个是fake图水印准确率
            with torch.no_grad():
                if isinstance(self.encoder_decoder, nn.DataParallel):
                    images_wm = self.encoder_decoder.module.encode_to_image(images, message_gt)
                else:
                    images_wm = self.encoder_decoder.encode_to_image(images, message_gt)
            deepfakes = stargan_layer((images_wm, images, labels))
            # self.save_images1(images,deepfakes,4,cnt)
            cnt+=1
            if isinstance(network.encoder_decoder, nn.DataParallel):
                damage_messages = self.encoder_decoder.module.decode_from_image(images_wm).float()
            else:
                damage_messages = self.encoder_decoder.decode_from_image(images_wm).float()
            
            if isinstance(network.encoder_decoder, nn.DataParallel):
                fake_messages = self.encoder_decoder.module.decode_from_image(deepfakes).float()
            else:
                fake_messages = self.encoder_decoder.decode_from_image(deepfakes).float()

            wm_acc_step = 1.0 - self.decoded_message_error_rate_batch(message_gt, damage_messages)
            fake_acc_step = 1.0 - self.decoded_message_error_rate_batch(message_gt, fake_messages)
            print(f"Watermark Accuracy on encoded images: {wm_acc_step:.4f}, Fake Image Watermark Accuracy: {fake_acc_step:.4f}")

        n_gpus = torch.cuda.device_count()
        print(n_gpus)
        # --------------------- A.载入autoencoder模型-------------------------------------------------
        ddconfig={
                "double_z": 0,
                "z_channels": 3,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                # 下采样率调节
                "ch_mult": [
                    1,
                    2,
                    4
                ],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0
            }
        if n_gpus > 1:
            autoencoder = torch.nn.DataParallel(AutoencoderKL(embed_dim=3,ddconfig=ddconfig)).to(self.device)
        else:
            autoencoder = AutoencoderKL(embed_dim=3,ddconfig=ddconfig).to(self.device)
        # 加载VAE权重
        state_dict = torch.load('/home/ldy/..workspace/zhou/repair/VAE_model/autoencoder_epoch_190.pth', map_location=self.device)

        if n_gpus > 1:
            # 如果是多卡，但权重没有“module.”前缀，需要加上
            if not list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict['module.' + k] = v
                state_dict = new_state_dict
            autoencoder.load_state_dict(state_dict)
        else:
            # 如果是单卡，但权重有“module.”前缀，需要去掉
            if list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v
                state_dict = new_state_dict
            autoencoder.load_state_dict(state_dict)
        
        autoencoder.eval()

        # ---------------------载入模型并完成fake image集的构建-----------------------------------------------------
        # 默认不计算准确率以节省内存，如果需要可以设置compute_accuracy=True
        val_dataloader = self.fake_data(val_dataloader, stargan_layer, autoencoder, message_length, compute_accuracy=True, clean_to_damage=True)
        train_dataloader = self.fake_data(train_dataloader, stargan_layer, autoencoder, message_length, compute_accuracy=False)


        # --------------------- B.UNET模型的构建-------------------------------------------------
        image_size = 32
        in_channels = 3          # 输入图像通道数（RGB）
        out_channels = 3         # 输出图像通道数（RGB）
        model_channels = 64      # 基础通道数，增大以提高特征表达能力
        attention_resolutions = [4, 2]  # 仅在4x4和2x2分辨率使用注意力，避免高分辨率时计算开销过大
        num_res_blocks = 2       # 每层残差块数量
        channel_mult = [1, 2, 4]  # 减少一层，避免32x32图像下特征图尺寸过小
        num_head_channels = 32   # 每个注意力头的通道数
        use_scale_shift_norm = True  # 使用FiLM风格的条件机制，增强时间步嵌入的影响
        resblock_updown = True   # 使用残差块进行上/下采样，保留更多细节
        use_spatial_transformer = True  # 使用更强大的SpatialTransformer代替普通注意力
        context_dim = 768        # 上下文维度（如使用文本条件）
        transformer_depth = 1    # Transformer层数

        # 创建UNet模型
        # 需要注意此处是否有残差连接
        model = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            attention_resolutions=attention_resolutions,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_spatial_transformer=use_spatial_transformer,
            context_dim=context_dim,
            transformer_depth=transformer_depth
        )

        # 移动到设备
        model = model.to(self.device)

        # 训练步骤
        # 训练参数
        num_epochs = 100
        save_interval = 10
        save_dir = './checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        # 训练循环
        train_losses = []
        val_losses = []

        acc_watermarkss = [[] for _ in range(9)] if more_val else []

        # --------------------- C.开始训练UNET模型-------------------------------------------------
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_bce_loss = 0.0
            epoch_mse_loss = 0.0
            for step, (repair_latents, repair_latents_prev, repair_t, repair_t_next, repair_damage, repair_step, repair_clean_message) in tqdm(enumerate(train_dataloader), desc=f"Epochs: {epoch+1}/{num_epochs} Train", total=len(train_dataloader)):
                # 将数据移动到设备
                repair_latents = repair_latents.to(self.device)
                repair_latents_prev = repair_latents_prev.to(self.device)
                repair_t = repair_t.to(self.device)
                repair_t_next = repair_t_next.to(self.device)
                repair_damage = repair_damage.to(self.device)
                # repair_img = autoencoder.module.decoder(repair_latents) if n_gpus > 1 else autoencoder.decoder(repair_latents)
                # repair_message = self.encoder_decoder.module.decode_from_image(repair_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(repair_img)
                # acc = 1.0 - self.decoded_message_error_rate_batch(repair_clean_message, repair_message)
                # with open("batch1.txt", "a") as f:
                #     f.write(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Watermark Accuracy: {acc:.4f},repair_step:{repair_step}\n")

                # 将时间步转换为整数timesteps (0-1000范围)
                # 这里时间步为什么要变成0-1000范围？
                timesteps = (repair_t * 10).float()
                
                # 创建合适维度的context tensor
                batch_size = repair_latents.shape[0]
                context = torch.zeros(batch_size, 1, 768).to(self.device)
                
                # 前向传播
                outputs = model(repair_latents, timesteps, context=context)

                repair_img = autoencoder.module.decoder(outputs) if n_gpus > 1 else autoencoder.decoder(outputs)
                repair_message = self.encoder_decoder.module.decode_from_image(repair_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(repair_img)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(repair_message, repair_clean_message)
                mse = torch.nn.functional.mse_loss(outputs, repair_latents_prev)
                
                # 计算损失
                loss = 0.5* mse + 0.3 * bce
                epoch_bce_loss += bce.item()
                epoch_mse_loss += mse.item()
                epoch_loss += loss.item()
                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
            # 每个epoch结束后打印学习率
            with open("batch111.txt", "a") as f:
                f.write(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, BCE Loss: {bce.item():.4f}, MSE Loss: {mse.item():.4f}\n")
            print(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            # 记录训练损失
            train_losses.append(epoch_loss / len(train_dataloader))
            model.eval()
            # 验证阶段
            if more_val:
                val_loss = 0.0
                val_step_count = 0
                acc_watermark = [0.0] * 15
                example_flag = 0
                imgs = []
                with torch.no_grad():
                    for step,(val_repair_latents, val_repair_latents_prev, val_repair_t, val_repair_t_next, val_repair_damage, val_step,repair_clean_message) in tqdm(enumerate(val_dataloader), desc=f"Epochs: {epoch+1}/{num_epochs} Val", total=len(val_dataloader)):
                        # 将数据移动到设备
                        val_repair_latents = val_repair_latents.to(self.device)
                        val_repair_latents_prev = val_repair_latents_prev.to(self.device)
                        val_repair_t = val_repair_t.to(self.device)
                        
                        # 将时间步转换为整数timesteps (0-1000范围)
                        val_timesteps = (val_repair_t * 1000).float()
                        val_step_count += 1
                        val_img = autoencoder.module.decoder(val_repair_latents_prev) if n_gpus > 1 else autoencoder.decoder(val_repair_latents_prev)
                        pred_img = autoencoder.module.decoder(val_repair_latents) if n_gpus > 1 else autoencoder.decoder(val_repair_latents)
                        val_messages = self.encoder_decoder.module.decode_from_image(val_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(val_img)
                        pred_message = self.encoder_decoder.module.decode_from_image(pred_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(pred_img)
                        # acc = 1.0 - self.decoded_message_error_rate_batch(repair_clean_message, pred_message)
                        acc_watermark[0]+=1.0 - self.decoded_message_error_rate_batch(repair_clean_message, pred_message)
                        acc_watermark[10] += 1.0 - self.decoded_message_error_rate_batch(repair_clean_message, val_messages)

                        for i in range(1,10):
                            # 创建合适维度的context tensor
                            val_batch_size = val_repair_latents.shape[0]
                            val_context = torch.zeros(val_batch_size, 1, 768).to(self.device)
                            
                            # 前向传播
                            val_outputs = model(val_repair_latents, val_timesteps, context=val_context)
                            
                            # 计算损失
                            val_loss_step = torch.nn.functional.mse_loss(val_outputs, val_repair_latents_prev)
                            val_loss += val_loss_step.item()
                            

                            # 计算水印准确率
                            self.encoder_decoder.eval()
                            val_img = autoencoder.module.decoder(val_repair_latents_prev) if n_gpus > 1 else autoencoder.decoder(val_repair_latents_prev)
                            pred_img = autoencoder.module.decoder(val_outputs) if n_gpus > 1 else autoencoder.decoder(val_outputs)
                            val_messages = self.encoder_decoder.module.decode_from_image(val_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(val_img)
                            pred_message = self.encoder_decoder.module.decode_from_image(pred_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(pred_img)
                            # Convert tensors to numpy arrays and adjust dimensions for SSIM calculation
                            pred_img_np = pred_img.detach().cpu().numpy()
                            val_clean_img_np = repair_clean_message.detach().cpu().numpy()

                            # # Calculate SSIM for each image in the batch
                            # ssim_scores = []
                            # for i in range(pred_img_np.shape[0]):
                            #     # Convert from CHW to HWC format for SSIM calculation
                            #     pred_hwc = np.transpose(pred_img_np[i], (1, 2, 0))
                            #     clean_hwc = np.transpose(val_clean_img_np[i], (1, 2, 0))
                                
                            #     # Calculate SSIM
                            #     ssim_score = ssim(clean_hwc, pred_hwc, multichannel=True, data_range=2.0)
                            #     ssim_scores.append(ssim_score)

                            # avg_ssim = np.mean(ssim_scores)
                            # print(f"Average SSIM: {avg_ssim:.4f}")
                            imgs.append(pred_img)
                            acc = 1.0 - self.decoded_message_error_rate_batch(repair_clean_message, pred_message)
                            acc_watermark[i] += acc

                            val_repair_latents = val_outputs.detach()  # 更新val_repair_latents为当前输出，准备下一步迭代
                            val_repair_t = val_repair_t+0.1

                        # 保存验证结果 - 添加在验证循环内部
                        if step == 0:  # 只保存第一个batch的结果作为示例
                            # 创建保存目录
                            val_save_dir = "/home/ldy/..workspace/zhou/repair/output_images/validation_results"
                            os.makedirs(val_save_dir, exist_ok=True)
                            # 使用autoencoder解码latent为图像进行可视化
                            n_save_val = min(8, val_outputs.shape[0])
                            for i in range(n_save_val):
                                # 解码预测结果
                                if n_gpus > 1:
                                    input_img = autoencoder.module.decoder(val_repair_latents[i:i+1])
                                    pred_img = autoencoder.module.decoder(val_outputs[i:i+1])
                                    target_img = autoencoder.module.decoder(val_repair_latents_prev[i:i+1])
                                else:
                                    input_img = autoencoder.decoder(val_repair_latents[i:i+1])
                                    pred_img = autoencoder.decoder(val_outputs[i:i+1])
                                    target_img = autoencoder.decoder(val_repair_latents_prev[i:i+1])

                                
                                # 逆标准化到[0,1]
                                pred_img = pred_img * 0.5 + 0.5
                                target_img = target_img * 0.5 + 0.5
                                input_img = input_img * 0.5 + 0.5
                                
                                # 拼接图像：输入-预测-目标
                                combined_img = torch.cat([input_img.squeeze(0), pred_img.squeeze(0), target_img.squeeze(0)], dim=2)

                                # 保存图像
                                save_path = os.path.join(val_save_dir, f'epoch_{epoch+1}_sample_{i}_input_pred_target.png')
                                save_image(combined_img.cpu(), save_path)
                # if example_flag == 0:
                #     # Ensure all images have the same shape before stacking
                #     if imgs and len(imgs) > 0:
                #         # Get the first image shape as reference
                #         reference_shape = imgs[0].shape
                #         # Filter images that match the reference shape
                #         valid_imgs = [img for img in imgs if img.shape == reference_shape]
                        
                #         if valid_imgs:
                #             imgs_tensor = torch.stack(valid_imgs)
                #             imgs_tensor = imgs_tensor.cpu()
                #             val_save_dir = "/home/ldy/..workspace/zhou/repair/output_images/validation_results"
                #             save_path = os.path.join(val_save_dir, f'epoch_{epoch+1}_sample_{i}_10times.png')
                #             save_image(imgs_tensor, save_path)
                #     example_flag = 1
                for i in range(0,11):
                    print(f"Watermark Accuracy at step {i}: {acc_watermark[i]/val_step_count:.4f}")
                    with open("batch11.txt", "a") as f:
                        f.write(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Watermark Accuracy at step {i}: {acc_watermark[i]/val_step_count:.4f}\n")
                    # acc_watermark[i] /= val_step_count if val_step_count > 0 else 1.0
                    # acc_watermarkss[i-1].append(acc_watermark[i])
                # # 计算平均验证损失
                #     avg_val_loss = val_loss / val_step_count if val_step_count > 0 else 0.0
                #     avg_acc_watermark = [acc / val_step_count if val_step_count > 0 else 0.0 for acc in acc_watermark]
                #     val_losses.append(avg_val_loss)
                #     acc_watermarkss.append(avg_acc_watermark)
                #     print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Watermark Accuracy: {avg_acc_watermark}")
            else:
                val_loss = 0.0
                val_step_count = 0
                acc_watermark = 0.0
                with torch.no_grad():
                    for val_step, (val_repair_latents, val_repair_latents_prev, val_repair_t, val_repair_t_next, val_repair_damage, val_clean_images, val_messages) in tqdm(enumerate(val_dataloader), desc=f"Epochs: {epoch+1}/{num_epochs} Val", total=len(val_dataloader)):
                        # 将数据移动到设备
                        val_repair_latents = val_repair_latents.to(self.device)
                        val_repair_latents_prev = val_repair_latents_prev.to(self.device)
                        val_repair_t = val_repair_t.to(self.device)
                        
                        # 将时间步转换为整数timesteps (0-1000范围)
                        val_timesteps = (val_repair_t * 0 + 1000).float()
                        
                        # 创建合适维度的context tensor
                        val_batch_size = val_repair_latents.shape[0]
                        val_context = torch.zeros(val_batch_size, 1, 768).to(self.device)
                        
                        # 前向传播
                        val_outputs = model(val_repair_latents, val_timesteps, context=val_context)
                        
                        # 计算损失
                        val_loss_step = torch.nn.functional.mse_loss(val_outputs, val_repair_latents_prev)
                        val_loss += val_loss_step.item()
                        val_step_count += 1

                        # 计算水印准确率
                        self.encoder_decoder.eval()
                        pred_img = autoencoder.module.decoder(val_outputs) if n_gpus > 1 else autoencoder.decoder(val_outputs)
                        pred_message = self.encoder_decoder.module.decode_from_image(pred_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(pred_img)
                        pred = (torch.sigmoid(pred_message) > 0.5).float()
                        message = (torch.sigmoid(val_messages) > 0.5).float()  # 假设val_messages是二进制的
                        acc = (pred == message).float().mean().item()
                        acc_watermark += acc
                        # 保存验证结果 - 添加在验证循环内部
                        if val_step == 0:  # 只保存第一个batch的结果作为示例
                            # 创建保存目录
                            val_save_dir = "/home/ldy/..workspace/zhou/repair/output_images/validation_results"
                            os.makedirs(val_save_dir, exist_ok=True)
                            # 使用autoencoder解码latent为图像进行可视化
                            n_save_val = min(8, val_outputs.shape[0])
                            for i in range(n_save_val):
                                # 解码预测结果
                                if n_gpus > 1:
                                    input_img = autoencoder.module.decoder(val_repair_latents[i:i+1])
                                    pred_img = autoencoder.module.decoder(val_outputs[i:i+1])
                                    target_img = autoencoder.module.decoder(val_repair_latents_prev[i:i+1])
                                    # clean_img = autoencoder.module.decoder(val_clean_images[i:i+1])
                                else:
                                    input_img = autoencoder.decoder(val_repair_latents[i:i+1])
                                    pred_img = autoencoder.decoder(val_outputs[i:i+1])
                                    target_img = autoencoder.decoder(val_repair_latents_prev[i:i+1])
                                    # clean_img = autoencoder.decoder(val_clean_images[i:i+1])

                                
                                # 逆标准化到[0,1]
                                pred_img = pred_img * 0.5 + 0.5
                                target_img = target_img * 0.5 + 0.5
                                input_img = input_img * 0.5 + 0.5
                                
                                # 拼接图像：输入-预测-目标
                                clean_img = val_clean_images[i:i+1]
                                clean_img = clean_img[0]
                                clean_img = clean_img * 0.5 + 0.5  # 逆标准化到[0,1]
                                combined_img = torch.cat([input_img.squeeze(0), pred_img.squeeze(0), target_img.squeeze(0)], dim=2)

                                # 保存图像
                                save_path = os.path.join(val_save_dir, f'epoch_{epoch+1}_sample_{i}_input_pred_target_clean.png')
                                save_image(combined_img.cpu(), save_path)

                # 计算平均验证损失
                avg_val_loss = val_loss / val_step_count if val_step_count > 0 else 0.0
                avg_acc_watermark = acc_watermark / val_step_count if val_step_count > 0 else 0.0
                val_losses.append(avg_val_loss)
                acc_watermarkss.append(avg_acc_watermark)
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Watermark Accuracy: {avg_acc_watermark:.4f}")
            # 更新学习率
            scheduler.step()
            # 保存模型 - 仅保存top3模型
            if (epoch + 1) % save_interval == 0:
                model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), model_save_path)
                
                # # 保持最多3个模型文件
                # saved_models = sorted([f for f in os.listdir(save_dir) if f.startswith('model_epoch_') and f.endswith('.pth')])
                # if len(saved_models) > 3:
                #     # 删除最旧的模型
                #     oldest_model = saved_models[0]
                #     os.remove(os.path.join(save_dir, oldest_model))
                #     print(f"Removed old model: {oldest_model}")
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
        plt.figure(figsize=(10, 5))
        if more_val:
            for i in range(1,10):
                plt.plot(acc_watermarkss[i-1], label=f'Watermark Accuracy {i}')
        else:
            plt.plot(acc_watermarkss, label='Watermark Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Watermark Accuracy')
        plt.savefig(os.path.join(save_dir, 'watermark_accuracy.png'))

        # ---------------------利用去噪网络开始训练-----------------------------------
        # 训练完成后，保存模型
        model_save_path = "/home/ldy/..workspace/zhou/repair/models/denoise_model.pth"
        torch.save(model.state_dict(), model_save_path) 
        # repair_dataloader: (当前步向量, 上一步向量, 步编号, damage阶段, t),下面是dalaloader的详细情况
        '''
        当前步向量(repair_latents_tensor):
        当前插值时间步的 latent 向量(如第10步的向量)。

        上一步向量(repair_latents_prev_tensor):
        上一个时间步的 latent 向量（如第9步的向量）。

        步编号（repair_steps_tensor）：
        当前的时间步编号（如10，表示第10步），编号从2开始（clean为1，damage为最大编号）。

        damage阶段编号（repair_damage_tensor）：
        该插值序列属于哪个损坏阶段（如1、2、3）。

        插值比例 t（repair_t_tensor）：
        当前步在 clean 和 damage latent 之间的插值比例（0~1，越大越接近损坏端）。
        '''
        # 下一步，利用这个dataloader直接开始训练。
        # 训练输入为: 1.repair_latent_tensor 2.repair_steps_tensor
        # 训练输出为: 1.repair_latent_prev_tensor

        # 越靠近image，时间步越小(real image的时间步为0)
        # 越靠近fake image 时间步越大(例如damage_degree为3的latent，初始时间步为30)

        # todo
        # todo
        # todo: 在network class下构建类扩散模型的denoise_network，完成去噪
        # ---------------------利用去噪网络开始训练-----------------------------------


    # def train_denoise(self, images: torch.Tensor, messages: torch.Tensor, labels: torch.Tensor, accumulation_steps=1):
    #     # 这里需要把encoder_decoder和discriminator的参数冻结掉
    #     for param in self.encoder_decoder.parameters():
    #         param.requires_grad = False

    #     for param in self.discriminator.parameters():
    #         param.requires_grad = False

    #     self.encoder_decoder.eval()
    #     self.discriminator.eval()

    #     # 这里需要enable_grad因为后面的去噪网络训练需要梯度下降
    #     with torch.no_grad():
    #         encoded_image = self.encoder_decoder.module.encode_to_image(images, messages)  # 使用 .module 访问方法

    #     # ---------------------载入模型并完成fake image集的构建-----------------------------------------------------
    #     stargan_layer = StarGAN_multi_step(
    #         g_conv_dim=64, c_dim=5, g_repeat_num=6, device=self.device,
    #         model_path="/home/ldy/..workspace/zhou/repair/models/200000-G.ckpt"
    #     )
    #     stargan_results = stargan_layer((encoded_image, images, labels))

    #     # 分别存储每个阶段的图像和message
    #     clean_images = []
    #     clean_messages = []
    #     transformed1_images = []
    #     transformed1_messages = []
    #     transformed2_images = []
    #     transformed2_messages = []
    #     transformed3_images = []
    #     transformed3_messages = []

    #     for img, step in stargan_results:
    #         if step == 0:
    #             clean_images.append(img)
    #             clean_messages.append(messages)
    #         elif step == 1:
    #             transformed1_images.append(img)
    #             msg1 = self.encoder_decoder.module.decode_from_image(img)
    #             transformed1_messages.append(msg1)
    #         # elif step == 2:
    #         #     transformed2_images.append(img)
    #         #     msg2 = self.encoder_decoder.module.decode_from_image(img)
    #         #     transformed2_messages.append(msg2)
    #         # elif step == 3:
    #         #     transformed3_images.append(img)
    #         #     msg3 = self.encoder_decoder.module.decode_from_image(img)
    #         #     transformed3_messages.append(msg3)

    #     # 合并batch
    #     if clean_images:
    #         clean_images = torch.cat(clean_images, dim=0)
    #         clean_messages = torch.cat(clean_messages, dim=0)
    #     if transformed1_images:
    #         transformed1_images = torch.cat(transformed1_images, dim=0)
    #         transformed1_messages = torch.cat(transformed1_messages, dim=0)
    #     # if transformed2_images:
    #     #     transformed2_images = torch.cat(transformed2_images, dim=0)
    #     #     transformed2_messages = torch.cat(transformed2_messages, dim=0)
    #     # if transformed3_images:
    #     #     transformed3_images = torch.cat(transformed3_images, dim=0)
    #     #     transformed3_messages = torch.cat(transformed3_messages, dim=0)

    #     # 打印各阶段样本数
    #     print(f"clean_images: {clean_images.shape[0] if isinstance(clean_images, torch.Tensor) else 0}")
    #     print(f"transformed1_images: {transformed1_images.shape[0] if isinstance(transformed1_images, torch.Tensor) else 0}")
    #     # print(f"transformed2_images: {transformed2_images.shape[0] if isinstance(transformed2_images, torch.Tensor) else 0}")
    #     # print(f"transformed3_images: {transformed3_images.shape[0] if isinstance(transformed3_images, torch.Tensor) else 0}")

    #     # 保存下来transformed0-3，检查一下是否正确
    #     save_dir = "/home/ldy/..workspace/zhou/repair/output_images/denoise_check"
    #     os.makedirs(save_dir, exist_ok=True)
    #     # 只保存当前batch的前8张
    #     n_save = min(8, clean_images.shape[0] if isinstance(clean_images, torch.Tensor) else 0)

    #     for i in range(n_save):
    #         imgs_to_save = []
    #         names = []
    #         if isinstance(clean_images, torch.Tensor):
    #             img = clean_images[i].detach().cpu()
    #             img = img * 0.5 + 0.5  # 逆标准化到[0,1]
    #             imgs_to_save.append(img)
    #             names.append("clean")
    #         if isinstance(transformed1_images, torch.Tensor):
    #             img = transformed1_images[i].detach().cpu()
    #             img = img * 0.5 + 0.5
    #             imgs_to_save.append(img)
    #             names.append("trans1")
    #         # if isinstance(transformed2_images, torch.Tensor):
    #         #     img = transformed2_images[i].detach().cpu()
    #         #     img = img * 0.5 + 0.5
    #         #     imgs_to_save.append(img)
    #         #     names.append("trans2")
    #         # if isinstance(transformed3_images, torch.Tensor):
    #         #     img = transformed3_images[i].detach().cpu()
    #         #     img = img * 0.5 + 0.5
    #         #     imgs_to_save.append(img)
    #         #     names.append("trans3")
    #         # 拼接
    #         if imgs_to_save:
    #             img_cat = torch.cat(imgs_to_save, dim=2)  # 横向拼接
    #             save_path = os.path.join(save_dir, f"sample_{i}_{'_'.join(names)}.png")
    #             save_image(img_cat, save_path)
    #             # print(f"Saved {save_path}")

    #     # 准确率测试
    #     # 原始图像
    #     if isinstance(clean_images, torch.Tensor):
    #         decoded_clean = self.encoder_decoder.module.decode_from_image(clean_images)
    #         acc_clean = 1.0 - self.decoded_message_error_rate_batch(clean_messages, decoded_clean)
    #         print(f"Accuracy (clean): {acc_clean:.4f}")
    #     # transformed1
    #     if isinstance(transformed1_images, torch.Tensor):
    #         acc_t1 = 1.0 - self.decoded_message_error_rate_batch(clean_messages, transformed1_messages)
    #         print(f"Accuracy (transformed1): {acc_t1:.4f}")
    #     # # transformed2
    #     # if isinstance(transformed2_images, torch.Tensor):
    #     #     acc_t2 = 1.0 - self.decoded_message_error_rate_batch(clean_messages, transformed2_messages)
    #     #     print(f"Accuracy (transformed2): {acc_t2:.4f}")
    #     # # transformed3
    #     # if isinstance(transformed3_images, torch.Tensor):
    #     #     acc_t3 = 1.0 - self.decoded_message_error_rate_batch(clean_messages, transformed3_messages)
    #     #     print(f"Accuracy (transformed3): {acc_t3:.4f}")


    #     # ---------------------利用预训练好的Auto Encoder将Image投影到Latent中去-----------------------------------------------------
    #     # 将他们投影到latent space中去
    #     n_gpus = torch.cuda.device_count()
    #     print(n_gpus)
    #     from ldm.models.autoencoder import VQModelInterface
    #     ddconfig={
    #             "double_z": 0,
    #             "z_channels": 3,
    #             "resolution": 256,
    #             "in_channels": 3,
    #             "out_ch": 3,
    #             "ch": 128,
    #             "ch_mult": [
    #                 1,
    #                 2,
    #                 4
    #             ],
    #             "num_res_blocks": 2,
    #             "attn_resolutions": [],
    #             "dropout": 0.0
    #         }

    #     lossconfig = {
    #             "target": "torch.nn.Identity"
    #         }
    #     if n_gpus > 1:
    #         autoencoder = torch.nn.DataParallel(VQModel(embed_dim=3,n_embed=8192,ddconfig=ddconfig)).to(self.device)
    #     else:
    #         autoencoder = VQModel(embed_dim=3,n_embed=8192,ddconfig=ddconfig).to(self.device)
    #     # 加载权重
    #     state_dict = torch.load('/home/ldy/..workspace/zhou/repair/results/last.ckpt', map_location=self.device)
    #     if n_gpus > 1:
    #         # 如果是多卡，但权重没有“module.”前缀，需要加上
    #         if not list(state_dict.keys())[0].startswith('module.'):
    #             from collections import OrderedDict
    #             new_state_dict = OrderedDict()
    #             for k, v in state_dict.items():
    #                 new_state_dict['module.' + k] = v
    #             state_dict = new_state_dict
    #         autoencoder.load_state_dict(state_dict)
    #     else:
    #         # 如果是单卡，但权重有“module.”前缀，需要去掉
    #         if list(state_dict.keys())[0].startswith('module.'):
    #             from collections import OrderedDict
    #             new_state_dict = OrderedDict()
    #             for k, v in state_dict.items():
    #                 new_state_dict[k[7:]] = v
    #             state_dict = new_state_dict
    #         autoencoder.load_state_dict(state_dict)
        
    #     autoencoder.eval()

    #     # 降维结果和损坏程度对应关系
    #     latent_list = []
    #     damage_list = []
    #     messages_list = []
    #     with torch.no_grad():
    #         # Image shape: [32, 3, 128, 128], Latent shape: [32, 3, 32, 32], Step shape: [1]
            
    #         for result, message in zip(stargan_results, messages):
    #             img, step = result
    #             if n_gpus > 1:
    #                 latent = autoencoder.module.encoder(img)
    #             else:
    #                 latent = autoencoder.encoder(img)
    #             latent_list.append(latent)
    #             damage_list.append(step)
    #             messages_list.append(message)
    #     print(f"Latent list length: {len(latent_list)}, Damage list length: {len(damage_list)}, Messages list length: {len(messages_list)}")

    #     clean_latent = latent_list[0]
    #     # 构建 (clean_latent, damage_latent, damage_level) 的配对
    #     clean_damage_pairs = []
    #     for i, damage_level in enumerate(damage_list[1:], 1):  # 跳过clean，从damage_list[1]开始
    #         damage_latent = latent_list[i]
    #         message = messages_list[i]
    #         clean_damage_pairs.append((clean_latent, damage_latent, damage_level,message))

    #     print(f"Clean latent shape: {clean_latent.shape}")
    #     print(f"Damage latent shape: {damage_latent.shape}")
    #     data_pair = []
    #     # 插值阶段
    #     num_interp = {1:10, 2:20, 3:30}  # 可自定义每个阶段插值数量
    #     for clean_latent, damage_latent, damage_level,message in clean_damage_pairs:
    #         # 生成编号序列：clean为1，damage为num_interp[damage]，中间插值点依次编号
    #         total_steps = num_interp[damage_level]
    #         interpolated_latents = []
    #         interpolated_steps = []
    #         for idx, t in enumerate(torch.linspace(0, 1, total_steps+2)):
    #             interp_latent = self.slerp(t, clean_latent, damage_latent)
    #             # print(interp_latent.shape, damage_level, idx + 1, float(t))
    #             # step_id: clean=1, damage=total_steps+1, 中间点=2~total_steps
    #             step_id = idx + 1
    #             interpolated_latents.append(interp_latent)
    #             interpolated_steps.append((damage_level, step_id, float(t)))
        
    #         # 将同一damage_level的相邻step组合
    #         for j in range(len(interpolated_latents) - 1):
    #             current_latent = interpolated_latents[j]
    #             next_latent = interpolated_latents[j + 1]
    #             current_step = interpolated_steps[j][2]
    #             next_step = interpolated_steps[j + 1][2]
    #             for current_img, next_img in zip(current_latent, next_latent):
    #                 # 在这个位置把batch消掉
    #                 # 组合 (当前步latent, 下一步latent, 当前步t, 下一步t, damage_level)
    #                 # 为每个样本分别添加对应的clean_image和message
    #                 current_idx = current_latent.tolist().index(current_img.tolist()) if hasattr(current_latent, 'tolist') else 0
    #                 single_clean_image = clean_images[current_idx:current_idx+1] if current_idx < clean_images.shape[0] else clean_images[0:1]
    #                 single_message = message[current_idx:current_idx+1] if current_idx < message.shape[0] else message[0:1]
                    
    #                 # print(f"Pairing: {current_img.shape}, {next_img.shape}, {current_step}, {next_step}, {damage_level}")
    #                 data_pair.append((current_img, next_img, current_step, next_step, damage_level, single_clean_image.squeeze(0), single_message.squeeze(0)))
    #     print(f"Data pair length: {len(data_pair)}")
        
    #     # 将data_pair转换为Tensor并创建DataLoader
    #     repair_latents_tensor = torch.stack([pair[0] for pair in data_pair])
    #     repair_latents_prev_tensor = torch.stack([pair[1] for pair in data_pair])
    #     repair_t_tensor = torch.tensor([pair[2] for pair in data_pair], dtype=torch.float32)
    #     repair_t_next_tensor = torch.tensor([pair[3] for pair in data_pair], dtype=torch.float32)
    #     repair_damage_tensor = torch.tensor([pair[4] for pair in data_pair], dtype=torch.long)
    #     # clean_images_tensor = torch.stack([pair[5] for pair in data_pair])
    #     messages_tensor = torch.stack([pair[6] for pair in data_pair])  # 如果需要messages

    #     print(f"repair_latents_tensor shape: {repair_latents_tensor.shape}")
    #     print(f"repair_latents_prev_tensor shape: {repair_latents_prev_tensor.shape}")
    #     print(f"repair_t_tensor shape: {repair_t_tensor.shape}")
    #     print(f"repair_t_next_tensor shape: {repair_t_next_tensor.shape}")
    #     print(f"repair_damage_tensor shape: {repair_damage_tensor.shape}")

    #     # 创建TensorDataset和DataLoader
    #     repair_dataset = TensorDataset(
    #         repair_latents_tensor, 
    #         repair_latents_prev_tensor, 
    #         repair_t_tensor,
    #         repair_t_next_tensor, 
    #         repair_damage_tensor,
    #         # clean_images_tensor,
    #         messages_tensor
    #     )
    #     repair_dataloader = DataLoader(repair_dataset, batch_size=32, shuffle=True)
    #     # 切分训练集和验证集
    #     train_size = int(0.8 * len(repair_dataset))
    #     val_size = len(repair_dataset) - train_size
    #     train_dataset, val_dataset = torch.utils.data.random_split(repair_dataset, [train_size, val_size])

    #     # 创建训练和验证的DataLoader
    #     train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #     val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    #     print(f"Train dataset size: {len(train_dataset)}")
    #     print(f"Validation dataset size: {len(val_dataset)}")
    #     # print(f"Created repair_dataloader with {len(repair_dataset)} samples")


    #     # 调整后的UNet参数配置
    #     image_size = 32
    #     in_channels = 3          # 输入图像通道数（RGB）
    #     out_channels = 3         # 输出图像通道数（RGB）
    #     model_channels = 64      # 基础通道数，增大以提高特征表达能力
    #     attention_resolutions = [4, 2]  # 仅在4x4和2x2分辨率使用注意力，避免高分辨率时计算开销过大
    #     num_res_blocks = 2       # 每层残差块数量
    #     channel_mult = [1, 2, 4]  # 减少一层，避免32x32图像下特征图尺寸过小
    #     num_head_channels = 32   # 每个注意力头的通道数
    #     use_scale_shift_norm = True  # 使用FiLM风格的条件机制，增强时间步嵌入的影响
    #     resblock_updown = True   # 使用残差块进行上/下采样，保留更多细节
    #     use_spatial_transformer = True  # 使用更强大的SpatialTransformer代替普通注意力
    #     context_dim = 768        # 上下文维度（如使用文本条件）
    #     transformer_depth = 1    # Transformer层数

    #     # 创建UNet模型
    #     model = UNetModel(
    #         image_size=image_size,
    #         in_channels=in_channels,
    #         out_channels=out_channels,
    #         model_channels=model_channels,
    #         attention_resolutions=attention_resolutions,
    #         num_res_blocks=num_res_blocks,
    #         channel_mult=channel_mult,
    #         num_head_channels=num_head_channels,
    #         use_scale_shift_norm=use_scale_shift_norm,
    #         resblock_updown=resblock_updown,
    #         use_spatial_transformer=use_spatial_transformer,
    #         context_dim=context_dim,
    #         transformer_depth=transformer_depth
    #     )

    #     # 移动到设备
    #     model = model.to(self.device)

    #     # 训练步骤
    #     # 训练参数
    #     num_epochs = 20
    #     save_interval = 10
    #     save_dir = './checkpoints'
    #     os.makedirs(save_dir, exist_ok=True)
    #     # 优化器
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #     # 学习率调度器
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    #     # 训练循环
    #     train_losses = []
    #     val_losses = []

    #     acc_watermarkss = [[] for _ in range(9)] if more_val else []
    #     for epoch in range(num_epochs):
    #         model.train()
    #         epoch_loss = 0.0
    #         for step, (repair_latents, repair_latents_prev, repair_t, repair_t_next, repair_damage,  message) in enumerate(train_dataloader):
    #             # 将数据移动到设备
    #             repair_latents = repair_latents.to(self.device)
    #             repair_latents_prev = repair_latents_prev.to(self.device)
    #             repair_t = repair_t.to(self.device)
    #             repair_t_next = repair_t_next.to(self.device)
    #             repair_damage = repair_damage.to(self.device)

    #             # 将时间步转换为整数timesteps (0-1000范围)
    #             timesteps = (repair_t * 1000).float()
                
    #             # 创建合适维度的context tensor
    #             batch_size = repair_latents.shape[0]
    #             context = torch.zeros(batch_size, 1, 768).to(self.device)
                
    #             # 前向传播
    #             outputs = model(repair_latents, timesteps, context=context)

    #             # 计算损失
    #             loss = torch.nn.functional.mse_loss(outputs, repair_latents_prev)
    #             epoch_loss += loss.item()
    #             # 反向传播
    #             loss.backward()
    #             if (step + 1) % accumulation_steps == 0:
    #                 # 更新参数
    #                 for param in model.parameters():
    #                     param.grad /= accumulation_steps

    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
    #         # 每个epoch结束后打印学习率
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    #         # 记录训练损失
    #         train_losses.append(epoch_loss / len(train_dataloader))
    #         # 验证阶段
    #         model.eval()
    #         val_loss = 0.0
    #         val_step_count = 0
    #         acc_watermark = 0.0
    #         with torch.no_grad():
    #             for val_step, (val_repair_latents, val_repair_latents_prev, val_repair_t, val_repair_t_next, val_repair_damage, val_messages) in enumerate(val_dataloader):
    #                 # 将数据移动到设备
    #                 val_repair_latents = val_repair_latents.to(self.device)
    #                 val_repair_latents_prev = val_repair_latents_prev.to(self.device)
    #                 val_repair_t = val_repair_t.to(self.device)
                    
    #                 # 将时间步转换为整数timesteps (0-1000范围)
    #                 val_timesteps = (val_repair_t * 1000).float()
                    
    #                 # 创建合适维度的context tensor
    #                 val_batch_size = val_repair_latents.shape[0]
    #                 val_context = torch.zeros(val_batch_size, 1, 768).to(self.device)
                    
    #                 # 前向传播
    #                 val_outputs = model(val_repair_latents, val_timesteps, context=val_context)
                    
    #                 # 计算损失
    #                 val_loss_step = torch.nn.functional.mse_loss(val_outputs, val_repair_latents_prev)
    #                 val_loss += val_loss_step.item()
    #                 val_step_count += 1

    #                 # 计算水印准确率
    #                 self.encoder_decoder.eval()
    #                 pred_img = autoencoder.module.decoder(val_outputs) if n_gpus > 1 else autoencoder.decoder(val_outputs)
    #                 pred_message = self.encoder_decoder.module.decode_from_image(pred_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(pred_img)
    #                 pred = (torch.sigmoid(pred_message) > 0.5).float()
    #                 message = (torch.sigmoid(val_messages) > 0.5).float()  # 假设val_messages是二进制的
    #                 acc = (pred == message).float().mean().item()
    #                 acc_watermark += acc
    #                 # 保存验证结果 - 添加在验证循环内部
    #                 if val_step == 0:  # 只保存第一个batch的结果作为示例
    #                     # 创建保存目录
    #                     val_save_dir = "/home/ldy/..workspace/zhou/repair/output_images/validation_results"
    #                     os.makedirs(val_save_dir, exist_ok=True)
    #                     # 使用autoencoder解码latent为图像进行可视化
    #                     n_save_val = min(8, val_outputs.shape[0])
    #                     for i in range(n_save_val):
    #                         # 解码预测结果
    #                         if n_gpus > 1:
    #                             input_img = autoencoder.module.decoder(val_repair_latents[i:i+1])
    #                             pred_img = autoencoder.module.decoder(val_outputs[i:i+1])
    #                             target_img = autoencoder.module.decoder(val_repair_latents_prev[i:i+1])
    #                             # clean_img = autoencoder.module.decoder(val_clean_images[i:i+1])
    #                         else:
    #                             input_img = autoencoder.decoder(val_repair_latents[i:i+1])
    #                             pred_img = autoencoder.decoder(val_outputs[i:i+1])
    #                             target_img = autoencoder.decoder(val_repair_latents_prev[i:i+1])
    #                             # clean_img = autoencoder.decoder(val_clean_images[i:i+1])

                            
    #                         # 逆标准化到[0,1]
    #                         pred_img = pred_img * 0.5 + 0.5
    #                         target_img = target_img * 0.5 + 0.5
    #                         input_img = input_img * 0.5 + 0.5
                            
    #                         # 拼接图像：输入-预测-目标
    #                         combined_img = torch.cat([input_img.squeeze(0), pred_img.squeeze(0), target_img.squeeze(0)], dim=2)

    #                         # 保存图像
    #                         save_path = os.path.join(val_save_dir, f'epoch_{epoch+1}_sample_{i}_input_pred_target.png')
    #                         save_image(combined_img.cpu(), save_path)
    #                     # # 保存预测结果和真实结果
    #                     # torch.save(val_outputs.cpu(), os.path.join(val_save_dir, f'val_outputs_epoch_{epoch+1}.pt'))
    #                     # torch.save(val_repair_latents_prev.cpu(), os.path.join(val_save_dir, f'val_targets_epoch_{epoch+1}.pt'))
                        
    #                     # # 可选：保存输入
    #                     # torch.save(val_repair_latents.cpu(), os.path.join(val_save_dir, f'val_inputs_epoch_{epoch+1}.pt'))
                        
    #                     # print(f"Saved validation results for epoch {epoch+1}")

    #         # 计算平均验证损失
    #         avg_val_loss = val_loss / val_step_count if val_step_count > 0 else 0.0
    #         avg_acc_watermark = [acc / val_step_count if val_step_count > 0 else 0.0 for acc in acc_watermark]
    #         val_losses.append(avg_val_loss)
    #         for i in range(len(avg_acc_watermark)):
    #             if i < len(acc_watermarkss):
    #                 acc_watermarkss[i].append(avg_acc_watermark[i])
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Watermark Accuracy: {avg_acc_watermark}")
    #         # 更新学习率
    #         scheduler.step()
    #         # 保存模型 - 仅保存top3模型
    #         if (epoch + 1) % save_interval == 0:
    #             model_save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
    #             torch.save(model.state_dict(), model_save_path)
                
    #             # 保持最多3个模型文件
    #             saved_models = sorted([f for f in os.listdir(save_dir) if f.startswith('model_epoch_') and f.endswith('.pth')])
    #             if len(saved_models) > 3:
    #                 # 删除最旧的模型
    #                 oldest_model = saved_models[0]
    #                 os.remove(os.path.join(save_dir, oldest_model))
    #                 print(f"Removed old model: {oldest_model}")
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(train_losses, label='Train Loss')
    #     plt.plot(val_losses, label='Validation Loss')
    #     # plt.plot(acc_watermarkss, label='Watermark Accuracy')
    #     plt.plot
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.title('Training and Validation Loss')
    #     plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    #     plt.figure(figsize=(10, 5))
    #     # plt.plot(train_losses, label='Train Loss')
    #     # plt.plot(val_losses, label='Validation Loss')
    #     plt.plot(acc_watermarkss, label='Watermark Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.legend()
    #     plt.title('Watermark Accuracy')
    #     plt.savefig(os.path.join(save_dir, 'watermark_accuracy.png'))

    #     # ---------------------利用去噪网络开始训练-----------------------------------
    #     # 训练完成后，保存模型
    #     model_save_path = "/home/ldy/..workspace/zhou/repair/models/denoise_model.pth"
    #     torch.save(model.state_dict(), model_save_path) 
    #     # repair_dataloader: (当前步向量, 上一步向量, 步编号, damage阶段, t),下面是dalaloader的详细情况
    #     '''
    #     当前步向量(repair_latents_tensor):
    #     当前插值时间步的 latent 向量(如第10步的向量)。

    #     上一步向量(repair_latents_prev_tensor):
    #     上一个时间步的 latent 向量（如第9步的向量）。

    #     步编号（repair_steps_tensor）：
    #     当前的时间步编号（如10，表示第10步），编号从2开始（clean为1，damage为最大编号）。

    #     damage阶段编号（repair_damage_tensor）：
    #     该插值序列属于哪个损坏阶段（如1、2、3）。

    #     插值比例 t（repair_t_tensor）：
    #     当前步在 clean 和 damage latent 之间的插值比例（0~1，越大越接近损坏端）。
    #     '''
    #     # 下一步，利用这个dataloader直接开始训练。
    #     # 训练输入为: 1.repair_latent_tensor 2.repair_steps_tensor
    #     # 训练输出为: 1.repair_latent_prev_tensor

    #     # 越靠近image，时间步越小(real image的时间步为0)
    #     # 越靠近fake image 时间步越大(例如damage_degree为3的latent，初始时间步为30)

    #     # todo
    #     # todo
    #     # todo: 在network class下构建类扩散模型的denoise_network，完成去噪
    #     # ---------------------利用去噪网络开始训练-----------------------------------
    

    def save_images(self, encoded_image: torch.Tensor, fake_image: torch.Tensor, batch_index: int):
        # 将图像从 [-1, 1] 还原到 [0, 1]
        # 更正：直接线性变换还原
        encoded_image = (encoded_image + 1) / 2
        fake_image = (fake_image + 1) / 2

        # 拼接图像：左边是encoded_image，右边是fake_image
        combined_image = torch.cat((encoded_image, fake_image), dim=3)  # 在宽度方向拼接

        # 创建保存路径
        save_dir = "/home/ldy/..workspace/zhou/repair/output_images"
        os.makedirs(save_dir, exist_ok=True)

        # 保存图像
        save_path = os.path.join(save_dir, f"batch_{batch_index}_combined.png")
        save_image(combined_image, save_path)
        print(f"Saved combined image to {save_path}")

    def save_images1(self, encoded_image: torch.Tensor, fake_image: torch.Tensor, batch_index: int, epoch: int):
        # 将图像从 [-1, 1] 还原到 [0, 1]
        # 更正：直接线性变换还原
        encoded_image = (encoded_image + 1) / 2
        encoded_image = encoded_image.clamp(0, 1)
        fake_image = (fake_image + 1) / 2
        fake_image = fake_image.clamp(0, 1)

        # 拼接图像：左边是encoded_image，右边是fake_image
        combined_image = torch.cat((encoded_image, fake_image), dim=3)  # 在宽度方向拼接

        # 创建保存路径
        save_dir = "/home/ldy/..workspace/zhou/repair/output_images"
        os.makedirs(save_dir, exist_ok=True)

        # 保存图像
        save_path = os.path.join(save_dir, f"batch_{epoch}_{batch_index}_combined.png")
        save_image(combined_image, save_path)
        print(f"Saved combined image to {save_path}")


    def train(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()
        self.discriminator.train()

        with torch.enable_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            train discriminator
            '''
            self.opt_discriminator.zero_grad()

            # RAW : target label for image should be "cover"(1)
            d_label_cover = self.discriminator(images)
            d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
            d_cover_loss.backward()

            # GAN : target label for encoded image should be "encoded"(0)
            d_label_encoded = self.discriminator(encoded_images.detach())
            d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
            d_encoded_loss.backward()

            self.opt_discriminator.step()

            '''
            train encoder and decoder
            '''
            self.opt_encoder_decoder.zero_grad()

            # GAN : target label for encoded image should be "cover"(0)
            g_label_decoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

            # RESULT : the decoded message should be similar to the raw message
            g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

            # full loss
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder

            g_loss.backward()
            self.opt_encoder_decoder.step()

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim_loss = SSIMLoss(window_size=5, reduction="mean")
            ssim = 1 - 2 * ssim_loss(encoded_images.detach(), images)

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder": g_loss_on_encoder,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": d_cover_loss,
            "d_encoded_loss": d_encoded_loss
        }
        return result

    def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.train()

        with torch.enable_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            train encoder and decoder
            '''
            self.opt_encoder_decoder.zero_grad()

            # RESULT : the decoded message should be similar to the raw message
            g_loss = self.criterion_MSE(decoded_messages, messages)

            g_loss.backward()
            self.opt_encoder_decoder.step()

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim(encoded_images.detach(), images, window_size=5, reduction="mean")

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": 0.,
            "g_loss_on_encoder": 0.,
            "g_loss_on_decoder": 0.,
            "d_cover_loss": 0.,
            "d_encoded_loss": 0.
        }
        return result

    def validation(self, images: torch.Tensor, messages: torch.Tensor):
        self.encoder_decoder.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # use device to compute
            images, messages = images.to(self.device), messages.to(self.device)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            '''
            validate discriminator
            '''
            # RAW : target label for image should be "cover"(1)
            d_label_cover = self.discriminator(images)
            d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

            # GAN : target label for encoded image should be "encoded"(0)
            d_label_encoded = self.discriminator(encoded_images.detach())
            d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

            '''
            validate encoder and decoder
            '''

            # GAN : target label for encoded image should be "cover"(0)
            g_label_decoded = self.discriminator(encoded_images)
            g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

            # RAW : the encoded image should be similar to cover image
            g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

            # RESULT : the decoded message should be similar to the raw message
            g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

            # full loss
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
                     self.decoder_weight * g_loss_on_decoder

            # psnr
            psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

            # ssim
            ssim_loss = SSIMLoss(window_size=5, reduction="mean")
            ssim = 1 - 2 * ssim_loss(encoded_images.detach(), images)

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

        result = {
            "error_rate": error_rate,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": g_loss_on_discriminator,
            "g_loss_on_encoder": g_loss_on_encoder,
            "g_loss_on_decoder": g_loss_on_decoder,
            "d_cover_loss": d_cover_loss,
            "d_encoded_loss": d_encoded_loss
        }

        return result, (images, encoded_images, noised_images, messages, decoded_messages)

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0.5)
        decoded_message = decoded_message.gt(0.5)
        error_rate = float(torch.sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator))

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def fake_data(self, dataloader: DataLoader, stargan_layer, autoencoder, message_length=64, compute_accuracy=False, clean_to_damage = False):
        n_gpus = torch.cuda.device_count()  # 添加这行来定义n_gpus
        batch_size = 16
        data_pair = []
        if clean_to_damage == True:
            for _, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="data preparing"):
                images = images.to(self.device)
                clean_message = torch.Tensor(np.random.choice([0,1], (images.shape[0], message_length))).to(self.device)
                with torch.no_grad():
                    encoded_image = self.encoder_decoder.module.encode_to_image(images, clean_message)  # 使用 .module 访问方法
                stargan_results = stargan_layer((encoded_image, images, labels))
            # 对比stargan_results的图片和原始图片的区别并打印
            torch.cuda.empty_cache()
            # 计算原始图片和各个转换阶段图片的差异
            for img, step in stargan_results:
                if step == 0:
                    # 计算原始编码图片和输入图片的差异
                    mse_diff = torch.nn.functional.mse_loss(img, images).item()
                    psnr_diff = kornia.losses.psnr_loss(img, images, max_val=2.0).item()
                    ssim_loss_fn = SSIMLoss(window_size=5, reduction="mean")
                    ssim_diff = 1 - 2 * ssim_loss_fn(img, images).item()
                    print(f"Step {step} (Clean) vs Original - MSE: {mse_diff:.6f}, PSNR: {psnr_diff:.4f}, SSIM: {ssim_diff:.4f}")
                    
                elif step == 1:
                    # 计算第一阶段转换图片和原始图片的差异
                    mse_diff = torch.nn.functional.mse_loss(img, images).item()
                    psnr_diff = kornia.losses.psnr_loss(img, images, max_val=2.0).item()
                    ssim_loss_fn = SSIMLoss(window_size=5, reduction="mean")
                    ssim_diff = 1 - 2 * ssim_loss_fn(img, images).item()
                    print(f"Step {step} (Transformed1) vs Original - MSE: {mse_diff:.6f}, PSNR: {psnr_diff:.4f}, SSIM: {ssim_diff:.4f}")
                    
                    # 还可以计算和clean图片的差异
                    if clean_images is not None:
                        mse_vs_clean = torch.nn.functional.mse_loss(img, clean_images).item()
                        psnr_vs_clean = kornia.losses.psnr_loss(img, clean_images, max_val=2.0).item()
                        ssim_vs_clean = 1 - 2 * ssim_loss_fn(img, clean_images).item()
                        print(f"Step {step} (Transformed1) vs Clean - MSE: {mse_vs_clean:.6f}, PSNR: {psnr_vs_clean:.4f}, SSIM: {ssim_vs_clean:.4f}")
                # Save some sample images for visualization
                if step <= 1:  # Only save for the first two steps to avoid too many images
                    save_dir = "/home/ldy/..workspace/zhou/repair/output_images/stargan_comparison"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Save a few sample images (first 4 from batch)
                    n_save = min(4, img.shape[0])

                    img_to_save     = self.denorm(img[:n_save].detach())
                    original_to_save = self.denorm(images[:n_save].detach())

                    for i in range(n_save):
                        # [C, H, W] 拼接到宽度维度 dim=2
                        combined = torch.cat([original_to_save[i], img_to_save[i]], dim=2)

                        save_path = os.path.join(save_dir, f"step_{step}_sample_{i}_comparison.png")
                        # save_image 期望输入已在 [0,1]，内部会乘以 255
                        save_image(combined.cpu(), save_path)        
                    
                    print(f"Saved {n_save} comparison images for step {step}")

                # 分别存储每个阶段的图像和message
                clean_images = []
                clean_messages = []
                transformed1_images = []
                transformed1_messages = []
                clean_bench_messages = []
                transformed1_bench_messages = []
                acc1 = 0.0
                image_total = []
                messages_list = []
                for img, step in stargan_results:
                    if step == 0:
                        clean_images.append(img)
                        clean_messages.append(clean_message)
                        clean_bench_messages.append(clean_message)

                    if step == 1:
                        transformed1_images.append(img)
                        msg1 = self.encoder_decoder.module.decode_from_image(img)
                        transformed1_messages.append(msg1)
                        transformed1_bench_messages.append(clean_message)
                # 合并batch
                damage_list = []
                bench_mess_list = []
                if clean_images:
                    clean_images = torch.cat(clean_images, dim=0)
                    clean_messages = torch.cat(clean_messages, dim=0)
                    clean_bench_messages = torch.cat(clean_bench_messages, dim=0)
                    image_total.append(clean_images)
                    messages_list.append(clean_messages)
                    bench_mess_list.append(clean_bench_messages)
                    damage_list.append(0)  # clean阶段的损坏程度为0
                if transformed1_images:
                    transformed1_images = torch.cat(transformed1_images, dim=0)
                    transformed1_messages = torch.cat(transformed1_messages, dim=0)
                    transformed1_bench_messages = torch.cat(transformed1_bench_messages, dim=0)
                    image_total.append(transformed1_images)
                    bench_mess_list.append(transformed1_bench_messages)
                    messages_list.append(transformed1_messages)
                    damage_list.append(1)  # transformed1阶段的损坏程度为1
                latent_list = []
                with torch.no_grad():
                    for idx, img in enumerate(image_total):
                        if n_gpus > 1:
                            latent = autoencoder.module.encoder(img)
                        else:
                            latent = autoencoder.encoder(img)
                        latent_list.append(latent)
                clean_latent = latent_list[0]
                # 构建 (clean_latent, damage_latent, damage_level) 的配对   
                clean_damage_pairs = []
                for i, damage_level in enumerate(damage_list[1:]):  # 跳过clean，从damage_list[1]开始
                    damage_latent = latent_list[damage_level]
                    mess = messages_list[damage_level]
                    clean_damage_pairs.append((clean_latent, damage_latent, damage_level, mess, bench_mess_list[damage_level]))
                for clean_latent, damage_latent, damage_level, message, bench_message in clean_damage_pairs:
                        current_latents = damage_latent
                        next_latents = clean_latent
                        current_step = 0
                        next_step = 10
                        bench_message = bench_message
                        for idx, (current_latent, next_latent, clean_mess) in enumerate(zip(current_latents, next_latents, bench_message)):
                            # 在这个位置把batch消掉
                            # 组合 (当前步latent, 下一步latent, 当前步t, 下一步t, damage_level)
                            # 确保每个样本对应正确的clean_image和message
                            data_pair.append((current_latent, next_latent, current_step, next_step, damage_level,torch.tensor(0, dtype=torch.long),clean_mess))
        else:
            for _, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="data preparing"):
            # for _, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                clean_message = torch.Tensor(np.random.choice([0,1], (images.shape[0], message_length))).to(self.device)
                with torch.no_grad():
                    encoded_image = self.encoder_decoder.module.encode_to_image(images, clean_message)  # 使用 .module 访问方法
                stargan_results = stargan_layer((encoded_image, images, labels))
            
                # 分别存储每个阶段的图像和message
                clean_images = []
                clean_messages = []
                transformed1_images = []
                transformed1_messages = []
                clean_bench_messages = []
                transformed1_bench_messages = []
                acc1 = 0.0
                image_total = []
                messages_list = []
                for img, step in stargan_results:
                    if step == 0:
                        clean_images.append(img)
                        clean_messages.append(clean_message)
                        clean_bench_messages.append(clean_message)

                    if step == 1:
                        transformed1_images.append(img)
                        msg1 = self.encoder_decoder.module.decode_from_image(img)
                        transformed1_messages.append(msg1)
                        transformed1_bench_messages.append(clean_message)
                # 合并batch
                damage_list = []
                bench_mess_list = []
                if clean_images:
                    clean_images = torch.cat(clean_images, dim=0)
                    clean_messages = torch.cat(clean_messages, dim=0)
                    clean_bench_messages = torch.cat(clean_bench_messages, dim=0)
                    image_total.append(clean_images)
                    messages_list.append(clean_messages)
                    bench_mess_list.append(clean_bench_messages)
                    damage_list.append(0)  # clean阶段的损坏程度为0
                if transformed1_images:
                    transformed1_images = torch.cat(transformed1_images, dim=0)
                    transformed1_messages = torch.cat(transformed1_messages, dim=0)
                    transformed1_bench_messages = torch.cat(transformed1_bench_messages, dim=0)
                    image_total.append(transformed1_images)
                    bench_mess_list.append(transformed1_bench_messages)
                    messages_list.append(transformed1_messages)
                    damage_list.append(1)  # transformed1阶段的损坏程度为1
                    acc1 += 1 - self.decoded_message_error_rate_batch(transformed1_messages, transformed1_bench_messages)
                    # print(1 - self.decoded_message_error_rate_batch(transformed1_messages, transformed1_bench_messages))
                # 合并所有图像和消息
                # print(f"Total images: {len(image_total)}, Total messages: {len(messages_list)}")
                # print(f"damage1baseline: {acc1 / len(transformed1_images) if len(transformed1_images) > 0 else 0}")

                # 将他们投影到latent space中去
                latent_list = []
                with torch.no_grad():
                    for idx, img in enumerate(image_total):
                        if n_gpus > 1:
                            latent = autoencoder.module.encoder(img)
                        else:
                            latent = autoencoder.encoder(img)
                        latent_list.append(latent)

                clean_latent = latent_list[0]
                # 构建 (clean_latent, damage_latent, damage_level) 的配对   
                clean_damage_pairs = []
                for i, damage_level in enumerate(damage_list[1:]):  # 跳过clean，从damage_list[1]开始
                    damage_latent = latent_list[damage_level]
                    mess = messages_list[damage_level]
                    clean_damage_pairs.append((clean_latent, damage_latent, damage_level, mess, bench_mess_list[damage_level]))
                # 插值阶段
                num_interp = {1:10, 2:20, 3:30}  # 可自定义每个阶段插值数量
                for clean_latent, damage_latent, damage_level, message, bench_message in clean_damage_pairs:
                    # 生成编号序列：clean为1，damage为num_interp[damage]，中间插值点依次编号
                    total_steps = num_interp[damage_level]
                    interpolated_latents = []
                    interpolated_steps = []
                    for idx, t in enumerate(torch.linspace(0, 1, total_steps+2)):
                        interp_latent = self.slerp(t, clean_latent, damage_latent)
                        step_id = idx + 1
                        interpolated_latents.append(interp_latent)
                        interpolated_steps.append((damage_level, step_id, float(t), bench_message))

                    # 将同一damage_level的相邻step组合
                    for j in range(len(interpolated_latents) - 1):
                        current_latents = interpolated_latents[j]
                        next_latents = interpolated_latents[j + 1]
                        current_step = interpolated_steps[j][2]
                        next_step = interpolated_steps[j + 1][2]
                        bench_message = interpolated_steps[j][3]
                        for idx, (current_latent, next_latent, clean_mess) in enumerate(zip(current_latents, next_latents, bench_message)):
                            # 在这个位置把batch消掉
                            # 组合 (当前步latent, 下一步latent, 当前步t, 下一步t, damage_level)
                            # 确保每个样本对应正确的clean_image和message
                            data_pair.append((current_latent, next_latent, current_step, next_step, damage_level,torch.tensor(interpolated_steps[j][1], dtype=torch.long),clean_mess))
            # 将data_pair转换为Tensor并创建DataLoader
        repair_latents_tensor = torch.stack([pair[0] for pair in data_pair])
        repair_latents_prev_tensor = torch.stack([pair[1] for pair in data_pair])
        repair_t_tensor = torch.tensor([pair[2] for pair in data_pair], dtype=torch.float32)
        repair_t_next_tensor = torch.tensor([pair[3] for pair in data_pair], dtype=torch.float32)
        repair_damage_tensor = torch.tensor([pair[4] for pair in data_pair], dtype=torch.long)
        repair_step = torch.stack([pair[5] for pair in data_pair])
        repair_clean_message = torch.stack([pair[6] for pair in data_pair])

        print(f"repair_latents_tensor shape: {repair_latents_tensor.shape}")
        print(f"repair_latents_prev_tensor shape: {repair_latents_prev_tensor.shape}")
        print(f"repair_t_tensor shape: {repair_t_tensor.shape}")
        print(f"repair_t_next_tensor shape: {repair_t_next_tensor.shape}")
        print(f"repair_damage_tensor shape: {repair_step.shape}")

        # 创建TensorDataset和DataLoader
        repair_dataset = TensorDataset(
            repair_latents_tensor, 
            repair_latents_prev_tensor, 
            repair_t_tensor,
            repair_t_next_tensor, 
            repair_damage_tensor,
            repair_step,
            repair_clean_message
        )
        repair_dataloader = DataLoader(repair_dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # 减少批次大小

        # 可选的水印准确率计算（需要大量内存）
        if compute_accuracy:
            max_acc = 0.0
            print("Computing watermark accuracy on a sample of data...")
            acc_watermark = [0.0] * 50  # 增加大小以适应更大的step值
            clean_acc_watermark = [0.0] * 50  # 用于记录clean阶段的准确率
            water_num = [0] * 50  # 用于记录每个step的样本数量
            for (repair_latents_tensor, 
                repair_latents_prev_tensor, 
                repair_t_tensor,
                repair_t_next_tensor, 
                repair_damage_tensor,
                repair_step,
                repair_clean_message) in tqdm(repair_dataloader, desc="watermark accuracy"):
                    

                with torch.no_grad():
                        # 清除缓存
                        torch.cuda.empty_cache()
                        
                        repair_current_img = autoencoder.module.decoder(repair_latents_tensor) if n_gpus > 1 else autoencoder.decoder(repair_latents_tensor)
                        repair_prev_img = autoencoder.module.decoder(repair_latents_prev_tensor) if n_gpus > 1 else autoencoder.decoder(repair_latents_prev_tensor)
                        
                        mess_curr = self.encoder_decoder.module.decode_from_image(repair_current_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(repair_current_img)
                        mess_prev = self.encoder_decoder.module.decode_from_image(repair_prev_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(repair_prev_img)
                        acc_clean = 1.0 - self.decoded_message_error_rate_batch(repair_clean_message, mess_curr)
                        acc = 1.0 - self.decoded_message_error_rate_batch(repair_clean_message, mess_prev)
                        max_acc = max(max_acc, acc)
                        max_acc = max(max_acc, acc_clean)
                        # print(f"Current batch accuracy: {acc:.4f}, Clean accuracy: {
                        # print (f"Current batch accuracy: {acc:.4f}")
                        step_idx = repair_step[0].item()
                        # print(f"Processing step {step_idx}, accuracy: {acc:.4f}, clean accuracy: {acc_clean:.4f}")
                        # 确保索引在有效范围内
                        if 0 <= step_idx < len(acc_watermark):
                            acc_watermark[step_idx] += acc
                            water_num[step_idx] += 1
                        if 0 <= step_idx < len(clean_acc_watermark):
                            clean_acc_watermark[step_idx] += acc_clean
                            # water_num[step_idx] += 1
                        # 清理中间变量
                        del repair_current_img, repair_prev_img, mess_curr, mess_prev
                        torch.cuda.empty_cache()
                

            
            # 计算平均准确率（基于样本数据）
            print(f"Watermark accuracy computed on {len(repair_dataloader)} sample batches:")
            for i in range(1, min(len(acc_watermark), 32)):  # 打印有效的步数范围
                if len(repair_dataloader) > 0:
                    acc_watermark[i] /= (water_num[i] if water_num[i] > 0 else 1.0)  # 避免除以0
                if acc_watermark[i] > 0:  # 只打印有数据的步数
                    print(f"Watermark Accuracy at step {i}: {acc_watermark[i]:.4f}")
            print("Clean Watermark Accuracy:")
            for i in range(1, min(len(clean_acc_watermark), 32)):
                if len(repair_dataloader) > 0:
                    clean_acc_watermark[i] /= (water_num[i] if water_num[i] > 0 else 1.0)
                if clean_acc_watermark[i] > 0:  # 只打印有数据的步数
                    print(f"Clean Watermark Accuracy at step {i}: {clean_acc_watermark[i]:.4f}")
            print(f"Max Watermark Accuracy: {max_acc:.4f}")
        else:
            print("Skipping watermark accuracy computation to save memory.")
        
        # 清理内存
        torch.cuda.empty_cache()
        
        return repair_dataloader
    
    def watermark_accuracy(self, dataloader: DataLoader, autoencoder):
        acc_watermark = [0.0] * 10
        for _,(repair_latents_tensor, 
            repair_latents_prev_tensor, 
            repair_t_tensor,
            repair_t_next_tensor, 
            repair_damage_tensor,
            repair_step) in tqdm(enumerate(dataloader), desc="watermark accuracy"):
            repair_current_img = autoencoder.module.decoder(repair_latents_tensor) if n_gpus > 1 else autoencoder.decoder(repair_latents_tensor)
            repair_prev_img = autoencoder.module.decoder(repair_latents_prev_tensor) if n_gpus > 1 else autoencoder.decoder(repair_latents_prev_tensor)
            mess_curr = self.encoder_decoder.module.decode_from_image(repair_current_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(repair_current_img)
            mess_prev = self.encoder_decoder.module.decode_from_image(repair_prev_img) if n_gpus > 1 else self.encoder_decoder.decode_from_image(repair_prev_img)
            acc = 1.0 - self.decoded_message_error_rate_batch(mess_curr, mess_prev)
            acc_watermark[repair_step[0].item()] += acc
        for i in range(1,10):
            acc_watermark[i] /= len(dataloader) if len(dataloader) > 0 else 1.0
            print(f"Watermark Accuracy at step {i}: {acc_watermark[i]:.4f}")


    






    



from torch.utils.data import DataLoader
from torchvision import transforms as T
from data_loader import CelebA
from utils.load_train_setting import *
from network.Network import *
# from network.Newwork import *
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
from network.vae import VQModel,AutoencoderKL

train_image_dir = "/home/ldy/..workspace/zhou/repair/latent-diffusion/data/CelebA-train/images"
val_image_dir = "/home/ldy/..workspace/zhou/repair/latent-diffusion/data/CelebA-val/images"
train_attr_path = "/home/ldy/..workspace/zhou/repair/latent-diffusion/data/CelebA-train/list_attr_celeba5001-15000.txt"
val_attr_path = "/home/ldy/..workspace/zhou/repair/latent-diffusion/data/CelebA-val/list_attr_celeba5000.txt"

selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
transform = T.Compose([
    T.CenterCrop(178), 
    T.Resize(128),     
    T.ToTensor(),     
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpus = torch.cuda.device_count()
if n_gpus > 1:
    print(f"Using {n_gpus} GPUs for DataParallel.")
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, only_decoder)
from network.ldm.models.autoencoder import VQModelInterface

if __name__ == "__main__":
    train_dataset = CelebA(
        image_dir=train_image_dir,
        attr_path=train_attr_path,
        selected_attrs=selected_attrs,
        transform=transform
    )
    # 将数据集规模减半
    train_subset_size = len(train_dataset)
    train_indices = list(range(train_subset_size))
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = CelebA(
        image_dir=val_image_dir,
        attr_path=val_attr_path,
        selected_attrs=selected_attrs,
        transform=transform
    )
    val_subset_size = len(val_dataset)//8
    val_indices = list(range(val_subset_size))
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_dataloader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)

    stargan_layer = StarGAN_multi_step(
            g_conv_dim=64, c_dim=5, g_repeat_num=6, device=device,
            model_path="/home/ldy/..workspace/zhou/repair/models/200000-G.ckpt"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs for DataParallel.")
    network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, only_decoder)
    network.train_denoise1(
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        more_val=True  # 设置为True以进行更多验证
    )