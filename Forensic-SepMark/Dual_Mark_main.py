import yaml
from easydict import EasyDict
import os
import time
from shutil import copyfile
import random
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from network.Dual_Mark import *
from utils import *


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
    torch.backends.cudnn.enabled = True


def main():

    seed_torch(42) # it doesnot work if the mode of F.interpolate is "bilinear"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('cfg/train_DualMark.yaml', 'r') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

    project_name = args.project_name
    epoch_number = int(os.environ.get("SEPMARK_EPOCHS", str(args.epoch_number)))
    batch_size = int(os.environ.get("SEPMARK_BATCH_SIZE", str(args.batch_size)))
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
    noise_layers_R = args.noise_layers.pool_R
    noise_layers_F = args.noise_layers.pool_F

    project_name += "_" + str(image_size) + "_" + str(message_length) + "_" + str(message_range) + "_" + str(lr) + "_" + str(beta1) + "_" + attention_encoder + "_" + attention_decoder
    for i in weight:
        project_name += "_" +  str(i)
    run_tag = os.environ.get("SEPMARK_RUN_TAG", "").strip()
    if run_tag:
        project_name += "_" + run_tag
    result_folder = "results/" + time.strftime(project_name + "_%Y_%m_%d_%H_%M_%S", time.localtime()) + "/"
    if not os.path.exists(result_folder): os.mkdir(result_folder)
    if not os.path.exists(result_folder + "images/"): os.mkdir(result_folder + "images/")
    if not os.path.exists(result_folder + "models/"): os.mkdir(result_folder + "models/")
    copyfile("cfg/train_DualMark.yaml", result_folder + "train_DualMark.yaml")
    writer = SummaryWriter('runs/'+ project_name + time.strftime("%_Y_%m_%d__%H_%M_%S", time.localtime()))

    network = Network(message_length, noise_layers_R, noise_layers_F, device, batch_size, lr, beta1, attention_encoder, attention_decoder, weight)
    init_ec = os.environ.get("SEPMARK_INIT_EC", "").strip()
    init_d = os.environ.get("SEPMARK_INIT_D", "").strip()
    if init_ec and os.path.exists(init_ec):
        network.load_model_ed(init_ec)
        print(f"[Init] Loaded encoder-decoder from {init_ec}")
    if init_d and os.path.exists(init_d):
        network.load_model_dis(init_d)
        print(f"[Init] Loaded discriminator from {init_d}")

    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    train_csv = os.path.join(dataset_path, "train.csv")
    val_csv = os.path.join(dataset_path, "val.csv")

    train_dataset = attrsImgDataset(train_dir, image_size, "celebahq", csv_path=train_csv)
    #train_dataset = maskImgDataset(os.path.join(dataset_path, "train_" + str(image_size)), image_size)

    val_dataset = attrsImgDataset(val_dir, image_size, "celebahq", csv_path=val_csv)
    #val_dataset = maskImgDataset(os.path.join(dataset_path, "val_" + str(image_size)), image_size)
    data_fraction = float(os.environ.get("SEPMARK_DATA_FRACTION", "1.0"))
    data_fraction = max(0.0, min(1.0, data_fraction))
    if data_fraction < 1.0:
        rng = np.random.RandomState(42)
        train_keep = max(1, int(len(train_dataset) * data_fraction))
        val_keep = max(1, int(len(val_dataset) * data_fraction))
        train_indices = rng.choice(len(train_dataset), size=train_keep, replace=False)
        val_indices = rng.choice(len(val_dataset), size=val_keep, replace=False)
        train_dataset = Subset(train_dataset, train_indices.tolist())
        val_dataset = Subset(val_dataset, val_indices.tolist())

    print(f"[Data] train={len(train_dataset)} val={len(val_dataset)} fraction={data_fraction} batch={batch_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStart training : \n\n")

    max_train_steps = int(os.environ.get("SEPMARK_MAX_TRAIN_STEPS", "0"))
    max_val_steps = int(os.environ.get("SEPMARK_MAX_VAL_STEPS", "0"))
    history_csv = result_folder + "history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "phase", "g_loss", "error_rate_C", "error_rate_R", "error_rate_F",
            "psnr", "ssim", "g_loss_on_discriminator", "g_loss_on_encoder_MSE",
            "g_loss_on_encoder_LPIPS", "g_loss_on_decoder_C", "g_loss_on_decoder_R",
            "g_loss_on_decoder_F", "d_loss"
        ])

    for epoch in range(1, epoch_number + 1):

        running_result = {
            "g_loss": 0.0,
            "error_rate_C": 0.0,
            "error_rate_R": 0.0,
            "error_rate_F": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder_MSE": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder_C": 0.0,
            "g_loss_on_decoder_R": 0.0,
            "g_loss_on_decoder_F": 0.0,
            "d_loss": 0.0
        }

        start_time = time.time()

        '''
        train
        '''
        train_total = len(train_dataloader) if max_train_steps <= 0 else min(len(train_dataloader), max_train_steps)
        train_bar = tqdm(enumerate(train_dataloader, 1), total=train_total, desc=f"Train E{epoch}", leave=False)
        for step, (image, mask) in train_bar:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

            result = network.train(image, message, mask)
            train_bar.set_postfix(
                g_loss=f"{float(result['g_loss']):.4f}",
                berC=f"{float(result['error_rate_C']):.4f}",
                berR=f"{float(result['error_rate_R']):.4f}",
                berF=f"{float(result['error_rate_F']):.4f}"
            )

            for key in result:
                print(key, float(result[key]))
                writer.add_scalar("Train/" + key, float(result[key]), (epoch - 1) * len(train_dataloader) + step)
                running_result[key] += float(result[key])
            if max_train_steps > 0 and step >= max_train_steps:
                break

        '''
        train results
        '''
        content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
        for key in running_result:
            content += key + "=" + str(running_result[key] / step) + ","
            writer.add_scalar("Train_epoch/" + key, float(running_result[key] / step), epoch)
        content += "\n"

        with open(result_folder + "/train_log.txt", "a") as file:
            file.write(content)
        print(content)
        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, "train", running_result["g_loss"] / step, running_result["error_rate_C"] / step,
                running_result["error_rate_R"] / step, running_result["error_rate_F"] / step,
                running_result["psnr"] / step, running_result["ssim"] / step,
                running_result["g_loss_on_discriminator"] / step, running_result["g_loss_on_encoder_MSE"] / step,
                running_result["g_loss_on_encoder_LPIPS"] / step, running_result["g_loss_on_decoder_C"] / step,
                running_result["g_loss_on_decoder_R"] / step, running_result["g_loss_on_decoder_F"] / step,
                running_result["d_loss"] / step
            ])

        '''
        validation
        '''

        val_result = {
            "g_loss": 0.0,
            "error_rate_C": 0.0,
            "error_rate_R": 0.0,
            "error_rate_F": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "g_loss_on_discriminator": 0.0,
            "g_loss_on_encoder_MSE": 0.0,
            "g_loss_on_encoder_LPIPS": 0.0,
            "g_loss_on_decoder_C": 0.0,
            "g_loss_on_decoder_R": 0.0,
            "g_loss_on_decoder_F": 0.0,
            "d_loss": 0.0
        }

        start_time = time.time()

        val_steps = len(val_dataloader)
        save_n = min(save_images_number, val_steps)
        if save_n > 0:
            saved_iterations = np.random.choice(np.arange(1, val_steps + 1), size=save_n, replace=False)
        else:
            saved_iterations = np.array([], dtype=np.int64)
        saved_all = None

        val_total = len(val_dataloader) if max_val_steps <= 0 else min(len(val_dataloader), max_val_steps)
        val_bar = tqdm(enumerate(val_dataloader, 1), total=val_total, desc=f"Val E{epoch}", leave=False)
        for step, (image, mask) in val_bar:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([-message_range, message_range], (image.shape[0], message_length))).to(device)

            result, (images, encoded_images, noised_images) = network.validation(image, message, mask)
            val_bar.set_postfix(
                g_loss=f"{float(result['g_loss']):.4f}",
                berC=f"{float(result['error_rate_C']):.4f}",
                berR=f"{float(result['error_rate_R']):.4f}",
                berF=f"{float(result['error_rate_F']):.4f}"
            )
            for key in result:
                print(key, float(result[key]))
                writer.add_scalar("Val/" + key, float(result[key]), (epoch - 1) * len(val_dataloader) + step)
                val_result[key] += float(result[key])

            if step in saved_iterations:
                if saved_all is None:
                    saved_all = get_random_images(image, encoded_images, noised_images)
                else:
                    saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)
            if max_val_steps > 0 and step >= max_val_steps:
                break

        if saved_all is not None:
            save_images(saved_all, epoch, result_folder + "images/", resize_to=None)

        '''
        validation results
        '''
        content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
        for key in val_result:
            content += key + "=" + str(val_result[key] / step) + ","
            writer.add_scalar("Val_epoch/" + key, float(val_result[key] / step), epoch)
        content += "\n"

        with open(result_folder + "/val_log.txt", "a") as file:
            file.write(content)
        print(content)
        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, "val", val_result["g_loss"] / step, val_result["error_rate_C"] / step,
                val_result["error_rate_R"] / step, val_result["error_rate_F"] / step,
                val_result["psnr"] / step, val_result["ssim"] / step,
                val_result["g_loss_on_discriminator"] / step, val_result["g_loss_on_encoder_MSE"] / step,
                val_result["g_loss_on_encoder_LPIPS"] / step, val_result["g_loss_on_decoder_C"] / step,
                val_result["g_loss_on_decoder_R"] / step, val_result["g_loss_on_decoder_F"] / step,
                val_result["d_loss"] / step
            ])

        '''
        save model
        '''
        path_model = result_folder + "models/"
        path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
        path_discriminator = path_model + "D_" + str(epoch) + ".pth"
        network.save_model(path_encoder_decoder, path_discriminator)

    writer.close()


if __name__ == '__main__':
    main()
