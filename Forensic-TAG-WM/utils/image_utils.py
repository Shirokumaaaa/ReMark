import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import random
from tampers.logo_putting.logo_putting import *

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img, seed, args):
    tamper_loc = None
    if args.jpeg_ratio is not None:
        img.save(f"tmp_{args.jpeg_ratio}.jpg", quality=args.jpeg_ratio)
        img = Image.open(f"tmp_{args.jpeg_ratio}.jpg")

    if args.logo_putting_num is not None:
        set_random_seed(seed)
        img = transforms.ToTensor()(img).unsqueeze(0)
        logo_instance = Logo(args)
        # Apply logo distortion
        img_tensor, ground_truth = logo_instance(img)
        # Convert tensor back to PIL image
        img = Image.fromarray(((img_tensor.squeeze(0).permute(1, 2, 0)).cpu().numpy() * 255).astype(np.uint8))
        if args.return_tamper_loc:
            tamper_loc = ground_truth.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy().astype(np.uint8)

    if args.random_crop_ratio is not None:
        set_random_seed(seed)
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * args.random_crop_ratio)
        new_height = int(height * args.random_crop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        end_x = start_x + new_width
        end_y = start_y + new_height
        
        padded_image = np.zeros_like(img) # black padding: normal
        # padded_image = (np.ones_like(img) * 255).astype(np.uint8) # white padding
        # padded_image = np.repeat((np.random.randint(0, 2, (width, height)) * 255)[:, :, np.newaxis], 3, axis=2).astype(np.uint8) # black or white padding
        # padded_image = np.random.normal(loc=127.5, scale=50, size=(width, height, c)).astype(np.uint8) # gaussian noise padding
        # padded_image = (np.random.random((width, height, c)) * 255).astype(np.uint8) # float uniform noise padding
        
        # choice = np.random.randint(0, 5)
        # if choice == 0:
        #     padded_image = np.zeros_like(img) # black padding: normal
        # elif choice == 1:
        #     padded_image = np.ones_like(img) * 255 # white padding: normal
        # elif choice == 2:
        #     padded_image = np.repeat((np.random.randint(0, 2, (width, height)) * 255)[:, :, np.newaxis], 3, axis=2).astype(np.uint8) # black or white padding
        # elif choice == 3:
        #     padded_image = np.random.normal(loc=127.5, scale=50, size=(width, height, c)).astype(np.uint8) # gaussian noise padding
        # elif choice == 4:
        #     padded_image = (np.random.random((width, height, c)) * 255).astype(np.uint8) # float uniform noise padding


        padded_image[start_y:end_y, start_x:end_x] = img[start_y:end_y, start_x:end_x]
        img = Image.fromarray(padded_image)
        if args.return_tamper_loc:
            tamper_loc = np.ones_like(img)
            tamper_loc[start_y:end_y, start_x:end_x] = 0

    if args.random_drop_ratio is not None:
        set_random_seed(seed)
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * args.random_drop_ratio)
        new_height = int(height * args.random_drop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        padded_image = np.zeros_like(img[start_y:start_y + new_height, start_x:start_x + new_width])
        img[start_y:start_y + new_height, start_x:start_x + new_width] = padded_image
        img = Image.fromarray(img)
        if args.return_tamper_loc:
            tamper_loc = np.zeros_like(img)
            tamper_loc[start_y:start_y + new_height, start_x:start_x + new_width] = 1

    if args.resize_ratio is not None:
        img_shape = np.array(img).shape
        resize_size = int(img_shape[0] * args.resize_ratio)
        img = transforms.Resize(size=resize_size)(img)
        img = transforms.Resize(size=img_shape[0])(img)

    if args.gaussian_blur_r is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.median_blur_k is not None:
        img = img.filter(ImageFilter.MedianFilter(args.median_blur_k))


    if args.gaussian_std is not None:
        img_shape = np.array(img).shape
        # g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = np.random.normal(0, args.gaussian_std, img_shape)
        g_noise = g_noise.astype(np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))

    if args.sp_prob is not None:
        c,h,w = np.array(img).shape
        prob_zero = args.sp_prob / 2
        prob_one = 1 - prob_zero
        rdn = np.random.rand(c,h,w)
        img = np.where(rdn > prob_one, np.zeros_like(img), img)
        img = np.where(rdn < prob_zero, np.ones_like(img)*255, img)
        img = Image.fromarray(img)

    if args.brightness_factor is not None:
        img = transforms.ColorJitter(brightness=args.brightness_factor)(img)

    if args.vertical_shift_ratio is not None or args.horizontal_shift_ratio is not None:
        width, height = img.size
        tx = int(args.horizontal_shift_ratio * width) if args.horizontal_shift_ratio is not None else 0
        ty = int(args.vertical_shift_ratio * height) if args.vertical_shift_ratio is not None else 0
        img = transforms.functional.affine(
            img, 
            angle=0, 
            translate=(tx, ty), 
            scale=1.0, 
            shear=0, 
            fill=0
        )
    
    if args.return_tamper_loc:
        if tamper_loc is None:
            tamper_loc = np.zeros_like(img)
        return img, tamper_loc

    else:
        return img


def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)
