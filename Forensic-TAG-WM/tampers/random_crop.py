import os
import cv2
import numpy as np
import random

def random_crop(image, crop_ratio):
    h, w, _ = image.shape
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    crop = image[top:top + new_h, left:left + new_w]
    return crop, top, left, new_h, new_w

def apply_padding(image, crop, h, w, top, left, new_h, new_w, method):
    padded_image = np.zeros((h, w, 3), dtype=np.uint8)
    if method == 'white':
        padded_image.fill(255)
    elif method == 'gaussian':
        padded_image = np.random.normal(128, 50, (h, w, 3)).astype(np.uint8)
    elif method == 'uniform':
        padded_image = np.random.uniform(0, 255, (h, w, 3)).astype(np.uint8)
    elif method == 'flip':
        padded_image = np.flip(image, axis=random.choice([0, 1]))
    padded_image[top:top + new_h, left:left + new_w] = crop
    return padded_image

def create_mask(h, w, top, left, new_h, new_w):
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[top:top + new_h, left:left + new_w] = 0
    return mask

def process_images(input_path, img_output_path, mask_output_path, crop_ratios, padding_methods):
    num = 0
    for ratio in crop_ratios:
        ratio_img_path = os.path.join(img_output_path, str(ratio))
        ratio_mask_path = os.path.join(mask_output_path, str(ratio))
        if not os.path.exists(ratio_img_path):
            os.makedirs(ratio_img_path)
        if not os.path.exists(ratio_mask_path):
            os.makedirs(ratio_mask_path)
            
        for image_name in os.listdir(input_path):
            image_path = os.path.join(input_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            crop, top, left, new_h, new_w = random_crop(image, ratio)
            
            method = random.choice(padding_methods)
            padded_image = apply_padding(image, crop, image.shape[0], image.shape[1], top, left, new_h, new_w, method)
            mask = create_mask(image.shape[0], image.shape[1], top, left, new_h, new_w)
            
            output_image_path = os.path.join(ratio_img_path, image_name)
            output_mask_path = os.path.join(ratio_mask_path, image_name)
            
            cv2.imwrite(output_image_path, padded_image)
            cv2.imwrite(output_mask_path, mask)
            
            print(f'save tampered img to {output_image_path}')
            print(f'save mask to {output_mask_path}')

            num += 1
            print('croped_num = ', num)
            
if __name__ == '__main__':
    # basic settings
    # root_dir = '/public/chenyuzhuo/models/image_watermarking_models/Gaussian-Shading/datasets/RTLO_dataset/TLT_center50/Stable-Diffusion-Prompts/Stable-Diffusion-2-1-base/inference_step50'
    root_dir = '/public/chenyuzhuo/models/image_watermarking_models/Gaussian-Shading/datasets/RTLO_dataset/TLT_intervals4/Stable-Diffusion-Prompts/Stable-Diffusion-2-1-base/inference_step50'
    second_names = ['DDIM', 'DEIS', 'DPMSolver', 'PNDM', 'UniPC']
    # third_names = ['512x512', '512x1024', '768x768', '768x1024', '1024x1024']
    third_names = ['512x512']
    fourth_input_name = 'images'
    fourth_output_img_name = 'tampered_images/RandomCrop'
    fourth_out_put_mask_name = 'tamper_loc_masks/RandomCrop'

    for second_name in second_names:
        for third_name in third_names:
            # if second_name == 'DPMSolver' and third_name == '512x512':
            #     continue
            input_path = os.path.join(root_dir, second_name, third_name, fourth_input_name)
            img_output_path = os.path.join(root_dir, second_name, third_name, fourth_output_img_name)
            mask_output_path = os.path.join(root_dir, second_name, third_name, fourth_out_put_mask_name)

            # settings for random crop
            crop_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            padding_methods = ['black', 'white', 'gaussian', 'uniform', 'flip']
            process_images(input_path, img_output_path, mask_output_path, crop_ratios, padding_methods)