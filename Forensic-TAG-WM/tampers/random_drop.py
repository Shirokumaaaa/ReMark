import os
import cv2
import numpy as np
import random

def random_drop(image, drop_ratio, padding_method):
    h, w, _ = image.shape
    new_h, new_w = int(h * drop_ratio), int(w * drop_ratio)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    drop = image[top:top + new_h, left:left + new_w]
    if padding_method == 'black':
        drop.fill(0)
    if padding_method == 'white':
        drop.fill(255)
    elif padding_method == 'gaussian':
        drop = np.random.normal(128, 50, (new_h, new_w, 3)).astype(np.uint8)
    elif padding_method == 'uniform':
        drop = np.random.uniform(0, 255, (new_h, new_w, 3)).astype(np.uint8)
    elif padding_method == 'flip':
        drop = np.flip(drop, axis=random.choice([0, 1]))
    image[top:top + new_h, left:left + new_w] = drop
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top:top + new_h, left:left + new_w] = 255
    
    return image, mask


def process_images(input_path, img_output_path, mask_output_path, drop_ratios, padding_methods):
    num = 0
    for ratio in drop_ratios:
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

            method = random.choice(padding_methods)
            droped_image, mask = random_drop(image, ratio, method)
        
            output_image_path = os.path.join(ratio_img_path, image_name)
            output_mask_path = os.path.join(ratio_mask_path, image_name)
            
            cv2.imwrite(output_image_path, droped_image)
            cv2.imwrite(output_mask_path, mask)
            
            print(f'save tampered img to {output_image_path}')
            print(f'save mask to {output_mask_path}')

            num += 1
            print('droped_num = ', num)
            
if __name__ == '__main__':
    # basic settings
    # root_dir = '/public/chenyuzhuo/models/image_watermarking_models/Gaussian-Shading/datasets/RTLO_dataset/TLT_center50/Stable-Diffusion-Prompts/Stable-Diffusion-2-1-base/inference_step50'
    root_dir = '/public/chenyuzhuo/models/image_watermarking_models/Gaussian-Shading/datasets/RTLO_dataset/TLT_intervals4/Stable-Diffusion-Prompts/Stable-Diffusion-2-1-base/inference_step50'
    second_names = ['DDIM', 'DEIS', 'DPMSolver', 'PNDM', 'UniPC']
    # third_names = ['512x512', '512x1024', '768x768', '768x1024', '1024x1024']
    third_names = ['512x512']
    fourth_input_name = 'images'
    fourth_output_img_name = 'tampered_images/RandomDrop'
    fourth_out_put_mask_name = 'tamper_loc_masks/RandomDrop'

    for second_name in second_names:
        for third_name in third_names:
            input_path = os.path.join(root_dir, second_name, third_name, fourth_input_name)
            img_output_path = os.path.join(root_dir, second_name, third_name, fourth_output_img_name)
            mask_output_path = os.path.join(root_dir, second_name, third_name, fourth_out_put_mask_name)

            # settings for random drop
            crop_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            padding_methods = ['black', 'white', 'gaussian', 'uniform', 'flip']
            process_images(input_path, img_output_path, mask_output_path, crop_ratios, padding_methods)