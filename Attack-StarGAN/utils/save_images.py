'''
Function to save images

By jzyustc, 2020/12/21

'''

import os
import numpy as np
import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def save_images(images, epoch, save_path, resize_to=None):
    """
    Save images to the specified path.
    Args:
        images: Tensor of images to save.
        epoch: Current epoch number (for logging purposes).
        save_path: Path to save the images.
        resize_to: Tuple (width, height) to resize images before saving.
    """
    if resize_to:
        images = torch.nn.functional.interpolate(images, size=resize_to, mode="bilinear", align_corners=False)
    vutils.save_image(images, save_path, nrow=1, padding=0, normalize=True)
    print(f"Saved images for epoch {epoch} into {save_path}...")


def get_random_images(images, encoded_images, noised_images):
    selected_id = np.random.randint(1, images.shape[0]) if images.shape[0] > 1 else 1
    image = images.cpu()[selected_id - 1:selected_id, :, :, :]
    encoded_image = encoded_images.cpu()[selected_id - 1:selected_id, :, :, :]
    noised_image = noised_images.cpu()[selected_id - 1:selected_id, :, :, :]
    return [image, encoded_image, noised_image]


def concatenate_images(saved_all, images, encoded_images, noised_images):
    saved = get_random_images(images, encoded_images, noised_images)
    if saved_all[2].shape[2] != saved[2].shape[2]:
        return saved_all
    saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
    saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
    saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
    return saved_all
