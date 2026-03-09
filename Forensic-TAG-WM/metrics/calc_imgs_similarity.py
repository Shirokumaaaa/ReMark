import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import torch

def calc_imgs_similarity(img1, img2):
    """
    Calculate multi-dimensional similarity metrics between two images
    """
    # Format conversion
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)   
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    # Ensure the images are in uint8 format
    if img1.dtype != np.uint8:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (img2 * 255).astype(np.uint8)

    # Preprocessing
    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0
    
    # Metrics calculation dictionary
    metrics = {}
    
    # Calculate MSE
    metrics['MSE'] = mse(img1_float, img2_float)
    
    # Calculate PSNR
    if metrics['MSE'] == 0:
        metrics['PSNR'] = float('inf')
    else:
        metrics['PSNR'] = 10 * np.log10(1.0 / metrics['MSE'])
    
    # Calculate SSIM
    metrics['SSIM'] = ssim(img1_float, img2_float, channel_axis=2, data_range=1.0)
    
    # Calculate MS-SSIM
    metrics['MS_SSIM'] = ssim(img1_float, img2_float, channel_axis=2, data_range=1.0,  multichannel=True, multiscale=True)
    
    return metrics