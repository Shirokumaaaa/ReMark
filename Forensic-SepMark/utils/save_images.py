'''
Function to save images

By jzyustc, 2020/12/21

'''

import os
import numpy as np
import torch
import torchvision.utils
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def save_images(saved_all, epoch, folder, resize_to=None):
	original_images, watermarked_images, noised_images = saved_all

	images = original_images[:original_images.shape[0], :, :, :].cpu()
	watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
	noised_images = noised_images[:noised_images.shape[0], :, :, :].cpu()

	# scale values to range [0, 1] from original range of [-1, 1]
	images = (images + 1) / 2
	watermarked_images = (watermarked_images + 1) / 2
	noised_images = (noised_images + 1) / 2
	diff_w2co = _normalize(torch.abs(images - watermarked_images))
	diff_w2no = _normalize(torch.abs(noised_images - watermarked_images))
	diff_o2n = _normalize(torch.abs(images - noised_images))

	if resize_to is None:
		resize_to = (256, 256)
	images = F.interpolate(images, size=resize_to, mode='bilinear', align_corners=False)
	watermarked_images = F.interpolate(watermarked_images, size=resize_to, mode='bilinear', align_corners=False)
	noised_images = F.interpolate(noised_images, size=resize_to, mode='bilinear', align_corners=False)
	diff_w2co = F.interpolate(diff_w2co, size=resize_to, mode='bilinear', align_corners=False)
	diff_w2no = F.interpolate(diff_w2no, size=resize_to, mode='bilinear', align_corners=False)
	diff_o2n = F.interpolate(diff_o2n, size=resize_to, mode='bilinear', align_corners=False)

	columns = ["Original", "Encoded", "Noised", "|Orig-Enc|", "|Orig-Noised|", "|Enc-Noised|"]
	rows = images.shape[0]
	tile_h, tile_w = resize_to
	pad = 8
	header_h = 28
	left_w = 58
	canvas_h = header_h + rows * (tile_h + pad) + pad
	canvas_w = left_w + len(columns) * (tile_w + pad) + pad
	canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

	draw_tensors = [images, watermarked_images, noised_images, diff_w2co, diff_o2n, diff_w2no]
	for r in range(rows):
		y = header_h + pad + r * (tile_h + pad)
		for c in range(len(columns)):
			x = left_w + pad + c * (tile_w + pad)
			tile = _to_uint8_image(draw_tensors[c][r])
			canvas[y:y + tile_h, x:x + tile_w] = tile

	img = Image.fromarray(canvas, mode="RGB")
	draw = ImageDraw.Draw(img)
	font = ImageFont.load_default()

	# Draw headers and row ids.
	for c, name in enumerate(columns):
		x = left_w + pad + c * (tile_w + pad)
		draw.text((x + 4, 8), name, fill=(20, 20, 20), font=font)
	for r in range(rows):
		y = header_h + pad + r * (tile_h + pad)
		draw.text((8, y + 6), f"#{r+1}", fill=(20, 20, 20), font=font)

	# Vertical separators make side-by-side comparison easier.
	for c in range(len(columns) + 1):
		x = left_w + c * (tile_w + pad)
		draw.line([(x, header_h), (x, canvas_h - 1)], fill=(180, 180, 180), width=1)

	filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
	img.save(filename)


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


def _normalize(input_tensor):
	output = input_tensor.clone()
	for i in range(output.shape[0]):
		min_val, max_val = torch.min(output[i]), torch.max(output[i])
		if float(max_val - min_val) < 1e-8:
			output[i] = torch.zeros_like(output[i])
		else:
			output[i] = (output[i] - min_val) / (max_val - min_val)

	return output


def _to_uint8_image(tensor_chw):
	arr = tensor_chw.detach().clamp(0, 1).permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255)
	return arr.to(torch.uint8).cpu().numpy()

	#min_val, max_val = torch.min(input_tensor), torch.max(input_tensor)
	#return (input_tensor - min_val) / (max_val - min_val)
