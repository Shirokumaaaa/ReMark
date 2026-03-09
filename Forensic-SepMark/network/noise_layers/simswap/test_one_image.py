import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from network.noise_layers.simswap.models.models import create_model
from network.noise_layers.simswap.test_options import TestOptions
import torch.nn as nn
import random
import warnings
from network.noise_layers.target_picker import resolve_target_images, random_target
warnings.filterwarnings("ignore")


class SimSwap(nn.Module):
    def __init__(self, temp="temp/", target="/home/likaide/sda4/wxs/Dataset/dual_watermark/celeba_256/val/"):
        super(SimSwap, self).__init__()
        self.target = target
        self.target_images = resolve_target_images(target)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        opt = TestOptions().parse()

        # torch.nn.Module.dump_patches = True
        self.model = create_model(opt)

    def get_target(self):
        picked = random_target(self.target_images)
        if picked is not None:
            return picked
        idx = random.randint(162771, 182637)
        return self.target + str(idx).zfill(6) + ".png"

    def forward(self, image_cover_mask):
        self.model.eval()
        image, cover_image = image_cover_mask[0], image_cover_mask[1]

        noised_image = torch.zeros_like(image)

        for i in range(image.shape[0]):
            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            pic_a = self.get_target()
            img_a = Image.open(pic_a).convert('RGB')
            img_a = self.transformer_Arcface(img_a)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).to(self.device)

            img_b = self.transformer(im)
            img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2]).to(self.device)

            # No training on deepfake layers: run inference-only and stay in tensor space.
            with torch.inference_mode():
                img_id_downsample = F.interpolate(img_id, size=(112, 112))
                latend_id = F.normalize(self.model.netArc(img_id_downsample), p=2, dim=1)
                img_fake = self.model(img_id, img_att, latend_id, latend_id, True)
                full = img_fake[0].detach().clamp(0, 1)

            # [0,1] -> [-1,1], keep everything on tensor path to avoid disk I/O.
            noised_image[i] = full.mul(2.0).sub(1.0).to(image.device)

        return noised_image
