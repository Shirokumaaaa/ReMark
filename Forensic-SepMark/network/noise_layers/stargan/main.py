import os
import torch
import torch.nn as nn
from PIL import Image
import random
from torchvision import transforms
from network.noise_layers.stargan.model import Generator
from network.noise_layers.target_picker import resolve_target_images, random_target


class StarGAN(nn.Module):

    def __init__(self, c_trg=3, image_size=256, temp="temp/", target="/home/likaide/sda4/wxs/Dataset/dual_watermark/celeba_256/val/"):
        super(StarGAN, self).__init__()
        self.c_trg = c_trg
        self.image_size = image_size
        self.target = target
        self.target_images = resolve_target_images(target)
        self.attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.G = Generator(64, 5, 6).to(self.device)
        G_path = os.path.join('network/noise_layers/stargan', str(self.image_size), '200000-G.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=self.device))
        self.G.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.regular_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def get_target(self):
        picked = random_target(self.target_images)
        if picked is not None:
            return picked
        idx = random.randint(162771, 182637)
        return self.target + str(idx).zfill(6) + ".png"

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def forward(self, image_cover_mask):
        image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]

        noised_image = torch.zeros_like(image)

        for i in range(image.shape[0]):
            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            try:
                fake = self.test(im, mask[i])
                noised_image[i] = self.transform(fake).unsqueeze(0).to(image.device)
            except Exception as e:
                print(f"Error in stargan: {e}")
                noised_image[i] = image[i]

        return noised_image

    def test(self, body_image, label):
        with torch.inference_mode():
            x_real = self.regular_image_transform(body_image).unsqueeze(0).to(self.device)
            c_org = label

            # Prepare input images and target domain labels.
            c_trg_list = self.create_labels(c_org, 5, selected_attrs=self.attrs)

            if self.c_trg is None:
                self.c_trg = random.randint(0, 4)
            # Translate images.
            x_fake = self.G(x_real, c_trg_list[self.c_trg].unsqueeze(0))

            fake = (self.denorm(x_fake[0]).permute(1, 2, 0) * 255.0).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            return fake

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[j] = 0
                else:
                    c_trg[i] = (c_trg[i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list
