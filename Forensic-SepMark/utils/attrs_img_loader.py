import os
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class attrsImgDataset(Dataset):

    def __init__(self, path, image_size, attr_path="celeba", csv_path=None):
        super(attrsImgDataset, self).__init__()
        self.image_size = image_size
        self.image_dir = path
        self.csv_path = csv_path
        if attr_path[0:len("celebahq")] != "celebahq":
            self.attr_path = 'network/noise_layers/stargan/list_attr_celeba.txt'
        else:
            self.attr_path = 'network/noise_layers/stargan/CelebAMask-HQ-attribute-anno.txt'

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.list = [] # os.listdir(path)]
        self.attr2idx = {}
        self.idx2attr = {}
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        attr_map = {}
        for i, line in enumerate(lines):
            split = line.split()
            basename = os.path.basename(split[0])
            filename = os.path.splitext(basename)[0] + ".jpg"
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            attr_map[filename] = label

        # Preferred path: explicit CSV with absolute paths.
        if self.csv_path is not None and os.path.isfile(self.csv_path):
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_path = row.get("img_path", "").strip()
                    if not image_path or not os.path.exists(image_path):
                        continue
                    basename = os.path.basename(image_path)
                    if basename in attr_map:
                        self.list.append([image_path, attr_map[basename]])
            return

        # Backward-compatible fallback: original folder-based discovery.
        for filename, label in attr_map.items():
            candidate = os.path.join(self.image_dir, filename)
            if os.path.exists(candidate):
                self.list.append([candidate, label])
                continue
            stem = os.path.splitext(filename)[0]
            png_candidate = os.path.join(self.image_dir, str(stem).zfill(5) + ".png")
            if os.path.exists(png_candidate):
                self.list.append([png_candidate, label])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        image_path, label = self.list[index]
        image = Image.open(image_path).convert("RGB")
        if image is not None:
            return self.transform(image), torch.FloatTensor(label), image_path

    def __len__(self):
        return len(self.list)
