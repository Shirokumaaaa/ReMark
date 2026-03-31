from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.dataset = []  # 存储所有图像和属性
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        # 读取属性文件
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()  # 属性名称
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        # 读取图像文件名和对应的属性
        lines = lines[2:]
        for line in lines:
            split = line.split()
            filename = split[0]
            values = split[1:]

            # 提取所选属性的值
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(1 if values[idx] == '1' else 0)

            self.dataset.append([filename, label])

        print(f'Finished preprocessing the CelebA dataset. Total samples: {len(self.dataset)}')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename, label = self.dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, num_workers=1):
    """Build and return a data loader."""
    # 定义图像预处理
    transform = T.Compose([
        T.CenterCrop(crop_size),
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 创建数据集
    dataset = CelebA(image_dir, attr_path, selected_attrs, transform)

    # 创建数据加载器
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader