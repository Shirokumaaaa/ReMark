'''
这个代码中包含两个类，分别是StarGAN和StarGAN_multiple_steps
第一个用于单步生成，即只进行一次deepfake变换
第二个用于进行3步Deepfake变换，得到一组图像
他们的本质逻辑是一样的
'''
import torch
import torch.nn as nn
from network.noise_layers.model_SG import Generator

class StarGAN(nn.Module):
    """
    StarGAN-based noise layer. Applies a StarGAN generator to the input image.
    """

    def __init__(self, g_conv_dim, c_dim, g_repeat_num, device, model_path):
        super(StarGAN, self).__init__()
        self.device = device
        self.generator = Generator(g_conv_dim, c_dim, g_repeat_num).to(device)
        
        # Load pre-trained model weights
        self.generator.load_state_dict(torch.load(model_path, map_location=device))

    def forward(self, image_and_cover):
        """
        Args:
            image_and_cover: A tuple (image, cover_image, label).
        Returns:
            Transformed image by the StarGAN generator.
        """
        image, cover_image, label = image_and_cover

        if label is None:
            raise ValueError("Label cannot be None. Ensure that the 'label' parameter is correctly passed to StarGAN.")

        # Modify the label: randomly change the first three dimensions (hair color) and reverse the third and fourth dimensions

        # ================ reverse hair colour =======================

        # modified_label = label.clone()

        # # Randomly change the first three dimensions (hair color) to a new one-hot encoding
        # batch_size = label.size(0)
        # random_hair_color = torch.zeros(batch_size, 3, device=self.device)
        # random_indices = torch.randint(0, 3, (batch_size,), device=self.device)
        # random_hair_color[torch.arange(batch_size), random_indices] = 1
        # modified_label[:, :3] = random_hair_color  # Replace the first three dimensions with the new one-hot encoding
        # # print("Modified label:", modified_label)
        # # Ensure all tensors are on the same device
        # image = image.to(self.device)
        # modified_label = modified_label.to(self.device)

        # ================ reverse hair colour =======================

        # Clone 原始标签
        c_trg_list = []
        modified_label = label.clone()

        # 按照 create_labels 的规则：只反转第三维度 (gender)，其他维度保持不变
        modified_label[:, 3] = 1 - modified_label[:, 3]
        c_trg_list.append(modified_label.to(self.device))

        # 确保张量在正确的 device 上
        image = image.to(self.device)
        # modified_label = modified_label.to(self.device)
        c_trg_list = torch.stack(c_trg_list, dim=0)
        c_trg_list = c_trg_list.squeeze(0)  # 去掉多余的维度
        # 送入 StarGAN 生成器
        transformed_image = self.generator(image, c_trg_list)
        return transformed_image
    


class StarGAN_multi_step(nn.Module):
    """
    StarGAN-based noise layer. Applies a StarGAN generator to the input image.
    """

    def __init__(self, g_conv_dim, c_dim, g_repeat_num, device, model_path):
        super(StarGAN_multi_step, self).__init__()
        self.device = device
        self.generator = Generator(g_conv_dim, c_dim, g_repeat_num).to(device)

        # Load pre-trained model weights
        self.generator.load_state_dict(torch.load(model_path, map_location=device))

        # 冻结 StarGAN 的所有参数
        for param in self.generator.parameters():
            param.requires_grad = False

    def forward(self, image_and_cover):
        """
        Args:
            image_and_cover: A tuple (image, cover_image, label).
        Returns:
            Transformed images by the StarGAN generator and their corresponding damage stages.
        """
        image, cover_image, label = image_and_cover

        if label is None:
            raise ValueError("Label cannot be None. Ensure that the 'label' parameter is correctly passed to StarGAN.")

        # 保证所有张量在同一设备
        image = image.to(self.device)
        modified_label = label.clone().to(self.device)

        # 再随机生成新的发色 one-hot 标签
        batch_size = label.size(0)
        random_hair_color = torch.zeros(batch_size, 3, device=self.device)
        random_indices = torch.randint(0, 3, (batch_size,), device=self.device)
        random_hair_color[torch.arange(batch_size), random_indices] = 1
        modified_label[:, :3] = random_hair_color  # 替换前三维为新的发色 one-hot
        transformed_image1 = self.generator(image, modified_label)

        # 再反转年龄
        modified_label[:, 4] = 1 - modified_label[:, 4]  # 反转年龄
        transformed_image2 = self.generator(transformed_image1, modified_label)

        # 先反转性别
        modified_label[:, 3] = 1 - modified_label[:, 3]  # 反转性别
        transformed_image3 = self.generator(transformed_image2, modified_label)

        # 返回每个阶段的图像及其对应的损坏阶段数
        return [
            (image, 0), # 即为clean image
            (transformed_image1, 1),
            (transformed_image2, 2),
            (transformed_image3, 3)
        ]
class StarGAN_five_step(nn.Module):
    """
    StarGAN-based noise layer. Applies a StarGAN generator to the input image.
    """

    def __init__(self, g_conv_dim, c_dim, g_repeat_num, device, model_path):
        super(StarGAN_five_step, self).__init__()
        self.device = device
        self.generator = Generator(g_conv_dim, c_dim, g_repeat_num).to(device)

        # Load pre-trained model weights
        self.generator.load_state_dict(torch.load(model_path, map_location=device))

        # 冻结 StarGAN 的所有参数
        for param in self.generator.parameters():
            param.requires_grad = False

    def forward(self, image_and_cover):
        """
        Args:
            image_and_cover: A tuple (image, cover_image, label).
        Returns:
            Transformed images by the StarGAN generator and their corresponding damage stages.
        """
        image, cover_image, label = image_and_cover

        if label is None:
            raise ValueError("Label cannot be None. Ensure that the 'label' parameter is correctly passed to StarGAN.")

        # 保证所有张量在同一设备
        image = image.to(self.device)
        modified_label = label.clone().to(self.device)
        # 对每个样本的三个发色分别生成三阶段的变换图像，并返回所有阶段的列表
        batch_size = label.size(0)
        results = [(image, 0)]  # 保留原始clean image
        modified_label = label.clone().to(self.device)
        modified_label[:,1]=1-modified_label[:,1]
        t1 = self.generator(image, modified_label)
        results.append((t1,1))
        modified_label[:,2]=1-modified_label[:,2]
        t2 = self.generator(t1, modified_label)
        results.append((t2,2))
        modified_label[:,0]=1-modified_label[:,3]
        t3 = self.generator(t2, modified_label)
        results.append((t3,3))
        # 再反转年龄
        modified_label[:, 4] = 1 - modified_label[:, 4]  # 反转年龄
        transformed_image2 = self.generator(t3, modified_label)
        results.append((transformed_image2, 4))

        # 先反转性别
        modified_label[:, 0] = 1 - modified_label[:, 0]  # 反转性别
        transformed_image3 = self.generator(transformed_image2, modified_label)
        results.append((transformed_image3, 5))

        # 返回每个阶段的图像及其对应的损坏阶段数
        return results
class StarGAN_three(nn.Module):
    """
    StarGAN-based noise layer. Applies a StarGAN generator to the input image.
    """

    def __init__(self, g_conv_dim, c_dim, g_repeat_num, device, model_path):
        super(StarGAN_three, self).__init__()
        self.device = device
        self.generator = Generator(g_conv_dim, c_dim, g_repeat_num).to(device)
        
        # Load pre-trained model weights
        self.generator.load_state_dict(torch.load(model_path, map_location=device))

    def forward(self, image_and_cover):
        """
        Args:
            image_and_cover: A tuple (image, cover_image, label).
        Returns:
            Transformed images by the StarGAN generator and their corresponding damage stages.
        """
        image, cover_image, label = image_and_cover

        if label is None:
            raise ValueError("Label cannot be None. Ensure that the 'label' parameter is correctly passed to StarGAN.")

        # 保证所有张量在同一设备
        image = image.to(self.device)
        modified_label = label.clone().to(self.device)

        # 再随机生成新的发色 one-hot 标签
        batch_size = label.size(0)
        random_hair_color = torch.zeros(batch_size, 3, device=self.device)
        random_indices = torch.randint(0, 3, (batch_size,), device=self.device)
        random_hair_color[torch.arange(batch_size), random_indices] = 1
        modified_label[:, :3] = random_hair_color  # 替换前三维为新的发色 one-hot
        transformed_image1 = self.generator(image, modified_label)

        # 再反转年龄
        modified_label[:, 4] = 1 - modified_label[:, 4]  # 反转年龄
        transformed_image2 = self.generator(transformed_image1, modified_label)

        # 先反转性别
        modified_label[:, 3] = 1 - modified_label[:, 3]  # 反转性别
        transformed_image3 = self.generator(transformed_image2, modified_label)
        return transformed_image3