import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from model import Generator

def load_generator(model_path, c_dim, device):
    """Load the pre-trained generator."""
    print(f"Loading the trained generator from {model_path}...")
    generator = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()  # Set to evaluation mode
    return generator

def preprocess_image(image_path, image_size):
    """Preprocess the input image."""
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # Assuming CelebA images
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def create_target_labels(c_dim, selected_attrs, device, original_label=None):
    """Create target domain labels for testing, avoiding the original label."""
    hair_color_indices = [selected_attrs.index(attr) for attr in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] if attr in selected_attrs]
    c_trg_list = []
    for i in range(c_dim):
        c_trg = torch.zeros(1, c_dim, device=device)  # Batch size is 1, ensure it's on the same device
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0
            c_trg[0, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[0, j] = 0
        else:
            c_trg[0, i] = 1  # Set non-hair attributes to 1

        # Skip the original label
        if original_label is not None and torch.equal(c_trg, original_label.to(device)):
            continue

        c_trg_list.append(c_trg)
    return c_trg_list

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def fake_single_image(image_path, model_path, output_dir, selected_attrs, c_dim, image_size, device, original_label):
    """Fake a single image using the pre-trained generator."""
    os.makedirs(output_dir, exist_ok=True)

    # Load the generator
    generator = load_generator(model_path, c_dim, device)

    # Preprocess the input image
    x_real = preprocess_image(image_path, image_size).to(device)

    # Convert original label to tensor
    original_label_tensor = torch.tensor(original_label, dtype=torch.float32).unsqueeze(0).to(device)

    # Create target domain labels
    c_trg_list = create_target_labels(c_dim, selected_attrs, device, original_label_tensor)

    # Translate the image
    with torch.no_grad():
        x_fake_list = [x_real]
        for c_trg in c_trg_list:
            x_fake_list.append(generator(x_real, c_trg))

        # Save the translated images
        x_concat = torch.cat(x_fake_list, dim=3)  # Concatenate along width
        result_path = os.path.join(output_dir, "fake_image.jpg")
        save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
        print(f"Saved fake image to {result_path}")

if __name__ == "__main__":
    # 配置参数
    image_path = "/home/ldy/..workspace/kei/stargan/sample_resized.jpg"  # 输入图片路径
    model_path = "/home/ldy/..workspace/kei/stargan/stargan/models/200000-G.ckpt"  # 预训练模型路径
    output_dir = "./outputs"  # 输出目录
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']  # 属性列表
    c_dim = len(selected_attrs)  # 属性维度
    image_size = 128  # 图像大小
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU 或 CPU

    # 原始标签
    original_label = [1, 0, 0, 1, 1]  # 假设原始标签

    # 对单张图片进行伪造
    fake_single_image(image_path, model_path, output_dir, selected_attrs, c_dim, image_size, device, original_label)