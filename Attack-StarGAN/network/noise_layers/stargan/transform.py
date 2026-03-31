from PIL import Image

def resize_image(image_path, output_path, size=(178, 178)):
    """
    Resize an image to the specified size and save it.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        size (tuple): Target size as (width, height).
    """
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    resized_image = image.resize(size, Image.LANCZOS)  # 使用 LANCZOS 替代 ANTIALIAS
    
    # 保存图片
    resized_image.save(output_path)
    print(f"Resized image saved to {output_path}")

# 示例用法
image_path = "/home/ldy/..workspace/kei/stargan/sample.jpg"  # 输入图片路径
output_path = "/home/ldy/..workspace/kei/stargan/sample_resized.jpg"  # 输出图片路径
resize_image(image_path, output_path)