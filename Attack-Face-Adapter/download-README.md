
## 1.首先创建环境
conda create -n FaceAdapter python=3.10
conda activate FaceAdapter

## 2.安装 PyTorch（GPU + CUDA 11.8）
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

## 3.安装项目 Python 依赖
进入项目根目录（`Face-Adapter-main`），**不要**在别的目录执行：
pip install -r requirements.txt

注：
- 不要先 `pip install diffusers` 或 `pip install transformers` 再装 requirements，容易版本冲突。
- 不要用 `pip install -r requirements.txt --upgrade`，会破坏锁定版本。

## 4. 下载模型
直接运行 `app.py` 或 `infer.py`，程序会首次运行时自动下载 FaceAdapter 权重到 `./checkpoints`。  
也可从 [Hugging Face](https://huggingface.co/FaceAdapter/FaceAdapter/tree/main) 手动下载到 `./checkpoints`。  
SD 与 VAE 会在首次加载时从 Hugging Face 下载（需 [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)、[stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) 或依赖自动下载）。


# 快速开始测试

本小节与仓库根目录 `README.md` 中的 **Quick Inference** 对应，用于跑通一次换脸推理。

**数据要求**：只需两个**图片目录**——**源脸**（提供身份）与**目标脸**（提供姿态/驱动）。若使用项目组统一数据集（如 `Dataset-CelebA_HQ`），可从 `train/`、`val/` 或 `test/` 中选取部分图片分别拷贝或软链到两个目录即可（无需 mask）。

**最小可跑命令**（在项目根目录 `Attack-Face-Adapter-main` 下执行）：

```bash
python infer.py -s <源脸目录路径> -t <目标脸目录路径> -o ./output -ckpt ./checkpoints
```

示例（使用自带 example）：
```bash
python infer.py -s ./example/src -t ./example/tgt -o ./output -ckpt ./checkpoints
```

**参数说明**：
- `-s` / `--source`：源脸图片目录（.jpg / .png / .jpeg）
- `-t` / `--target`：目标脸图片目录
- `-o` / `--output`：结果保存目录（其下会生成 `swap/`、`concat/`、`drive/` 等）
- `-ckpt` / `--checkpoint`：权重目录，默认 `./checkpoints`
- `-r` / `--crop_ratio`：人脸裁剪范围，默认 `0.81`，不宜过大以免影响质量（参见主 README）
- `-b` / `--base_model`：基座扩散模型，默认 `runwayml/stable-diffusion-v1-5`，可改为社区模型如 `frankjoshua/toonyou_beta6`

**输出**：换脸图在 `-o` 指定目录下的 `swap/`，对比图在 `concat/`。

更多用法见根目录 **README.md**（如安装说明、Quick Inference、换用社区模型等）。

---

# 可能遇到问题（踩坑清单）

### 1. `ImportError: cannot import name 'clear_device_cache' from 'accelerate.utils.memory'`

- **原因**：`peft` 版本过高，与当前 `accelerate` 不兼容。
- **解决**：严格使用本仓库 `环境配置需求/requirements.txt`，其中已固定 `peft==0.10.0`、`accelerate==0.24.0`。若已乱装，可执行：
  ```powershell
  pip install peft==0.10.0 accelerate==0.24.0
  ```

### 2. NumPy 报错或版本冲突

- **要求**：必须使用 `numpy==1.26.4`，不要升级到 2.x。
- **解决**：`pip install numpy==1.26.4`

### 3. 同时安装过 `opencv-python` 和 `opencv-python-headless``

- **现象**：导入 cv2 报错或行为异常。
- **解决**：只保留一种。本仓库使用 headless：
  ```powershell
  pip uninstall opencv-python opencv-python-headless -y
  pip install opencv-python-headless==4.9.0.80
  ```
  若需本地 GUI（如 `cv2.imshow`），再改为只装 `opencv-python==4.9.0.80`。

### 4. InsightFace / onnxruntime 报错

- **现象**：`FaceAnalysis` 初始化失败或 CUDA Provider 报错。
- **解决**：本仓库使用 `onnxruntime-gpu==1.23.2`。若 GPU/驱动过老，可改用 CPU：
  ```powershell
  pip uninstall onnxruntime-gpu -y
  pip install onnxruntime==1.18.0
  ```
  推理会变慢但可跑通。

### 5. 模型下载慢或超时

- 使用 HF 镜像：`set HF_ENDPOINT=https://hf-mirror.com` 后再运行。
- 或在能访问外网的机器下载好 `checkpoints` 与 HF 缓存，拷贝到本机后使用 `--use_cache` 与 `--cache_dir`。


### 6. diffusers / transformers 版本不符

- **禁止**随意升级到 diffusers 0.3x 或 transformers 5.x，API 会变。  
- 严格按 `环境配置需求/requirements.txt` 安装。

---