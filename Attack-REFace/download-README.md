#环境配置

## 1. 创建并激活环境（如果之前还没创建）
conda create -n REFace python=3.10.13   # 只需要执行一次
conda activate REFace

## 2. 安装 NumPy（固定在 1.x）
pip install "numpy==1.26.4"

## 3. 安装 PyTorch 三件套（CUDA 11.7）
pip install "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" "torchaudio==0.13.1+cu117" -f https://download.pytorch.org/whl/torch_stable.html

## 4. 安装 PyTorch Lightning 与 TorchMetrics（与 torch 1.13.1 兼容）
pip install "pytorch-lightning==1.9.0" "torchmetrics==0.11.4"

## 5. 安装 / 修复 setuptools（提供 pkg_resources，固定到兼容版本）
pip install "setuptools==68.2.2"
 
## 6. 安装 omegaconf
pip install omegaconf

## 7. 安装项目其他依赖（不会改动上面的核心版本）
pip install -r requirements.txt

## 8. 安装 GitHub 依赖（taming-transformers）
pip install "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"

## 9. 如需 dlib（可选，推荐用 conda）
conda install -c conda-forge "dlib==19.24.2"

## 10. 下载模型与依赖
- **REFace 权重**：从 [Hugging Face](https://huggingface.co/Sanoojan/REFace/blob/main/last.ckpt) 下载 `last.ckpt`，或从 [REFace 仓库](https://huggingface.co/Sanoojan/REFace/tree/main) 获取。
- **其他依赖**（人脸解析、ArcFace、landmark 等）：见项目根目录 **README.md** 的 “Other dependencies” 小节，需放到 `Other_dependencies/` 等对应路径；也可直接从 [Hugging Face REFace](https://huggingface.co/Sanoojan/REFace/tree/main) 下载并替换 `Other_dependencies` 文件夹。
- **Stable Diffusion 预训练**：训练或部分脚本需 SD v1-4，见主 README 的 “Download the pretrained model of Stable Diffusion”。

---

# 快速开始测试

本小节与仓库根目录 **README.md** 的 Demo / Testing 对应，便于快速跑通一次换脸。


**方式一：在标准数据集上测试（CelebA-HQ / FFHQ）**  
若有 **CelebA-HQ** 结构（`CelebA-HQ-img/` + `CelebA-HQ-mask/Overall_mask/`），可用 test bench 脚本。  
先创建 Overall mask：若只有按类别拆分的 mask，在项目根目录运行 `python process_CelebA_mask.py`（参见主 README “Data preparing”）。  
然后执行：
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_test_bench.py \
    --outdir ./results \
    --config models/REFace/configs/project_ffhq.yaml \
    --ckpt path/to/last.ckpt \
    --dataset "CelebA" \
    --dataset_dir path/to/CelebAMask-HQ \
    --ddim_steps 50 \
    --n_samples 10
```
将 `path/to/last.ckpt`、`path/to/CelebAMask-HQ` 换成实际路径。  
或直接运行：`sh inference_test_bench.sh`（需先按脚本内注释配置变量）。

**方式三：自选源/目标文件夹**  
若已准备好源图文件夹和目标图文件夹，可运行：
```bash
sh inference_selected.sh
```
或按脚本内说明传入 `--target_folder`、`--src_folder` 等（参见 `scripts/inference_swap_selected.py`）。

**使用项目组统一数据集（仅图片 + CSV）**：  
若只有类似 `Dataset-CelebA_HQ` 的图片与 `train/val/test.csv`（无现成 mask），REFace 的 test bench 需要 **人脸裁剪图 + mask**。可选做法：(1) 若有 CelebA-Mask-HQ 的按类别 mask，放到约定目录后运行 `process_CelebA_mask.py` 生成 Overall_mask，再按方式一跑 `inference_test_bench.py`；(2) 或用脚本根据 CSV 生成裁剪与 mask 再跑 `scripts/one_inference.py` 等（需自行准备预处理流程）。

更多说明（数据准备、训练、测试 benchmark）见项目根目录 **README.md**。
