# ReMark

一个针对deepfake攻击的水印恢复的框架。

这个仓库不是单一模型的精简发布版，而是一个实验工作台，当前同时包含：

- `ReMark` 主模块：在 latent 空间中修复 deepfake 后受损的含水印图像
- 多种生成式水印方法
- 多种 deepfake攻击后端
- 多种数据集：CelebA-HQ、FFHQ、Stable Diffusion Prompt

主线目标可以概括为：

```text
原图 -> 水印编码 -> 含水印图 -> Deepfake / Attack -> 受损图
                                                     |
                                                     v
                                             ReMark 修复模块
                                                     |
                                                     v
                                             水印解码 / 验证
```

ReMark 的核心思想不是重写已有水印模型，而是在攻击之后增加一个独立修复器，尽量把受损图像拉回更接近原始含水印图像的状态，从而提升水印提取准确率。

## 仓库概览

### 1. 主控模块

- `Forensic-Remark_Module/`
  - 当前整仓库的主入口。
  - 包含 `train_stage1.py`、`train_stage2.py`、`configs/`、`data_manifests/`、`wm_adapters/`、`attacks/`、`tools/`。
  - 负责把不同的水印模型和不同的攻击模型接到同一套 ReMark 训练/评估流程中。

### 2. 水印方法

这些目录既可以单独使用，也可以通过 `Forensic-Remark_Module/wm_adapters/` 接入 ReMark：

- `Forensic-FIN/`
- `Forensic-LaWa/`
- `Forensic-LampMark/`
- `Forensic-MaskWM/`
- `Forensic-SepMark/`
- `Forensic-SleeperMark/`
- `Forensic-TAG-WM/`
- `Forensic-TrustMark/`

当前在 ReMark 代码中已注册的水印适配器包括：

- [`FIN`](https://ojs.aaai.org/index.php/AAAI/article/view/25633)
- [`SepMark`](https://doi.org/10.1145/3581783.3612471)
- [`LampMark`](https://dl.acm.org/doi/10.1145/3664647.3680869)
- [`LaWa`](https://arxiv.org/abs/2408.05868)
- [`SleeperMark`](https://arxiv.org/abs/2412.04852)
- [`TAG-WM`](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_TAG-WM_Tamper-Aware_Generative_Image_Watermarking_via_Diffusion_Inversion_Sensitivity_ICCV_2025_paper.html)
- [`MaskWM`](https://arxiv.org/abs/2504.12739)
- [`TrustMark`](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html)

### 3. 攻击方法

这些目录提供 deepfake / face swap / reenactment 后端，供 ReMark 调用：

- `Attack-DiffSwap/`
- `Attack-Face-Adapter/`
- `Attack-REFace/`
- `Attack-StarGAN/`
- `Attack-arc2face_wrapper/`

当前在 ReMark 中已注册的攻击适配器包括：

- [`stargan2`](https://openaccess.thecvf.com/content_CVPR_2020/html/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.html)
- [`SimSwap`](https://dl.acm.org/doi/abs/10.1145/3394171.3413630)
- [`Arc2face_wrapper`](https://openaccess.thecvf.com/content/ICCV2025W/I-HFM/html/Papantoniou_ID-Consistent_Precise_Expression_Generation_with_Blendshape-Guided_Diffusion_ICCVW_2025_paper.html)
- [`DiffSwap`](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_DiffSwap_High-Fidelity_and_Controllable_Face_Swapping_via_3D-Aware_Masked_Diffusion_CVPR_2023_paper.html)
- [`REFace`](https://ieeexplore.ieee.org/abstract/document/10943471)
- [`Face_Adapter`](https://link.springer.com/chapter/10.1007/978-3-031-72973-7_2)

### 4. 数据与通用工具

- `Dataset-CelebA_HQ/`
- `Dataset-FFHQ/`
- `common/`
  - 公共逻辑，目前主要是 landmark -> bit 的编码工具。
- `tools/`
  - 顶层数据清洗脚本，例如 landmark bit cache 和 clean subset 生成。

## 推荐阅读顺序

如果你是第一次接触这个仓库，建议按这个顺序看：

1. 根目录 `README.md`：先理解整仓库的角色分工。
2. `Forensic-Remark_Module/README.md`：更细的设计说明和实验上下文。
3. `Forensic-Remark_Module/readme_research.md`：latent geometry 研究笔记。
4. 你实际要用的水印方法和攻击方法各自的 README。

## ReMark 主线流程

`Forensic-Remark_Module/` 当前是最值得先跑通的部分。

### Stage 1

训练一个 VAE 风格的重建器，把 deepfake 后的受损图像映射回可修复的 latent 空间表示。

入口：

```bash
cd Forensic-Remark_Module
python train_stage1.py --config configs/stage1_vae.yaml
```

多卡：

```bash
cd Forensic-Remark_Module
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_stage1.py \
  --config configs/stage1_vae.yaml
```

### Stage 2

在 Stage 1 的 latent 空间上训练一个 U-Net 修复器，逐步把受损 latent 推回原始含水印 latent,最终实现水印的修复。

入口：

```bash
cd Forensic-Remark_Module
python train_stage2.py --config configs/stage2_unet.yaml
```

多卡：

```bash
cd Forensic-Remark_Module
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_stage2.py \
  --config configs/stage2_unet.yaml
```

手动指定 Stage 1 checkpoint：

```bash
cd Forensic-Remark_Module
python train_stage2.py \
  --config configs/stage2_unet.yaml \
  --stage1-ckpt runs/<stage1_run>/checkpoints/vae/best.pth
```

续训：

```bash
cd Forensic-Remark_Module
python train_stage1.py --config configs/stage1_vae.yaml --resume <run_name>
python train_stage2.py --config configs/stage2_unet.yaml --resume <run_name>
```

训练输出默认会写到：

```text
Forensic-Remark_Module/runs/<stage>_<timestamp>/
  config.yaml
  train.log
  samples/
  checkpoints/
```

## 运行前必须先改的配置

当前仓库中的默认 YAML 仍然带有本机路径和本地实验状态，第一次运行前请先检查：

- `Forensic-Remark_Module/configs/stage1_vae.yaml`
- `Forensic-Remark_Module/configs/stage2_unet.yaml`

另外要特别注意：这两个基础 YAML 目前对应的是不同时间点的本地实验快照，不保证默认就是一组互相匹配的配置；切换水印模型、攻击模型或数据集时，请把 Stage 1 和 Stage 2 成对检查。

重点关注这些字段：

- `data.train_csv`
- `data.val_csv`
- `wm_model`
- `attacks.online`
- `attacks.offline`
- `attack_options.*`
- `paths.stage1_checkpoint`（Stage 2）

当前 `configs/experiments/` 目录是空的，所以现阶段以编辑基础配置文件为主。

## 数据组织与 manifest 格式

ReMark 训练数据由 CSV manifest 驱动，面对不同的训练需求时不需要挪动图片，仅需修改csv文件即可。`Forensic-Remark_Module/data/dataset.py` 支持的典型格式如下。

最小格式：

```csv
img_path
/abs/path/to/image1.png
/abs/path/to/image2.png
```

带离线攻击图：

```csv
img_path,fake_path
/abs/path/to/image1.png,/abs/path/to/fake1.png
```

带水印缓存：

```csv
img_path,wm_path
/abs/path/to/image1.png,/abs/path/to/wm1.npy
```

如果 CSV 中还带有额外列：

- 数值列会被当成属性标签，例如 `Black_Hair, Blond_Hair, Brown_Hair, Male, Young`
- 字符串列会被保留为文本条件，例如 prompt 或 replay 元信息

这也是为什么 `ffhq_10k_train_with_attrs.csv` 这类 manifest 可以直接支持条件攻击。

## 在线攻击与 replay 攻击

在 ReMark 里，攻击分成两类：

- 在线攻击
  - 在训练循环里直接生成 fake 图
- replay / 离线攻击
  - 先离线生成结果，再通过 CSV 或 JSONL 映射回训练样本

有些deepfake攻击方式，如`diffswap`、`arc2face`等，直接用在线攻击速度较慢，所以设计了离线攻击，虽然这种方式在一定程度上降低了攻击的多样性，但在时间较紧，算力有限的情况下是一个很好的选择。

因此，`Forensic-Remark_Module/tools/` 下有很多辅助脚本专门用来：

- 构建 replay 索引
- 把离线结果转成 manifest
- 为 FFHQ / CelebA-HQ 生成训练清单
- 做迁移评估、假阳性评估和 latent 几何分析

比较常用的脚本包括：

- `Forensic-Remark_Module/tools/analyze_latent_geometry.py`
- `Forensic-Remark_Module/tools/build_manifest_from_replay_jsonl.py`
- `Forensic-Remark_Module/tools/build_ffhq_diffswap_replay_index.py`
- `Forensic-Remark_Module/tools/build_ffhq_reface_replay_index.py`
- `Forensic-Remark_Module/tools/build_ffhq_faceadapter_replay_index.py`
- `Forensic-Remark_Module/tools/prepare_ffhq_ffpp_10k.py`

## 数据预处理脚本

### landmark bit cache 与 clean subset

顶层 `common/landmark_bits.py` 提供统一的 landmark bit 编码逻辑，通常用于 landmark 条件类水印实验。

构建 cache：

```bash
python tools/build_landmark_bit_cache.py \
  --csv /path/to/train.csv \
  --cache-dir landmark_bit_cache
```

过滤失败样本并生成 clean 子集：

```bash
python tools/prepare_landmark_clean_subset.py \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/val.csv \
  --cache-dir landmark_bit_cache \
  --out-root Dataset-CelebA_HQ_10k_landmark_clean_20260324 \
  --lampmark-manifest-dir Forensic-LampMark/data_manifests
```

## 环境与依赖

这个仓库目前没有统一的根目录 `requirements.txt`，因为它本质上是多个项目的组合。建议把依赖理解为两层：

### 1. 你要跑的主模块

如果你只想跑 ReMark 主线，优先保证：

- `Forensic-Remark_Module/`
- 你选定的水印后端
- 你选定的攻击后端

三者的依赖同时可用。

### 2. 各子项目自己的环境文件

目前仓库里可直接参考的依赖文件包括：

- `Attack-DiffSwap/requirements.txt`
- `Attack-Face-Adapter/requirements.txt`
- `Attack-REFace/environment.yml`
- `Attack-REFace/requirements.txt`
- `Forensic-LaWa/environment.yml`
- `Forensic-TAG-WM/requirements.txt`
- `Forensic-TrustMark/python/requirements.txt`
- `Forensic-TrustMark/python/pyproject.toml`

实际使用时请特别注意：

- 很多脚本和 YAML 里还保留着绝对路径，例如 `/mnt/personal_workspace/...`、`/home/ldy/miniconda3/...`
- 模型权重大多不随仓库分发，需要你按各子项目 README 另行下载
- 某些攻击或水印模块依赖独立的 conda 环境

## 当前仓库的定位

为了避免误解，这里明确说明一下当前状态：

- 这是研究整合仓库，不是 one-click release。
- 根目录 README 的目标是给你一张地图，而不是替代每个子项目原始文档。
- 许多子目录是上游项目、复现实验代码或本地适配版本的组合。
- `Forensic-Remark_Module/README.md` 中还保留了大量实验记录和研究日志，适合做上下文参考，但不应把它当成唯一的快速开始文档。

## 建议的上手路线

如果你想最快熟悉整个仓库，建议按下面顺序：

1. 先确认你要用哪一个水印方法，例如 `lampmark` 或 `sepmark`。
2. 再确认你要用哪一个攻击后端，例如 `simswap`、`stargan2` 或 `diffswap`。
3. 安装相应依赖。
4. 修改 `Forensic-Remark_Module/configs/stage1_vae.yaml`，先跑通 Stage 1。
5. 等 Stage 1 产出稳定 checkpoint 后，再启动 Stage 2。
6. 如果需要研究解释性或迁移性，再看 `tools/analyze_latent_geometry.py` 和各类 replay / eval 工具。

## 引用与致谢

如果你使用这个仓库做研究，请同时引用：

- ReMark 对应论文或项目说明
- 你实际使用到的水印方法原论文
- 你实际使用到的攻击方法原论文

各个子目录通常已经附带自己的 README、论文链接和许可证说明，请以对应子项目为准。
