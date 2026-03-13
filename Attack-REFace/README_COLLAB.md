# REFace 协作与快速上手说明（供协作者与 Agent 使用）

本文档面向 **后续使用者 / 协作者 / Agent**，目标是：

- 明确 `Attack-REFace` 在项目中的角色；
- 给出 **数据预处理脚本** 与 **推理脚本** 的「一键可复制命令」；
- 兼顾离线/弱网环境（本项目场景），尽量避免在线下载。

---

## 1) 分支角色与整体目标

- **目录**：`Attack-REFace`，对应论文  
  *Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models*（WACV 2025）。
- **在项目中的角色**：作为换脸攻击模型，与 `Attack-Face-Adapter` 并列，用于对上游水印/检测等方法做攻击评测。
- **数据约定**：
  - 训练与论文评测使用 **CelebAMask‑HQ / FFHQ** 等数据；
  - 项目内常见根目录约定为：`../Dataset-CelebA_HQ`；  
  - 推理/测试时，只需要若干 **源脸目录 `Source/`** 与 **目标脸目录 `Target/`** 即可。

---

## 2) 关键文件与入口（给人看，也给 Agent 看）

| 用途 | 路径 |
|------|------|
| 协作/快速上手说明 | 本文件 `Attack-REFace/README_COLLAB.md` |
| 原始项目说明 | `Attack-REFace/README.md` |
| 训练入口 | `Attack-REFace/main.py` |
| 推理入口（批量评测 CelebA/FFHQ） | `Attack-REFace/inference_test_bench.sh` + `scripts/inference_test_bench.py` |
| 推理入口（自定义 Source/Target 目录） | `Attack-REFace/inference_selected.sh` + `scripts/inference_swap_selected.py` |
| GUI Demo | `Attack-REFace/Demo.sh` |
| CelebA-HQ 子集与软链接视图构建脚本 | `Attack-REFace/tools/prepare_celebahq_eval_inputs.py` |
| 预训练模型（REFace） | `Attack-REFace/models/REFace/checkpoints/last.ckpt` |
| 预训练 CLIP（本地离线目录） | `Attack-REFace/Other_dependencies/clip-vit-large-patch14/` |

> **建议**：在自动化脚本 / Agent 逻辑中优先调用 `*.sh` 包装脚本（如 `inference_selected.sh`），避免重复拼接长命令。

---

## 3) 环境与必要文件

### 3.1 Conda 环境

参考原始 `README.md` 或项目统一规范，在服务器上创建环境，例如：

```bash
conda create -n "REFace" python=3.10.13 -y
conda activate REFace
sh setup.sh
```

> 若使用多台机器（本地 Windows + 远程 Linux），请保证 **远程 Linux 服务器** 上的 `REFace` 环境已经创建完毕。

### 3.2 预训练 REFace 模型与 CLIP

- **预训练 REFace 模型 `last.ckpt`**：统一放在  
  `Attack-REFace/models/REFace/checkpoints/last.ckpt`  
  （若需要兼容原始脚本，可在同目录下复制一份为 `saved.ckpt`）。
- **预训练 CLIP 权重目录**：统一放在  
  `Attack-REFace/Other_dependencies/clip-vit-large-patch14/`  
  且相关配置（如 `configs/train.yaml`、`models/REFace/configs/project_ffhq.yaml` 中的 `cond_stage_config.params.version`）已在本仓库中配置为指向该目录。协作者在**同一台服务器**上使用时，无需重复配置，只要路径保持不变即可复用。 

---

## 4) 使用 CelebA-HQ 快速构建评测子集（prepare_celebahq_eval_inputs.py）

`Attack-REFace/tools/prepare_celebahq_eval_inputs.py` 脚本用于从项目统一的 CelebA-HQ 数据集中，选取一小部分图像，生成：

- 一对 **CSV 文件**：记录选取的源/目标图片；
- 一对 **软链接目录**：后续作为 Face-Swap 输入的 `Source/`、`Target/` 目录。

### 4.1 脚本行为概览

脚本核心逻辑（简化）：

- 从 `--dataset-root/<split>` 中按数字顺序（或文件名）读取前 N 张 `.jpg`；
- 将选中的路径写入 CSV（`source.csv` / `target.csv`）；
- 在 `faceswap_inputs/source` 与 `faceswap_inputs/target` 下创建软链接视图。

默认参数（在 `Attack-REFace` 根目录运行）：

- `--dataset-root`: `../Dataset-CelebA_HQ`
- `--source-split`: `test`
- `--target-split`: `test`
- `--num-source`: `8`
- `--num-target`: `256`
- `--out-dir`: `Attack-REFace/data/celebahq_eval`
- `--faceswap-input-dir`: `Attack-REFace/data/faceswap_inputs`

### 4.2 一键构建评测输入（复制即可）

```bash
cd Attack-REFace

# 从 ../Dataset-CelebA_HQ/test 中抽取 8 张源脸、256 张目标脸
python tools/prepare_celebahq_eval_inputs.py \
  --dataset-root ../Dataset-CelebA_HQ \
  --source-split test \
  --target-split test \
  --num-source 8 \
  --num-target 256
```

运行结束后，你会得到：

- CSV：
  - `data/celebahq_eval/source.csv`
  - `data/celebahq_eval/target.csv`
- 软链接目录（真正作为换脸输入）：
  - `data/faceswap_inputs/source/`
  - `data/faceswap_inputs/target/`

> **不修改原始数据集**：脚本只对 `data/faceswap_inputs` 目录进行增删软链接，**不会改写 `../Dataset-CelebA_HQ` 下的任何文件**。

---

## 5) 使用 REFace 模型做推理（不训练，只推理）

以下流程假设：

- 你已经完成了第 3 节中的预训练模型与 CLIP 配置；
- 已经按第 4 节准备好了 `data/faceswap_inputs/source` 与 `data/faceswap_inputs/target`（也可以用你自己的 Source/Target 目录，只需在脚本参数中改路径）。

### 5.1 自定义 Source/Target 目录的换脸推理（inference_selected.sh）

1. 在服务器上，确认当前目录为 `Attack-REFace`，并激活环境：

```bash
conda activate REFace
cd /path/to/Attack-REFace
export PYTHONPATH=$(pwd)
```

2. 若使用 `tools/prepare_celebahq_eval_inputs.py` 生成的目录，可以将 `inference_selected.sh` 中的路径改为：

```bash
target_path="data/faceswap_inputs/target"
source_path="data/faceswap_inputs/source"
CKPT="models/REFace/checkpoints/last.ckpt"
CONFIG="models/REFace/configs/project_ffhq.yaml"
```

3. 一键运行换脸推理：

```bash
bash inference_selected.sh
```

4. 输出结果位置（默认）：

```text
examples/FaceSwap/Swap_outs/results/
```

其中会包含：

- 最终换脸结果图；
- 中间 inpaint 图、mask 图、参考人脸等（具体结构见脚本）。

### 5.2 按论文设置跑 CelebA / FFHQ benchmark（inference_test_bench.sh）

`Attack-REFace/inference_test_bench.sh` 封装了作者在 README 中给出的 `scripts/inference_test_bench.py` 典型用法，可一次性在 CelebA / FFHQ 上跑出测试集结果。

1. 确保脚本中的 `CKPT` 指向你的 `last.ckpt`：

```bash
CKPT="models/REFace/checkpoints/last.ckpt"
```

2. 执行：

```bash
cd Attack-REFace
conda activate REFace
export PYTHONPATH=$(pwd)
bash inference_selected.sh
```

3. 默认输出路径（可在脚本中查看/修改）：

- CelebA 结果：`results/CelebA/REFace/`
- FFHQ 结果：`results/FFHQ/REFace/`

---

## 6) 训练 REFace（仅在需要复现实验时）

**注意**：训练 REFace 需要完整数据集和 Stable Diffusion v1‑4 预训练权重，开销较大；如果你只关心推理，可以跳过本节。

按照原始 `README.md`：

1. 从 Hugging Face 下载 Stable Diffusion v1‑4 原始 ckpt：
   - `https://huggingface.co/CompVis/stable-diffusion-v-1-4-original`
   - 放到：`Attack-REFace/pretrained_models/sd-v1-4.ckpt`
2. 运行一次权重转换脚本：

```bash
cd Attack-REFace
python scripts/modify_checkpoints.py
```

3. 使用作者推荐命令启动训练：

```bash
python -u main.py \
  --logdir models/REFace/ \
  --pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \
  --base configs/train.yaml \
  --scale_lr False
```

训练完成后，会在 `models/REFace/checkpoints/` 下生成新的 ckpt，可替换前面推理脚本中的 `last.ckpt`。

---

## 7) 文档与协作建议

- 若修改了推理入口（例如新增新的 `*.sh` 或 Python 脚本），请在本文件的「关键文件与入口」与「推理流程」中同步更新。
- 若统一了新的数据集根目录（如 `../Dataset-FFHQ`），请在第 4 节和第 5 节中补充注释与示例命令。
- 修改环境依赖与版本时，请同步修改 `setup.sh` / `requirements.txt` 并在这里注明兼容建议（例如 GPU/驱动版本）。 

# REFace 协作与分支说明（供协作者与 Agent 快速上手）

本文档描述本目录在当前项目中的分支角色、关键文件、注意事项和快速上手步骤，便于协作者或 Agent 阅读后快速理解并参与开发/复现。

---

## 1) 当前分支/复现目标

- **目录**：`Attack-REFace-main`，对应论文 Realistic and Efficient Face Swapping (REFace, WACV 2025)。
- **在项目中的角色**：项目组内**负责复现的 REFace 攻击模型**，与 `Attack-Face-Adapter-main` 并列，供上游水印/检测等模块做攻击评测。
- **数据约定**：项目组统一使用 **Dataset-CelebA_HQ**。REFace 推理有多种模式：
  - **GUI Demo**：单张源图+目标图，无需预先准备目录结构。
  - **Test bench（CelebA/FFHQ）**：需要 **CelebA-HQ 结构**（`CelebA-HQ-img/` + `CelebA-HQ-mask/Overall_mask/`）；若仅有按类别拆分的 mask，需先运行本目录下的 `process_CelebA_mask.py` 生成 Overall_mask。
  - **自选文件夹 / 视频帧**：需要**目标人脸裁剪图**（如 `0.png, 1.png, ...`）与**对应 mask** 两个目录，或通过脚本从 Dataset-CelebA_HQ 的 CSV 做预处理得到。

---

## 2) 关键文件与入口（Agent 可解析）

| 用途 | 路径 |
|------|------|
| 协作/快速上手说明 | 本文件 `README_COLLAB.md` |
| 环境、下载与快速测试 | `download-README.md` |
| 原始项目说明、Demo、Testing | `README.md` |
| 生成 Overall_mask（合并按类别 mask） | `process_CelebA_mask.py` |
| GUI Demo | `sh Demo.sh` |
| 数据集 test bench 推理 | `scripts/inference_test_bench.py`，或 `sh inference_test_bench.sh` |
| 自选源/目标文件夹推理 | `scripts/inference_swap_selected.py`，或 `sh inference_selected.sh` |
| 单次/批量（含 Flask） | `scripts/one_inference.py` |
| 视频换脸 | `scripts/inference_swap_video.py` |
| 配置示例 | `models/REFace/configs/`（如 `project_ffhq.yaml`） |
| 权重 | `last.ckpt`（见 README / Hugging Face）；其他依赖见 `Other_dependencies/` |
| 依赖版本 | `requirements.txt`；PyTorch/Lightning 等见 download-README |

**最快可跑方式**：在本目录执行 `sh Demo.sh`，在浏览器中上传源图与目标图即可换脸。

**使用 CelebA-HQ 结构跑 test bench**（需先有 Overall_mask）：
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_test_bench.py \
  --outdir ./results --config models/REFace/configs/project_ffhq.yaml \
  --ckpt path/to/last.ckpt --dataset "CelebA" --dataset_dir path/to/CelebAMask-HQ \
  --ddim_steps 50 --n_samples 10
```

---

## 3) 写作者/协作者注意事项

- **依赖与版本**
  - 严格按 **download-README.md** 顺序安装：NumPy 1.x、PyTorch 1.13.1+cu117、PyTorch Lightning、taming-transformers 等，避免跳过或调换导致不兼容。
  - `Other_dependencies/` 下的人脸解析、ArcFace、landmark 等需从 README 或 Hugging Face 获取，路径不要随意改动，以免脚本找不到权重。

- **数据与路径**
  - Test bench 依赖 **CelebA-HQ 目录结构**（`CelebA-HQ-img/`、`CelebA-HQ-mask/Overall_mask/`）或 FFHQ 对应结构；仅当已有按类别 mask 时可用 `process_CelebA_mask.py` 生成 Overall_mask，该脚本**不会**从 RGB 图自动生成 mask。
  - 若项目组只有 **Dataset-CelebA_HQ**（仅图片 + train/val/test.csv），直接跑 test bench 会缺 mask；应使用 Demo.sh 做单张测试，或先实现“由 CSV 生成裁剪+mask”的预处理再调 `one_inference.py` 等。

- **代码与配置**
  - 推理入口在 **scripts/** 下，不是仓库根目录的 `main.py`（训练用）。修改脚本时注意 `--config`、`--ckpt`、`--Base_dir`、`--dataset_dir` 等与 README/download-README 一致。
  - 配置文件（如 `models/REFace/configs/project_ffhq.yaml`）中涉及数据路径、mask 类别等时，与当前数据约定保持一致。

- **文档**
  - 环境、下载、快速测试以 **download-README.md** 为准；算法、Demo、Training、Test Benchmark 以 **README.md** 为准。变更入口或数据流程时请同步更新 download-README 和本文档。

---

## 4) 协作者/Agent 快速上手流程

1. **读文档顺序**：先读本文件（`README_COLLAB.md`）了解分支与注意点，再读 **download-README.md** 完成环境与模型/依赖下载。
2. **环境**：按 download-README 创建 conda 环境（如 `REFace`）、按顺序安装 NumPy、PyTorch、Lightning、taming-transformers、dlib（可选）等。
3. **权重与依赖**：下载 REFace 的 `last.ckpt`；将人脸解析、ArcFace、landmark 等放到 `Other_dependencies/`（见 README 或 Hugging Face）。
4. **跑通测试**：  
   - **方式 A**：执行 `sh Demo.sh`，在浏览器中上传源图、目标图。  
   - **方式 B**：若有 CelebA-HQ 结构，先运行 `python process_CelebA_mask.py`（如需要），再按 download-README 执行 `scripts/inference_test_bench.py` 或 `sh inference_test_bench.sh`。
5. **输出**：Demo 在浏览器中直接显示；test bench 结果在 `--outdir` 指定目录下。

---

## 5) 与项目其他部分的关系

- **数据集**：统一使用 **Dataset-CelebA_HQ**。REFace 的 test bench 需要“图片 + Overall_mask”的目录结构；若仅有图片与 CSV，需额外预处理生成 mask 或使用 Demo 做单张测试。
- **上游/下游**：本目录作为“攻击模型”被其他模块调用或对比；修改脚本参数或配置路径时请考虑对调用方的影响。
- **与 Attack-Face-Adapter**：同为项目组复现的换脸模型，数据源一致；Face-Adapter 只需两个图片目录无需 mask，REFace 多数流程需要裁剪图与 mask，入口与数据准备不同（见 `Attack-Face-Adapter-main/README_COLLAB.md`）。

---

## 6) 变更本文档时建议

- 若分支用途、数据约定或入口发生变更，请更新“当前分支/复现目标”和“关键文件与入口”。
- 若依赖、脚本路径或环境步骤变更，请同步更新 **download-README.md** 并在“写作者注意事项”中注明。
- 新增重要脚本或配置时，在“关键文件与入口”中补充路径，便于 Agent 解析。
