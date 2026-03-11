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
