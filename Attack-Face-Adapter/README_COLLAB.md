# Face-Adapter 协作与分支说明（供协作者与 Agent 快速上手）

本文档描述本目录在当前项目中的分支角色、关键文件、注意事项和快速上手步骤，便于协作者或 Agent 阅读后快速理解并参与开发/复现。

---

## 1) 当前分支/复现目标

- **目录**：`Attack-Face-Adapter-main`，对应论文 Face Adapter for Pre-Trained Diffusion Models（换脸/身份控制）。
- **在项目中的角色**：项目组内**负责复现的 Face-Adapter 攻击模型**，与 `Attack-REFace-main` 并列，供上游水印/检测等模块做攻击评测。
- **数据约定**：项目组统一使用 **Dataset-CelebA_HQ**（仓库内通常为 `../Dataset-CelebA_HQ`）。本模型推理只需**两个图片目录**（源脸、目标脸），无需 mask；可从 Dataset-CelebA_HQ 的 `train/`、`val/`、`test/` 中选取图片组成两个目录使用。

---

## 2) 关键文件与入口（Agent 可解析）

| 用途 | 路径 |
|------|------|
| 协作/快速上手说明 | 本文件 `README_COLLAB.md` |
| 环境与下载、快速测试 | `download-README.md` |
| 原始项目说明、Quick Inference | `README.md` |
| 推理入口（命令行） | `infer.py` |
| 推理入口（Web/交互） | `app.py` |
| 人脸几何/裁剪工具（无 Dataset 类） | `data/datasets_faceswap.py` |
| 权重与 SD/VAE 缓存 | `./checkpoints`（需自行下载或首次运行拉取） |
| 依赖版本锁定 | `requirements.txt`（勿用 `--upgrade`） |

**最小可跑命令**（在**本目录**下执行）：
```bash
python infer.py -s ./example/src -t ./example/tgt -o ./output -ckpt ./checkpoints
```
使用项目组数据集时，将 `-s`、`-t` 指向从 Dataset-CelebA_HQ 准备的两个图片目录即可。

---

## 3) 写作者/协作者注意事项

- **依赖与版本**
  - 严格按 `download-README.md` 和 `requirements.txt` 安装，**不要**先单独 `pip install diffusers`/`transformers` 再装 requirements，易冲突。
  - **禁止**对 `requirements.txt` 使用 `pip install -r requirements.txt --upgrade`，会破坏锁定版本。
  - NumPy 需 1.x（如 `1.26.4`），不要升级到 2.x；peft/accelerate 版本见 download-README 踩坑清单。

- **数据与路径**
  - 推理输入为**两个目录**：`--source`（源脸）、`--target`（目标脸），支持 `.jpg`/`.png`/`.jpeg`。
  - 与项目组统一数据对接时，应使用 **Dataset-CelebA_HQ** 的路径或由其 CSV 派生的两个目录，避免写死其他项目本地路径。

- **代码与配置**
  - 人脸检测与 5 点由 InsightFace 在运行时完成；`data/datasets_faceswap.py` 仅提供几何变换工具，**不包含** PyTorch Dataset 类。
  - 若新增“从 CSV 读路径”等逻辑，建议保持与现有“两目录”接口兼容，或在本目录内增加薄封装脚本，便于协作者统一使用。

- **文档**
  - 环境、下载、快速测试以 **download-README.md** 为准；算法与引用以 **README.md** 为准。修改环境或入口时请同步更新 download-README 和本文档。

---

## 4) 协作者/Agent 快速上手流程

1. **读文档顺序**：先读本文件（`README_COLLAB.md`）了解分支与注意点，再读 **download-README.md** 完成环境与模型下载。
2. **环境**：按 download-README 创建 conda 环境（如 `FaceAdapter`）、安装 PyTorch 与 `requirements.txt`，且在本目录下执行 pip。
3. **权重**：运行一次 `infer.py` 或从 Hugging Face 将 Face-Adapter 与 SD/VAE 放到 `./checkpoints`（或按 download-README 指定位置）。
4. **跑通测试**：使用自带 example 或从 Dataset-CelebA_HQ 准备两目录，执行：
   ```bash
   python infer.py -s <源脸目录> -t <目标脸目录> -o ./output -ckpt ./checkpoints
   ```
5. **输出**：换脸图在 `-o` 目录下的 `swap/`，对比图在 `concat/`。

---

## 5) 与项目其他部分的关系

- **数据集**：统一使用 **Dataset-CelebA_HQ**（`train/val/test` + CSV）。本模型不读 CSV，需由调用方或脚本从 CSV 准备两个图片目录再传入。
- **上游/下游**：本目录作为“攻击模型”被其他模块（如 Forensic-SepMark、水印评测流程）调用或对比；修改接口时请考虑对调用方的影响。
- **与 Attack-REFace**：同为项目组复现的换脸模型，数据源一致，但 REFace 需要人脸裁剪与 mask，入口与脚本不同（见 `Attack-REFace-main/README_COLLAB.md`）。

---

## 6) 变更本文档时建议

- 若分支用途、数据约定或入口发生变更，请更新“当前分支/复现目标”和“关键文件与入口”。
- 若依赖或环境步骤变更，请同步更新 **download-README.md** 并在“写作者注意事项”中注明。
- 新增重要脚本或配置时，在“关键文件与入口”中补充路径，便于 Agent 解析。
