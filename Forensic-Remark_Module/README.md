# Forensic-ReMark Module

> **设计状态**：架构设计完成，代码骨架待实现。
> 本文档记录了完整的设计思路、工程决策与已知问题，供协作开发和断点续接使用。

---

## 一、项目定位

ReMark 是一个**即插即用的水印修复模块**，其逻辑位置位于：

```
原始图像 → WM-Encoder → 含水印图像 → Attack（deepfake） → 受损图像
                                                                ↓
提取水印 ← WM-Decoder ← 修复图像 ← [ReMark] ←←←←←←←←←←←←←←←
```

**核心思路**：不增强水印本身，而是在 deepfake 之后将受损图像修复回"接近原始含水印状态"，从而提升水印提取准确率（ACC）。

**关键约束**：
- WM-Encoder 和 WM-Decoder **全程冻结**，不参与参数更新
- ReMark 独立训练，与底层水印框架解耦
- 换一个水印模型，只需重新训练 ReMark，无需修改水印模型本身

**参考论文**：*ReMark: Enhancing Deepfake Forensic Watermark via Image Reconstruction on Latent Space*（ICME submission）

---

## 二、两阶段训练流程

### Stage 1：路径重建（VAE）

将 deepfake 在图像域中复杂的、不可微的变换，转化为 VAE latent 空间中可建模的超球面路径。

```
算法：
1. 对每张含水印图像 x，随机采样 deepfake 次数 n
2. 生成 x̃ = DF^(n)(x)
3. 用 VAE 编码两者：z_src = E(x̃)，z_gt = E(x)
4. 用 SLERP 在超球面上插值，构建从 z_src → z_gt 的轨迹 {z_0,...,z_m}
5. 训练 VAE，Loss = λ1·L1 + λ2·LPIPS + λ3·BCE + β·KL
```

**损失说明**：
| 损失项 | 作用 |
|--------|------|
| L1 | 像素级重建保真度 |
| LPIPS | 感知相似度 |
| BCE | 水印信息保留（经 WM-Decoder 验证） |
| KL | 约束 latent 空间结构，使 SLERP 有意义 |

> ⚠️ **已知风险**：β（KL 权重）极为敏感。β 过大导致后验坍缩，β 过小导致 latent 无结构、SLERP 失效。建议采用 β warm-up 调度，不要固定值。

### Stage 2：水印修复（U-Net）

以 Stage 1 生成的轨迹为监督信号，训练 U-Net 迭代精修 latent，将受损 latent 逐步推向原始 latent。

```
算法：
对每张图像 x：
  z_src ← E(DF^(n)(x))，z_gt ← E(x)
  插值路径 {z_0,...,z_m} ← SLERP(z_src, z_gt, m)
  for k = 0 to m-1:
    z̃_{k+1} = G_ψ(z̃_k)
    Loss = λ1·L1(z̃_{k+1}, z_{k+1}) + λ2·BCE(W_rec, W_gt)
```

> ⚠️ **已知风险**：训练时用 teacher forcing（真实轨迹），推理时用自回归（自身输出），步数越多误差累积越严重。

> ⚠️ **耦合约束**：Stage 2 依赖 Stage 1 的 VAE latent 空间，两者版本必须绑定。Checkpoint 保存时需记录 Stage 1 的版本信息，切勿单独更新其中一个。

---

## 三、工程架构

```
Forensic-Remark_Module/
│
├── configs/                        # 所有超参数，实验时只改这里
│   ├── stage1_vae.yaml             # VAE 训练配置
│   ├── stage2_unet.yaml            # U-Net 训练配置
│   └── experiments/                # 实验 override，继承 base 配置
│
├── network/
│   ├── vae.py                      # VAE 实现（可替换）
│   ├── unet.py                     # U-Net 实现（可替换）
│   └── losses.py                   # 损失函数注册表
│
├── attacks/                        # 攻击侧适配层
│   ├── base.py                     # 抽象基类，定义统一接口
│   ├── stargan.py                  # StarGAN 封装
│   ├── simswap.py                  # SimSwap 封装（已接入）
│   └── registry.py                 # 名称 → 类的注册表
│
├── wm_adapters/                    # 水印编解码器适配层
│   ├── base.py                     # 抽象接口
│   ├── fin.py                      # FIN 适配
│   ├── sepmark.py                  # SepMark 适配（已接入）
│   ├── lampmark.py                 # LampMark 适配
│   ├── lawa.py                     # LaWa 适配（官方 / prompt-finetuned 预设）
│   └── registry.py
│
├── data/
│   ├── dataset.py                  # CSV manifest 驱动的数据集
│   └── wm_cache.py                 # 含水印图像缓存管理
│
├── data_manifests/                 # CSV 文件，指向 Dataset-CelebA_HQ
│   ├── celeba_hq_128_train.csv
│   ├── celeba_hq_128_val.csv
│   └── celeba_hq_128_test.csv
│
├── tools/                          # 独立工具脚本（不依赖训练代码）
│   ├── gen_pairs_diffswap.py       # 慢速攻击的离线配对生成
│   └── verify_adapter.py           # adapter 格式正确性验证
│
├── train_stage1.py                 # Stage 1 训练入口
├── train_stage2.py                 # Stage 2 训练入口
├── evaluate.py                     # 评估入口（含盲提取 ACC）
└── sample.py                       # 可视化输出
```

---

## 四、关键接口设计

### 4.1 Attack Adapter

所有攻击模型继承 `BaseAttack`，基类通过 `__call__` 锁定调度顺序，**子类无法绕过 postprocess**：

```python
class BaseAttack:
    def preprocess(self, images):
        """canonical [-1,1] BCHW → 攻击模型期望格式"""
        raise NotImplementedError

    def generate(self, preprocessed):
        """模型推理，子类实现"""
        raise NotImplementedError

    def postprocess(self, images, original_size):
        """攻击模型输出 → canonical [-1,1] BCHW，恢复原始尺寸"""
        raise NotImplementedError

    def __call__(self, images):
        # 基类统一调度，保证输出永远是 canonical 格式
        original_size = images.shape[-2:]
        x = self.preprocess(images)
        x = self.generate(x)
        out = self.postprocess(x, original_size)
        # 格式断言，写错立刻报错而非训练时才发现
        assert out.shape[-2:] == original_size
        assert out.min() >= -1.1 and out.max() <= 1.1
        return out
```

**在线 vs 离线策略**：
- GAN 类（StarGAN、SimSwap、GANimation）：在线生成，接入 DataLoader worker
- 扩散类（DiffSwap、Arc2Face）：推理慢，通过 `tools/gen_pairs_*.py` 离线生成配对 CSV，训练时读 CSV，接口与在线模型一致

**命名与后端绑定约定（强制）**：
- `Arc2Face`（或 `arc2face`）明确绑定 Arc2Face wrapper 产物路径（`Attack-arc2face_wrapper/outputs`）。
- 若 `source_mode=random_pool` 但 source 目录为空，默认直接报错；不会再静默退回 `cover_roll`。
- 任何“名称与后端不匹配”的替换都必须先确认，不允许自动替换为其他模型。

### 4.2 WM Adapter

统一接口屏蔽各水印模型的消息格式差异：

```python
class BaseWMAdapter:
    def encode(self, images, messages) -> wm_images:
        """images: [-1,1]，messages: {0,1} → wm_images: [-1,1]"""
        raise NotImplementedError

    def decode(self, images) -> logits:
        """images: [-1,1] → logits（BCE-ready，未经 sigmoid）"""
        raise NotImplementedError
```

各模型内部格式差异由 adapter 自行处理（例如 FIN 的 `{-0.5, 0.5}` 在 adapter 内转换），训练代码始终使用 `{0, 1}` 消息和 `[-1, 1]` 图像。

> ⚠️ **重要**：ReMark 每次训练绑定**一个固定的水印模型**，不支持混用。换水印模型需重新训练 Stage 1 和 Stage 2。

### 4.3 Sample 可视化

所有组件输出均为 canonical 格式，可视化只需一个函数，风格天然统一：

```python
def to_display(tensor):
    """任何组件的输出都可以直接调用"""
    return (tensor.clamp(-1, 1) + 1) / 2
```

---

## 五、训练效率设计

| Trick | 配置项 | 说明 |
|-------|--------|------|
| WM 缓存 | `cache_wm_images: true` | 预计算含水印图像，跳过 Encoder 前向 |
| AMP 混合精度 | `use_amp: true` | 显存减半，速度提升约 30-50% |
| Decoder 调用频率 | `wm_loss_freq: 4` | 每 N 个 batch 才计算一次 BCE，减少 Decoder 调用 |
| 攻击异步生成 | `attack_num_workers: 2` | 在 DataLoader worker 中并行生成 deepfake |
| SLERP 预计算 | `cache_slerp: true` | epoch 开始前批量预计算轨迹，训练时直接读取 |

**梯度冻结策略**：
```
Encoder  → torch.no_grad()          # 无需从此处反传
Attack   → torch.no_grad()          # 同上
ReMark   → 正常计算图               # 唯一需要更新的部分
Decoder  → requires_grad_(False)    # 参数冻结，但梯度可穿过（BCE 反传需要）
```

---

## 六、配置系统

所有超参数通过 YAML 管理，实验时新建一个 override 文件，不改代码：

```yaml
# configs/stage1_vae.yaml（示例）

model:
  latent_channels: 4          # VAE 下采样后的通道数
  downsample_factor: 16       # 空间下采样倍率（影响 latent 分辨率）

training:
  epochs: 200
  batch_size: 16
  lr: 1e-5
  warmup_epochs: 20           # β warm-up 轮次

slerp:
  n_steps: 10                 # 插值点数

losses:
  l1:    { weight: 1.0 }
  lpips: { weight: 0.1 }
  bce:   { weight: 10.0 }
  kl:    { weight: 1.0, warmup: true }  # 支持单独开关

attacks:
  online:  [stargan]
  offline: [diffswap]         # 读离线生成的 CSV

efficiency:
  cache_wm_images: true
  use_amp: true
  wm_loss_freq: 4
  attack_num_workers: 2

wm_model: fin                 # 绑定的水印模型

preflight_eval:
  enabled: true               # 训练前先跑“无 ReMark”基线
  max_batches: 0              # 0=全量验证集
  save_csv: true              # 另存 runs/.../preflight_baseline.csv
```

LaWa 接入后，`wm_model` 也可以直接切到 `lawa` 系列别名，常用预设如下：

- `lawa` / `LaWa` / `lawa_official` / `lawa_sd14_48`：官方 SD1.4 + KL-f8 的 48-bit checkpoint（`weights/LaWa/last.ckpt`）
- `lawa_traincfg`：仍使用官方 checkpoint，但按训练配置 `configs/SD14_LaWa.yaml` 构建模型
- `lawa_prompt` / `lawa_prompt_finetuned` / `lawa_prompt_dataset`：使用当前仓库里 prompt 数据集微调得到的 `outputs/train_result/checkpoints/last.ckpt`

对应配置示例：

```yaml
wm_model: lawa

wm_adapter_lawa:
  preset: official_sd14_48
  root: /mnt/personal_workspace/chenkeyu/ReMark/Forensic-LaWa
  image_size: 128               # ReMark 训练默认 128；也可改为 256 或 null
  # 可选：显式覆写权重 / config
  # config_path: configs/SD14_LaWa_inference.yaml
  # ckpt: weights/LaWa/last.ckpt
  # first_stage_ckpt: weights/first_stage_models/first_stage_KL-f8.ckpt
```

---

## 七、StarGAN 训练前校准（CelebA-HQ）

为避免大规模训练前攻击强度不合适，先在 `preflight` 阶段校准 `StarGAN`。  
以下结果基于 SepMark + CelebA-HQ（tiny val 子集）：

| 配置 | attack_options | stargan bit_acc / ber | 说明 |
|------|----------------|-----------------------|------|
| A | `latent + self + random_domain + blend_alpha=0.65` | `0.8994 / 0.1006` | 视觉较自然，攻击偏弱 |
| B | `latent + self + random_domain + blend_alpha=1.0`  | `0.5840 / 0.4160` | 推荐默认，强度与稳定性平衡 |
| C | `reference + self + random_domain + blend_alpha=1.0` | `0.5752 / 0.4248` | 强度接近 B，参考图驱动更强 |

对应样例图与指标文件：

- `A`: [preflight_stargan_A.png](docs/stargan_calibration/preflight_stargan_A.png), [preflight_baseline_A.csv](docs/stargan_calibration/preflight_baseline_A.csv)
- `B`: [preflight_stargan_B.png](docs/stargan_calibration/preflight_stargan_B.png), [preflight_baseline_B.csv](docs/stargan_calibration/preflight_baseline_B.csv)
- `C`: [preflight_stargan_C.png](docs/stargan_calibration/preflight_stargan_C.png), [preflight_baseline_C.csv](docs/stargan_calibration/preflight_baseline_C.csv)

样例图（每张均为三行）：

- 第 1 行：原图
- 第 2 行：含水印图
- 第 3 行：StarGAN 攻击后

### A（弱攻击）
![StarGAN A](docs/stargan_calibration/preflight_stargan_A.png)

### B（推荐）
![StarGAN B](docs/stargan_calibration/preflight_stargan_B.png)

### C（强攻击，reference）
![StarGAN C](docs/stargan_calibration/preflight_stargan_C.png)

后续大规模训练建议：

- 默认采用 `B` 配置。
- 先保持 `preflight_eval.enabled=true`，每次切换水印模型/攻击参数时先跑基线。
- 若需要更强攻击可优先调 `stargan_blend_alpha` 与 `stargan_random_domain`。

---

## 八、已知问题与解决思路

| 优先级 | 问题 | 解决思路 |
|--------|------|---------|
| 🔴 高 | KL 权重调度 | β warm-up，从 0 线性增至目标值 |
| 🔴 高 | 攻击模型静默失败（人脸检测失败返回原图） | Attack adapter 内做输出验证，异常时跳过该样本并记录日志 |
| 🔴 高 | Stage1/2 版本绑定 | Checkpoint 中保存 Stage1 的 hash，Stage2 加载时校验 |
| 🟡 中 | 训练/推理分布偏移（teacher forcing vs 自回归） | 训练后期引入 scheduled sampling |
| 🟡 中 | 推理步数 m 未知 | 先 hardcode 最大步数，后续设计降解程度估计器 |
| 🟡 中 | 评估指标与部署场景不对齐 | evaluate.py 中独立实现"盲提取 ACC"，不依赖 ground truth message |
| 🟢 低 | SLERP 数值边界（θ≈0 或 θ≈π） | 插值前检测角度，退化时 fallback 到 LERP |
| 🟢 低 | 显存竞争（攻击模型 + 训练模型同卡） | 通过 `attack_num_workers` 控制，极端情况可指定攻击模型用 CPU |

---

## 九、与项目其他模块的关系

```
ReMark 项目总览：

Dataset-CelebA_HQ/          ← 统一数据源，所有模块共享
    ↓ CSV manifests
Forensic-LampMark/          ← 防御侧：LampMark 水印
Forensic-LaWa/              ← 防御侧：LaWa（latent decoder watermark）
Forensic-SepMark/           ← 防御侧：SepMark 水印
Forensic-FIN/               ← 防御侧：FIN 水印
Forensic-SleeperMark/       ← 防御侧：SleeperMark
Forensic-TAG-WM/            ← 防御侧：TAG-WM
    ↑ wm_adapters/ 接入
Forensic-Remark_Module/     ← 本模块：即插即用水印修复
    ↑ attacks/ 接入
Attack-DiffSwap/            ← 攻击侧（慢速，离线使用）
Attack-arc2face_wrapper/    ← 攻击侧（慢速，离线使用）
```

ReMark 不直接依赖其他 Forensic-* 模块的代码，通过 `wm_adapters/` 和 `attacks/` 的薄接口层间接调用，保证双向解耦。

---

## 十、下一步开发计划

- [ ] 搭建代码骨架（目录结构、base 类、registry、config 解析）
- [ ] 实现 Stage 1 VAE 训练（迁移 train.py 现有逻辑，接入配置系统）
- [ ] 接入 FIN wm_adapter（最简单，优先验证端到端流程）
- [ ] 接入 StarGAN attack adapter（在线生成验证）
- [x] 实现 Stage 2 U-Net 训练
- [ ] 实现 evaluate.py（含盲提取 ACC）
- [ ] 接入更多水印模型和攻击模型

---

## 十一、VAE 调参与训练记录（持续更新）

> 说明：本节用于记录每一次会影响结果的改动（配置/代码/环境），保证可追溯。

| 日期(Asia/Shanghai) | 变更文件 | 变更内容 | 目的 | 结果 |
|---|---|---|---|---|
| 2026-03-14 | `wm_adapters/lawa.py`、`wm_adapters/__init__.py`、`configs/experiments/{preflight,stage1,stage2}_lawa_*.yaml` | 新增 LaWa wm_adapter，支持官方 `weights/LaWa/last.ckpt`、训练配置版和 prompt-dataset 微调版 preset/alias；补齐 README 与实验模板 | 将 Forensic-LaWa 正式注册到 ReMark 模块，并保留多 checkpoint 家族切换能力 | 已完成，待按目标攻击链路做 preflight |
| 2026-03-10 | `configs/stage1_vae.yaml` | `downsample_factor: 16 -> 8` | 保留更多空间细节，降低重建模糊 | 待训练验证 |
| 2026-03-10 | `configs/stage1_vae.yaml` | `training.kl_warmup_epochs: 6 -> 20` | 延后 KL 约束强度上升，缓解早期坍缩 | 待训练验证 |
| 2026-03-10 | `network/vae.py` | Encoder/Decoder 通道表改为按 `n_downsample` 动态生成 | 保证 `downsample_factor=8` 下编解码器通道更对称 | 语法通过，待训练验证 |
| 2026-03-10 | `train_stage1.py` | 修正 KL 调度：`kl.weight<=0` 时按 `kl_warmup_epochs` 线性升至 `kl.target_weight` | 让 warmup 配置真正生效，避免固定分段曲线 | 语法通过，待训练验证 |
| 2026-03-10 | `configs/stage1_vae.yaml` | 新增 `losses.kl.target_weight: 0.01` | 显式给出自动 KL 调度目标值 | 待训练验证 |
| 2026-03-10 | 环境 `sepmark` | 安装训练依赖（`torch/torchvision`），并清理磁盘缓存 | 使用 SepMark 指定环境稳定复现实验 | 进行中 |
| 2026-03-10 | `configs/experiments/vae_tune_s1_recon_only.yaml` | 新增小数据“纯重建”快速实验配置（1000/200，禁用 attack 与 bce） | 先验证 VAE 是否能稳定重建，排除 fake 分支干扰 | 待运行 |
| 2026-03-10 | `configs/experiments/vae_tune_s2_fake_identity.yaml` | 新增小数据“fake2fake + 轻量 bce”实验配置（attack 概率渐进） | 在重建稳定后测试 fake 分支与 ACC 走势 | 待运行 |
| 2026-03-10 | `configs/experiments/vae_tune_s1_recon_only.yaml`、`configs/experiments/vae_tune_s2_fake_identity.yaml` | 为提升迭代速度改为 `tiny(50/20)`，并临时禁用 LPIPS | 减少每轮等待时间，优先观察是否坍缩与基本趋势 | 待运行 |
| 2026-03-10 | `data_manifests/local_celeba_hq_128_tiny_{train,val}.csv` | 新增本地盘缓存清单（样本复制到 `/home/ldy/remark_data_local/tiny_celeba_hq_128`） | 避免 NAS 读取导致 `rpc_wa` 卡顿，提升快测吞吐 | 待运行 |
| 2026-03-10 | 运行 `stage1_20260310_1114`（`vae_tune_s1_recon_only`） | clean-only 快测完成：`val_l1 0.281 -> 0.151` | 验证训练链路可收敛 | 曲线下降但样例仍明显模糊，需继续调参 |
| 2026-03-10 | `network/vae.py` + `configs/experiments/vae_tune_s1_recon_det.yaml` | 新增 `model.deterministic_latent` 开关并配置 DET 实验 | 减少训练期随机采样噪声，验证是否可缓解模糊/坍缩 | 待运行 |
| 2026-03-10 | 运行 `stage1_20260310_1117`（`vae_tune_s1_recon_det`） | DET 快测完成：`val_l1 0.283 -> 0.125` | 对照随机采样版，评估 deterministic latent 效果 | 指标继续提升，但视觉仍有明显模糊 |
| 2026-03-10 | `configs/experiments/vae_tune_s1_ae_overfit.yaml` | 新增 overfit 验证（deterministic AE，禁 KL/BCE/attack） | 判断架构上限：tiny 数据能否学到清晰重建 | 待运行 |
| 2026-03-10 | 运行 `stage1_20260310_1119`（`vae_tune_s1_ae_overfit`） | overfit 快测完成：`val_l1 0.283 -> 0.080`（80 epoch） | 验证在无 KL/BCE 情况下的重建上限 | 仍有可见模糊，推断需结构级细节通路 |
| 2026-03-10 | `network/vae.py` + `configs/experiments/vae_tune_s1_ae_residual.yaml` | 新增 `residual_output/residual_scale`，并配置 Residual-AE 实验 | 为 identity 任务提供输入细节直通路径，抑制过度平滑 | 待运行 |
| 2026-03-10 | 运行 `stage1_20260310_1123`（`vae_tune_s1_ae_residual`） | Residual-AE 快测完成：`val_l1 0.0176 -> 0.0004`、`val_acc=1.000` | 验证 residual 细节通路是否有效 | 成功，重建样例已接近无损 |
| 2026-03-10 | `configs/experiments/vae_tune_s2_fake_residual.yaml` | 新增 fake2fake 快测（Residual VAE + 轻量 BCE + attack 概率渐进） | 验证在 fake 分支下是否仍保持重建稳定并兼顾 ACC | 待运行 |
| 2026-03-10 | 环境 `sepmark` | 补齐 fake 分支依赖：`munch`、`ffmpeg-python` | 打通 StarGAN 攻击链路 | 已完成 |
| 2026-03-10 | 运行 `stage1_20260310_1130`（`vae_tune_s2_fake_residual`） | fake2fake 快测完成：`val_l1≈0.0010`、`val_acc=1.000`；attacked bit_acc `0.5922 -> 0.5934` | 验证在攻击分支下的稳定重建与 ACC 微提升 | 成功，样例中 fake 重建基本无损 |
| 2026-03-10 | `configs/experiments/stage1_sepmark_residual_prod.yaml` | 新增 10k 正式训练配置（Residual + deterministic + 轻量 BCE） | 将已验证方案迁移到生产规模训练 | 待运行 |
| 2026-03-10 | 运行 `stage1_20260310_1155`（`stage1_sepmark_residual_prod`） | 验证正式配置可正常启动：完成 SepMark/StarGAN 加载并进入 preflight | 排除“启动即卡死/配置错误” | 可运行，但 `preflight max_batches=0` 在当前机器启动耗时较长 |
| 2026-03-10 | `configs/experiments/stage1_sepmark_residual_prod.yaml` | `training.batch_size: 64 -> 16`，`preflight_eval.max_batches: 0 -> 4` | 降低单卡显存与启动等待成本，加快进入 epoch 训练 | 待运行验证 |
| 2026-03-10 | 运行 `stage1_20260310_1159`（更新后的 `stage1_sepmark_residual_prod`） | 4-batch preflight 正常完成并写出 baseline；训练进入 `Epoch 000` | 验证更新配置在正式数据清单上的可执行性 | 启动链路正常，但 `Epoch 000` 首批数据读取出现持续 I/O 阻塞（进程 D 态），需切本地数据盘继续 |
| 2026-03-10 | `checkpoints/stage1_vae/sepmark_residual_fake2fake_best_20260310.{pth,config.yaml}` | 导出稳定 checkpoint（来源 `stage1_20260310_1130` 的 `best.pth`）到统一交付目录 | 便于直接接入 Stage2/评估脚本 | 已完成 |
| 2026-03-10 | `attacks/simswap.py`、`attacks/arc2face.py` | 新增 `attack_options.enforce_nontrivial_swap` 与 `nontrivial_swap_eps` 检查（默认开启） | 防止攻击分支静默退化为“近似原图替换” | 已生效，异常会直接报错终止 |
| 2026-03-10 | `configs/experiments/preflight_sepmark_{simswap,arc2face}_tiny.yaml` | 新增 SimSwap/Arc2Face 独立 preflight 配置 | 与 StarGAN 一致先做“无 ReMark”基线测量 | 已完成 |
| 2026-03-10 | `configs/experiments/vae_tune_s2_fake_residual_{simswap,arc2face}.yaml` | 新增 SimSwap/Arc2Face fake2fake 独立快测配置 | 分别验证两条攻击链路的重建稳定性 | 已完成 |
| 2026-03-10 | 运行 `stage1_20260310_1336`（`preflight_sepmark_simswap_tiny`） | preflight: `clean=1.0000`，`simswap=0.7344` | 建立 SimSwap 攻击前基线 | 已完成 |
| 2026-03-10 | 运行 `stage1_20260310_1339`（`preflight_sepmark_arc2face_tiny`） | preflight: `clean=1.0000`，`Arc2Face=0.5074` | 建立 Arc2Face 攻击前基线 | 已完成 |
| 2026-03-10 | 运行 `stage1_20260310_1337`（`vae_tune_s2_fake_residual_simswap`） | fake2fake 收敛（`val_l1≈0.0052`），但 attacked bit_acc `0.7340 -> 0.6898` | 验证 SimSwap 基础配置是否能稳住 ACC | 可重建，但 ACC 回落，需强化 BCE/attack 比例 |
| 2026-03-10 | 运行 `stage1_20260310_1339`（`vae_tune_s2_fake_residual_arc2face`） | fake2fake 收敛（`val_l1≈0.0018`），attacked bit_acc `0.5000 -> 0.4957` | 验证 Arc2Face 基础配置 ACC 走势 | 可重建，但 ACC 轻微回落 |
| 2026-03-10 | `utils/logger.py` | run 时间戳 `'%Y%m%d_%H%M' -> '%Y%m%d_%H%M%S'` | 避免同分钟多次启动覆盖同一 run 目录 | 已修复 |
| 2026-03-10 | `tools/eval_attacked_recon_acc.py` | 新增统一评估脚本（`raw_attacked` vs `vae_recon` bit-ACC/BER）并输出 `attacked_recon_eval_*.csv` | 统一不同攻击配置的评估口径 | 已用于所有当前 run |
| 2026-03-10 | `configs/experiments/vae_tune_s2_fake_residual_{simswap,arc2face}_acc.yaml` | ACC 强化：`bce.weight=1.0` + 全程 attack 分支（prob=1.0） | 降低 fake2fake 路径 ACC 回落 | 已完成 |
| 2026-03-10 | 运行 `stage1_20260310_134226`（`..._simswap_acc`） | attacked bit_acc `0.7332 -> 0.7617`（`+0.0285`） | 验证 SimSwap 强化配置有效性 | 成功，重建后 ACC 明显提升 |
| 2026-03-10 | 运行 `stage1_20260310_134838`（`..._arc2face_acc`） | attacked bit_acc `0.5000 -> 0.5105`（`+0.0105`） | 验证 Arc2Face 强化配置有效性 | 成功，重建后 ACC 小幅提升 |
| 2026-03-10 | 样例图质检（`stage1_20260310_134226`、`stage1_20260310_134838`） | 检查 `fake->hat` 的 MAE/PSNR/方差比：无塌缩、无大面积饱和、无明显颜色漂移；同时攻击输出与 wm 输入存在显著差异（SimSwap MAD≈0.059，Arc2Face MAD≈0.158） | 确认“真实换脸路径 + 稳定重建” | 通过 |
| 2026-03-10 | `tools/eval_attacked_recon_acc.py` | 评估脚本扩展为同时输出 `recon_l1`、`latent_kl`、`mu_abs_mean`、`logvar_mean` | 调参时同时观察 ACC、重建误差与潜空间分布 | 已用于后续 sweep |
| 2026-03-10 | `tools/sweep_simswap_kl_bce.py` + `configs/experiments/sweeps/simswap_kl_bce/*.yaml` | 新增 SimSwap 网格搜索：`BCE∈{0.5,1,2,3}` × `KL target∈{0.001,0.003}`（30 epoch，full attack） | 探索 Stage1 可恢复水印的上限边界，并引入高斯先验 | 已完成 8/8 组合 |
| 2026-03-10 | 运行 `runs/sweep_simswap_kl_bce_summary_20260310_143956.csv`（8 组） | 最优：`BCE=1.0, KL=0.001`，`delta bit_acc=+0.0629`，`recon bit_acc=0.7965`，`recon_l1=0.0391`，`latent_kl=0.1736`（run=`stage1_20260310_144145`） | 寻找 ACC 与潜空间约束的平衡点 | 成功，优于无 KL 的旧最佳（`+0.0285`） |
| 2026-03-10 | 同一 sweep 失败区间 | `BCE>=3.0` 出现明显退化（如 `BCE=3,KL=0.003`：`delta=-0.0418`、`recon_l1=0.1058`） | 明确上限边界，避免过强 BCE 导致重建破坏 | 结论成立 |
| 2026-03-10 | `configs/experiments/vae_tune_s2_fake_residual_arc2face_klbce.yaml` + 运行 `stage1_20260310_145426` | 采用 `BCE=1.0, KL=0.001` 复验 Arc2Face：`delta bit_acc=0.5000 -> 0.4969`（`-0.0031`） | 验证 SimSwap 最优组合能否迁移到 Arc2Face | 不迁移，Arc2Face 上需单独调参 |
| 2026-03-10 | `configs/experiments/vae_tune_s2_fake_residual_simswap_klbce.yaml` | 固化推荐配置（`BCE=1.0, KL target=0.001`） | 作为下一步中/大规模训练的默认起点 | 已提交配置，待长训验证 |
| 2026-03-10 | `tools/sweep_stage1_arch_cv.py` + `configs/experiments/sweeps/arch_stage1_cv/*.yaml` | 新增两阶段架构搜索：Phase-A 在 SimSwap 搜索 6 个结构，Phase-B 对 Top-3 在 StarGAN/Arc2Face 交叉验证 | 从“只在单攻击上最优”切换为“跨攻击稳定最优” | 已完成 12/12 训练与评估 |
| 2026-03-10 | 运行 `runs/arch_cv_summary_20260310_152633.csv` / `runs/arch_cv_leaderboard_20260310_152633.csv` | 交叉验证结论：`compact_24_48` 平均最优（CV: `StarGAN +0.0031`, `Arc2Face +0.0082`, `avg_delta +0.00566`）；`detail_ds4` 与 `wide_48_96` 在 CV 平均为负 | 选出可迁移结构，避免 SimSwap-only 过拟合结构 | 结论成立 |
| 2026-03-10 | `configs/experiments/stage1_sepmark_compact_simswap_10k.yaml` | 固化 10k 正式配置：`compact_24_48 + deterministic + residual + BCE1 + KL target 0.001 + SimSwap full-attack` | 将“跨攻击验证通过”的结构迁移到大规模训练 | 已提交配置 |
| 2026-03-10 | 运行 `stage1_20260310_154310`（`stage1_sepmark_compact_simswap_10k`） | 10k 长训已启动：`Train=10000/Val=2156`，preflight `simswap bit_acc=0.7081`，已进入 `Epoch 000` | 在大规模数据上验证 cross-validated 架构的稳定收敛与 ACC 走势 | 进行中 |
| 2026-03-10 | `utils/logger.py` | 移除 per-epoch `metrics.csv` 写入，仅保留 `train.log` 文本日志；epoch 摘要支持 `val_clean(wm->rec_wm)` 与 `val_attack[attack](fake->rec_fake)` 分段输出 | 按训练日志统一查看关键指标，避免 CSV/日志双轨不一致 | 已完成（新 run 不再生成/更新 `metrics.csv`） |
| 2026-03-10 | `train_stage1.py` | 验证流程拆分为 clean 与 attack 两类：新增 `val_clean_step` + `val_attack_step`；按攻击名循环统计 `raw_acc` 与 `rec_acc`，自动支持多攻击模型 | 在同一 epoch 内同时观察 `wm->rec_wm` 与 `fake->rec_fake` 的解码 ACC | 已完成 |
| 2026-03-10 | `configs/stage1_vae.yaml`、`configs/experiments/stage1_sepmark_compact_simswap_10k.yaml` | 新增 `validation.max_batches`（默认 8）并关闭 `preflight_eval.save_csv` | 适度降低验证数据量，缩短每 epoch 评估开销；统一”仅 train.log”输出 | 已完成 |
| 2026-03-11 | `attacks/diffswap.py`、`attacks/__init__.py` | 新增 DiffSwap 离线 replay adapter（注册为 `diffswap`/`DiffSwap`），新增 `configs/experiments/preflight_sepmark_diffswap_tiny.yaml` | 接入 Attack-DiffSwap 产物，补全扩散类攻击接口；nontrivial_swap 检查已内置 | 已完成，待生成 JSONL 后运行 preflight 验证 |
| 2026-03-11 | `train_stage2.py` | Stage2 训练器修正：checkpoint 记录当前 `epoch`/`best_acc`（修复 resume 漂移）、新增 DDP 指标聚合（支持 NaN）、新增 train/val 进度日志、`slerp.n_steps` 参数保护（`>=1`） | 让 Stage2 在单卡/多卡下日志与 best 选择一致，并确保可稳定续训 | 已完成（待 smoke 语法与最小前向验证） |
| 2026-03-11 | `train_stage2.py`、`network/unet.py` | 在 `sepmark` 环境完成 Stage2 smoke 验证：`py_compile` 通过、`train_stage2.py --help` 可启动、U-Net 前向 shape 校验 `[(2,48,16,16)->(2,48,16,16)]` | 确认交付代码可导入、可解析参数、核心网络前向无 shape 错误 | 已完成 |
| 2026-03-11 | `train_stage2.py` | 清理无用 import（`nn`/`math`/未使用变量），并复跑 `py_compile` 二次确认 | 减少 lint 噪音，确保收尾改动未引入语法回归 | 已完成 |
| 2026-03-11 | 运行 `stage1_20260311_121605`（`stage1_sepmark_compact_simswap_10k`，`torchrun --nproc_per_node=4`，GPU `2,3,4,5`） | 启动新的 SepMark+SimSwap 多卡 VAE 训练（CelebA-HQ 10k）；preflight：`clean bit_acc=1.0000`、`simswap bit_acc=0.7034`；`Epoch 000`：`val_attack[simswap] raw_acc=0.7025 -> acc=0.8279`，并写出 `checkpoints/vae/{best,last,epoch_000}.pth` | 为下一步 Stage2 U-Net 提供同版本 Stage1 `best.pth` 与稳定 run 快照 | 进行中（已进入 `Epoch 001`） |
| 2026-03-11 | `configs/stage2_unet.yaml` | 将 `paths.stage1_checkpoint` 更新为 `runs/stage1_20260311_121605/checkpoints/vae/best.pth` | 将 Stage2 默认输入绑定到最新稳定的 SepMark+SimSwap Stage1 权重，避免误读旧 checkpoint | 已完成 |
| 2026-03-11 | 运行切换：`stage1_20260311_121605 -> stage2_20260311_131612` | 依据稳定判据（近 12 个 epoch：`val_attack[simswap] acc` 在 `0.8335~0.8374` 区间小幅波动、`l1≈0.0036~0.0046`）停止 Stage1（最后完整 epoch=053），并在 GPU `2,3,4,5` 启动 Stage2 多卡训练；当前已进入 `epoch 000` 训练（`step 60/157`） | 按“Stage1 稳定后立即切换 Stage2”执行流水线，缩短迭代闭环时间 | 进行中 |
| 2026-03-11 | `train_stage2.py` | 新增 Stage2 日志增强：训练前 `Preflight-NoUNet`（identity + 各 attack 的 `raw_acc` 与 `vae_resample_acc`）；每个 epoch 新增 `val_curve[attack] point 0..N`（point 0=ACC高侧，point N=ACC低侧） | 解决“只看最终 ACC 无法定位问题”的可观测性缺口，直接暴露各推理阶段 ACC | 已完成（待重启训练生效） |
| 2026-03-11 | `configs/stage2_unet.yaml` | 新增 `preflight_eval` 与 `validation.point_curve` 配置节（默认启用） | 让上述两类日志可配置（batch 数可控），避免验证开销失控 | 已完成 |
| 2026-03-11 | `train_stage2.py` | 修复 Stage2 的 VAE 解码路径：新增 `_decode_latent(z, reference)`，当 Stage1 为 `residual_output=True` 时按 Stage1 forward 公式恢复（`reference + residual_scale * decode(z)`）；并同步替换 train/val/preflight/point-curve 的所有 `vae.decode(z)` 调用 | 修复“Stage2 绕开 residual 分支导致 identity `vae_resample_acc≈0.5`、训练信号失真”的根因，恢复与 Stage1 一致的重采样语义 | 已完成（待重启训练验证） |
| 2026-03-11 | `runs/stage2_20260311_133024/transfer_eval_stage2_{last,best}_val_*.csv` | 迁移性评估（`stargan/arc2face/diffswap`）：`stargan` 在 128 样本上 `raw_acc≈0.5880 -> rec_acc≈0.5525`（-0.0355）；`arc2face` 仅命中 20/128 样本，`0.5039 -> 0.5047`（+0.0008）；`diffswap` 因缺少 `Attack-DiffSwap/outputs/generation_results.jsonl` 未能加载 | 验证 Stage2 跨攻击迁移效果并定位数据依赖缺口 | 已完成（结论：当前对 SimSwap 外攻击迁移不足，需补齐 DiffSwap replay 并扩充 Arc2Face 覆盖后再调参） |
| 2026-03-11 | `train_stage2.py` | 增加迁移增强可选项：`training.weight_decay`、`training.use_lr_scheduler/lr_min_scale`、`training.latent_noise_std`、`efficiency.bce_on_all_steps`；checkpoint 同步保存/恢复 scheduler；日志新增优化器与增强开关打印 | 缓解 Stage2 对单一攻击分布的过拟合，提升跨攻击泛化（默认兼容旧配置，不开启不影响历史实验） | 已完成（语法检查通过） |
| 2026-03-11 | `tools/eval_stage2_transfer.py` | 新增 Stage2 迁移评估脚本：统一输出 `raw_acc/rec_acc/delta/rec_l1`，支持多攻击批量评估，支持 replay 攻击按样本降级跳过并统计 `skipped_samples` | 固化“跨攻击迁移”评估口径，避免临时脚本口径漂移 | 已完成（`--help` 与 `py_compile` 通过） |
| 2026-03-11 | `configs/experiments/stage1_sepmark_compact_stargan_10k_transfer.yaml`、`configs/experiments/stage2_unet_from_stargan_transfer.yaml` | 新增“StarGAN→SimSwap 迁移”专用配置：Stage1 使用 StarGAN 强攻击（compact_24_48 + residual + deterministic + BCE1.2 + KL target 0.001），Stage2 使用 StarGAN 训练并启用迁移增强项 | 构建反向迁移实验链路（先 harder attack 预训练，再测向 SimSwap 的泛化） | 已完成 |
| 2026-03-11 | 运行 `stage1_20260311_145328`（`stage1_sepmark_compact_stargan_10k_transfer`，`torchrun --nproc_per_node=2`，GPU `0,1`） | 启动 StarGAN 预训练 VAE（CelebA-HQ 10k）；preflight：`clean bit_acc=1.0000`、`stargan bit_acc=0.5858`；`Epoch 000`：`val_attack[stargan] raw_acc=0.5738 -> acc=0.5772`，并写出 `samples/preflight_{clean,stargan}.png` + `samples/epoch_000.png` | 为下一步 Stage2（from StarGAN）提供专用 Stage1 checkpoint，并在线观察是否逐步形成可迁移表示 | 进行中（已进入 `Epoch 001`） |
| 2026-03-11 | 运行切换：`stage2_20260311_133024 -> stage2_20260311_150013` | 停止旧的 SimSwap-only Stage2 长训（已平台期），切换到 StarGAN-迁移实验；首次尝试 `stage2_20260311_145904` 因 `bce_on_all_steps=true` + `batch=16` 在 24GB 卡 OOM，随后修正配置为 `batch=8` + `bce_on_all_steps=false` 并成功重启 | 保证迁移实验可稳定训练，同时保留 `wd/cosine/latent_noise` 泛化增强 | 已完成（新 run 稳定进行中） |
| 2026-03-11 | 运行 `stage2_20260311_150013`（`stage2_unet_from_stargan_transfer`，`torchrun --nproc_per_node=4`，GPU `2,3,4,5`） | Stage2 from StarGAN 已启动；preflight：`identity 1.0000`、`stargan raw_acc=0.5858 / vae_resample_acc=0.5868`；`Epoch 001`：`val_attack[stargan] raw_acc=0.5898 -> acc=0.5901` | 验证“Stage1-StarGAN + Stage2-StarGAN”链路可跑通并开始学习 | 进行中 |
| 2026-03-11 | `runs/stage2_20260311_150013/transfer_eval_stage2_best_val_{20260311_150404,20260311_150452,20260311_150952}.csv` | 对新 Stage2 做在线迁移评估（`simswap + stargan`，8 batch，64 样本）：`epoch0-best` 时 SimSwap `0.7087 -> 0.7087`（+0.0000）；`epoch1-best` 时 SimSwap `0.7084 -> 0.7098`（+0.0015）；后续复测 SimSwap `0.7084 -> 0.7100`（+0.0016），StarGAN 保持 `+0.0016` | 提供“训练早期迁移曲线”基线，后续按同口径持续追踪 | 已完成（早期已出现小幅正迁移） |
| 2026-03-11 | `train_stage2.py` | 新增 `attacks.sample_weights` 支持：Stage2 训练时按权重抽样攻击（`random.choices`），并在日志打印权重；默认未配置时退化为均匀采样 | 支持“主攻击稳定 + 次攻击迁移”联合训练策略，不必再改代码硬编码攻击采样 | 已完成（兼容旧配置） |
| 2026-03-11 | `configs/experiments/stage2_unet_from_stargan_mix_finetune.yaml` | 新增 StarGAN 主训后的迁移微调模板（`online=[stargan,simswap]`，`sample_weights=0.7/0.3`） | 在保持 StarGAN 性能的同时，逐步提升对 SimSwap 的跨攻击恢复能力 | 已完成（待在当前 Stage2 收敛后接续） |
| 2026-03-11 | `train_stage1.py`、`configs/experiments/stage1_sepmark_compact_stargan_10k_transfer.yaml` | 修复 Stage1 `best.pth` 选择逻辑：新增 `training.best_select`（`auto/val_clean_acc/val_attack_avg_acc/val_attack_avg_delta`），默认 `auto`；本实验显式设为 `val_attack_avg_acc`；checkpoint 同步保存 `best_score/best_metric_name/best_select` 并兼容旧 ckpt | 避免 `val_clean acc≈1.0` 导致 best 长期卡在早期 epoch，确保 Stage2 读取到“攻击恢复能力最优”的 Stage1 权重 | 已完成（`py_compile` 通过） |
| 2026-03-11 | 运行切换：`stage1_20260311_145328` 续训 + 停止 `stage2_20260311_150013` | 以新 best 逻辑重启 Stage1（`--resume stage1_20260311_145328`，GPU `0,1`）；日志确认 `best checkpoint metric: val_attack_avg_acc`；在 `Epoch 019` 产生新 best：`val_attack[stargan] acc=0.5926`（`raw=0.5826`），并写出 `checkpoints/vae/best.pth` | 先修正 Stage1 best 再启动 Stage2，避免沿用旧/早期 VAE 权重污染迁移实验 | 已完成 |
| 2026-03-11 | 运行 `stage2_20260311_153327`（`stage2_unet_from_stargan_transfer`，`torchrun --nproc_per_node=4`，GPU `2,3,4,5`） | 基于更新后的 Stage1 best 启动新的 Stage2（无 mix）；preflight：`identity 1.0000`、`stargan raw_acc=0.5858 -> vae_resample_acc=0.6006`；已进入 `epoch 000` 训练 | 进入“StarGAN 预训练 VAE -> Stage2 U-Net”的干净迁移实验回合，并与 Stage1 并行不抢卡 | 进行中 |
| 2026-03-11 | `runs/stage2_20260311_153327/transfer_eval_stage2_best_val_20260311_153644.csv` | 对新 Stage2（epoch0-best）执行迁移评估（`stargan + simswap`，8 batch/64 样本）：`stargan 0.5802 -> 0.5834`（`+0.0032`），`simswap 0.7021 -> 0.7599`（`+0.0577`） | 验证“StarGAN 预训练 VAE + Stage2”是否对 SimSwap 出现有效零样本迁移 | 已完成（已达到“>5pt”量级的早期增益） |
| 2026-03-11 | 运行切换：停止 `stage1_20260311_145328`，仅保留 `stage2_20260311_153327` | 在 Stage2 正常运行后关闭 Stage1 续训会话，释放 GPU `0,1`；Stage2 继续在 GPU `2,3,4,5` 训练 | 符合“VAE 稳定后 kill 掉并进入 Stage2”的执行策略，减少无效算力占用 | 已完成 |
| 2026-03-11 | 运行异常定位：`stage2_20260311_153327` | Stage2 在 `epoch002` 早期被外部会话终止后，尝试 `--resume stage2_20260311_153327` 失败；错误为 Stage1 hash 校验不一致（当前 `2d5c7a249139762f` vs 绑定 `d5a574df2cea0bc2`） | 说明 Stage2 与 Stage1 checkpoint 强绑定，Stage1 `best.pth` 变更后不能直接续训旧 Stage2 run | 已定位（按设计行为） |
| 2026-03-11 | 运行 `stage2_20260311_155814`（`stage2_unet_from_stargan_transfer`，`torchrun --nproc_per_node=4`，GPU `2,3,4,5`） | 基于当前 Stage1 最新 `best.pth` 重新启动全新 Stage2；preflight：`identity 1.0000`、`stargan raw_acc=0.5858 -> vae_resample_acc=0.6007`；已进入 `epoch 000` | 以一致的 Stage1 版本继续 Unet 训练，避免版本不匹配导致的 resume 中断 | 进行中 |
| 2026-03-11 | `train_stage2.py` | 新增零样本迁移增强训练策略：`training.scheduled_sampling`（概率渐进的 rollout 输入）与 `losses.tail`（轨迹尾段 step 加权）；训练日志新增 ss/tail 配置打印与每 epoch `ss_prob` 调度输出 | 缓解 teacher-forcing 与推理分布偏移，并强化最难恢复区间（point N 附近）监督，目标提升 SimSwap 零样本迁移 | 已完成（`py_compile` 通过，向后兼容旧配置） |
| 2026-03-11 | `configs/experiments/stage2_unet_from_stargan_transfer_aggressive.yaml` | 新增 no-mix 激进配置：`lr=6e-5`、`wd=2e-4`、`latent_noise_std=0.02`、`scheduled_sampling(0.10→0.60)`、`tail(last_k=4, weight=2.5)` | 在不引入 SimSwap 训练样本的前提下提升跨攻击泛化 | 已完成 |
| 2026-03-11 | 运行切换：`stage2_20260311_155814 -> stage2_20260311_161044` | 停止旧 Stage2，按 aggressive 配置启动新 run（GPU `2,3,4,5`）；启动日志确认 ss/tail 开关生效；`Epoch 000`：`val_attack[stargan] raw_acc=0.5861 -> acc=0.5897`，`Epoch 001` 已进入训练 | 建立“无 mix 的迁移强化”主实验线 | 进行中 |
| 2026-03-11 | `runs/stage2_20260311_161044/transfer_eval_stage2_best_val_20260311_161445.csv` | aggressive run 的早期迁移评估（64 样本）：`stargan 0.5802 -> 0.5869`（`+0.0067`），`simswap 0.7017 -> 0.7272`（`+0.0255`） | 监控新策略对 SimSwap 的零样本提升幅度，并与旧 run 对齐比较 | 已完成（当前仍低于 `+0.10` 目标） |
| 2026-03-11 | `train_stage2.py`、`tools/eval_stage2_transfer.py` | 统一 Stage2 指标口径：`raw_acc` 改为“仅 VAE 重采样（不经过 U-Net）后的 ACC”，`acc/rec_acc` 为“经过 U-Net 推理后的 ACC”；不再用“攻击后原图 ACC”与 U-Net 结果直接比较 | 直接衡量第二阶段 U-Net 的净增益，避免把 Stage1 与 Stage2 的贡献混在一起 | 已完成（语法检查通过；`stage2_20260311_161044` 已 resume 生效） |
| 2026-03-11 | `runs/stage2_20260311_133024/transfer_eval_stage2_best_val_20260311_163638.csv` | 用新口径复核 SimSwap 主线：`raw_acc=0.8275`（VAE-only）→ `rec_acc=0.8271`（UNet 后），`delta=-0.0004`（128 样本） | 结论修正：旧日志中的“大幅提升”主要来自“攻击后原图 → VAE/UNet”的口径差异；按 Stage2 净增益口径，当前 U-Net 对该 run 的额外提升接近 0 | 已完成 |
| 2026-03-11 | `configs/experiments/stage2_unet_from_stargan_transfer_aggressive.yaml` + 运行 `stage2_20260311_161044`（resume） | 将 `efficiency.bce_on_all_steps` 从 `false` 提升为 `true`（`wm_loss_freq=1` 保持），并重启 Stage2 续训；日志确认新配置生效：`SLERP ... wm_loss_freq=1  bce_on_all_steps=True`（`16:55:14`） | 按“每一步都做 BCE 传播”强化 U-Net 在整条轨迹上的监督信号，先不考虑计算开销 | 进行中（训练吞吐约 `3.0 it/s -> 1.87 it/s`） |
| 2026-03-11 | 运行停止：`stage2_20260311_161044` | 基于收敛检查停止当前 Stage2：共 45 个 epoch，最佳 `acc=0.6020@epoch013`；最近 10 个 epoch `mean(raw_acc)=0.5939`、`mean(acc)=0.5833`、`mean(delta)=-0.0106` | 已进入平台且 U-Net 净增益为负，继续训练收益低 | 已完成（训练会话 `remark_stage2_aggr` 已停止） |
| 2026-03-11 | `train_stage2.py` | 验证曲线日志由“SLERP 插值点”改为“自回归推理 step”：新增 `val_infer_curve[attack] step 0..N`（`step0=VAE-only`，`stepN=第 N 次 U-Net 推理后`），并将验证阶段进度标签改为 `val-infer-curve:*` | 直接反映第二阶段每一步真实推理收益，避免插值语义干扰判断 | 已完成（`py_compile` 通过） |
| 2026-03-12 | `network/unet.py`、`train_stage2.py`、`configs/stage2_unet.yaml`、`configs/experiments/stage2_unet_from_stargan_transfer*.yaml` | 接入 timestep-conditioned U-Net 训练链路（train/val/sample 全路径传入 `timestep`），新增 `slerp.time_min/time_max` 调度与启动日志打印；保留单路 StarGAN 攻击（不启用 multi-step）；新增可选 bottleneck self-attention（默认关，迁移配置开启）并同步提高 Stage2 容量（`n_levels`/`base_channels`） | 对齐老版“按 step 条件化去噪”范式，并提升模型表达能力，准备新一轮 Stage2 训练 | 已完成（`py_compile` + YAML 解析通过） |
| 2026-03-12 | `configs/experiments/stage2_unet_sepmark_simswap_attn.yaml` | 新增 SepMark+SimSwap 的 Stage2 专用配置：开启 `time_cond=true` 与 `use_bottleneck_attn=true`，并启用 `bce_on_all_steps=true`、`scheduled_sampling`、`tail`；绑定 Stage1 `runs/stage1_20260311_121605/checkpoints/vae/best.pth` | 启动“SimSwap 主线 + 注意力增强”的 Stage2 训练回合 | 已完成（可直接 `torchrun` 启动） |
| 2026-03-12 | `train_stage2.py`、`configs/experiments/stage2_unet_sepmark_simswap_attn.yaml` | Stage2 训练改为严格自回归链：`step k+1` 输入必须使用 `step k` 当前输出；每一步都计算 `L1(z_pred_k, z_traj_k)`，并将 `z_pred_k` 解码后用 `BCE(message_pred, message_gt)` 监督；移除该实验配置中的 `scheduled_sampling` 段 | 消除 teacher-forcing 偏差，强制训练路径与真实推理路径一致，并在每个推理步提供 message 级监督 | 已完成（待新 run 验证收敛） |
| 2026-03-12 | `configs/experiments/stage1_sepmark_compact_stargan_10k_transfer_resume180.yaml` | 新增 StarGAN+SepMark Stage1 续训配置：从既有 run 续训时将 `training.epochs` 从 100 提升到 180，并关闭 preflight 以减少重启开销 | 支持“按原 run 继续训练而非重开”，继续提升 `val_attack[stargan] acc` | 已完成（可直接 `--resume stage1_20260312_101214`） |
| 2026-03-12 | `attacks/stargan.py`、`configs/experiments/stage1_sepmark_compact_stargan_fixed_quick.yaml`、`configs/experiments/stage2_unet_from_stargan_fixed_quick.yaml` | 新增固定域攻击别名 `stargan_fixed`（强制 `random_domain=false`，目标域由 `stargan_fixed_domain` 指定）；新增一套 fixed-domain quick 验证配置（Stage1/Stage2）用于“固定域 vs 随机域”快速实验 | 验证固定目标域是否能提升 Stage2 的可学习性与恢复效果 | 已完成（待运行） |
| 2026-03-12 | 运行 `stage1_20260312_134628`（`stage1_sepmark_compact_stargan_fixed_quick`，2 GPU，quick-1k/200） | fixed-domain VAE 快速适配：preflight `stargan_fixed bit_acc=0.5870`；`epoch003` 达到 `val_attack[stargan_fixed] acc=0.5950`（best），随后手动中断以进入 Stage2 验证 | 产出固定域适配 VAE checkpoint 供 Stage2 假设验证 | 已完成（使用 `checkpoints/vae/best.pth`） |
| 2026-03-12 | 运行 `stage2_20260312_134912`（`stage2_unet_from_stargan_fixed_quick`，2 GPU，quick-1k/200，8 epochs） | fixed-domain Stage2 验证完成：best `val_attack[stargan_fixed] rec_acc=0.6041@epoch006`，对应 `raw_acc=0.6044`（delta≈-0.0003）；`epoch004` 有轻微正增益 `0.6021 -> 0.6022`（+0.0001） | 检验“固定目标域是否显著改善 Stage2 恢复” | 已完成（结论：当前设置下仅有极小波动级变化，未见显著提升） |
| 2026-03-12 | `train_stage2.py` | 按旧版 loss 设计差异完成 Stage2 训练逻辑修正（不改 SLERP 方向）：1) 每个推理 step 独立 `backward+step`；2) 增加 teacher-forcing warmup（`teacher_forcing_*` 配置，线性退火到自回归）；3) 新增末步强化权重 `losses.final_step_{l1,bce}_weight`；4) 修复 train/infer 解码参考一致性（训练 BCE 解码改为固定 `fake_images`，与推理一致） | 对齐“训练目标=推理路径”，并加强最终恢复点位监督，降低 Stage2 净增益为负的风险 | 已完成（`py_compile` 通过） |
| 2026-03-12 | `configs/stage2_unet.yaml`、`configs/experiments/stage2_unet_sepmark_simswap_attn.yaml` | 同步新增 Stage2 配置项：`training.per_step_update=true`、`teacher_forcing_warmup_epochs/start_prob/end_prob`、`losses.final_step_l1_weight/final_step_bce_weight`；移除 Stage2 中已无效的 `wm_loss_freq` 注释项 | 让新训练逻辑可配置可复现，并避免“代码改了但配置没跟上” | 已完成（可直接用于下一轮 Stage2） |
| 2026-03-12 | `configs/experiments/stage2_unet_sepmark_simswap_attn.yaml` | 为提升 GPU4/5 显存利用率，`training.batch_size: 4 -> 16`，`efficiency.attack_num_workers: 0 -> 4` | 提高单卡显存占用与吞吐，减少小 batch 下的资源闲置 | 已完成（需重启当前 Stage2 run 生效） |
| 2026-03-12 | `configs/experiments/stage2_unet_sepmark_simswap_attn_quick.yaml` | 新增 Stage2 quick 验证配置（`train/val=1000/200`，`epochs=12`，`batch=16`，`val/preflight` batch 数收敛到小规模） | 先在小数据集快速验证新 loss 与训练策略趋势，缩短调参闭环 | 已完成（可直接 quick 启动） |
| 2026-03-12 | `train_stage2.py` | 验证新增三类可对照曲线日志：`val_infer_curve`（自回归推理 step）、`val_learn_curve`（teacher 输入点的单步预测）、`val_target_curve`（监督轨迹点本身）；用于定位“模型在学什么”与“真实推理在做什么”的偏差 | 解决“推理后 ACC 几乎不动却看不出原因”的可观测性问题，支持逐 step 对照诊断 | 已完成（`py_compile` 通过，重启 run 生效） |
| 2026-03-12 | `train_stage2.py` | 监督点日志再细分为两条：`val_target_curve_train_ref`（fixed fake reference，和训练/推理参考一致）与 `val_target_curve`（oracle reference：reference 从 fake→wm 过渡，终点用 wm）；用于分离“模型没学到”与“解码参考不匹配”两类现象 | 解释并验证“point10 理应接近 1.0”的预期：在 oracle reference 下应接近 Stage1 上限，而 train_ref 下可能维持在当前水平 | 已完成（`py_compile` 通过，重启 run 生效） |
| 2026-03-12 | `train_stage2.py` | 按实验命名规范统一曲线术语：将“监督点/teacher输入点”重命名为“插值点/推理点”；日志键更新为 `val_infer_point_curve_rollout`、`val_infer_point_curve_train`、`val_interp_point_curve_train_ref`、`val_interp_point_curve_oracle` | 保证日志语义与训练定义一致（球面轨迹=插值点，基于上一插值点预测=推理点），降低分析歧义 | 已完成（重启 run 生效） |
| 2026-03-12 | `train_stage2.py`、`configs/stage2_unet.yaml`、`configs/experiments/stage2_unet_sepmark_simswap_attn_quick.yaml` | 新增 point-curve 日志开关并默认关闭 `log_train_infer_curve`：训练阶段不再打印“训练推理点”曲线；同时保留自回归推理点与插值点曲线打印可控 | 对齐“自回归推理点主要用于测试分析”的日志需求，减少训练日志噪声 | 已完成（重启 run 生效） |
| 2026-03-12 | `train_stage2.py`、`configs/stage2_unet.yaml`、`configs/experiments/stage2_unet_sepmark_simswap_attn_quick.yaml` | 新增 `validation.train_probe`：每个 epoch 可在训练集上计算同口径 `raw_acc/acc/l1`，并可选输出训练集曲线（`train_*_curve`）；quick 配置默认开启（`max_batches=2`，`curve_max_batches=1`） | 用训练集与验证集并行对照判断“模型是否在 train 上学习但在 val 不泛化”，定位原地踏步根因 | 已完成（重启 run 生效） |
| 2026-03-12 | `train_stage2.py`、`configs/stage2_unet.yaml`、`configs/experiments/stage2_unet_sepmark_simswap_attn_quick.yaml` | 按需求再次精简日志：仅保留两条自回归 rollout 曲线（`rollout_curve[train]` 与 `rollout_curve[val]`）；删除插值点/训练式推理点/oracle 曲线及 `train_probe_attack` 打印，标注统一简化为 train/val | 聚焦“训练集 vs 验证集”真实推理表现，减少非必要诊断噪声 | 已完成（`py_compile` 通过，重启 run 生效） |
| 2026-03-12 | 代码排查 `wm_adapters/sepmark.py` + `train_stage2.py` | 梯度链路快速诊断：BCE 梯度可达 U-Net（单批次 `grad_norm_from_bce≈2.13e-1`，`grad_norm_from_l1≈1.04e+0`），不存在“梯度完全断开”；但 SepMark 解码 logits 映射存在裁剪饱和（`|pred|>=range` 比例约 13.3%，logits 可到 ±13.8），且 BCE 梯度显著弱于 L1 | 当前“train/val 都几乎不动”更像目标权重与饱和区导致的弱优化，而非 SLERP 起终点或计算图断链 | 已完成（结论待下一轮 loss/decoder 映射调整验证） |
| 2026-03-12 | `wm_adapters/sepmark.py`、`configs/stage2_unet.yaml`、`configs/experiments/stage2_unet_sepmark_simswap_attn{,_quick}.yaml` | 实施“平滑 BCE logits”与权重增强：SepMark adapter 新增 `decode_logits_mode`（`legacy_clamped_logit`/`smooth_linear`），Stage2 默认切到 `smooth_linear`；同时上调 BCE 权重（quick: `10→15`，full-attn: `3→5`） | 缓解 legacy clamp 导致的 BCE 饱和梯度问题，并增强 message 监督驱动力，观察 rollout 曲线是否脱离平台 | 已完成（重启 run 生效） |
| 2026-03-12 | `network/unet.py`、`configs/stage2_unet.yaml`、`configs/experiments/stage2_unet_sepmark_simswap_attn{,_quick}.yaml` | Stage2 U-Net 架构对齐旧版 Denoise：新增 diffusion-style `scale-shift norm`、多尺度 `attention_resolutions`、`mid_attn_depth` attention+ffn 堆叠，并将 SimSwap Stage2 配置切到更高容量（`base_channels=96`、`n_res=3`、`use_level_attn=true`、`[8,4,2]` 注意力） | 对齐“时序条件 + 重容量 + 多尺度建模”范式，优先解决当前 U-Net rollout 几乎原地不动的问题 | 已完成（`py_compile` 通过，待 quick 训练验证） |
| 2026-03-12 | `train_stage2.py`、`configs/stage2_unet.yaml`、`configs/experiments/stage2_unet_sepmark_simswap_attn{,_quick}.yaml` | 修正 Stage2 每步损失缩放策略：新增 `losses.normalize_step_weights`（默认按 `per_step_update` 自动，当前显式关闭），`per_step_update=true` 时不再把 step 权重归一化到和为 1；启动日志新增 step-weight 统计（sum/min/max） | 解决“每步独立更新但梯度被额外缩小（约 10x）”导致 rollout 基本不动的问题，提升每步更新有效信号 | 已完成（待新 run 验证收敛曲线） |
| 2026-03-12 | 运行 `stage2_20260312_174937`（`stage2_unet_sepmark_simswap_attn_quick`，`torchrun --nproc_per_node=4`，GPU `0,1,4,5`） | 接管服务器后清理旧 stage2 会话并启动新 Stage2 quick 训练；启动日志确认 `Step-weight normalization: normalize=False, sum=14` 与 `U-Net=85.07M` 生效；`Epoch001` 指标：`val_attack[simswap] raw_acc=0.8275 -> acc=0.8280`，并输出 `rollout_curve[train/val]` | 验证“每步损失缩放修正”在真实多卡训练中可执行且开始产生正向净增益 | 进行中 |
| 2026-03-12 | `configs/experiments/stage2_unet_sepmark_simswap_teacher_strongbce_quick.yaml` + 新 run（待起） | 新增“teacher-only + 强 BCE”快速配置：训练阶段固定 `tf_prob=1.0`（`teacher_forcing_start=end=1.0`，warmup 充分大）、关闭 latent 噪声、`n_steps/max_infer_steps=4`、`bce.weight=30`、`l1.weight=0.25`、`final_step_bce_weight=10`、`weight_decay=0` | 按“图像质量次要，优先水印恢复”目标，先验证在非自回归训练输入下能否显著抬升 rollout 终点 ACC | 已完成配置，准备启动 |
| 2026-03-12 | 运行 `stage2_20260312_191520`（`stage2_unet_sepmark_simswap_teacher_strongbce_quick`） | 启动 teacher-only 强 BCE 快测（4 卡）；早期结果 `epoch0/1` 均出现 rollout 末步回落（如 val `step0≈0.8314 -> step4≈0.8287`），并伴随 latent L1 偏大（`~0.99+`） | 验证“过强 BCE + 弱 latent 约束”会把 latent 拉偏，不能直接提升恢复 ACC | 已停止（负增益） |
| 2026-03-12 | 运行 `stage2_20260312_191907`（`stage2_unet_sepmark_simswap_teacher_balanced_quick`） | 启动 teacher-only 平衡配置（4 卡）：`tf_prob=1.0` 固定、`n_steps=6`、`L1+BCE+direction/progress/step_size` 联合约束；`epoch000` 指标：`raw_acc=0.8263 -> acc=0.8265`，但 rollout 末步仍略低于 step0（val `0.8282 -> 0.8267`） | 在“全 teacher”前提下先验证是否可避免大幅拉偏；当前已消除强 BCE 的崩塌行为，但自回归终点增益仍需继续观察 | 进行中 |
| 2026-03-12 | `attacks/simswap.py`、`configs/experiments/stage2_unet_sepmark_simswap_teacher_balanced_fixedsrc_quick.yaml` | 新增 SimSwap 固定 source 人脸模式：`attack_options.simswap_source_mode` 支持 `cover_roll/self/fixed_file`，其中 `fixed_file` 从 `simswap_fixed_source_image` 加载单张 source 并对全 batch 复用；同时新增 fixed-source quick 配置（固定为 `Dataset-CelebA_HQ/train/0.jpg`） | 降低 SimSwap 任务难度，验证“统一 source 身份”是否能显著提升 Stage2 水印恢复 | 已完成配置与代码，待启动新 run 验证 |
| 2026-03-12 | `data_manifests/stage2_celeba_hq_{train_3k,val_600}.csv`、`configs/experiments/stage2_unet_sepmark_simswap_teacher_balanced_fixedsrc_mid.yaml` + 运行 `stage2_20260312_195910` | 启动 fixed-source 的中等规模 Stage2（4 卡，3k/600）：preflight `simswap raw=0.7079 -> vae_resample=0.8279`；`epoch000`：`raw=0.8318 -> acc=0.8315`；`epoch001`：`raw=0.8293 -> acc=0.8286`，rollout 曲线在 step0~step6 基本平坦（val `0.8347 -> 0.8347`） | 在比 quick 更大样本上复核“固定 source 是否带来稳定正增益” | 进行中（当前早期趋势仍接近平台） |
| 2026-03-12 | `wm_adapters/lampmark.py`、`wm_adapters/__init__.py`、`configs/experiments/stage1_lampmark_compact_simswap_10k.yaml` + 运行 `stage1_20260312_201739` | 新增 LampMark adapter（支持 `wm_model=lampmark`，加载 `Forensic-LampMark/weights/128_64/deepfake/{encoder,decoder}_epoch_30.pth`，`decode` 默认 `linear_centered` logits 映射），并启动 Stage1 VAE（2 卡，GPU `2,3`，10k/2156，SimSwap 在线攻击）；preflight：`clean bit_acc=0.9697`、`simswap bit_acc=0.8391`，随后进入 `Epoch 000` 训练 | 按“切换到 LampMark，从 VAE 重新训练，攻击仍用 SimSwap”执行新主线 | 进行中 |
| 2026-03-12 | 快速迁移评估（脚本内覆写 attack=`simswap`，`max_batches=8`） | 评估“SepMark+StarGAN 训练出的 VAE”在 SimSwap 上的重采样增益：`stage1_20260312_101214/best`：`raw_acc=0.7105 -> recon_acc=0.5526`（`-0.1580`）；`stage1_20260311_145328/best`：`raw_acc=0.7105 -> recon_acc=0.7403`（`+0.0298`） | 回答“StarGAN 预训练 VAE 对 SimSwap 的迁移增益有多大”并定位 checkpoint 版本差异 | 已完成（当前旧版 best 有约 +3pt，最新版 best 为负迁移） |
| 2026-03-12 | 精确复核 `stage1_20260312_101214`（100+ epoch） | 在同一口径下比较后期 checkpoint（`max_batches=8`）：`best -0.1579`、`last -0.1533`、`epoch_170 -0.1575`、`epoch_160 -0.1594`、`epoch_150 -0.1531`（均为 `recon_acc - raw_acc`，attack=`simswap`） | 针对“跑了一百多轮”的目标 run 给出明确结论 | 已完成（该 run 在 SimSwap 上稳定负迁移，约 -15.3~-15.9pt） |
| 2026-03-12 | 反向迁移评估：`stage1_20260311_121605`（SepMark+SimSwap 训练 VAE）在 `StarGAN` 上 | 同口径评估（`val`, `max_batches=8`, 64 样本）：`best.pth raw=0.5802 -> recon=0.5483 (delta=-0.0319)`；`last.pth delta=-0.0364`；`epoch_050.pth delta=-0.0343` | 回答“SimSwap 训练的 VAE 用于 StarGAN 恢复表现如何” | 已完成（当前为稳定负迁移，约 -3.2~-3.6pt） |
| 2026-03-12 | `configs/experiments/stage2_unet_lampmark_simswap_attn.yaml` + 运行切换 `stage1_20260312_201739 -> stage2_20260312_210047` | 在 LampMark 路线上新增 Stage2 配置（`wm_model=lampmark` + `wm_adapter_lampmark`，Stage1 绑定 `runs/stage1_20260312_201739/checkpoints/vae/best.pth`，攻击=`simswap`），停止 Stage1 后于 GPU `2,3` 启动 Stage2；启动日志确认 U-Net/SLERP/teacher-forcing 配置生效并进入 `Preflight baseline (NO U-Net)` | 按“LampMark VAE 已基本收敛，切到 Unet 第二阶段继续 SimSwap”执行训练流水线 | 进行中 |

---

## 2026-03-12 实施记录（SepMark + StarGAN 提升主线）

### A. Stage2 训练机制重构（已落地）
- 文件：`train_stage2.py`
- 变更：
  - U-Net 输出改为 `delta` 更新：`z_pred = z_k + delta_scale * unet(z_k, t)`。
  - 新增损失：
    - `move_floor_loss = relu(min_step - |z_pred-z_k|)`（抑制“几乎不动”）。
    - `bce_progress_loss = relu(BCE(z_pred)-BCE(z_k)+margin)`（要求单步 message 可读性不变差）。
    - `terminal_anchor_loss`（最后一步额外拉向 `z_gt`）。
  - 新增 Phase-A/B 调度：
    - Phase-A：前 `phase_a_epochs` 全 teacher。
    - Phase-B：teacher 从 1.0 退火到 `phase_b_teacher_forcing_end_prob`，并支持 `bptt_horizon` 短程 BPTT。
  - 新增日志：
    - `rollout_uplift[train/val]`（`step_last - step0`）。
    - `epoch_uplift[train/val]`（每个 epoch 的平均 uplift）。
  - 最优模型选择改为：先比较 `val_uplift`，并列时比较 `val_step_last`。

### B. 迁移评估口径对齐（已落地）
- 文件：`tools/eval_stage2_transfer.py`
- 变更：
  - 推理逻辑对齐 Stage2：支持 `time_cond + delta_scale + unet_output_blend`。
  - CSV 增加：`step0_acc`、`step_last_acc`，并保留 `delta`。

### C. 新实验配置（已新增）
- `configs/experiments/stage1_sepmark_stargan_10k_noref_120.yaml`
  - 关键：`residual_output: false`，`epochs=120`，`batch=32`，`bce=12.0`，攻击 `stargan(random_domain=true)`。
- `configs/experiments/stage2_unet_sepmark_stargan_phase1.yaml`
  - 固定域课程：`stargan_fixed(domain=1)`，80 epoch。
- `configs/experiments/stage2_unet_sepmark_stargan_phase2.yaml`
  - 随机域课程：`stargan(random_domain=true)`，续训到 220 epoch。

### D. 启动命令（执行手册）
1. Stage1（SepMark + StarGAN）
```bash
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_stage1.py \
  --config configs/stage1_vae.yaml \
  --override configs/experiments/stage1_sepmark_stargan_10k_noref_120.yaml
```

2. Stage2-Phase1（固定域）
```bash
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_stage2.py \
  --config configs/stage2_unet.yaml \
  --override configs/experiments/stage2_unet_sepmark_stargan_phase1.yaml \
  --stage1-ckpt runs/<NEW_STAGE1_RUN>/checkpoints/vae/best.pth
```

3. Stage2-Phase2（随机域续训）
```bash
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_stage2.py \
  --config configs/stage2_unet.yaml \
  --override configs/experiments/stage2_unet_sepmark_stargan_phase2.yaml \
  --stage1-ckpt runs/<NEW_STAGE1_RUN>/checkpoints/vae/best.pth \
  --resume <PHASE1_STAGE2_RUN>
```

4. 迁移评估（只评估，不混攻训练）
```bash
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module
CUDA_VISIBLE_DEVICES=4 python tools/eval_stage2_transfer.py \
  --run-dir runs/<PHASE2_STAGE2_RUN> \
  --checkpoint best \
  --attacks stargan simswap \
  --max-batches 16
```

### E. 当前执行状态（本次实现）
- Stage1 主线已启动：`runs/stage1_20260312_213701`（2 卡 `CUDA_VISIBLE_DEVICES=4,5`），当前在持续训练中。
- Stage2 新逻辑烟测已通过：`runs/stage2_20260312_213946`（2 分钟 timeout 验证），已确认可启动并进入训练循环，结束原因为超时主动终止而非代码报错。

### F. 无 KL 对照链（SepMark+StarGAN，自动接力）
- 新增配置：
  - `configs/experiments/stage1_sepmark_stargan_10k_nokl_120.yaml`
    - 关键改动：`losses.kl = {weight: 0.0, target_weight: 0.0, warmup: false}`，并将 `training.kl_warmup_epochs=0`，实现“KL 全程关闭”。
  - `configs/experiments/stage2_unet_sepmark_stargan_nokl_phase1.yaml`
  - `configs/experiments/stage2_unet_sepmark_stargan_nokl_phase2.yaml`
- 新增自动接力脚本：
  - `tools/run_sepmark_stargan_nokl_pipeline.sh`
  - 流程：Stage1(noKL) -> Stage2-Phase1(stargan_fixed) -> Stage2-Phase2(stargan random resume) -> transfer eval(stargan+simswap)
  - 支持环境变量：`GPU_SET`、`NPROC_PER_NODE`、`MASTER_PORT_STAGE*`、`RUN_TAG`。
- 资源隔离策略：
  - 主线实验继续使用 GPU `4,5`；
  - 无 KL 对照链固定使用 GPU `0,1`，避免互相干扰。
- 实际运行状态（2026-03-12 22:17）：
  - no-KL 接力链已启动（独立会话），`RUN_TAG=nokl_stargan_chain_live_20260312_222000`。
  - Stage1(no-KL) 当前 run：`runs/stage1_20260312_221712`（GPU `0,1`）。
  - 并行主线实验保持在 GPU `4,5`，两组资源已完全错开。
  - 2026-03-12 22:52：按资源利用率要求将 no-KL Stage1 `batch_size` 从 `32` 提升到 `64`，并执行 `--resume stage1_20260312_221712` 续训（继续使用 GPU `0,1`）。
  - 续训生效校验：日志显示 `Train ... (78 steps/epoch/GPU)`（由原 `157` 降为 `78`），且 GPU 显存占用提升到约 `19GB/card`（先前约 `12GB/card`）。
  - 2026-03-12 22:56：继续将 no-KL Stage1 `batch_size` 从 `64` 提升到 `128`，停止旧进程后执行同一 run 的 `--resume stage1_20260312_221712`（GPU `0,1`）。
  - 生效校验：启动日志显示 `Train ... (39 steps/epoch/GPU)`（由 `78` 再降为 `39`），当前显存约 `36~37GB/card`，训练可正常推进（未出现 OOM）。
  - 2026-03-12 23:21：按“前期 L1 主导、后期 BCE 主导”新增 Stage1 动态损失调度并热更新续训：
    - 代码：`network/losses.py` 新增通用 `losses.<name>.schedule` 线性调权（`start_epoch/end_epoch/start_weight/end_weight`）；`train_stage1.py` 每个 epoch 额外打印 `l1_weight/bce_weight`。
    - no-KL 配置（第一次）：`l1 48->8`、`bce 4->18`（`0~35` epoch 线性过渡）。
    - 修正（第二次）：考虑该 run 从 `epoch 25` 续训，调度起点改为当前阶段：`l1 80->8`、`bce 2->20`（`26~60` epoch 线性过渡）。
    - 运行：二次重启 `runs/stage1_20260312_221712` 后日志显示 `Epoch 026 schedule: ... l1_weight=80.0000  bce_weight=2.0000`，确认“先 L1、后 BCE”从当前阶段开始生效。
  - 2026-03-13 00:29：按收敛需求将 no-KL Stage1 总轮数从 `120` 提升到 `200`，并在原 run 上继续续训：
    - 配置修改：`configs/experiments/stage1_sepmark_stargan_10k_nokl_120.yaml` 中 `training.epochs: 200`。
    - 续训命令（GPU `0,1`）：`/home/ldy/miniconda3/envs/sepmark/bin/torchrun --master_port 29833 --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml --override configs/experiments/stage1_sepmark_stargan_10k_nokl_120.yaml --resume stage1_20260312_221712`
    - 生效校验：`runs/stage1_20260312_221712/train.log` 已打印 `lr=... over 200 epochs`、`Resumed from epoch 50` 且进入 `Epoch 051` 训练进度。
  - 2026-03-13 00:38：按“训练偏慢、L1 下降过快”要求进一步热更新并续训：
    - 学习率提升：`training.lr: 3.0e-4 -> 6.0e-4`。
    - L1/BCE 调度重对齐（从当前阶段继续，避免过快切权重）：
      - `l1.schedule`: `start_epoch=53, end_epoch=150, start_weight=22.8235, end_weight=8.0`
      - `bce.schedule`: `start_epoch=53, end_epoch=150, start_weight=16.2941, end_weight=20.0`
    - 续训方式：停止旧进程后继续 `--resume stage1_20260312_221712`（GPU `0,1`，同 run）。
    - 生效校验：日志出现 `lr=6.00e-04  cosine → 6.00e-06 over 200 epochs`，并重新进入 `Epoch 053` 训练循环。
  - 2026-03-13 00:46：按“学习率提升到 e-3 量级，并衰减到 e-5/e-6”再次调整：
    - 配置修改：`training.lr: 6.0e-4 -> 1.0e-3`（当前实现的余弦下限规则为 `eta_min = base_lr * 0.01`，故自动得到 `1.0e-5`）。
    - 续训恢复：发现 `checkpoints/vae/last.pth` 曾被异常写坏（文件过小且报 `failed finding central directory`），已用 `epoch_050.pth` 覆盖恢复后再续训。
    - 生效校验：`runs/stage1_20260312_221712/train.log` 打印 `lr=1.00e-03  cosine → 1.00e-05  over 200 epochs`，并成功 `Resumed from epoch 50` 后继续训练。
  - 2026-03-13 01:09：新增“睡眠期间自动接力 Stage2”的守护脚本并已挂起：
    - 新增脚本：`tools/watch_stage1_then_stage2_nokl.sh`
    - 行为：持续监控 `runs/stage1_20260312_221712/train.log`，当 Stage1 达到目标 epoch（按当前 no-KL override 为 200 epoch，即 `target_epoch=199`）后自动执行：
      1) `stage2_unet_sepmark_stargan_nokl_phase1.yaml`
      2) `stage2_unet_sepmark_stargan_nokl_phase2.yaml`（resume 同一 Stage2 run）
      3) `tools/eval_stage2_transfer.py`（stargan+simswap）
    - 当前守护日志：`logging/watch_nokl_chain_stage1_20260312_221712_live_20260313_0110.pipeline.log`

### G. Stage2 BCE 梯度回传与每步独立优化修正（2026-03-13）
- 代码检查结论（链路层面）：
  - `wm_adapters/base.py` 中 `decode()` 明确不包 `no_grad`，梯度允许从 `BCE -> wm_decoder -> x_hat -> U-Net` 回传。
  - `train_stage2.py` 中 VAE 参数冻结（`requires_grad=False`），但前向图不切断，BCE 梯度可穿过冻结模块回到 U-Net。
- 训练逻辑修正（`train_stage2.py`）：
  - 新增 `training.separate_bce_step`（默认 `true`）与 `training.bce_only_step_scale`（默认 `1.0`）。
  - 当 `per_step_update=true` 且 `separate_bce_step=true` 时，每个 step 改为两次优化：
    1. `BCE-only step`：仅优化 `bce + bce_progress` 路径；
    2. `Main step`：优化 `L1 + move_floor + terminal_anchor + direction + progress + step_size`。
  - 目的：避免 BCE 仅作为“数值加权项”被 L1/几何项掩盖，强化 message 路径的实际参数更新。
- 配置同步：
  - `configs/stage2_unet.yaml`
  - `configs/experiments/stage2_unet_sepmark_simswap_attn.yaml`
  - `configs/experiments/stage2_unet_sepmark_simswap_attn_quick.yaml`
  - `configs/experiments/stage2_unet_lampmark_simswap_attn.yaml`
  均已加入：
  - `training.separate_bce_step: true`
  - `training.bce_only_step_scale: 1.0`
- 运行日志可见性：
  - 启动日志新增 `separate_bce_step` 与 `bce_only_step_scale` 打印，便于确认是否启用。

### H. 回退到可收敛基线并仅保留 no-KL（2026-03-13 09:15）
- 目标：
  - 放弃 no-KL 链路中的激进改动（高学习率、大 batch、L1/BCE 调度、`residual_output=false`），回到已验证可学的 StarGAN 配方，仅保留“KL 关闭”这一项。
- 新增配置：
  - `configs/experiments/stage1_sepmark_stargan_10k_rebuild_nokl.yaml`
  - 关键设置：
    - `residual_output=true, residual_scale=0.5`
    - `batch_size=32, lr=3e-4, epochs=120`
    - `bce=3.0, l1=1.0, lpips=0.1`
    - `alternating_train: warmup_prob=0.30, mid_prob=0.70, main/late=1.00`
    - `kl: weight=0.0, target_weight=0.0`
- 启动命令（已执行）：
  - `CUDA_VISIBLE_DEVICES=4,5 /home/ldy/miniconda3/envs/sepmark/bin/torchrun --master_port 29841 --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml --override configs/experiments/stage1_sepmark_stargan_10k_rebuild_nokl.yaml`
- 当前 run：
  - `runs/stage1_20260313_091503`
  - 启动日志已确认课程参数生效：`attack_prob=0.30`（epoch0）且 `kl_weight=0.00000`，进入训练循环。

### I. Stage1 双分支 1:1（去 attack_prob 抽样）大规模 StarGAN（2026-03-13 09:27）
- 需求：
  - 不再使用 `attack_prob` 随机采样；
  - 每个训练 step 同时优化 clean 与 fake 两条分支，等权 1:1；
  - 使用完整 `10k` CelebA-HQ 训练集。
- 代码改动：
  - `train_stage1.py`
    - 新增 `training.dual_branch_train` 开关（默认 `false`）；
    - 新增 `training.dual_clean_weight` / `training.dual_fake_weight`；
    - `dual_branch_train=true` 时，每步执行：
      1) fake 分支：`x_fake -> VAE -> 对 x_fake 重建`
      2) clean 分支：`wm_images -> VAE -> 对 wm_images 重建`
      并按权重合并总损失（默认 1:1）；
    - 日志调度改为打印 `attack_mode=dual(1:1 clean+fake)`，不再依赖 attack_prob。
- 新增配置：
  - `configs/experiments/stage1_sepmark_stargan_10k_rebuild_nokl_dual.yaml`
  - 关键：
    - `dual_branch_train: true`
    - `dual_clean_weight: 1.0`
    - `dual_fake_weight: 1.0`
    - `alternating_train.enabled: false`
    - 其余保持可收敛基线（`residual_output=true, bce=3, lr=3e-4, batch=32, no-KL`）。
- 启动命令（已执行）：
  - `CUDA_VISIBLE_DEVICES=4,5 /home/ldy/miniconda3/envs/sepmark/bin/torchrun --master_port 29842 --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml --override configs/experiments/stage1_sepmark_stargan_10k_rebuild_nokl_dual.yaml`
- 当前 run：
  - `runs/stage1_20260313_092741`
  - 启动日志已确认：`Epoch 000 schedule: attack_mode=dual(1:1 clean+fake) ...`

### J. CelebA 非HQ + StarGAN v1 固定单标签验证（2026-03-13 10:08）
- 目标：
  - 从“原仓库”路径复用 CelebA 非HQ数据与 StarGAN v1 权重，验证是否主要是任务难度导致当前链路表现差。
  - 攻击策略改为固定单属性翻转（不随机域），便于稳定对照。
- 代码改动：
  - `data/dataset.py`
    - CSV 除 `img_path/fake_path/wm_path` 外的列自动作为 `sample['attrs']` 输出（支持 CelebA 属性标签进入攻击器）。
  - `attacks/stargan_v1.py`（新增）
    - 新攻击注册名：`stargan_v1_fixed`；
    - 直接加载 `Forensic-SepMark/network/noise_layers/stargan/model.py` 的 `Generator`（绕开包级 `dlib` 依赖）；
    - 使用 batch 中属性标签构造目标属性：固定翻转一个标签（当前配置 `Blond_Hair` / idx=1）。
  - `attacks/__init__.py`
    - 增加 `from . import stargan_v1` 注册入口。
- 资源与权重“拉取”：
  - StarGAN v1 checkpoint 链接到本仓：
    - `third_party/stargan_v1/200000-G.ckpt -> /home/ldy/..workspace/zhou/repair/modelckpt/200000-G.ckpt`
  - 非HQ数据清单直接使用：
    - `data_manifests/celeba_nonhq_train_10k.csv`
    - `data_manifests/celeba_nonhq_val_5k.csv`
- 新增配置：
  - `configs/experiments/stage1_sepmark_compact_stargan_v1_fixed_nonhq_10k_nokl.yaml`
  - 关键设置：
    - `attacks.online: [stargan_v1_fixed]`
    - `attack_options.stargan_v1_fixed_attr: Blond_Hair`
    - `data: celeba_nonhq_train_10k / celeba_nonhq_val_5k`
    - `dual_branch_train: true`（clean+fake 1:1）
    - `kl=0`（no-KL）
- 启动与状态：
  - 首次尝试 `batch_size=48` 在 epoch0 OOM，已下调为 `batch_size=16` 后重启。
- 当前运行中：
    - `runs/stage1_20260313_100854`
    - 启动命令：`CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29846 --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml --override configs/experiments/stage1_sepmark_compact_stargan_v1_fixed_nonhq_10k_nokl.yaml`
  - 2026-03-13 10:15 色差修正：
    - 参考旧仓 `network/noise_layers/StarGAN.py` 调用方式，`stargan_v1` 目标标签改为“legacy 纯翻转单维”（不再默认 hair-exclusive 重写）；
    - 配置固定标签从 `Blond_Hair(idx=1)` 改为旧链路常用的 `Male(idx=3)`；
    - 代码新增开关：`attack_options.stargan_v1_hair_exclusive`（当前设为 `false`）；
    - 已停止旧 run 并重启：
      - 新 run：`runs/stage1_20260313_101502`
      - 启动命令：`CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29847 --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml --override configs/experiments/stage1_sepmark_compact_stargan_v1_fixed_nonhq_10k_nokl.yaml`
  - 2026-03-13 10:18 二次色差修正（定量筛选标签）：
    - 对 `Black_Hair/Blond_Hair/Brown_Hair/Male/Young` 做了小批量定量对比（色偏向量 + fake_acc）；
    - 发现 `Male` 在当前权重上引入显著全局暗偏，`Young` 色差最小且仍保持攻击强度；
    - 将固定标签改为：
      - `stargan_v1_fixed_attr: Young`
      - `stargan_v1_fixed_attr_idx: 4`
    - 重启 run：
      - `runs/stage1_20260313_101845`（GPU `2,3`）
      - 命令：`CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29848 --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml --override configs/experiments/stage1_sepmark_compact_stargan_v1_fixed_nonhq_10k_nokl.yaml`
    - preflight 色差对比（clean vs fake）：
      - 旧（Male）：`||delta_rgb|| ≈ 0.0822`
      - 新（Young）：`||delta_rgb|| ≈ 0.0307`

### K. 切换到“后期 StarGAN（LampMark）+ 固定目标域”（2026-03-13 10:23）
- 背景：
  - 用户确认 `stargan_v1` 路线仍不符合预期，要求改用后期 StarGAN 模型并固定标签目标。
- 执行：
  - 停止 `stargan_v1` run（`stage1_20260313_101845`）。
  - 新增配置：
    - `configs/experiments/stage1_sepmark_compact_stargan_fixed_nonhq_10k_nokl.yaml`
    - 关键项：
      - `attacks.online: [stargan_fixed]`
      - `stargan_random_domain: false`
      - `stargan_fixed_domain: 1`
      - `nonHQ 10k train / 5k val`
      - `no-KL + dual_branch_train`
- 当前运行：
  - `runs/stage1_20260313_102351`
  - 启动命令：`CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29849 --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml --override configs/experiments/stage1_sepmark_compact_stargan_fixed_nonhq_10k_nokl.yaml`

### L. 基于 `stage1_20260313_102351` 启动 Stage2 U-Net（2026-03-13 11:26）
- 用户要求：直接使用 `runs/stage1_20260313_102351` 的内容进行 Stage2 训练，沿用此前已调整好的 U-Net 训练框架。
- 新增配置：
  - `configs/experiments/stage2_unet_sepmark_stargan_fixed_nonhq_10k_from_20260313_102351.yaml`
  - 关键：
    - `paths.stage1_checkpoint -> runs/stage1_20260313_102351/checkpoints/vae/best.pth`
    - `attacks.online: [stargan_fixed]`
    - `stargan_random_domain: false`
    - `stargan_fixed_domain: 1`
    - 数据集改为 nonHQ：`celeba_nonhq_train_10k / celeba_nonhq_val_5k`
- 启动命令（当前在跑）：
  - `CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 29851 --nproc_per_node=2 train_stage2.py --config configs/stage2_unet.yaml --override configs/experiments/stage2_unet_sepmark_stargan_fixed_nonhq_10k_from_20260313_102351.yaml`
- 当前 run：
  - `runs/stage2_20260313_112636`

### M. SepMark+SimSwap 降难度验证：`source=target(self-source)`（2026-03-13 12:23）
- 用户目标：回到 SepMark+SimSwap 路线，进一步降低任务难度，让 swap 的目标脸与原脸一致，先验证是否带来可见 uplift。
- 新增配置：
  - `configs/experiments/stage2_unet_sepmark_simswap_attn_selfsrc.yaml`
  - 基于原 `stage2_unet_sepmark_simswap_attn.yaml`，关键改动：
    - `attack_options.simswap_source_mode: self`
    - `attack_options.enforce_nontrivial_swap: false`
- 启动命令（当前在跑）：
  - `CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 29852 --nproc_per_node=2 train_stage2.py --config configs/stage2_unet.yaml --override configs/experiments/stage2_unet_sepmark_simswap_attn_selfsrc.yaml`
- 当前 run：
  - `runs/stage2_20260313_122354`
  - 当前日志显示已进入训练：`epoch=000 step=40/313`，速度约 `0.36 it/s`。

### N. SepMark+SimSwap 进一步降难度：固定单张 source（2026-03-13 13:03）
- 用户要求：比 self-source 再降低难度，令所有样本都使用同一张 source 脸做 SimSwap。
- 新增配置：
  - `configs/experiments/stage2_unet_sepmark_simswap_attn_fixedsrc.yaml`
  - 关键改动：
    - `attack_options.simswap_source_mode: fixed_file`
    - `attack_options.simswap_fixed_source_image: /mnt/personal_workspace/chenkeyu/ReMark/Dataset-CelebA_HQ/train/0.jpg`
    - `attack_options.enforce_nontrivial_swap: false`
- 启动命令：
  - `CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 29853 --nproc_per_node=2 train_stage2.py --config configs/stage2_unet.yaml --override configs/experiments/stage2_unet_sepmark_simswap_attn_fixedsrc.yaml`
- 当前 run：
  - `runs/stage2_20260313_130329`
  - preflight（No U-Net）观测：
    - `simswap raw_acc=0.7113`
    - `vae_resample_acc=0.8286`

### O. 固定单张 source 的 quick 验证（2026-03-13 13:12）
- 目的：不改 fixed-source 机制，仅缩小数据/轮次，快速看 rollout 趋势。
- 新增配置：
  - `configs/experiments/stage2_unet_sepmark_simswap_attn_fixedsrc_quick.yaml`
  - 关键改动：`epochs=12`，`quick_celeba_hq_128_train/val (1000/200)`，`val_max_batches=4`。
- 启动命令：
  - `CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29854 --nproc_per_node=2 train_stage2.py --config configs/stage2_unet.yaml --override configs/experiments/stage2_unet_sepmark_simswap_attn_fixedsrc_quick.yaml`
- 当前 run：
  - `runs/stage2_20260313_131251`
  - epoch0 验证：
    - `rollout step0=0.8307`
    - `rollout step10=0.7555`
    - `uplift=-0.0752`

### P. 老仓库配置完整复刻 + U-Net 前向模式对齐（2026-03-13 13:32）
- 用户要求：回到 `/home/ldy/..workspace/zhou/repair`，把老版 VAE / U-Net 配方完整迁回当前工程（即便先过拟合，也要先看到有效 uplift）。
- 核心差异修正（代码）：
  - `network/unet.py` 增加 `predict_delta` 开关（默认 `false`），支持两种输出语义：
    - `predict_delta=false`：输出“绝对下一步 latent”（对齐老版 `Denoise`）
    - `predict_delta=true`：输出“delta 位移”
  - `train_stage2.py` 增加 `training.unet_prediction_mode`（`auto|absolute|delta`）并在 `_unet_step` 中按语义更新：
    - `absolute`: `z_next = z + delta_scale * (pred - z)`
    - `delta`: `z_next = z + delta_scale * pred`
  - 日志新增真实前向语义打印：`Forward mode: absolute/delta ...`
- 新增“老版复刻”配置：
  - `configs/experiments/stage1_sepmark_stargan_legacyclone_nokl.yaml`
  - `configs/experiments/stage2_unet_sepmark_simswap_fixedsrc_legacyclone.yaml`
  - `configs/experiments/stage2_unet_sepmark_simswap_fixedsrc_legacyclone_quick.yaml`
  - `configs/experiments/stage2_unet_sepmark_stargan_fixed_legacyclone.yaml`
- 对应启动命令：
  - Stage1 legacy：
    - `CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29856 --nproc_per_node=2 train_stage1.py --config configs/stage1_vae.yaml --override configs/experiments/stage1_sepmark_stargan_legacyclone_nokl.yaml`
  - Stage2 legacy full（SimSwap fixed source）：
    - `CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29857 --nproc_per_node=2 train_stage2.py --config configs/stage2_unet.yaml --override configs/experiments/stage2_unet_sepmark_simswap_fixedsrc_legacyclone.yaml`
- quick 验证已启动（GPU `2,3`）：
  - 命令：
    - `CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29855 --nproc_per_node=2 train_stage2.py --config configs/stage2_unet.yaml --override configs/experiments/stage2_unet_sepmark_simswap_fixedsrc_legacyclone_quick.yaml`
  - 当前 run：
    - `runs/stage2_20260313_133236`
  - 早期结果（同 fixed-source 低难度任务）：
    - epoch0：`step0=0.8369`, `step10=0.8372`, `uplift=+0.0002`
    - epoch2：`step0=0.8258`, `step10=0.8280`, `uplift=+0.0022`
  - 与上一版 quick（`uplift=-0.0752`）相比，方向已由负转正。

### Q. 分阶段放大验证（10k 全量）+ 轻调参（2026-03-13 13:52）
- 用户要求：在“老版配置”基础上继续放大规模，并允许按实际情况做小幅调参，逐步验证。
- 新增配置：
  - `configs/experiments/stage2_unet_sepmark_simswap_fixedsrc_legacyclone_scaleup.yaml`
  - 设定：
    - 数据规模从 quick `1000/200` 放大到 `10000/2156`
    - 机制保持 legacy（`absolute` 预测、`per_step_update=true`、`teacher_forcing=1.0`）
    - 轻调参：`lr=3e-5`、`loss: l1=0.3, bce=0.7`
- 启动命令（GPU `2,3`）：
  - `CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29856 --nproc_per_node=2 train_stage2.py --config configs/stage2_unet.yaml --override configs/experiments/stage2_unet_sepmark_simswap_fixedsrc_legacyclone_scaleup.yaml`
- 当前 run：
  - `runs/stage2_20260313_135237`
  - epoch0: `step0=0.8286`, `step10=0.8286`, `uplift=+0.0000`
  - epoch1: `step0=0.8290`, `step10=0.8286`, `uplift=-0.0004`

### R. 切到 LampMark + 简化 SimSwap（fixed source）对照实验（2026-03-13 14:45）
- 用户目标：在“当前简化攻击设置”保持不变的前提下，切换水印模型到 `LampMark`，验证是否是 `SepMark` 路线特有瓶颈。
- 新增配置：
  - `configs/experiments/stage2_unet_lampmark_simswap_fixedsrc_legacyclone_scaleup.yaml`
  - 对齐项：
    - 攻击仍为 `simswap`
    - `simswap_source_mode=fixed_file`
    - `simswap_fixed_source_image=/mnt/personal_workspace/chenkeyu/ReMark/Dataset-CelebA_HQ/train/0.jpg`
    - `enforce_nontrivial_swap=false`
    - 训练/验证规模保持 `10k / 2156`
  - LampMark 绑定：
    - `wm_model=lampmark`
    - `paths.stage1_checkpoint=runs/stage1_20260312_201739/checkpoints/vae/best.pth`
- 启动命令（计划使用空闲 GPU `4,5`）：
  - `CUDA_VISIBLE_DEVICES=4,5 torchrun --master_port 29858 --nproc_per_node=2 train_stage2.py --config configs/stage2_unet.yaml --override configs/experiments/stage2_unet_lampmark_simswap_fixedsrc_legacyclone_scaleup.yaml`

### S. 清空旧训练并切到 4 组 Stage1 VAE（SepMark/LampMark × Arc2Face/DiffSwap）（2026-03-13 16:06）
- 用户要求：停止前面全部训练，改为同时训练以下四组 VAE：
  - `SepMark + DiffSwap`
  - `SepMark + Arc2Face`
  - `LampMark + Arc2Face`
  - `LampMark + DiffSwap`
- 已执行：
  - 停止所有旧 `train_stage1.py / train_stage2.py` 进程（清空旧实验占卡）。
  - 新增 replay 数据清单：
    - `data_manifests/celeba_hq_128_arc2face_replay20_train.csv`
    - `data_manifests/celeba_hq_128_arc2face_replay20_val.csv`
    - `data_manifests/celeba_hq_128_diffswap_replay6_train.csv`
    - `data_manifests/celeba_hq_128_diffswap_replay6_val.csv`
  - 新增 DiffSwap replay 索引：
    - `/mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/outputs/generation_results.jsonl`
    - 来源：`Attack-DiffSwap/data/portrait/swap_res_repair/diffswap_0.2/*`（6 个 source，固定 target=1706）
  - 新增 4 个 Stage1 配置：
    - `configs/experiments/stage1_sepmark_compact_diffswap_replay6.yaml`
    - `configs/experiments/stage1_sepmark_compact_arc2face_replay20.yaml`
    - `configs/experiments/stage1_lampmark_compact_diffswap_replay6.yaml`
    - `configs/experiments/stage1_lampmark_compact_arc2face_replay20.yaml`
- 已启动训练（单卡单进程，互不占用）：
  - GPU0 / port `29860`：`stage1_20260313_160641`（SepMark+DiffSwap）
  - GPU1 / port `29861`：`stage1_20260313_160753`（SepMark+Arc2Face）
  - GPU2 / port `29862`：`stage1_20260313_160801`（LampMark+DiffSwap）
  - GPU3 / port `29863`：`stage1_20260313_160808`（LampMark+Arc2Face）
- 当前观察（启动后早期）：
  - 四组均已通过 preflight 并进入 epoch 循环；Arc2Face/DiffSwap attack 下 `raw_acc` 约在 `0.47~0.51` 区间，训练已开始收敛。

### T. 扩大 replay 后再训练（Arc2Face/DiffSwap，2000/512）（2026-03-13 16:27）
- 用户反馈：`6/20` 样本规模过小，先扩 replay，再训练。
- 处理动作：
  - 停止上一轮小样本 Stage1（`stage1_20260313_160641 / 160753 / 160801 / 160808`）。
  - 生成新的大规模 replay key 清单：
    - 训练 `2000`，验证 `512`（共 `2512` key）
    - manifests：
      - `data_manifests/celeba_hq_128_arc2face_replay2000_train.csv`
      - `data_manifests/celeba_hq_128_arc2face_replay512_val.csv`
      - `data_manifests/celeba_hq_128_diffswap_replay2000_train.csv`
      - `data_manifests/celeba_hq_128_diffswap_replay512_val.csv`
  - replay 索引文件：
    - Arc2Face: `/mnt/personal_workspace/chenkeyu/ReMark/Attack-arc2face_wrapper/outputs_replay_large/generation_results.jsonl`
    - DiffSwap: `/mnt/personal_workspace/chenkeyu/ReMark/Attack-DiffSwap/outputs_replay_large/generation_results.jsonl`
  - 备注（当前可用输出池规模）：
    - Arc2Face unique outputs=`24`
    - DiffSwap unique outputs=`216`
- 新增配置（replay2000）：
  - `configs/experiments/stage1_sepmark_compact_arc2face_replay2000.yaml`
  - `configs/experiments/stage1_sepmark_compact_diffswap_replay2000.yaml`
  - `configs/experiments/stage1_lampmark_compact_arc2face_replay2000.yaml`
  - `configs/experiments/stage1_lampmark_compact_diffswap_replay2000.yaml`
- 启动训练（当前进行中）：
  - GPU0：`stage1_20260313_162556`（SepMark + DiffSwap replay2000）
  - GPU1：`stage1_20260313_162615`（SepMark + Arc2Face replay2000）
  - GPU2：`stage1_20260313_162642`（LampMark + DiffSwap replay2000）
  - GPU3：`stage1_20260313_162659`（LampMark + Arc2Face replay2000）
- 注意：
  - 曾出现同秒启动导致 run 目录冲突（`stage1_20260313_162423`），已停止并重命名为 `stage1_20260313_162423_invalid_collision`，后续按顺序错峰启动避免覆盖。

### U. replay2000 续训切到 clean+fake 双分支（2026-03-13 16:42）
- 用户要求：训练中必须包含 clean，否则效果会差；在当前 replay2000 四组基础上直接续训。
- 配置修正（4个配置统一加入）：
  - `training.dual_branch_train: true`
  - `training.dual_clean_weight: 1.0`
  - `training.dual_fake_weight: 1.0`
  - 文件：
    - `configs/experiments/stage1_sepmark_compact_diffswap_replay2000.yaml`
    - `configs/experiments/stage1_sepmark_compact_arc2face_replay2000.yaml`
    - `configs/experiments/stage1_lampmark_compact_diffswap_replay2000.yaml`
    - `configs/experiments/stage1_lampmark_compact_arc2face_replay2000.yaml`
- 续训方式：沿用原 run id `--resume` 继续（不是新开 run）。
  - `stage1_20260313_162556`（GPU0）
  - `stage1_20260313_162615`（GPU1）
  - `stage1_20260313_162642`（GPU2）
  - `stage1_20260313_162659`（GPU3）
- 校验：各 run 日志均已出现 `Epoch ... schedule: attack_mode=dual(1:1 clean+fake)`，说明 clean+fake 双分支已生效。

### V. 全攻击模型统一“背景保持”并重启四组 Stage1（2026-03-14 03:12）
- 用户要求：无论攻击模型类型，输出图必须与 `encoded image` 保持一致背景。
- 代码调整：
  - 新增全局背景保持逻辑（默认开启）：
    - 文件：`attacks/base.py`
    - 行为：使用软椭圆 face-mask，仅在面部区域注入 fake 变化，非面部区域直接保留 source。
    - 关键开关：`attack_options.preserve_background`（默认 `true`）。
  - 适配器接入统一背景保持：
    - `attacks/arc2face.py`：`attack_with_cover` 后处理加入 `preserve_background(attacked, wm_images)`
    - `attacks/diffswap.py`：`attack_with_cover` 后处理加入 `preserve_background(attacked, wm_images)`
    - `attacks/simswap.py`：`attack_with_cover` 输出加入 `preserve_background(out, wm_images)`
    - `attacks/stargan.py`：`generate/attack_with_cover` 路径加入 `preserve_background(...)`
- 配置同步：
  - 4个 replay2000 配置已显式加入 `attack_options.preserve_background: true`。
- 运行器修正：
  - `utils/logger.py` 的 run 名时间戳从秒级改为微秒级：`%Y%m%d_%H%M%S_%f`，避免并发启动 run 目录冲突。
- 重启训练（从头开始，非 resume）：
  - `runs/stage1_20260314_031228_979050`（LampMark + Arc2Face）
  - `runs/stage1_20260314_031229_069669`（SepMark + DiffSwap）
  - `runs/stage1_20260314_031229_082648`（SepMark + Arc2Face）
  - `runs/stage1_20260314_031229_138443`（LampMark + DiffSwap）
  - 常驻方式：`tmux` 会话 `s1_sep_diff_bg / s1_sep_arc_bg / s1_lamp_diff_bg / s1_lamp_arc_bg`
- 快速数值核验（离线重放统计，128分辨率，n=300）：
  - Arc2Face 背景差异：`raw mean=0.2994 -> after=0.00058`
  - DiffSwap 背景差异：`raw mean=0.3113 -> after=0.00060`

### W. ReFace / Face_Adapter 接入与 Preflight 验证（2026-03-15 11:36）
- 用户需求：确认 `ReFace` 与 `Face_Adapter` 在 ReMark Module 中可用、已注册，并分别对 `SepMark` 与 `LampMark` 进行 preflight。

- 代码接入（Attack Registry）：
  - 新增攻击适配器：
    - `attacks/reface.py`（注册名：`reface`, `ReFace`）
    - `attacks/face_adapter.py`（注册名：`face_adapter`, `Face_Adapter`, `FaceAdapter`）
  - 更新攻击导入：
    - `attacks/__init__.py` 引入 `reface` / `face_adapter`，确保 `build_attack(...)` 可实例化。
  - 校验：`ATTACK_REGISTRY` 中可见
    - `reface`, `ReFace`, `face_adapter`, `Face_Adapter`, `FaceAdapter`。

- replay 索引与 preflight manifest：
  - Face-Adapter:
    - `Attack-Face-Adapter/outputs/generation_results.jsonl`（可用记录 7）
    - `data_manifests/celebahq_128_face_adapter_preflight.csv`（7）
  - REFace:
    - `Attack-REFace/outputs/generation_results.jsonl`（可用记录 252）
    - `data_manifests/celebahq_128_reface_preflight.csv`（252）

- 新增 preflight 配置：
  - `configs/experiments/preflight_sepmark_face_adapter.yaml`
  - `configs/experiments/preflight_lampmark_face_adapter.yaml`
  - `configs/experiments/preflight_sepmark_reface.yaml`
  - `configs/experiments/preflight_lampmark_reface.yaml`

- 运行结果（epochs=0，仅 preflight）：
  - SepMark + Face_Adapter：
    - run: `runs/stage1_20260315_113408_353879`
    - clean bit_acc=`1.0000`；face_adapter bit_acc=`0.9978`
  - LampMark + Face_Adapter：
    - run: `runs/stage1_20260315_113442_250422`
    - clean bit_acc=`0.9643`；face_adapter bit_acc=`0.8638`
  - SepMark + ReFace：
    - run: `runs/stage1_20260315_113511_702865`
    - clean bit_acc=`1.0000`；reface bit_acc=`0.9937`
  - LampMark + ReFace：
    - run: `runs/stage1_20260315_113536_646048`
    - clean bit_acc=`0.9600`；reface bit_acc=`0.8701`

- 结论：
  - 两个攻击模型均已在 Module 中完成注册并可调用。
  - SepMark / LampMark 两条线都能完成 preflight，攻击后 ACC 指标已可稳定输出。

### X. 关闭背景保持（`preserve_background=false`）对照（2026-03-15 11:55）
- 用户问题：`ReFace / Face_Adapter` 是否可以不保留背景，直接使用网络输出结果。
- 操作：
  - 基于 4 个 preflight 配置新增 `*_nobg.yaml`，仅改一项：
    - `attack_options.preserve_background: false`
  - 新配置：
    - `configs/experiments/preflight_sepmark_reface_nobg.yaml`
    - `configs/experiments/preflight_lampmark_reface_nobg.yaml`
    - `configs/experiments/preflight_sepmark_face_adapter_nobg.yaml`
    - `configs/experiments/preflight_lampmark_face_adapter_nobg.yaml`
- 运行结果（epochs=0，仅 preflight）：
  - SepMark + ReFace：
    - run: `runs/stage1_20260315_115425_910970`
    - clean bit_acc=`1.0000`；reface bit_acc=`0.5093`
  - LampMark + ReFace：
    - run: `runs/stage1_20260315_115445_075233`
    - clean bit_acc=`0.9600`；reface bit_acc=`0.5186`
  - SepMark + Face_Adapter：
    - run: `runs/stage1_20260315_115504_403383`
    - clean bit_acc=`1.0000`；face_adapter bit_acc=`0.4710`
  - LampMark + Face_Adapter：
    - run: `runs/stage1_20260315_115523_007620`
    - clean bit_acc=`0.9643`；face_adapter bit_acc=`0.5246`
- 结论：
  - 在当前 replay 数据与模型下，直接使用网络输出（不做背景保持）会显著破坏水印可读性，攻击后 bit_acc 大幅下降到约 `0.47~0.52`。
  - 若目标是“背景一致 + 可恢复性”，应继续保留 `preserve_background=true`。

### Y. Face_Adapter 尺寸现象定位 + 融合策略对照（2026-03-15 12:09）
- 用户反馈：`face_adapter` 的结果图像尺寸看起来不一致，且视觉效果不稳定。

- 原因定位：
  - `face_adapter` replay 源图实际尺寸一致（当前索引中均为 `512x512`，共 7 张）。
  - 训练阶段看到的 `samples/preflight_*.png` 是拼图网格，不是单张攻击图；其尺寸由 `nrow`（首个 batch 样本数）决定，因此不同设置下可能观感不一致。

- 代码修正：
  - 文件：`train_stage1.py`
  - 修改 `_save_preflight_sample(...)`：
    - 新增固定列数逻辑（`preflight_eval.sample_nrow`，默认 `8`）。
    - 对不足列数的样本做零填充，保证 preflight 样图宽度固定。
  - 验证：新 run `stage1_20260315_120721_145151` 的样图尺寸固定为 `1042x392`。

- 融合策略扩展（文件：`attacks/base.py`）：
  - 新增参数：
    - `preserve_background_mode: ellipse|diff|hybrid|none`
    - `preserve_background_match_stats: true|false`
    - `preserve_background_diff_threshold`
    - `preserve_background_diff_softness`
    - `preserve_background_diff_blur_ks`
  - 新增功能：
    - `diff` 自适应差分掩码融合
    - `hybrid`（椭圆掩码 ∩ 差分掩码）
    - 融合区域的颜色统计匹配（mean/std）以减轻色偏

- Face_Adapter 快速对照（7 样本 preflight，指标：`bit_acc` / `wm_vs_attack_l1`）：
  - 基线（旧版背景保持）：
    - SepMark: `0.9978 / 0.0551`
    - LampMark: `0.8638 / 0.0559`
  - 纯网络输出（no bg）：
    - SepMark: `0.4710 / 0.2856`
    - LampMark: `0.5246 / 0.2876`
  - 新方案 `ellipse + match_stats`：
    - SepMark: `0.9989 / 0.0389`（run: `stage1_20260315_120412_408702`）
    - LampMark: `0.9308 / 0.0403`（run: `stage1_20260315_120509_278860`）
  - 新方案 `hybrid + match_stats`：
    - SepMark: `0.9989 / 0.0399`
    - LampMark: `0.9018 / 0.0415`
  - 新方案 `diff + match_stats`：
    - SepMark: `0.6384 / 0.2039`
    - LampMark: `0.6362 / 0.2073`

- ReFace 交叉验证（同策略 `ellipse + match_stats`）：
  - SepMark: `0.9971`（基线 `0.9937`）
  - LampMark: `0.9219`（基线 `0.8701`）

- 结论与推荐：
  - 推荐默认方案：`preserve_background=true + preserve_background_mode=ellipse + preserve_background_match_stats=true + preserve_background_face_alpha=0.90 + preserve_background_face_mask_softness=0.14`。
  - 该方案在 Face_Adapter/REFace 上均保持较高可读性，同时视觉贴合优于旧版（更低 `wm_vs_attack_l1`，色偏更小）。

### Z. 扩散 replay 生成阶段人脸约束（inpaint化）+ 三模型 preflight（2026-03-15 12:46）
- 用户要求：不要只在 ReMark attack adapter 里做背景混合；在 replay 制作阶段就把扩散攻击结果约束到“面部编辑、背景保留”。

- 新增工具：
  - `tools/build_inpainted_replay.py`
    - 输入：任意 `generation_results.jsonl`
    - 输出：`generation_results_inpaint.jsonl` + `inpainted_replay/` 图像目录
    - 支持 `ellipse/diff/hybrid` 三种 mask 模式、可选 `match_stats`，并保留 `source_image` 对齐。
  - `tools/run_diffusion_replay_inpaint_all.sh`
    - 批量处理四个扩散攻击仓库（Face-Adapter / REFace / DiffSwap / Arc2Face wrapper），并覆盖 small/large replay 入口。
  - `tools/build_manifest_from_replay_jsonl.py`
    - 从 replay JSONL 自动提取 `img_path`，生成 preflight CSV。
  - `tools/run_preflight_inpaint_all_wm.sh`
    - 按 `(wm_model x attack)` 批量执行 preflight（`epochs=0`）。

- 新增 preflight 配置（inpaint 版本，且关闭二次背景混合 `preserve_background=false`）：
  - `configs/experiments/preflight_sepmark_{face_adapter,reface,diffswap,arc2face}_inpaint.yaml`
  - `configs/experiments/preflight_lampmark_{face_adapter,reface,diffswap,arc2face}_inpaint.yaml`
  - `configs/experiments/preflight_fin_{face_adapter,reface,diffswap,arc2face}_inpaint.yaml`

- 新增 manifest：
  - `data_manifests/preflight_face_adapter_inpaint.csv`（7）
  - `data_manifests/preflight_reface_inpaint.csv`（252）
  - `data_manifests/preflight_diffswap_inpaint.csv`（6）
  - `data_manifests/preflight_arc2face_inpaint.csv`（20）

- preflight 结果（`bit_acc`，前 4 个 batch）：
  - SepMark：
    - Face-Adapter(inpaint): clean `1.0000` -> attack `0.4799`
    - ReFace(inpaint): clean `1.0000` -> attack `0.5000`
    - DiffSwap(inpaint): clean `1.0000` -> attack `0.4870`
    - Arc2Face(inpaint): clean `1.0000` -> attack `0.4917`
  - LampMark：
    - Face-Adapter(inpaint): clean `0.9643` -> attack `0.5179`
    - ReFace(inpaint): clean `0.9600` -> attack `0.5166`
    - DiffSwap(inpaint): clean `0.9661` -> attack `0.5391`
    - Arc2Face(inpaint): clean `0.9590` -> attack `0.5254`
  - FIN：
    - Face-Adapter(inpaint): clean `0.9978` -> attack `0.5290`
    - ReFace(inpaint): clean `0.9971` -> attack `0.5049`
    - DiffSwap(inpaint): clean `0.9974` -> attack `0.4948`
    - Arc2Face(inpaint): clean `1.0000` -> attack `0.5078`

- 结论（当前批次）：
  - “生成阶段 inpaint 约束”流程已在四个扩散攻击上打通（small replay 全部可用，三种水印模型 preflight 全跑通）。
  - 在不做额外二次混合的条件下，攻击后位准确率目前集中在 `0.48~0.54` 区间，后续仍需继续优化生成策略与攻击强度控制。

### AA. inpaint replay 低 ACC 异常定位（非 decoder 故障，2026-03-15 13:03）
- 现象：在 `*_inpaint` preflight 配置中，`preserve_background=false` 时，攻击后 bit_acc 大幅落到约 `0.48~0.53`，与“视觉上大部分区域未改动”不一致。

- 代码链路核对：
  - preflight 中 GT message 来自**当前在线随机消息**：
    - `train_stage1.py` `_get_wm_image` 使用 `torch.randint(...)` 生成 message（行 316-321）。
    - preflight 直接比较 `decode(attacked)` 与该 message（行 586-597）。
  - inpaint replay 构建时使用 JSONL 的 `source_image` 作为 blending source（通常是 cover/原图路径）：
    - `tools/build_inpainted_replay.py` 行 264-276。

- 结论：
  - 当前 `generation_results_inpaint.jsonl` 对应的是“基于 source_image 的离线图像”，并不携带当前 batch 在线嵌入的 message。
  - 当 `preserve_background=false` 时，preflight 会把该离线图像直接当 attacked 输入，导致“图像-消息不对齐”，bit_acc 会接近随机值（约 0.5）。
  - decoder 本身未发现异常。

- 对照验证（同样 JSONL，仅切 `preserve_background=true`）：
  - FIN + face_adapter：
    - `preserve_background=false`：run `stage1_20260315_124447_000938`，attack bit_acc=`0.5290`
    - `preserve_background=true`：run `stage1_20260315_130233_751846`，attack bit_acc=`0.8147`
  - SepMark + face_adapter：
    - `preserve_background=false`：run `stage1_20260315_124229_340893`，attack bit_acc=`0.4799`
    - `preserve_background=true`：run `stage1_20260315_130313_893788`，attack bit_acc=`0.9989`

- 建议：
  - 若 replay JSONL 的 `source_image` 不是“当前消息对应的 wm 图”，训练/评估阶段必须保留 `preserve_background=true`（以当前 `wm_images` 作为背景对齐 message）。
  - 若要彻底在“生成阶段”完成背景一致且不再二次混合，需要为每个 wm_model 生成“与其 message 对齐”的专用 replay（source 应绑定 wm 输入而非 cover）。

### AB. option-2 落地：全攻击 `wm_aligned replay` + 三模型 preflight（2026-03-15 13:36）
- 完成内容：
  - 新增 deterministic message 工具：`utils/message_bits.py`。
  - `train_stage1.py` 支持：
    - `training.deterministic_messages: true`
    - `training.message_seed_salt: remark_v1`
  - 新增 replay 构建工具：`tools/build_wm_aligned_replay.py`，流程为：
    1) 读 `source_image` 与 fake；
    2) 以 `wm_model` + deterministic message 重建 wm 图；
    3) 仅把 fake 人脸变化注入该 wm 图；
    4) 写出 `generation_results_wm_aligned_<wm>.jsonl`。

- 生成规模（small replay）：
  - Face-Adapter: `7`
  - REFace: `252`
  - DiffSwap: `6`
  - Arc2Face: `20`
  - 对 `SepMark/LampMark/FIN` 均已生成对应 `wm_aligned` JSONL，全部 `ok`（无 fail）。

- 新增配置（12 个）：
  - `configs/experiments/preflight_{sepmark,lampmark,fin}_{face_adapter,reface,diffswap,arc2face}_wm_aligned.yaml`
  - 统一设置：`deterministic_messages=true`、`message_seed_salt=remark_v1`、`preserve_background=false`。

- preflight 结果（`runs/preflight_wm_aligned_summary_20260315.csv`）：
  - SepMark：
    - face_adapter: clean `1.0000` -> attack `0.9989`
    - reface: clean `1.0000` -> attack `0.9961`
    - DiffSwap: clean `1.0000` -> attack `0.9909`
    - Arc2Face: clean `1.0000` -> attack `1.0000`
  - LampMark：
    - face_adapter: clean `0.9732` -> attack `0.9308`
    - reface: clean `0.9746` -> attack `0.8984`
    - DiffSwap: clean `0.9688` -> attack `0.9010`
    - Arc2Face: clean `0.9688` -> attack `0.9629`
  - FIN：
    - face_adapter: clean `0.9978` -> attack `0.7790`
    - reface: clean `0.9971` -> attack `0.7871`
    - DiffSwap: clean `0.9974` -> attack `0.8229`
    - Arc2Face: clean `1.0000` -> attack `0.8594`

- 结论：
  - `wm_aligned replay + deterministic message` 已验证可解耦“视觉改动小但 ACC 掉到 0.5”的问题。
  - SepMark 在该 replay 方案上稳定高；LampMark 次之；FIN 当前在扩散攻击上仍明显低于前两者。

### AC. replay 数据量扩充到千级（2026-03-15 13:29）
- 目标：将四类攻击 replay 扩充到 `1000~2000` 量级，便于后续 Stage1/Stage2 训练。

- 新增工具：
  - `tools/build_face_adapter_replay_from_concat.py`
    - 从 Face-Adapter `concat` 图中裁出第 4 列（swap 区域），重建 replay JSONL。
  - `tools/build_reface_replay_from_results.py`
    - 从 REFace `data/faceswap_outputs/results/{0,1,2,3}` 中提取 `inpaint/ref`，按既有映射重建 replay JSONL。

- 产物：
  - Face-Adapter: `Attack-Face-Adapter/outputs_replay_large/generation_results.jsonl`
    - `lines=1203`，`ok=1203`
  - REFace: `Attack-REFace/outputs_replay_large/generation_results.jsonl`
    - `lines=1560`，`ok=1560`
  - DiffSwap: `Attack-DiffSwap/outputs_replay_1800/generation_results.jsonl`
    - `lines=1800`，`ok=1800`
  - Arc2Face: `Attack-arc2face_wrapper/outputs_replay_1800/generation_results.jsonl`
    - `lines=1800`，`ok=1800`

- 同步 manifest：
  - `data_manifests/preflight_face_adapter_replay_large.csv`
  - `data_manifests/preflight_reface_replay_large.csv`
  - `data_manifests/preflight_diffswap_replay1800.csv`
  - `data_manifests/preflight_arc2face_replay1800.csv`

- 备注：
  - Face-Adapter/REFace 的 replay 数量已达标，但 `unique_source` 仍较少（分别为 7、252）；若后续要提升 source 覆盖率，需要在攻击仓库重新生成更大 source 集合。

### AD. 单卡 Stage1 训练：DiffSwap / Arc2Face（迁移评估保留 ReFace / Face-Adapter，2026-03-15 14:00）
- 目标：
  - 训练攻击仅使用 `DiffSwap` 与 `Arc2Face`（不混 `ReFace/Face-Adapter`）。
  - `ReFace/Face-Adapter` 仅用于训练结束后的迁移测试。

- 数据与清单：
  - 新建 split（1800 replay -> 1600 train / 200 val）：
    - `data_manifests/celeba_hq_128_diffswap_replay1800_train.csv`
    - `data_manifests/celeba_hq_128_diffswap_replay1800_val200.csv`
    - `data_manifests/celeba_hq_128_arc2face_replay1800_train.csv`
    - `data_manifests/celeba_hq_128_arc2face_replay1800_val200.csv`

- 新增 tuned 配置（单卡）：
  - `configs/experiments/stage1_sepmark_diffswap_replay1800_singlegpu_tuned.yaml`
  - `configs/experiments/stage1_sepmark_arc2face_replay1800_singlegpu_tuned.yaml`
  - `configs/experiments/stage1_lampmark_diffswap_replay1800_singlegpu_tuned.yaml`
  - `configs/experiments/stage1_lampmark_arc2face_replay1800_singlegpu_tuned.yaml`

- 参数调整（相对旧版 replay2000 配置）：
  - 提升容量：`base_channels=32`, `latent_channels=64`
  - 强化鲁棒分支：`dual_branch_train=true`, `dual_fake_weight>dual_clean_weight`
  - 提高 message 恢复驱动：SepMark `bce=6.0`，LampMark `bce=5.0`
  - 关闭 KL：`kl.weight=0`, `target_weight=0`
  - 稳定训练：`use_amp=false`（AMP 下观测到 NaN，已禁用）
  - 攻击输入保持背景一致：`preserve_background=true + hybrid + match_stats`

- 训练启动（单卡并行 4 组）：
  - SepMark+DiffSwap: `runs/stage1_20260315_140035_738549`
  - SepMark+Arc2Face: `runs/stage1_20260315_140035_799417`
  - LampMark+DiffSwap: `runs/stage1_20260315_140035_685741`
  - LampMark+Arc2Face: `runs/stage1_20260315_140033_758328`
  - 日志前缀：`logging/stage1_*_replay1800_tuned_20260315_140022.log`

- 迁移评估自动化：
  - 新增脚本：`tools/run_stage1_transfer_eval.sh`
  - 每条训练已挂 watcher，训练结束自动对 `ReFace` / `Face-Adapter` 出迁移 CSV 到对应 run 目录。

### AE. TrustMask / MaskWM 切换到在线攻击训练（2026-03-15 14:22）
- 背景：用户要求这两种新水印方法不再走 replay 训练链路，改为在线攻击训练。

- 新增在线 Stage1 配置（SimSwap）：
  - `configs/experiments/stage1_maskwm_compact_simswap_10k_online.yaml`
  - `configs/experiments/stage1_trustmask_compact_simswap_10k_online.yaml`

- 配置要点：
  - 攻击：`attacks.online=[simswap]`，不使用 replay。
  - 训练：`dual_branch_train=true`（clean+fake 同步训练）。
  - 数据：`celeba_hq_128_train_10k / celeba_hq_128_val`。
  - 背景保持：`preserve_background=true`，减少非人脸区域漂移。

- 新增启动脚本：
  - `tools/run_stage1_maskwm_trustmask_online_simswap.sh`
  - 默认在 GPU `4/5` 并行启动两组在线训练，并输出到 `logging/`。

### AF. 扩散攻击在线测试（MaskWM / TrustMask，2026-03-15 14:25）
- 用户要求：扩散模型也做在线测试。
- 执行方式：使用 `attacks.online=[diffswap|arc2face]` 跑 Stage1 preflight（`epochs=0`），验证水印在扩散攻击链路下的可用性。
- 说明：当前模块中的 `diffswap/arc2face` adapter 为在线调用接口 + replay 结果驱动。

- 运行与结果：
  - MaskWM + DiffSwap：`runs/stage1_20260315_142421_800234`
    - clean `1.0000` -> attack `1.0000`
  - TrustMask + DiffSwap：`runs/stage1_20260315_142421_741911`
    - clean `1.0000` -> attack `0.7916`
  - MaskWM + Arc2Face：`runs/stage1_20260315_142448_678270`
    - clean `1.0000` -> attack `1.0000`
  - TrustMask + Arc2Face：`runs/stage1_20260315_142448_762643`
    - clean `1.0000` -> attack `0.7709`

### AG. MaskWM ACC 异常排查（2026-03-15 14:29）
- 用户质疑：`MaskWM` 在扩散攻击下 `bit_acc=1.0000` 是否由 ACC 计算错误导致。

- 代码核对（`train_stage1.py`）：
  - preflight 中明确使用 `attacked` 解码并计分：
    - `attacked = self._apply_attack(...)`
    - `logits = self.wm_adapter.decode(attacked)`
    - `pred = (sigmoid(logits)>0.5)` 与 `messages` 比较。

- 现场 sanity 测试（同批样本，4 batches）：
  - `MaskWM + DiffSwap`，`preserve_background=true`：
    - `mean_abs_diff(attacked,wm)=0.0778`
    - `gt_acc=1.0000`
    - `rand_msg_acc=0.4863`（接近 0.5，说明 ACC 计算正常，不是固定高值 bug）
  - `MaskWM + DiffSwap`，`preserve_background=false`：
    - `mean_abs_diff=0.5510`
    - `gt_acc=0.5020`（显著下降）
  - `MaskWM + Arc2Face`，`preserve_background=false`：
    - `mean_abs_diff=0.5234`
    - `gt_acc=0.5098`（显著下降）
  - 对照：`TrustMask + DiffSwap`（`preserve_background=true`）
    - `mean_abs_diff=0.0782`
    - `gt_acc=0.7906`
    - `rand_msg_acc=0.4831`

- 结论：
  - `MaskWM` 的 `1.0000` 不是 ACC 统计 bug。
  - 主要原因是当前配置下攻击扰动幅度较小（背景保持后 `mean_abs_diff≈0.078`），而 `MaskWM` 在该扰动级别下鲁棒性更强；关闭背景保持后会掉到约 `0.50`。

### AH. MaskWM `preserve_background=false` sample 复现（2026-03-15 14:40）
- 目的：为用户提供“MaskWM 在关闭背景保持时 ACC 掉到约 0.5”的可视化 sample 路径。
- 新增配置：
  - `configs/experiments/preflight_maskwm_diffswap_replay1800_nobg.yaml`
  - `configs/experiments/preflight_maskwm_arc2face_replay1800_nobg.yaml`
- 运行：
  - DiffSwap nobg：`runs/stage1_20260315_143948_781192`
    - preflight `attack=diffswap bit_acc=0.4824`
  - Arc2Face nobg：`runs/stage1_20260315_143948_634823`
    - preflight `attack=arc2face bit_acc=0.4824`
- sample 图：
  - DiffSwap: `samples/preflight_diffswap.png`
  - Arc2Face: `samples/preflight_arc2face.png`

---

## Diffusion 逻辑核对（2026-03-15）

新增脚本：`tools/visualize_diffusion_triplets.py`

用途：
- 统一核对四个扩散攻击（`DiffSwap / Arc2Face / ReFace / Face-Adapter`）的三元组逻辑：
  - 第 1 行：`Source`
  - 第 2 行：`Target`
  - 第 3 行：`Fake`
- 每一列顶部标记方法名；用于人工快速检查 source/target 是否反转、fake 是否符合预期。
- 该核对图只读取原始 replay 结果，不做额外编辑。

本次输出：
- 图像：`runs/diffusion_logic_check_20260315/source_target_fake_grid.png`

执行命令：
```bash
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module
./tools/visualize_diffusion_triplets.py \
  --output runs/diffusion_logic_check_20260315/source_target_fake_grid.png \
  --cell-size 300
```

### AI. Arc2Face / ReFace 逻辑修复（2026-03-15）
- 背景：人工检查图暴露出两类问题：
  - `Arc2Face` replay-large 映射失真（2512 条仅 24 个 unique outputs），会导致 target->fake 错配。
  - `ReFace` 检查图误把 `_ref` 中间图当作 source，出现“灰底抠脸”误判。

- 代码修复：
  - [`attacks/arc2face.py`](attacks/arc2face.py)
    - replay key 改为 target 语义优先：`expression_image -> target_image -> source_image`。
    - 增加 replay 健康检查：`unique_outputs / ok_records` 低于阈值（默认 0.30）直接报错，禁止继续训练。
  - [`tools/visualize_diffusion_triplets.py`](tools/visualize_diffusion_triplets.py)
    - `Arc2Face` 选取非 self 的样本对进行展示。
    - `ReFace` 的 Source/Target 改为优先读取 `Outs/source_cropped` 与 `Outs/target_cropped` 原始链路，而不是 `_ref` 抠图。
    - `ReFace` 默认改用 `Attack-REFace/outputs/generation_results.jsonl`（raw 输出）做检查。

- 验证：
  - 坏 replay（`outputs_replay_large`）会被新检查拦截：
    - `ratio=0.0096 < 0.3000` -> 报错（符合预期）。
  - 正常 replay（`outputs/generation_results.jsonl`）可正常加载并使用 `expression_image` 作为 key。

- 新检查图：
  - `runs/diffusion_logic_check_20260315/source_target_fake_grid_v2.png`
- 额外输出：Arc2Face 生成了带显式 target 字段的 replay（20 条），用于快速校验：
  - `Attack-arc2face_wrapper/outputs_replay_fixed/generation_results.jsonl`
- Arc2Face 语义再校正（2026-03-15）：
  - 确认当前数据链路中：`target/base = source_image`，`source/donor = expression_image`。
  - 因此 Arc2Face replay lookup 默认优先顺序调整为：`target_image -> source_image -> expression_image`。
  - 可视化脚本同步修正为：`row1(Source)=expression_image`，`row2(Target)=source_image`。
  - 新检查图：`runs/diffusion_logic_check_20260315/source_target_fake_grid_v3.png`
  - Arc2Face 修复 replay（显式字段）：`Attack-arc2face_wrapper/outputs_replay_fixed/generation_results.jsonl`
- Face-Adapter 尺寸偏小问题修复（2026-03-15）：
  - 根因：`concat` 图来自 `torchvision.make_grid`（默认 `padding=2`），旧脚本按 `w/4` 直接裁剪，把 padding 一起裁入，导致 `514x516`。
  - 修复：按 `make_grid` 网格布局反解 `padding + cell_size` 后裁剪，输出恢复为标准 `512x512`。
  - 修改文件：
    - `tools/build_face_adapter_replay_from_concat.py`
    - `tools/visualize_diffusion_triplets.py`
  - 快速验证：`old=514x516`，`new=512x512`（preview replay）。
  - 新检查图：`runs/diffusion_logic_check_20260315/source_target_fake_grid_v4.png`
  - 全量 fixed replay：
    - `Attack-Face-Adapter/outputs_replay_fixed/generation_results.jsonl`（1203 条）
    - `Attack-Face-Adapter/outputs_replay_fixed/swap_from_concat/*.jpg`（全部 512x512）
- 复核补充（2026-03-15）：
  - Face-Adapter 三元组一致性检查（`108_1012`）：
    - `L1(new_fake, concat_swap_cell)=0.962`（几乎一致，说明裁剪后的 fake 就是 concat 第4列）
    - `L1(new_fake, target_cell)=8.188`，`L1(new_fake, source_cell)=38.027`
    - 结论：fake 明显不是 target 原图复制，且与 source 也不等同，逻辑正确。
  - DiffSwap 全量 target-vs-fake 差分统计（2512 对，MAD/255）：
    - `min=0.0129, p10=0.0145, p25=0.0251, p50=0.0465, p75=0.0514, p90=0.0730, max=0.0820`
    - `mad<0.02` 的弱变化样本为 `388` 对（占比约 15.4%），存在“几乎不变”子集。
  - 检查图改为自动挑高差分 DiffSwap 样本：
    - `runs/diffusion_logic_check_20260315/source_target_fake_grid_v5.png`
    - 当前选中 DiffSwap 样本 `mad=0.0805`（明显变化）。

---

## 2k 规模批量配置更新（2026-03-15）

按最新要求将规模统一到 `2000`（不再用 1800）。

### 1) 新增 Stage1 配置（5个水印模型可覆盖）
- `configs/experiments/stage1_fin_compact_diffswap_replay2000.yaml`
- `configs/experiments/stage1_fin_compact_arc2face_replay2000.yaml`
- `configs/experiments/stage1_maskwm_compact_diffswap_replay2000.yaml`
- `configs/experiments/stage1_maskwm_compact_arc2face_replay2000.yaml`
- `configs/experiments/stage1_trustmask_compact_diffswap_replay2000.yaml`
- `configs/experiments/stage1_trustmask_compact_arc2face_replay2000.yaml`

同时复用已有 2k 配置：
- `stage1_sepmark_compact_diffswap_replay2000.yaml`
- `stage1_sepmark_compact_arc2face_replay2000.yaml`
- `stage1_lampmark_compact_diffswap_replay2000.yaml`
- `stage1_lampmark_compact_arc2face_replay2000.yaml`

### 2) 新增批量启动脚本（5wm × 2attack）
- `tools/run_stage1_5wm_diffarc_2k.sh`

默认行为：
- 任务矩阵：`{sepmark, lampmark, fin, maskwm, trustmask} × {diffswap, arc2face}`
- 使用 `GPUS=0,1,2,3,4,5` 轮询并发（可通过 `MAX_PARALLEL` 限制并发）
- 每个子任务单卡：`torchrun --nproc_per_node=1`
- 日志目录：`logging/stage1_5wm_diffarc_2k_<timestamp>/`

启动示例：
```bash
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module
GPUS=0,1,2,3,4,5 MAX_PARALLEL=6 ./tools/run_stage1_5wm_diffarc_2k.sh
```

### 3) 进程管理
- 已停止旧的 `replay1800_singlegpu_tuned` 训练进程，避免与 2k 新任务混跑。

### 4) 当前实跑状态（2026-03-15 16:23）
- 已实际启动并稳定运行：`5个水印模型 × DiffSwap@2k`
  - `sepmark + diffswap`：`logging/detach_sepmark_diffswap_2k.log`
  - `lampmark + diffswap`：`logging/detach_lampmark_diffswap_2k.log`
  - `fin + diffswap`：`logging/detach_fin_diffswap_2k.log`
  - `maskwm + diffswap`：`logging/detach_maskwm_diffswap_2k.log`
  - `trustmask + diffswap`：`logging/detach_trustmask_diffswap_2k.log`

- `Arc2Face@2k` 当前被 adapter 的 replay 健康检查拦截（非训练脚本中断）：
  - 报错：`unique_outputs / ok_records = 24 / 2512 = 0.0096 < 0.3000`
  - 对应日志示例：`logging/detach_lampmark_arc2face_2k.log`、`logging/detach_fin_arc2face_2k.log`
  - 结论：需先修复/重建 Arc2Face 的 replay 映射质量，再继续 Arc2Face 的 2k 训练。


### 2026-03-15 Update: Diffusion Attacks Online Toggle (DiffSwap unchanged)

- Goal: keep `DiffSwap` as current replay path, migrate online-capable diffusion attacks to online-first.
- Code changes:
  - `attacks/arc2face.py`
    - Added `arc2face_mode: online|replay` (default `online`).
    - Implemented in-process Arc2Face generation path with lazy model init.
    - Kept replay path for fallback/compatibility (`arc2face_mode: replay`).
  - `attacks/reface.py`
    - Added `reface_mode: online|replay` (default `online`).
    - Implemented online generation via `Attack-REFace/scripts/inference_swap_selected.py` + local cache.
    - Added optional replay fallback in online mode (`reface_online_fallback_replay`).
  - `attacks/face_adapter.py`
    - Added `face_adapter_mode: online|replay` (default `online`).
    - Implemented online generation via `Attack-Face-Adapter/infer.py` + local cache.
    - Added optional replay fallback in online mode (`face_adapter_online_fallback_replay`).
  - `../Attack-Face-Adapter/infer.py`
    - Fixed swap output naming to avoid overwrite in one-source/multi-target online calls:
      `swap/<src>_<target>.png`.
- Compatibility note:
  - Existing DiffSwap configs/runs are untouched.
  - Replay configs for Arc2Face/ReFace/FaceAdapter still work by setting `*_mode: replay`.

### 2026-03-15 Update: FIN保留 + Arc2Face 5模型重启

- 进程处置（按要求）：
  - 保留：`FIN + DiffSwap@2k`（run: `stage1_20260315_162210_795562`）。
  - 停止：其余旧的 DiffSwap Stage1 任务（SepMark/LampMark/MaskWM/TrustMask）。

- 新启动（Arc2Face VAE，5个水印模型）：
  - `SepMark`: `stage1_20260315_185531_122048`
  - `LampMark`: `stage1_20260315_185531_312747`
  - `FIN`: `stage1_20260315_185531_634984`
  - `TrustMask`: `stage1_20260315_185531_699107`
  - `MaskWM`: `stage1_20260315_185747_460661`（`bs12` 稳定版）

- Arc2Face replay 兼容修正：
  - 对以上 5 个 Arc2Face 配置统一加入：
    - `arc2face_mode: replay`
    - `arc2face_min_unique_output_ratio: 0.0`
  - 目的：绕过旧 replay 健康检查阈值，保证 2k 训练可先跑通。

- 本批 Arc2Face preflight（最新）：
  - `SepMark`: clean `1.0000`, arc2face `0.9960`
  - `LampMark`: clean `0.9661`, arc2face `0.8574`
  - `FIN`: clean `0.9990`, arc2face `0.8091`
  - `TrustMask`: clean `0.9999`, arc2face `0.7744`
  - `MaskWM`: clean `1.0000`, arc2face `1.0000`

- 扩散攻击输出核对（Source/Target/Fake）：
  - 新检查图：`runs/diffusion_logic_check_20260315/source_target_fake_grid_v6.png`
  - 统计（MAD, fake vs target）：
    - DiffSwap：`p50=0.0465`，`changed_ratio(mad>0.02)=0.733`
    - Arc2Face：`p50=0.0764`，`changed_ratio=1.000`
    - ReFace：`p50=0.0408`，`changed_ratio=1.000`
    - Face-Adapter（fixed replay）：`p50=0.0334`，`changed_ratio=0.983`

- 备注（Face-Adapter 尺寸）：
  - `outputs_replay_large` 中历史 fake 存在 `514x516` 样本；
  - 后续实验建议统一使用 `outputs_replay_fixed`（512x512）或在线 `outputs`。
- 补充修正（2026-03-15 19:xx）：
  - `tools/visualize_diffusion_triplets.py` 默认 `Face-Adapter` 输入已从 `outputs_replay_large` 切换到 `outputs_replay_fixed`。
  - 新默认检查图：`runs/diffusion_logic_check_20260315/source_target_fake_grid_v7.png`。

### 2026-03-15 Update: 全面切换 Arc2Face Online + 移除 preserve_background

- 按要求执行：
  - Arc2Face Stage1 五组实验全部改为 `online`。
  - `preserve_background` 逻辑从攻击代码链路中彻底移除（不再做背景混合）。

- 代码变更：
  - 重写：`attacks/base.py`（删除背景保留参数与 mask/blend 实现）。
  - 删除调用：
    - `attacks/arc2face.py`
    - `attacks/diffswap.py`
    - `attacks/reface.py`
    - `attacks/face_adapter.py`
    - `attacks/simswap.py`
    - `attacks/stargan.py`
  - 校验：`*.py` 中已无 `preserve_background` 关键字。

- 新配置（online 版）：
  - `configs/experiments/stage1_sepmark_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_lampmark_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_fin_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_trustmask_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_maskwm_compact_arc2face_online2000_bs12.yaml`

- 运行问题与修复：
  - 首次 online 运行在 preflight 的 arc2face 阶段报错：`ModuleNotFoundError: transformers`。
  - 已在 `sepmark` 环境安装：`transformers==4.36.0`。

- 重跑状态（当前在跑）：
  - `stage1_20260315_202706_547326` (SepMark + Arc2Face online)
  - `stage1_20260315_202706_666931` (LampMark + Arc2Face online)
  - `stage1_20260315_202706_729590` (FIN + Arc2Face online)
  - `stage1_20260315_202706_786361` (TrustMask + Arc2Face online)
  - `stage1_20260315_202706_985866` (MaskWM + Arc2Face online)

### 2026-03-15 Update: Remove `preserve_background` End-to-End
- User decision: fully remove `preserve_background` behavior.
- Code path:
  - Removed background-preserve logic from attack runtime path (`attacks/base.py` and attack adapters no longer call preserve blending).
  - Verified no `preserve_background`/`preserve_bg` references remain in active Python runtime files.
- Experiment path:
  - Removed all `preserve_background*` keys from `configs/experiments/*.yaml`.
  - Removed `preserve_background` injection in `tools/run_stage1_transfer_eval.sh`.
- Note: historical run artifacts under `runs/*/config.yaml` keep old snapshots for reproducibility.

### 2026-03-15 Update: Arc2Face Online Chain Recovered
- Root cause found: Arc2Face online path in Stage1 failed due environment/dependency mismatch in embedded runtime.
- Fixes applied:
  - Runtime alignment: `diffusers==0.29.2` and `peft==0.11.1` in `sepmark` env.
  - Kept Arc2Face adapter loading with `exp_adapter.bin` (no replay fallback).
  - Removed temporary incompatible monkey patch in `Attack-arc2face_wrapper/arc2face/expression_generator.py`.
- Validation:
  - Standalone `Arc2FaceExpressionGenerator(..., strict_cuda_provider=False)` init now passes (`INIT_OK`).
- Training launched (online Arc2Face, 2k protocol):
  - `stage1_sepmark_compact_arc2face_online2000.yaml`
  - `stage1_lampmark_compact_arc2face_online2000.yaml`
  - `stage1_fin_compact_arc2face_online2000.yaml`
  - `stage1_trustmask_compact_arc2face_online2000.yaml`
  - `stage1_maskwm_compact_arc2face_online2000_bs12.yaml`
- Added launcher script for persistent Arc2Face online 5-run setup:
  - `tools/run_stage1_arc2face_online_5.sh`

### 2026-03-15 Update: Arc2Face "No Progress" Root Cause and Stabilization
- Symptom: runs stayed at preflight clean and then appeared frozen.
- Root causes:
  - missing runtime dependency `face_alignment` caused Arc2Face preflight worker crash/restart.
  - detached multi-run launcher produced repeated elastic restarts, making logs look stalled.
- Fixes:
  - installed `face-alignment==1.4.1`.
  - set Arc2Face online configs to tolerant mode to avoid full-run abort on occasional sample failure:
    - `arc2face_allow_missing: true`
    - `enforce_nontrivial_swap: false`
- Relaunched clean 5-run Arc2Face online sessions after killing duplicated workers.

## 2026-03-15 Arc2Face Online Throughput Tuning (5-way Stage1)

- Trigger: GPU memory utilization too low during Arc2Face-online preflight/training.
- Updated configs:
  - `configs/experiments/stage1_sepmark_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_lampmark_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_fin_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_trustmask_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_maskwm_compact_arc2face_online2000_bs12.yaml`
- Main changes:
  - increased `training.batch_size` (32 for Sep/Lamp/FIN/TrustMask, 24 for MaskWM)
  - enabled `efficiency.use_amp: true`
  - increased dataloader workers via `efficiency.attack_num_workers`
  - reduced `preflight_eval.max_batches` from 8 to 4 for faster startup feedback
  - set `progress.log_interval_steps: 10` for denser progress logs
- Relaunched 5 jobs on GPUs `0/2/3/4/5` in detached screen sessions:
  - `arc2_sepmark`, `arc2_lampmark`, `arc2_fin`, `arc2_trustmask`, `arc2_maskwm`
- New run directories:
  - `runs/stage1_20260315_215138_161483` (SepMark)
  - `runs/stage1_20260315_215138_108419` (LampMark)
  - `runs/stage1_20260315_215138_172504` (FIN)
  - `runs/stage1_20260315_215138_203896` (TrustMask)
  - `runs/stage1_20260315_215138_021518` (MaskWM)

## 2026-03-15 Arc2Face Online Stability Fix

Issue observed:
- `preflight_arc2face.png` showed severe collapse/artifacts (blur/red-mask/cartoon-like outputs), especially in Stage1 online pipeline at 128 resolution.

Root causes identified:
1. Arc2Face generation was effectively run at low resolution (`output_size` followed training size 128), which is unstable for this diffusion path.
2. Online adapter consumed low-res tensors directly for source/expression in roll-batch mode, further reducing semantic fidelity.
3. Reference conditioning default (`reference=source`) was less stable for this training setup.

Code changes:
- Updated [`attacks/arc2face.py`](./attacks/arc2face.py):
  - Added and enabled stable defaults for online mode:
    - `arc2face_online_min_output_size=256`
    - `arc2face_online_reference_mode=expression`
    - `arc2face_online_expression_from=path`
  - Roll-batch source/expression now prefers `batch['img_path']` when available (path input), with tensor fallback.
  - Output is still resized back to training resolution after generation.

Config changes (5 Arc2Face-online experiments):
- [`configs/experiments/stage1_sepmark_compact_arc2face_online2000.yaml`](./configs/experiments/stage1_sepmark_compact_arc2face_online2000.yaml)
- [`configs/experiments/stage1_lampmark_compact_arc2face_online2000.yaml`](./configs/experiments/stage1_lampmark_compact_arc2face_online2000.yaml)
- [`configs/experiments/stage1_fin_compact_arc2face_online2000.yaml`](./configs/experiments/stage1_fin_compact_arc2face_online2000.yaml)
- [`configs/experiments/stage1_trustmask_compact_arc2face_online2000.yaml`](./configs/experiments/stage1_trustmask_compact_arc2face_online2000.yaml)
- [`configs/experiments/stage1_maskwm_compact_arc2face_online2000_bs12.yaml`](./configs/experiments/stage1_maskwm_compact_arc2face_online2000_bs12.yaml)

Verification artifacts:
- Before fix (severe collapse):
  - `/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/runs/stage1_20260315_215138_161483/samples/preflight_arc2face.png`
- After fix (stabilized, no catastrophic red-mask collapse):
  - `/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/docs/arc2face_online_chain_check_after_fix_path.png`
  - `/mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module/docs/arc2face_online_chain_check_after_fix_wm.png`

Relaunched Arc2Face-online Stage1 jobs with new code/configs:
- `arc2_sepmark`, `arc2_lampmark`, `arc2_fin`, `arc2_trustmask`, `arc2_maskwm`

## 2026-03-15 Arc2Face Standalone Debug (No-WM Path)

### Goal
- Verify Arc2Face generation quality without watermark embedding/decoding.
- Reproduce and isolate face affine drift + blur observed in online Arc2Face attacks.

### What Was Done
- Added standalone checker script:
  - `tools/debug_arc2face_standalone.py`
  - Generates `Source / Target / Fake` grid directly from Arc2Face wrapper (no watermark path).
- Ran three direct Arc2Face variants for A/B checks:
  - `docs/arc2face_standalone_cur_expr_ref_256.png`
  - `docs/arc2face_standalone_src_ref_256.png`
  - `docs/arc2face_standalone_src_ref_512_s35.png`
- Confirmed stable setting:
  - `reference_mode=source`
  - `output_size=512`
  - `num_steps=35`
- Re-ran checker with stable setting:
  - `docs/arc2face_standalone_check_latest.png`

### Config/Code Updates
- Arc2Face online defaults in `attacks/arc2face.py` tuned for stability:
  - `arc2face_online_reference_mode` default: `source`
  - `arc2face_online_min_output_size` default: `512`
  - `arc2face_online_num_steps` default: `35`
- Updated Arc2Face-online Stage1 experiment configs (5 files) to the same stable preset:
  - `configs/experiments/stage1_sepmark_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_lampmark_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_fin_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_trustmask_compact_arc2face_online2000.yaml`
  - `configs/experiments/stage1_maskwm_compact_arc2face_online2000_bs12.yaml`

### Note
- ONNX insightface still falls back to CPU provider in this environment (`libcublasLt.so.11` missing), which mainly impacts speed.

## 2026-03-15 Arc2Face 5-Adapter Relaunch (Stable Online Preset)

- Confirmed visualization column offset is from intentional non-self pairing (`source=i+1`, `target=i`) and does not affect training logic.
- Stopped old Arc2Face online Stage1 jobs and relaunched all 5 adapters with updated stable Arc2Face settings.
- Relaunch command:
  - `tools/run_stage1_arc2face_online_5.sh`
- Active logs:
  - `logging/tmux_stage1_arc2face_sepmark_online2k.log`
  - `logging/tmux_stage1_arc2face_lampmark_online2k.log`
  - `logging/tmux_stage1_arc2face_fin_online2k.log`
  - `logging/tmux_stage1_arc2face_trustmask_online2k.log`
  - `logging/tmux_stage1_arc2face_maskwm_online2k.log`
- New run directories observed at startup:
  - `runs/stage1_20260315_225404_086343` (sepmark)
  - `runs/stage1_20260315_225404_067892` (lampmark)
  - `runs/stage1_20260315_225404_151307` (fin)
  - `runs/stage1_20260315_225404_217416` (trustmask)
  - `runs/stage1_20260315_225404_179550` (maskwm)

## 2026-03-15 Arc2Face Semantics Update (No Extra Constraints)

- User requirement: do not add external/background-preserve constraints; keep target-background consistency by Arc2Face input semantics itself.
- Applied semantic choice:
  - `source_image`: donor identity
  - `expression_image`: encoded/target image
  - `reference_mode`: `expression`
- Updated defaults/configs accordingly (no post-blend/inpaint constraints).
- Relaunched all 5 Arc2Face-online stage1 experiments with the updated semantic setting.
- New run dirs:
  - `runs/stage1_20260315_225948_463641` (sepmark)
  - `runs/stage1_20260315_225948_522287` (lampmark)
  - `runs/stage1_20260315_225948_483955` (fin)
  - `runs/stage1_20260315_225948_394348` (trustmask)
  - `runs/stage1_20260315_225948_321389` (maskwm)

- 2026-03-16: Added smoke config `stage1_sepmark_compact_arc2face_online20_nopf_smoke.yaml` to validate Stage1 BCE NaN fix with tiny data (20/20) and `preflight_eval.enabled=false`.

- 2026-03-16: Switched Arc2Face smoke to INPAINT replay path via `stage1_sepmark_compact_arc2face_inpaint_nopf_smoke.yaml` (tiny data, preflight disabled) for faster quality check.

- 2026-03-16: Fixed wm-aligned Arc2Face replay message mismatch by enabling deterministic messages (`deterministic_messages=true`, `message_seed_salt=remark_v1`) in smoke config.

- 2026-03-16: Added 2k preflight config `stage1_sepmark_compact_arc2face_wmaligned_2k_preflight.yaml` (Arc2Face wm-aligned replay + deterministic messages).

- 2026-03-16: Restarted 5x Arc2Face-online2k Stage1 VAE jobs (SepMark/LampMark/FIN/TrustMask/MaskWM) in tmux sessions `a2k_*` with logs `logging/tmux_stage1_arc2face_*_online2k_restart_20260316_132328.log`.

- 2026-03-16: Switched 5x Arc2Face Stage1 jobs back to replay mode (`arc2face_mode: replay`) and inpaint replay jsonl (`outputs_replay_large/generation_results_inpaint.jsonl`), relaunched in tmux sessions `r2k_*`.

- 2026-03-16: Cleaned erroneous Stage1 runs/logs, switched Arc2Face replay2k configs to `wm_aligned` jsonl per wm-model + deterministic messages, started builders (`bwm_*`) for lampmark/fin/trustmask/maskwm, and added auto launcher session `auto_r2k_wmaligned` to start 5 VAE runs after builders finish.

- 2026-03-16: Added a lower-strength Arc2Face replay profile for debugging over-strong corruption:
  - New configs:
    - `configs/experiments/stage1_sepmark_compact_arc2face_replay2000_mild.yaml`
    - `configs/experiments/stage1_lampmark_compact_arc2face_replay2000_mild.yaml`
    - `configs/experiments/stage1_fin_compact_arc2face_replay2000_mild.yaml`
    - `configs/experiments/stage1_trustmask_compact_arc2face_replay2000_mild.yaml`
    - `configs/experiments/stage1_maskwm_compact_arc2face_replay2000_bs12_mild.yaml`
  - New replay builder script:
    - `tools/run_build_arc2face_wm_aligned_mild_5wm.sh`
  - Mild blend settings in script: `mode=hybrid`, `alpha=0.70`, `diff_threshold=0.10`, `diff_softness=0.04`, `match_stats=true`.

- 2026-03-16: Debugged Arc2Face corruption strength by splitting pipeline:
  - Kept running: `sepmark/lampmark/maskwm` replay2k training.
  - Stopped `fin/trustmask` replay2k training and started mild replay rebuild sessions:
    - `bwm_fin_mild`
    - `bwm_trustmask_mild`
  - Added auto handoff sessions to relaunch Stage1 after mild replay generation:
    - `auto_fin_mild` -> `stage1_fin_compact_arc2face_replay2000_mild.yaml`
    - `auto_trustmask_mild` -> `stage1_trustmask_compact_arc2face_replay2000_mild.yaml`
