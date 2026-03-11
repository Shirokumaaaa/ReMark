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
│   ├── lampmark.py                 # LampMark 适配（待接入）
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
- [ ] 实现 Stage 2 U-Net 训练
- [ ] 实现 evaluate.py（含盲提取 ACC）
- [ ] 接入更多水印模型和攻击模型

---

## 十一、VAE 调参与训练记录（持续更新）

> 说明：本节用于记录每一次会影响结果的改动（配置/代码/环境），保证可追溯。

| 日期(Asia/Shanghai) | 变更文件 | 变更内容 | 目的 | 结果 |
|---|---|---|---|---|
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
