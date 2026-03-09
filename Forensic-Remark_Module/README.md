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
│   ├── simswap.py                  # SimSwap 封装（待接入）
│   └── registry.py                 # 名称 → 类的注册表
│
├── wm_adapters/                    # 水印编解码器适配层
│   ├── base.py                     # 抽象接口
│   ├── fin.py                      # FIN 适配
│   ├── sepmark.py                  # SepMark 适配（待接入）
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
```

---

## 七、已知问题与解决思路

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

## 八、与项目其他模块的关系

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

## 九、下一步开发计划

- [ ] 搭建代码骨架（目录结构、base 类、registry、config 解析）
- [ ] 实现 Stage 1 VAE 训练（迁移 train.py 现有逻辑，接入配置系统）
- [ ] 接入 FIN wm_adapter（最简单，优先验证端到端流程）
- [ ] 接入 StarGAN attack adapter（在线生成验证）
- [ ] 实现 Stage 2 U-Net 训练
- [ ] 实现 evaluate.py（含盲提取 ACC）
- [ ] 接入更多水印模型和攻击模型
