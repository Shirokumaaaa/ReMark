# Face-Adapter / REFace 与 ReMark 集成说明（供后续协作者使用）

本文档仅记录 **本次集成改动**，帮助后续协作者快速理解：

- 这两个攻击模型是如何「注册」进 ReMark Module 的；
- 训练/评测时如何在 ReMark 下调用它们；
- 需要在各自 Attack 仓库额外准备哪些离线产物（JSONL）。

---

## 1. 集成目标与整体思路

- **目标**：让 `Attack-Face-Adapter` 与 `Attack-REFace` 像 `Attack-arc2face_wrapper` 一样，作为「慢攻击 / 离线 replay 模型」接入 `Forensic-Remark_Module`，用于 Stage1/评测。
- **思路**：完全沿用现有 Arc2Face/DiffSwap 的做法：
  - 在 `Forensic-Remark_Module/attacks/` 下为两者各写一个 **Adapter 类**，继承 `BaseAttack`，通过 `@register_attack` 注册；
  - 不在 ReMark 内部直接 import 大模型，而是统一通过  
    「**离线伪造图 + generation_results.jsonl manifest**」→ replay；
  - `train_stage1.py` 和 `tools/eval_attacked_recon_acc.py` 不大改，只在原有攻击加载逻辑上补充对新名字的支持。

---

## 2. ReMark Module 侧的改动概览

### 2.1 新增/修改的文件

- **新增 Adapter 文件**
  - `attacks/face_adapter.py`  
    - 类名：`FaceAdapterAttack`  
    - 注册名：`'face_adapter'` / `'FaceAdapter'`  
    - 功能：读取 `Attack-Face-Adapter` 生成的 `generation_results.jsonl` + 伪造图，做离线 replay。
  - `attacks/reface.py`  
    - 类名：`REFaceAttack`  
    - 注册名：`'reface'` / `'REFace'`  
    - 功能：读取 `Attack-REFace` 生成的 `generation_results.jsonl` + 伪造图，做离线 replay。

- **更新注册入口**
  - `attacks/__init__.py`  
    - 新增：
      - `from . import face_adapter  # registers FaceAdapterAttack`
      - `from . import reface        # registers REFaceAttack`

- **更新训练脚本攻击加载逻辑**
  - `train_stage1.py`  
    - 原来只加载 `cfg.attacks.online`；  
    - 现在改为：
      ```python
      attack_names = list(getattr(cfg.attacks, 'online', [])) + list(getattr(cfg.attacks, 'offline', []))
      for name in attack_names:
          ...
          self.online_attacks[name] = build_attack(name, cfg)
      ```
    - 日志文案从 `Online attack loaded` 调整为 `Attack loaded`，语义统一。

- **README 结构补充（仅描述层面）**
  - `README.md` 中工程目录树里补充了：
    - `attacks/face_adapter.py`：Face-Adapter 离线 replay 封装（待接入）
    - `attacks/reface.py`：REFace 离线 replay 封装（待接入）
  - 示例配置中 `attacks.offline` 增加了：
    - `offline: [diffswap, arc2face, face_adapter, reface]`

> 注：Adapter 类接口与 `DiffSwapAttack` / `Arc2FaceAttack` 完全对齐：  
> - `preprocess / generate / postprocess` 走 `BaseAttack` 统一入口；  
> - 训练和评估优先调用 `attack_with_cover(wm_images, images, batch=batch)` 分支；  
> - 内部通过 `batch['img_path']` + JSONL replay 找到对应伪造图。

---

## 3. Attack 仓库侧的补充脚本（生成 JSONL）

为避免修改原有推理逻辑，只在各自攻击仓库下新增了一个 **小工具脚本**，负责将「输入→输出」的 CSV 映射转成统一的 `generation_results.jsonl`，字段与 `Attack-arc2face_wrapper` 完全一致。

### 3.1 Attack-Face-Adapter

- **新增脚本**
  - `Attack-Face-Adapter/tools/build_generation_results_from_csv.py`

- **输入 CSV 要求**
  - 至少两列：
    - `source_image`：原始/含水印图路径，需与 ReMark CSV 中的 `img_path` 一致（用于 replay key）；
    - `output_image`：Face-Adapter 推理得到的换脸结果路径（绝对/相对路径均可，只要与 JSONL 中写的一致）。
  - 可选列：
    - `index`：自定义样本 id（不填则使用行号）；
    - `expression_image` / `reference_image` / `ok`：若有会一并写入 JSONL，没有则用空字符串/默认值。

- **生成 JSONL 的典型命令**

  在 `Attack-Face-Adapter` 根目录：

  ```bash
  python tools/build_generation_results_from_csv.py \
    --mapping-csv data/celebahq_eval/face_adapter_pairs.csv \
    --output-dir outputs
  ```

  运行后会在 `outputs/` 下得到：

  ```text
  outputs/generation_results.jsonl
  ```

  每行记录示意：

  ```json
  {
    "index": 0,
    "source_image": "/abs/path/to/source.jpg",
    "expression_image": "",
    "reference_image": "",
    "ok": true,
    "outputs": ["/abs/path/to/faceadapter_swap.png"],
    "error": null
  }
  ```

### 3.2 Attack-REFace

- **新增脚本**
  - `Attack-REFace/tools/build_generation_results_from_csv.py`

- **输入 CSV 要求**
  - 与 Face-Adapter 版完全一致：
    - `source_image`：原始/含水印图路径（ReMark 的 `img_path`）；
    - `output_image`：REFace 推理结果路径。

- **生成 JSONL 的典型命令**

  在 `Attack-REFace` 根目录：

  ```bash
  python tools/build_generation_results_from_csv.py \
    --mapping-csv data/celebahq_eval/reface_pairs.csv \
    --output-dir outputs
  ```

  生成 `outputs/generation_results.jsonl`，字段与 Arc2Face wrapper 完全一致。

---

## 4. 在 ReMark 训练/评测中使用 Face-Adapter / REFace

### 4.1 配置文件示意

以 `configs/experiments/...yaml` 为例（伪代码）：

```yaml
attacks:
  online:  [stargan]                       # 需要的在线攻击（可为空）
  offline: [diffswap, arc2face, face_adapter, reface]

attack_options:
  # DiffSwap / Arc2Face 的配置略

  # Face-Adapter
  face_adapter_results_jsonl: D:/GitHub/ReMark/Attack-Face-Adapter/outputs/generation_results.jsonl
  face_adapter_outputs_base:  D:/GitHub/ReMark/Attack-Face-Adapter
  face_adapter_allow_missing: false        # 若 JSONL 缺 key，是否退回原图
  face_adapter_replay_key:    source_image

  # REFace
  reface_results_jsonl:       D:/GitHub/ReMark/Attack-REFace/outputs/generation_results.jsonl
  reface_outputs_base:        D:/GitHub/ReMark/Attack-REFace
  reface_allow_missing:       false
  reface_replay_key:          source_image
```

说明：

- 将 `face_adapter` / `reface` 填到 `attacks.offline` 后，`train_stage1.py` 会自动加载并放到内部的 `online_attacks` 字典中；
- 训练阶段：
  - clean 分支：重建 `wm_images`；
  - attack 分支：按概率调用攻击（包括离线攻击），重建 `attacked`；
- 评估脚本 `tools/eval_attacked_recon_acc.py` 会遍历 `cfg.attacks.online` 中的名字；若需要在 eval 中同时覆盖 Face-Adapter/REFace，可在配置里将它们也加入 `online` 列表，或后续统一扩展 eval 脚本的攻击来源。

---

## 5. 给后续协作者的使用建议

1. **新增攻击模型时**：
   - 优先参考 `arc2face.py` / `diffswap.py` / `face_adapter.py` / `reface.py` 的写法，保持「离线 replay + JSONL」接口一致；
   - 在 `attacks/__init__.py` 中补一行 import，让 `ATTACK_REGISTRY` 自动注册。

2. **复现实验时**：
   - 优先使用各 Attack 仓库自带的 `tools/prepare_celebahq_eval_inputs.py` 生成 `source.csv` / `target.csv` 与软链接目录；
   - 跑完推理后，通过本次新增的 `build_generation_results_from_csv.py` 生成 `generation_results.jsonl`；
   - 最后在 ReMark 的 YAML 里指向对应 JSONL 路径即可，无需改 ReMark 代码。

3. **调试问题时**：
   - 若攻击 replay 报「missing replay entry」「No valid replay entries loaded」之类错误，优先检查：
     - JSONL 中的 `source_image` 与 ReMark CSV 中的 `img_path` 是否一致（绝对/相对路径）；
     - `attack_options.*_results_jsonl` 路径是否正确；
     - JSONL 是否为空、或所有记录 `ok=false`。

