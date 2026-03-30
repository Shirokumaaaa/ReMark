# ReMark Latent Geometry Research Note

## 1. 当前研究判断

我们当前**不再先验假设**：

`watermark accuracy 相近的 latent 会在 AE / VAE 空间里自然聚类。`

原因是：

- 仅有重建损失（如 `L1`）和任务损失（如 `BCE`）时，latent 一般不会被显式约束为按某个下游语义指标自动排布。
- 标准 VAE 的 prior / KL 约束主要促进平滑性与可采样性，不等价于“按 watermark recoverability 聚类”。
- 即使是 `beta-VAE` 一类更偏向因子化表示的方法，也不能直接推出“会自然形成按水印恢复率分簇的几何结构”。

我们目前更相信的工作假设是：

`forge -> original 的 latent 位移存在共享结构。`

更具体地说：

- 未必是“高 accuracy 的点彼此更近”；
- 更可能是“从 forge latent 指向 original latent 的修复方向具有共性”；
- 因而沿着 pair-wise repair path 前进时，watermark accuracy 会表现出较稳定的单调上升。

这个假设来自当前实验现象，而不是理论定论。

---

## 2. 研究目标

在现阶段，我们不优先上更复杂的模型，而是先做 4 组验证，判断究竟是：

1. `latent clustering` 假设更合理；
2. 还是 `shared repair displacement / repair map` 假设更合理。

---

## 3. 四组验证设计

### 3.1 Accuracy 分桶聚类检验

把样本按 watermark accuracy 分桶，例如每 `0.1` 一个桶，检查 forge latent 是否按准确率自然聚类。

核心指标：

- 桶内平均欧氏距离
- 桶内平均余弦距离
- 桶间平均欧氏距离
- 桶间平均余弦距离
- silhouette
- Davies-Bouldin
- latent 上的 kNN 回归，看邻域 accuracy 是否一致

判读逻辑：

- 如果只是 pair 路径单调，但“同桶样本并不更近”，那就不支持全局聚类假设。

### 3.2 位移共享性检验

对每个样本对定义：

`delta_z_i = z_original_i - z_forge_i`

检查这些 repair displacement 是否共享结构。

核心指标：

- 不同 `delta_z_i` 之间的 cosine similarity 分布
- 对 `delta_z` 做 PCA，查看前几个主成分解释方差
- 用简单线性映射 `M z_f + b ~= z_o` 做回归，测 latent MSE
- 将线性映射预测出的 latent 解码后，测 watermark accuracy 是否相对 forge latent 提升

判读逻辑：

- 如果 `delta_z` 之间方向相近、低维 PCA 解释方差高、线性映射就能带来 accuracy uplift，那么更支持“repair map”而不是“latent clustering”。

### 3.3 路径单调性鲁棒性检验

不仅测当前的 `slerp`，还测：

- 线性插值
- 小随机扰动后的局部路径
- 只沿 `delta_z` 的 PCA 主方向的路径

对每条路径采样多个点，解码后测 watermark accuracy，并统计：

- monotonic rate
- fully monotonic fraction
- Spearman 相关
- path AUC
- 起点到终点 uplift

判读逻辑：

- 如果“只要从 forge 朝 original 方向走，大部分路径都提升”，说明 recoverability 更像一个连续低维因子，而不是离散簇。

### 3.4 latent 对 accuracy 的可预测性检验

训练一个很小的 probe：

`s(z) -> a_hat`

当前计划对 forge latent 试两种 probe：

- 线性层
- 两层 MLP

评估指标：

- `R^2`
- Pearson
- Spearman
- 分桶分类准确率

判读逻辑：

- 如果简单 probe 就能预测 accuracy，说明 latent 中已经存在“recoverability axis”；
- 即使它不形成聚类，也足够支持后续沿 latent 方向做修复。

---

## 4. 已实现内容

已经新增一个离线分析脚本：

`tools/analyze_latent_geometry.py`

它会直接复用当前 ReMark 的：

- `run_dir/config.yaml`
- Stage1 checkpoint
- watermark adapter
- attack adapter
- VAE encode / decode

并输出：

- `summary.json`
- `samples.csv`
- `bucket_stats.csv`
- `bucket_between.csv`
- `path_sample_summary.csv`
- `path_curves.csv`
- `probe_predictions.csv`

### 当前脚本默认分析的 latent 定义

- `z_forge`: `E(attacked_wm_image)`
- `z_original`: `E(wm_image)`

这里的 `wm_image` 指含水印原图，不是 clean cover image。

### 当前脚本默认的 accuracy 定义

分析用的主指标是**sample-level bit accuracy**，而不是仅看全数据集平均值。

脚本同时会保存：

- `raw_attacked_acc`
- `forge_recon_acc`
- `original_selfref_acc`
- `original_forgeref_acc`

其中：

- `forge_recon_acc` 是 forge latent 经 VAE decode 后的 sample-level ACC；
- `original_selfref_acc` 是 original latent 用 original reference decode 的 ACC；
- `original_forgeref_acc` 是 original latent 用 forge reference decode 的 ACC。

之所以区分后两者，是因为当前 Stage1 / Stage2 存在 `residual_output` 情况，decode 时 reference 选择会影响结果。

### 路径评估中的 decode 参考图

当前默认采用：

`decode_reference = forge`

这是为了与当前 Stage2 的 repair 语义保持一致，即：

- 在 residual VAE 下，整个 repair path 都以 forge / attacked image 作为 decode reference。

如果需要，也可以改成：

- `original`
- `blend`

---

## 5. 推荐的第一轮运行方式

运行前提：

- 需要使用项目训练环境，例如 `sepmark`；
- 当前机器上的系统默认 `python` 不带完整 PyTorch 训练依赖，直接运行可能失败。

先做小规模验证：

```bash
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module
/home/ldy/miniconda3/envs/sepmark/bin/python tools/analyze_latent_geometry.py \
  --run-dir runs/stage1_20260313_092741 \
  --checkpoint best \
  --split val \
  --attack auto \
  --max-batches 8
```

如果第一轮结果有信号，再跑更完整的验证：

```bash
cd /mnt/personal_workspace/chenkeyu/ReMark/Forensic-Remark_Module
/home/ldy/miniconda3/envs/sepmark/bin/python tools/analyze_latent_geometry.py \
  --run-dir runs/stage1_20260313_092741 \
  --checkpoint best \
  --split val \
  --attack auto \
  --max-batches 0 \
  --path-max-samples 256
```

---

## 6. 结果解释优先级

建议按下面顺序判断：

### 情况 A：聚类弱，但位移共享强

表现可能是：

- silhouette 低
- Davies-Bouldin 不好
- 同桶样本并不显著更近
- 但 `delta_z` cosine similarity 偏高
- `delta_z` PCA 低维解释率高
- 线性 repair map 或 mean-delta baseline 就能带来明显 uplift

这会支持：

`shared repair displacement / repair map` 假设

### 情况 B：路径单调性稳定，但全局聚类仍弱

表现可能是：

- `slerp / linear / local / pca` 多种路径都大体单调
- Spearman 为正
- monotonic rate 较高
- 但全局分桶聚类指标仍一般

这会支持：

`recoverability 是连续因子，不一定是离散簇`

### 情况 C：probe 可预测性强

表现可能是：

- 线性 probe 就有较高 Pearson / Spearman / R²
- 或者两层 MLP 明显优于线性 probe

这说明：

- latent 中已经编码了 watermark recoverability 信息；
- 后续可以优先考虑“沿 recoverability 方向修复”，而不必强求显式聚类。

### 情况 D：以上都不成立

如果出现：

- 聚类弱
- 位移共享弱
- 路径不稳定
- probe 也预测不动

那说明当前 Stage1 latent 几何并没有形成足够稳定的 repair structure；
这时才值得考虑引入显式几何约束或结构化训练目标。

---

## 7. 当前状态说明

目前是：

- 研究判断已整理完成；
- 四组验证方案已形式化；
- 对应分析脚本已实现；
- **但还没有在目标 run 上产出正式实证结果。**

所以当前可以对外表述为：

`我们已经完成了从“经验观察”到“可验证分析框架”的落地，但尚未完成本轮实证统计。`

---

## 8. 希望 ChatGPT 5.4 重点帮看什么

如果把这份文档发给 ChatGPT 5.4，建议重点让它评估：

1. 这 4 组分析是否足以区分 `latent clustering` 与 `repair map` 两种假设；
2. 当前指标设计里是否有统计偏差或定义不清的问题；
3. `original_selfref_acc` 与 `original_forgeref_acc` 的区分是否合理；
4. 路径鲁棒性检验中，`local_perturbed` 与 `pca_topk` 的定义是否还需要更严谨；
5. 如果验证结果支持 repair-map 假设，下一步模型设计应该优先学什么形式的 mapping。
