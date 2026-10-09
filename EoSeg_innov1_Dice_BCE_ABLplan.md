# EoSeg 创新点1实施计划：Weighted Dice + Weighted BCE + Active Boundary Loss

## 0. 本次修订说明

1. **Boundary Loss 类型错误**：当前项目 `.codex` 规则已经明确第一创新点 C1 为 **Active Boundary Loss（ABL）**，不是自定义 Boundary Dice。因此第一创新点应严格定义为：

```text
Weighted Dice(0.6) + Weighted BCE(0.4) + λ_ABL × Active Boundary Loss
```

本计划后续全部以当前服务器、当前 ViT-L baseline 和 ABL 为准。

---

# 1. 当前正式实验环境与冻结 baseline

## 1.1 当前服务器环境

| 项目 | 当前设置 |
|---|---|
| 训练位置 | 远程服务器 |
| GPU | NVIDIA RTX 5090D 32GB |
| 主创新工作区 | `Z:\EoSe-main-innovation_point` |
| official baseline | `Z:\EoSeg-main-official`，冻结，只读 |
| 数据集 | GlaS / Warwick_QU_Dataset |
| 输入尺寸 | 224 × 224 |

## 1.2 当前 Backbone 与预训练权重

当前 backbone 应统一写为：

```text
Backbone: DINOv2 ViT-L/14 reg4
代码标识: vit_large_patch14_reg4_dinov2
Pretrained weight: dinov2_vitl14_reg4_pretrain.pth
```

需要特别区分：

- `vit_large_patch14_reg4_dinov2` 是当前模型使用的 **backbone/模型架构标识**；
- `dinov2_vitl14_reg4_pretrain.pth` 是对应的 **预训练权重文件**；
- 两者不是同一个概念，论文和实验记录中应分别记录。

## 1.3 当前冻结训练项

第一创新点 C1 只能改变 Loss 中的 ABL 项，其余 baseline 条件继续冻结：

```text
Backbone: DINOv2 ViT-L/14 reg4
Pretrained: dinov2_vitl14_reg4_pretrain.pth
Input: 224x224
Loss baseline: Weighted Dice(0.6) + Weighted BCE(0.4)
Mask path: mask-only query mask average
Decoder norm: GroupNorm
lr: 5e-5
backbone_lr: 1e-5
decoder_lr: 1e-4
LLRD: 0.8
warmup_steps: [500, 1000]
gradient_clip_val: 1.0
physical batch: 2
accumulate_grad_batches: 7
effective batch: 14
seed: 0
threshold: 0.5
checkpoint monitor: metrics/val_dice
main evaluation: single-view, no TTA
```

### RTX 5090D 需要注意的一点

RTX 5090D 32GB 显存比本地 RTX 4060 充足很多，但**第一创新点实验不能因为显存更大就直接提高 batch size**。

原因不是显卡不够，而是当前 baseline 已经冻结：

```text
physical batch = 2
accumulate_grad_batches = 7
effective batch = 14
```

如果 C1 同时把 batch size 改大，就无法判断性能变化到底来自 ABL 还是训练批量变化。因此 C1 主实验继续保持原 batch 配置。如果以后需要单独研究更大 batch，应建立新的实验组，不能混入 C1 消融。

---

# 2. 当前代码链路：Loss 实际在哪里生效

| 环节 | 文件 | 当前行为 | 与创新点1的关系 |
|---|---|---|---|
| 训练配置 | `configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml` | 指向 `MedicalBinarySegmentation`，网络为 `EoSegMultiQSegFusion` | 增加 ABL 参数应从这里传入 |
| 模型输出 | `models/eoseg.py` / `EoSegMultiQSegFusion` | 最后三个 ViT block 产生并融合 query mask/class 输出 | C1 不修改这里 |
| 前景 logits/probs | `training/medical_binary_segmentation.py` / `_foreground_logits_and_probs` | mask-only 路径对 query mask logits 求平均，输出 1 通道 foreground logits，再 sigmoid 得到 probs | ABL 应基于这里的最终预测构造 |
| GT | `_targets_to_binary_masks` | 合并为 `[B,1,H,W]` 二值 mask | 作为 ABL 的 GT 来源 |
| 基础损失 | `WeightedDiceBCE` | `0.6 × WeightedDice + 0.4 × WeightedBCE` | 必须保留不变 |
| 训练入口 | `_shared_step` | 计算预测、GT、loss、metrics 和 log | 在这里组合基础损失与 ABL |

第一创新点的核心原则是：

> **不改 EoSegMultiQSegFusion，不改 ViT 三层融合，不改 decoder，只在最终训练目标中加入 ABL。**

这样 C1 的性能变化才能归因于边界监督，而不会和后续 C2 的 decoder 融合创新混在一起。

---

# 3. 当前基础 Loss：Weighted Dice + Weighted BCE

当前 baseline：

```text
L_base = 0.6 × L_Dice + 0.4 × L_BCE
```

## 3.1 Weighted Dice

Dice 主要约束预测区域与真实区域的整体重叠：

```text
区域层面：预测腺体区域和 GT 是否整体重合
```

## 3.2 Weighted BCE

BCE 主要进行逐像素二分类监督：

```text
像素层面：当前像素究竟应该是前景还是背景
```

## 3.3 第一创新点的正确形式

C1 不替换 Dice，也不替换 BCE，而是在 baseline 上增加 ABL：

```text
L_total
= L_base + λ_ABL × L_ABL
= 0.6 × L_Dice
+ 0.4 × L_BCE
+ λ_ABL × L_ABL
```

因此论文中的创新点应描述为：

> 在原有区域重叠监督和逐像素二分类监督基础上，引入 Active Boundary Loss，对预测轮廓与真实轮廓之间的方向性偏差施加显式约束。

---

# 4. 为什么这里应该使用 Active Boundary Loss，而不是 Boundary Dice

当前项目的 C1 已经被冻结为 **Active Boundary Loss（ABL）**，所以不要再自行换成 Boundary Dice。

ABL 和简单的 Boundary Dice 不同。

## 4.1 Boundary Dice 的思路

Boundary Dice 通常是：

1. 从预测 mask 和 GT 中提取边缘；
2. 计算两张边缘图的 Dice；
3. 直接提高轮廓重叠程度。

它实现简单，但这不是当前项目中已经定义的 C1。

## 4.2 Active Boundary Loss 的核心

ABL 的重点不是简单比较两条边界是否重叠，而是：

1. 根据当前网络输出检测 **预测边界**；
2. 从 GT 边界计算 **距离信息**；
3. 对预测边界位置寻找朝向真实边界的局部方向；
4. 将边界对齐转化为可训练的方向分类问题；
5. 让错误的预测边界在优化过程中逐渐向 GT 边界移动。

所以它更强调：

```text
预测边界应该往哪个方向移动，才能靠近真实边界
```

而不是只有：

```text
预测边界和真实边界重叠多少
```

## 4.3 对上一版文档中“无需 SciPy”的纠正

上一版写了：

```text
无需 SciPy 距离变换
```

这个说法不适用于当前确定的 ABL。

官方 ABL 实现使用：

```python
from scipy.ndimage import distance_transform_edt
```

来构造距离图。

当前项目的 `requirements.txt` 本身已经包含 `scipy`，因此服务器环境不需要因为 ABL 额外改变整个依赖体系；但我们仍然要承认：

- 距离变换主要发生在 CPU/NumPy/SciPy 一侧；
- RTX 5090D 并不能消除这一部分 CPU 计算；
- ABL 的距离图计算可能增加 dataloader/step 时间；
- 后续如果确认 ABL 有效，可以考虑缓存 GT boundary distance map 来优化速度。

所以推荐 ABL 的原因应该是：

> **它就是当前论文第一创新点冻结的边界监督方案，并且其目标与改善腺体轮廓对齐高度一致。**

而不是“因为 RTX 4060 比较弱，所以选一个简单 loss”。

---

# 5. ABL 与当前二值分割输出之间的适配

这是代码实现中最需要注意的一点。

## 5.1 当前模型输出是 1 通道 logits

当前 mask-only 路径最终得到：

```text
foreground_logits: [B, 1, H, W]
foreground_probs:  sigmoid(foreground_logits)
```

BCEWithLogitsLoss 对这种 1 通道二值输出没有问题。

但是官方 ABL 的预测边界检测通过不同类别 logits 的局部分布差异完成，因此它天然更适合：

```text
[B, C, H, W]
```

且 `C >= 2`。

如果直接把 `[B,1,H,W]` 输入依赖 softmax/KL 的 ABL，单通道 softmax 恒为 1，就无法形成有效的类别分布差异。

## 5.2 二值任务应转换为等价的 2 类 logits

对当前 foreground logit `z`：

```text
p_fg = sigmoid(z)
```

可以构造两类 logits：

```python
abl_logits = torch.cat(
    [torch.zeros_like(foreground_logits), foreground_logits],
    dim=1,
)
```

得到：

```text
abl_logits: [B, 2, H, W]
```

因为：

```text
softmax([0, z])_foreground = sigmoid(z)
```

所以这种转换不会改变当前模型所表达的前景概率，只是把同一个二分类预测表示成 ABL 需要的二通道形式。

GT 则从：

```text
[B,1,H,W]
```

转换为：

```python
abl_target = targets[:, 0].long()
```

即：

```text
[B,H,W]
0 = background
1 = foreground
```

---

# 6. 具体代码应该怎么修改

## 6.1 新增文件：`training/active_boundary_loss.py`

不要把整套 ABL 全部塞进 `medical_binary_segmentation.py`，建议单独建立：

```text
training/active_boundary_loss.py
```

职责：

```text
ActiveBoundaryLoss
├── GT boundary 提取
├── GT boundary distance map
├── predicted boundary 提取
├── 8 邻域方向 GT 构造
├── 方向预测 loss
└── distance weighting
```

实现时以官方 ABL 思路为基准，但需要做以下工程适配。

### 必须修改的设备写法

官方旧实现中存在类似：

```python
.cuda()
```

当前工程应全部改为 device-safe 写法，例如：

```python
device = logits.device
x = torch.zeros(..., device=device)
```

否则代码会过度绑定默认 CUDA 设备，不利于 Lightning/DDP/设备管理。

### NumPy 类型兼容

不要继续使用已经废弃的：

```python
np.bool
```

改成：

```python
bool
# 或 np.bool_
```

### ABL 返回空边界时

官方逻辑在找不到有效预测边界时可能返回 `None`。

当前训练代码中不要让整个 step 崩溃，应统一处理为：

```python
if abl_loss is None:
    abl_loss = foreground_logits.new_zeros(())
```

同时可以记录一次 debug 计数，但不要影响反向传播。

---

## 6.2 修改 `training/medical_binary_segmentation.py`

### A. 保留现有基础损失

不要删除或重写当前：

```text
WeightedDiceLoss
WeightedBCE
WeightedDiceBCE
```

因为这是 C0 baseline 的冻结实现。

### B. 新增 ABL 模块

在 `MedicalBinarySegmentation.__init__()` 中新增：

```python
self.active_boundary_loss = ActiveBoundaryLoss(...)
```

并增加可配置参数，例如：

```text
active_boundary_loss_weight
abl_max_n_ratio
abl_label_smoothing
abl_max_clip_dist
```

第一版尽量沿用官方 ABL 默认内部超参数；真正需要筛选的主要变量只保留：

```text
λ_ABL
```

这样消融更干净。

### C. 在 `_shared_step` 中组合三项 Loss

当前逻辑：

```text
foreground_logits
foreground_probs
targets
→ WeightedDiceBCE
```

修改后：

```python
base_loss = self.criterion(
    foreground_logits,
    foreground_probs,
    targets,
)

abl_logits = torch.cat(
    [torch.zeros_like(foreground_logits), foreground_logits],
    dim=1,
)
abl_target = targets[:, 0].long()

abl_loss = self.active_boundary_loss(abl_logits, abl_target)
if abl_loss is None:
    abl_loss = foreground_logits.new_zeros(())

loss = base_loss + self.active_boundary_loss_weight * abl_loss
```

### D. 独立记录 loss 日志

至少记录：

```text
train/loss
train/loss_base
train/loss_abl
val/loss
val/loss_base
val/loss_abl
```

如果当前 `WeightedDiceBCE` 方便拆分，还可以继续记录：

```text
loss_dice
loss_bce
loss_abl
```

这样后续才能判断：

- ABL 是否正常下降；
- ABL 是否数值尺度过大；
- ABL 是否压制 Dice/BCE；
- 总 Dice 下降是否是 λ_ABL 过大造成。

---

## 6.3 新建 C1 专用 YAML

不要覆盖 official baseline YAML。

建议在：

```text
Z:\EoSe-main-innovation_point\configs\glas\
```

新建独立配置，例如：

```text
vit_query_mul_scale_fusion_protocol_b_c1_abl.yaml
```

保持所有 baseline 配置不变，只增加 ABL 参数，例如：

```yaml
model:
  init_args:
    active_boundary_loss_weight: 0.10
    abl_max_n_ratio: 0.01
    abl_label_smoothing: 0.20
    abl_max_clip_dist: 20.0
```

注意：

```text
Dice = 0.6
BCE = 0.4
```

继续保持，不要为了让三个权重相加等于 1 而重新归一化。

---

## 6.4 新建 C1 启动脚本

建议复制：

```text
configs/glas/run_protocol_b.sh
```

为：

```text
configs/glas/run_protocol_b_c1_abl.sh
```

增加：

```text
ABL_LOSS_WEIGHT
```

并让 run name 明确包含：

```text
c1_abl
seed
fold
abl weight
```

例如：

```text
c1_abl_w0p10_seed0_fold0
```

不能和 official baseline 的 runs 混用。

---

# 7. 为什么 ABL 仍然接最终 foreground prediction，而不是 ViT 中间层

| 位置 | 是否推荐 | 原因 |
|---|---:|---|
| 最终 foreground logits + GT | **推荐** | 与最终分割边界直接对应，梯度可继续反传至 decoder 和 backbone |
| 每个 query mask | 暂不推荐 | query 没有固定实例/类别对应关系，逐 query 强制同一 GT 边界会产生额外假设 |
| ViT 最后三层特征 | 不用于 C1 | 会把 loss 创新与 C2 三层融合创新混合 |
| threshold 后二值 mask | 不用于训练 | 阈值离散化会破坏需要的连续优化信息 |

因此 C1 的数据流应是：

```text
ViT-L backbone
    ↓
EoSegMultiQSegFusion
    ↓
query mask logits
    ↓ mean(query)
foreground_logits [B,1,H,W]
    ├── sigmoid → Weighted Dice
    ├── logits  → Weighted BCE
    └── [0, logits] → 2-class logits → Active Boundary Loss
```

---

# 8. 实验执行顺序

## 阶段 0：环境与 baseline 核对

训练前确认：

```text
GPU = RTX 5090D 32GB
Backbone = vit_large_patch14_reg4_dinov2
Pretrained = dinov2_vitl14_reg4_pretrain.pth
Workspace = Z:\EoSe-main-innovation_point
Input = 224x224
Batch = 2
Accumulation = 7
Effective batch = 14
Seed = 0
```

同时确认没有把结果写入：

```text
Z:\EoSeg-main-official\runs
```

---

## 阶段 1：代码级 sanity check

### Test 1：ABL 二通道 logits 正确性

随机输入：

```text
foreground_logits: [2,1,224,224]
```

构造：

```text
abl_logits: [2,2,224,224]
```

确认：

```text
softmax(abl_logits)[:,1]
≈ sigmoid(foreground_logits[:,0])
```

### Test 2：forward/backward

确认：

```text
base_loss finite
abl_loss finite
total_loss finite
loss.backward() success
```

### Test 3：ABL weight = 0 回归

设置：

```text
active_boundary_loss_weight = 0.0
```

必须退化为：

```text
0.6 × Weighted Dice + 0.4 × Weighted BCE
```

并与 C0 baseline 在相同输入下保持数值一致（浮点误差范围内）。

### Test 4：AMP/设备检查

确认 5090D 上 Lightning mixed precision 路径中：

```text
ABL 不出现 NaN/Inf
CPU distance transform 与 GPU tensor 转换正常
无硬编码 .cuda() 设备冲突
```

---

## 阶段 2：预先冻结 λ_ABL；Fold0 仅做数值诊断

第一轮只允许把 `λ_ABL` 作为事前声明的唯一 Loss 变量；正式论文主实验开始前必须冻结一个值。当前 Protocol B 的 Fold0 `Test_Folder` 会参与 checkpoint 选择与最终评估，不能再用该 Test_Folder 的 Dice/IoU/BF1 从下表挑选 λ，并声称五折是未调参的确认性结果。若确需筛选 λ，另在 `Train_Folder` 内部做调参诊断，锁定 λ 后再按冻结 Protocol B 重跑正式五折；诊断结果不得混入主表。

候选范围仅供另立调参诊断参考：

```text
λ_ABL
```

建议：

| 实验 | Dice | BCE | λ_ABL | 用途 |
|---|---:|---:|---:|---|
| C0 | 0.6 | 0.4 | 0.00 | baseline 回归 |
| C1-A | 0.6 | 0.4 | 0.05 | 弱 ABL |
| C1-B | 0.6 | 0.4 | 0.10 | 初始候选 |
| C1-C | 0.6 | 0.4 | 0.20 | 检查过强边界监督 |

这一阶段：

- fold 固定；
- seed=0；
- batch 固定；
- optimizer 固定；
- LR 固定；
- scheduler 固定；
- checkpoint selection 固定；
- 不修改 C2 decoder 结构。

Fold0 的 loss 曲线和有限值检查可诊断实现是否正常，但不能以 Fold0 `Test_Folder` 指标直接选择有利 λ。若训练数值异常需改变 λ，应记录原因、建立新的诊断 run，并重新冻结正式实验方案。

---

## 阶段 3：正式 5-fold

在正式评价前按上述非 Test_Folder 规则冻结 λ_ABL 后，再跑完整 5 folds。

每 fold：

```text
Train_Folder → train
Test_Folder → per-epoch val_dice / checkpoint selection
同一 Test_Folder → final single-view test
```

结果必须标记：

```text
protocol = author_protocol_b_test_as_val
selection_bias = true
threshold = 0.5
TTA = false（主实验）
```

主结果汇总：

```text
Dice mean ± sample std
IoU mean ± sample std
```

---

# 9. 边界指标

既然第一创新点声称改善边界，仅报告 Dice/IoU 证据不够直接。

当前正式边界指标已冻结为 Boundary-F1（BF1）：224×224、阈值 0.5、8 邻域内边界、欧氏距离 2 px 容差、逐图后折内宏平均；完整定义以 `Z:\.codex\METRICS_PROTOCOL.md` 为准。HD95 可另作补充分析，不得取代既定 BF1 或事后改变其定义。

但第一版不要用边界指标重新选择 checkpoint，checkpoint 仍按照冻结规则：

```text
metrics/val_dice
```

这样不会因为加入新 metric 改变 baseline 选择口径。

---

# 10. ABL 实现时需要重点避免的坑

## 10.1 不能直接把 1-channel logits 输入基于 softmax/KL 的 ABL

错误：

```text
[B,1,H,W] → softmax over channel
```

结果只有一个类别，softmax 恒为 1，预测边界 KL 信息失效。

正确：

```text
[B,1,H,W]
→ cat([0, foreground_logit])
→ [B,2,H,W]
```

## 10.2 不要修改 Dice/BCE baseline 权重

C1 只能增加 ABL：

```text
0.6 Dice + 0.4 BCE + λ_ABL ABL
```

不要修改为：

```text
0.5 Dice + 0.3 BCE + 0.2 ABL
```

否则 baseline 也被改变。

## 10.3 不要因为 5090D 显存大就同步增大 batch

显卡升级只改变可用算力，不改变本轮实验的公平性规则。

## 10.4 注意 ABL 的 CPU 距离变换

如果训练明显变慢，优先分析：

```text
SciPy distance_transform_edt
CPU → GPU tensor copy
```

而不是误以为 5090D 没吃满就是模型实现有问题。

确认 ABL 有收益后，可以再考虑：

```text
在 Dataset/DataLoader 阶段预计算或缓存 GT boundary distance map
```

但缓存优化应该保证数学定义不变。

## 10.5 不要修改 C2 的最后三层融合

第一创新点 C1：

```text
只增加 ABL
```

第二创新点 C2：

```text
只修改最后三层 ViT Decoder 融合
```

最后只有在 C1、C2 都独立验证后，才能建立 C3：

```text
C1 + C2
```

---

# 11. 推荐的实际修改顺序

1. 保持 `Z:\EoSeg-main-official` 完全不动。
2. 只在 `Z:\EoSe-main-innovation_point` 开发 C1。
3. 新增 `training/active_boundary_loss.py`。
4. 将官方 ABL 思路改造成当前工程可用、device-safe 的实现。
5. 在 `medical_binary_segmentation.py` 中实例化 ABL。
6. 将当前 1-channel foreground logits 转成等价 2-class logits。
7. 保留 `WeightedDiceBCE` 完全不变。
8. 使用 `L_total = L_base + λ_ABL × L_ABL`。
9. 新建 C1 专用 YAML，不覆盖 baseline YAML。
10. 新建 C1 专用 run script / run suffix。
11. 先做 `λ_ABL=0` 回归测试。
12. 再做 ABL forward/backward sanity check。
13. 预先冻结单一 λ_ABL；如确需筛选，只在 Train_Folder 内部诊断，不用 Fold0 Test_Folder 选值。
14. 确定单一 λ 后再跑完整 5 folds，并披露 Protocol B 的 Test_Folder 仍用于 checkpoint 选择。
15. 记录 Dice、IoU、ABL loss 曲线，并增加边界敏感指标。
16. C1 完成后再继续 C2 三层融合，不同时修改。

---

# 12. 论文中建议的表述

第一创新点不要写成“提出 Dice Loss”，也不要写成“提出 Boundary Dice”。

推荐表述：

> **在原有 Weighted Dice 与 Weighted BCE 联合区域监督的基础上，引入 Active Boundary Loss，通过对预测边界与真实边界之间的局部方向关系进行显式建模，使错误轮廓在训练过程中逐步向真实边界对齐，从而强化模型对腺体轮廓、相邻腺体分界及局部边界细节的学习。**

对应数学形式：

```text
L_total
= 0.6 L_Dice
+ 0.4 L_BCE
+ λ_ABL L_ABL
```

---

# 13. 当前 C1 推荐配置摘要

```text
Experiment group: C1
Workspace: Z:\EoSe-main-innovation_point
GPU: RTX 5090D 32GB
Backbone: vit_large_patch14_reg4_dinov2
Pretrained: dinov2_vitl14_reg4_pretrain.pth
Input: 224x224
Mask path: mask-only query mask average
Decoder norm: GroupNorm
Dice weight: 0.6
BCE weight: 0.4
ABL: enabled
λ_ABL: first test 0.10
ABL max_N_ratio: 0.01
ABL label smoothing: 0.20
ABL max_clip_dist: 20.0
Physical batch: 2
Gradient accumulation: 7
Effective batch: 14
Seed: 0
Threshold: 0.5
Checkpoint monitor: metrics/val_dice
Main evaluation: single-view
```

其中 `λ_ABL=0.10` 只是第一轮 Fold0 的起始候选值，不应在没有实验结果前写成最终论文超参数。

---

# 14. 本轮最重要的三个结论

1. **当前实验环境不再是 RTX 4060，而是服务器 RTX 5090D 32GB。**
2. **当前 backbone 是 DINOv2 ViT-L/14 reg4，预训练权重是 `dinov2_vitl14_reg4_pretrain.pth`。**
3. **第一创新点不是 Boundary Dice，而是当前项目规则已经确定的 Active Boundary Loss：`WeightedDiceBCE + λ_ABL × ABL`。**

因此下一步代码修改应围绕 **“将 ABL 正确适配当前单通道二值 logits”** 展开，而不是实现上一版计划里的 Boundary Dice。

---

# 15. 独立入口与可复现消融约束（2026-10-01）

本节是后续实施门禁，**只规定将来应如何建立入口，不表示 C1 代码、训练或结果已经完成**。C1 的模型、Loss、配置、脚本、日志和 checkpoint 只在 `Z:\EoSe-main-innovation_point` 开发/保存；`Z:\EoSeg-main-official` 的 C0 稳定 baseline 及其 BF1 评价材料保持只读。每个创新点有独立的实验入口，不允许先打开某创新点、再手改同一份 YAML 把它当成另一组结果。

| 消融组 | 基础 Dice/BCE | ABL | Decoder 结构创新 | 入口与结果隔离 |
|---|---:|---:|---:|---|
| C0 | 开 | 关 | 关 | 已冻结的 official baseline；引用既有结果，不覆盖 |
| C1 | 开 | 开 | 关 | innovation_point 的 C1 专用 YAML + 启动脚本/命令 + 独立 run/checkpoint |
| C2 | 开 | 关 | 开 | innovation_point 的 C2 专用 YAML + 启动脚本/命令 + 独立 run/checkpoint |
| C3 | 开 | 开 | 开 | C1/C2 独立验证后新建组合入口与独立 run/checkpoint |

本表的 `C3` 仅指 I1+I2 组合消融，不代表计划中的第三创新点 I3。I3 尚未确定；确定后必须另设独立入口与实验编号，再设计含 I3 的组合组，不能覆盖 C0–C3。

推荐 C1 文件名仍为第 6.3、6.4 节所列的 `vit_query_mul_scale_fusion_protocol_b_c1_abl.yaml` 与 `run_protocol_b_c1_abl.sh`；C2、C3 必须有不同的文件名和 run namespace。通用 Python 训练代码可以共享，前提是 ABL、Decoder 新模块均能由配置显式关闭；默认/关闭态不能改变 C0 数值路径。C1 必须显式关 C2，C2 必须显式关 ABL，C3 才允许二者同时开启。每组从同一冻结 DINOv2 ViT-L 预训练初始化规则和同一训练 seed 出发，不默认加载另一消融组训练后的 checkpoint。若将来做 warm-start，应另立实验，不能放入 C0/C1/C2/C3 主消融。

每个启动入口至少核验并记录：实际展开配置、实验组与开关、ABL 的 λ、ViT-L 预训练权重路径/哈希、fold manifest、seed、基础与分模块学习率、训练预算、checkpoint 选择规则、single-view/阈值、代码 commit、输出目录。用唯一 run 名称防覆盖；各组五折和多 seed 的路径彼此分离。正式训练前须验证 `λ_ABL=0` 时 C1 共用代码与 C0 的基础 Loss/前景概率在同输入下数值等价，并验证 C2 的 ABL 确实关闭、C1 的 Decoder 创新确实关闭；C3 的两开关均开启。这样 C1/C2/C3 都可单独从其入口重跑，才构成有效消融。

历史 92.56% Dice、86.66% IoU 来自旧 ViT-S/Train-Test/TTA 口径，不作为当前 ViT-L Protocol B 下 C1 的已完成结果。新增 C1 正式结果只能在五折完成后，按相同 BF1 定义与 C0 成对报告；不得从原历史指标直接推断增益。
