# Codex 执行任务：ABL 的 PDB 采样改为每张图独立计算（只修改这一项）

> 项目：EoSeg / GlaS / ViT-L/14 reg4 / C1 Active Boundary Loss  
> 工作目录：`/Disk/HDD/Sk/EoSeg-main/EoSe-main-innovation_point`  
> 当前 Git 分支：`fix/abl-per-image-sampling`（用户已创建）  
> 任务性质：在**服务器当前本地源码中直接编辑**，不使用 `.patch`，不覆盖 official baseline。  
> 本次唯一算法变量：ABL 预测边界（PDB）的 KL 阈值 ε 和 1% 像素上限从**整个 batch 共享**改为**每张图独立**。

## 0. 给 Codex 的硬性约束——先阅读

用户当前工作树有大量已修改（`M`）和未跟踪（`??`）文件，包括 `training/active_boundary_loss.py`、`scripts/diagnose_abl_sampling.py`、`tests/`、`runs/`。这不是需要清理的异常状态。

1. **以服务器本地实际文件为准**，先逐一读取文件再编辑；不要直接将 GitHub 文件整份覆盖到本地。
2. **不要执行** `git reset --hard`、`git clean`、`git restore`、删除 `runs/`、删除 checkpoints、覆盖既有实验记录，也不要执行 `git apply`。
3. 不自动 `git add`、`git commit`、`git push`，不另建分支；完成后报告修改的文件和验证结果即可。
4. 只允许变更以下文件（除非本地结构有差异，需要先说明并获得用户同意）：
   - **必改**：`training/active_boundary_loss.py`
   - **必改**：`tests/test_active_boundary_loss.py`（增加独立性测试，保留现有测试）
   - **必改**：`scripts/diagnose_abl_sampling.py`（该脚本目前复刻了旧的 batch 级算法，必须与新实现同步）
   - **仅核对兼容性，必要时才修改**：`scripts/diagnose_abl_boundary_overlay.py`（引用了 `audit_batch()` 的结果字段）
5. **本次不修改** `training/medical_binary_segmentation.py`、EoSeg 模型、数据集/五折划分、训练入口/YAML、Loss 权重、EarlyStopping、梯度累积、学习率、随机种子、指标及评价阈值。
6. **保留** ABL 的 `isdetach=True`，保留 `[0, z]` 双通道转换、方向损失、GT 距离变换、距离加权、`max_n_ratio=0.01`、`max_clip_dist=20`、`label_smoothing=0.2`。
7. 不启动实际训练，也不要覆盖此前的 `experiment_a`、`experiment_b`、`experiment_c` 输出。

## 1. 先理解问题和目标

目前 `ActiveBoundaryLoss._predicted_boundary()` 中有类似代码：

```python
threshold = 1e-5
max_points = height * width * self.max_n_ratio
active = kl_map > threshold
for _ in range(256):
    if int(active.sum().item()) <= max_points:
        break
    threshold *= 1.2
    active = kl_map > threshold
```

**问题不在 1% 的计算公式，而在 `active.sum()` 包含了 batch 维度。** 例如 `B=2`、`H=W=224`、`max_n_ratio=0.01`，两张图共用 `224×224×0.01 = 501.76` 个初始 PDB 点的总上限，导致每张图与同 batch 的另一张图互相竞争。

**期望行为**：每张图各有其独立的上限 `H×W×0.01`。对每张图，ε 从 `1e-5` 开始，候选点过多时反复 `ε *= 1.2`，直到该图候选数不超过上限（最多 256 次迭代）。

注意：

- 每张图 ε **独立自适应**，不是把两张图总上限简单设成 `B×H×W×0.01`，然后仍使用共享 ε。
- ε 是随当前预测 logits 动态计算的筛选阈值，**不是**新的可学习参数，也不应根据 epoch 手工调整。
- KL 筛选的离散性保持不变；保留后续 3×3 dilation。上限针对**膨胀前**选中点，膨胀后像素数可超过 1%。
- 对 224×224 图像，初始像素点数必须满足 `count <= 501.76`，由于 count 为整数，即最多 **501** 点（膨胀前）。
- 不能改变其他梯度路径，也不能修改 ABL loss 公式。

## 2. 必改文件一：`training/active_boundary_loss.py`

找到 `class ActiveBoundaryLoss` 中的 `_predicted_boundary()`。请先核对本地该方法的实际签名与返回类型。如果它仍与下方结构一致，可以将**该方法整体**改成下面的代码；不要替换类的其他部分。

```python
def _predicted_boundary(self, logits: torch.Tensor) -> torch.Tensor:
    """Find PDB pixels using an independent adaptive KL cap for each image.

    Input:  logits [B, C, H, W]
    Output: boundary mask [B, H, W] of dtype torch.bool
    """
    _, _, height, width = logits.shape
    if height < 2 or width < 2:
        return torch.zeros(
            (logits.shape[0], height, width),
            dtype=torch.bool,
            device=logits.device,
        )

    kl_vertical = self._kl_divergence(
        logits[:, :, 1:, :], logits[:, :, :-1, :]
    ).sum(dim=1, keepdim=True)
    kl_horizontal = self._kl_divergence(
        logits[:, :, :, 1:], logits[:, :, :, :-1]
    ).sum(dim=1, keepdim=True)
    kl_vertical = F.pad(kl_vertical, (0, 0, 0, 1))
    kl_horizontal = F.pad(kl_horizontal, (0, 1, 0, 0))
    kl_map = kl_vertical + kl_horizontal  # [B, 1, H, W]

    # Each image independently receives max_n_ratio * H * W PDB candidates.
    max_points_per_image = height * width * self.max_n_ratio
    per_image_masks = []

    for image_kl_map in kl_map:  # [1, H, W], one image at a time
        epsilon = 1e-5
        image_active = image_kl_map > epsilon

        for _ in range(256):
            if int(image_active.sum().item()) <= max_points_per_image:
                break
            epsilon *= 1.2
            image_active = image_kl_map > epsilon
        else:
            # Retain the old implementation's iteration-limit fallback,
            # but apply it only to the affected image.
            image_active = torch.zeros_like(image_active)

        per_image_masks.append(image_active)

    active = torch.stack(per_image_masks, dim=0)  # [B, 1, H, W]

    # Keep the original 3x3 dilation: it is performed AFTER the 1% cap.
    active = F.max_pool2d(
        active.float(), kernel_size=3, stride=1, padding=1
    ) > 0
    return active[:, 0]  # [B, H, W]
```

**不要**改变 `_direction_loss()` 中的：

```python
if self.isdetach:
    neighbor_logits = neighbor_logits.detach()
```

也**不要**改变 `binary_logits_to_two_class()` 或 `forward()` 的损失路径。

## 3. 必改文件二：`tests/test_active_boundary_loss.py`

在现有 `ActiveBoundaryLossTests(unittest.TestCase)` 中，**新增**以下两个测试方法；不要删除原有梯度、AMP、零损失、二通道等测试。

```python
def test_pdb_sampling_invariant_to_batch_companion_and_order(self):
    # For identical logits, image A's selection must not depend on B.
    gen = torch.Generator().manual_seed(20261011)
    a = torch.randn(1, 1, 64, 64, generator=gen) * 4.0
    b = torch.randn(1, 1, 64, 64, generator=gen) * 2.0
    abl = ActiveBoundaryLoss(max_n_ratio=0.01)

    def pdb(foreground_logits):
        return abl._predicted_boundary(
            binary_logits_to_two_class(foreground_logits)
        )

    single_a = pdb(a)
    single_b = pdb(b)
    pair_ab = pdb(torch.cat((a, b), dim=0))
    pair_ba = pdb(torch.cat((b, a), dim=0))

    self.assertTrue(torch.equal(single_a[0], pair_ab[0]))
    self.assertTrue(torch.equal(single_b[0], pair_ab[1]))
    self.assertTrue(torch.equal(single_a[0], pair_ba[1]))
    self.assertTrue(torch.equal(single_b[0], pair_ba[0]))


def test_constant_companion_does_not_change_other_image_pdb(self):
    gen = torch.Generator().manual_seed(20261012)
    active_image = torch.randn(1, 1, 64, 64, generator=gen) * 3.0
    blank = torch.zeros_like(active_image)
    abl = ActiveBoundaryLoss(max_n_ratio=0.01)

    single = abl._predicted_boundary(
        binary_logits_to_two_class(active_image)
    )
    paired = abl._predicted_boundary(
        binary_logits_to_two_class(torch.cat((blank, active_image), dim=0))
    )

    self.assertFalse(bool(paired[0].any()))
    self.assertTrue(torch.equal(single[0], paired[1]))
```

**额外建议：** 将 `_predicted_boundary()` 在膨胀前的中间采样数量，放在诊断脚本中断言 `<= 0.01*H*W`；不要用膨胀后数量断言 1%，否则会产生误报。

这些测试应能使旧的 batch 共享阈值实现暴露问题，并验证新实现具有 batch-size/order 不变性。

## 4. 必改文件三：`scripts/diagnose_abl_sampling.py`

**为什么必须修改**：实验 A 的 `audit_batch()` 自己又实现了一遍 batch 级 ε 更新，还用它检查与 `abl._predicted_boundary(logits)` 是否相等。如果只改训练核心代码，实验 A 诊断脚本会继续使用旧算法，出现假性不一致或错误日志。

### 4.1 替换 `audit_batch()` 中的 KL 采样片段

保留当前 `kl_map` 的计算、方向 GT/权重统计、CSV 写出和模型加载方式，只把原来类似以下的**整 batch 共享**片段：

```python
threshold = 1e-5
cap = height * width * abl.max_n_ratio
selected = kl_map > threshold
initial_counts = selected[:, 0].sum(dim=(1, 2)).tolist()
threshold_updates = 0
for _ in range(256):
    if int(selected.sum().item()) <= cap:
        break
    threshold *= 1.2
    threshold_updates += 1
    selected = kl_map > threshold
else:
    selected = torch.zeros_like(selected)
iteration_limit_hit = threshold_updates == 256
```

替换为：

```python
cap = height * width * abl.max_n_ratio  # PER IMAGE, not per batch
initial_counts = (kl_map > 1e-5)[:, 0].sum(dim=(1, 2)).tolist()

selected_images = []
thresholds_per_image = []
updates_per_image = []
limit_hits_per_image = []

for image_kl_map in kl_map:
    epsilon = 1e-5
    image_selected = image_kl_map > epsilon
    updates = 0

    for _ in range(256):
        if int(image_selected.sum().item()) <= cap:
            break
        epsilon *= 1.2
        updates += 1
        image_selected = image_kl_map > epsilon
    else:
        image_selected = torch.zeros_like(image_selected)

    selected_images.append(image_selected)
    thresholds_per_image.append(epsilon)
    updates_per_image.append(updates)
    limit_hits_per_image.append(updates == 256)

selected = torch.stack(selected_images, dim=0)  # [B,1,H,W]
```

之后仍然使用原来的：

```python
before = selected[:, 0]
after = (
    F.max_pool2d(selected.float(), kernel_size=3, stride=1, padding=1) > 0
)[:, 0]

if not torch.equal(after, abl._predicted_boundary(logits)):
    raise RuntimeError("Diagnostic selector differs from ABL._predicted_boundary")

# NEW: per-image hard assertion, BEFORE dilation.
for image_index, sample_id in enumerate(sample_ids):
    selected_count = int(before[image_index].sum().item())
    if selected_count > cap:
        raise AssertionError(
            f"Per-image PDB cap exceeded for {sample_id}: {selected_count} > {cap}"
        )
```

### 4.2 更新每图和每 batch 的统计字段

在原先生成的 `image_rows.append({...})` 字典中，保留原始统计字段，**再增加**：

```python
"cap_per_image": cap,
"threshold_updates": updates_per_image[image_index],
"final_kl_threshold": thresholds_per_image[image_index],
"iteration_limit_hit": limit_hits_per_image[image_index],
```

原先 `batch_row` 内 `cap_for_whole_batch`、单值 `threshold_updates`、单值 `final_kl_threshold` 已经不能代表新版算法。请删除旧的误导字段，改成：

```python
"cap_per_image": cap,
"nominal_cap_for_batch": cap * len(sample_ids),
"initial_kl_candidate_count": int(sum(initial_counts)),
"cap_exceeded_initially": any(count > cap for count in initial_counts),
"threshold_updates_total": sum(updates_per_image),
"threshold_updates_per_image": ";".join(map(str, updates_per_image)),
"final_kl_thresholds_per_image": ";".join(map(str, thresholds_per_image)),
"iteration_limit_hit": any(limit_hits_per_image),
```

这里是字典**字段替换示意**，不是完整 `batch_row`；保留其他字段，如 `batch_index`、`sample_ids`、`selected_before_dilation`、`selected_after_dilation`、`valid_direction_count`、`abl_returns_none`、`abl_loss_unweighted` 等。

在总 `summary.json` 中新增：

```python
"sampling_scope": "per_image",
"cap_per_image": cap,  # 若该变量仅存在于 audit_batch 局部作用域，请在 main 中从 abl.max_n_ratio 和 img_size 重算
```

原来 summary 的 `caveat` 里如果还有 “batch-wide cap”，必须改成 “per-image independent cap”；历史实验 A 文件**不要回写覆盖**。

### 4.3 检查实验 C 兼容性

` scripts/diagnose_abl_boundary_overlay.py ` 会导入 `audit_batch()`，还会在可视化 metadata 中存储 `abl_batch_sampling_row`。请确认它没有索引已经移除的旧字段；需要时仅更新其 metadata 说明、字段读取，不更改绘图颜色、样本、阈值和 ROI。

## 5. 测试与验收标准

先检查、再测试；**不进行训练**。推荐 Codex 在当前目录执行：

```bash
pwd
git status --short
python -m compileall -q training/active_boundary_loss.py tests/test_active_boundary_loss.py scripts/diagnose_abl_sampling.py
python -m unittest discover -s tests -p 'test_active_boundary_loss.py' -v
python -m unittest discover -s tests -p 'test_boundary_f1.py' -v
git diff --check
```

如本地环境缺失测试依赖，应如实报告并提供安装/运行方案，不得虚称测试通过。

**必须满足的行为：**

1. 同一张图的同一 logits，在 `batch=1`、`batch=2`、交换图像顺序后，`_predicted_boundary()` 的 bool mask 完全相同。
2. 对每张图独立满足 `selected_before_dilation <= H*W*max_n_ratio`，而不是只保证整个 batch 满足一张图的上限。
3. 旧版方向损失 `_direction_loss()`、`neighbor_logits.detach()`、GT 距离权重、`binary_logits_to_two_class()` 不变。
4. 空 PDB 的 `None` 行为不变；梯度正常反传；测试用二类概率仍等价于原 sigmoid。
5. 实验 A 的诊断复刻函数与新版 `_predicted_boundary()` 输出一致，不出现 `Diagnostic selector differs` 异常。
6. 既有实验 A/B/C 的输出未被修改；以后若重新运行诊断，请选择**新输出目录**，如 `experiment_a_per_image_fix/`，并标明这是改进版采样，而不是历史数据重算。
7. 原先的阈值 0.5、BF1 容差 2 px、五折清单、基本训练参数和 checkpoint 选择规则全部不变。

**测试解释：** 如果分别把 batch=1 和 batch=2 送入整套神经网络重新推理，极小的浮点误差可能使边界点出现在不同阈值一侧。用于验证采样机制本身是否与 batch 无关，应把**完全相同的预计算 logits** 交给采样函数，而不是要求两次混合精度网络推理逐位一致。

## 6. Codex 完成时必须输出的报告

请最终用中文说明：

- 本地实际读取并修改了哪些文件；
- `_predicted_boundary()` 的核心变化及独立 ε 机制；
- 实验 A 和实验 C 诊断脚本是否兼容；
- 新增的测试名、测试通过数量；
- 是否遇到本地未提交文件、与 GitHub 不同的实现，如何避免覆盖；
- 明确确认**没有**改 Loss、`detach()`、λ、EarlyStopping、训练 epoch、评价阈值、官方 baseline；
- 不执行训练，不提交/推送，不清理原有工作区。

---

## 7. 后续研究（**此次不实现**）

1. 验证 `detach()` 与 Dice+BCE 梯度之间的关系（当前 `isdetach=True`，不要再次修改）。
2. 设计末 20% epoch 才启用 ABL 的独立实验（需同时考虑早停机制和公平训练预算）。
3. 比较新采样与旧采样在固定 0.5 阈值下的 Dice、IoU、BF1，并保留独立命名的实验记录。

> **给 Codex 的最终要求：本次只修复“batch 共用 ε/1% 上限”的问题；不要顺手做任何其他创新或训练策略改动。**
