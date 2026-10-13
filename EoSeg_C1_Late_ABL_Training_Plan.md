# EoSeg C1-Late：后 20% Epoch 启用 ABL 的受控实验实施 Plan

> **供 Codex 在服务器本地直接编辑源码并执行测试使用。**
>
> 版本：2026-10-12；状态：**代码、独立配置、启动脚本和本机单元/799→800 切换检查已完成；服务器正式训练尚未启动**。
>
> 项目根目录：`/Disk/HDD/Sk/EoSeg-main/EoSe-main-innovation_point`；`EoSeg-main-official` 只读。
>
> 先阅读本地 `AGENTS.md`、`PROJECT_EXPERIMENT_PRIORITY.md`、`EoSeg_innov1_Dice_BCE_ABLplan.md`、`LOSS_PROTOCOL.md`、`PROTOCOL_B_SELECTION_OVERRIDE.md`，以**服务器实际文件**为准。GitHub 的 `main` 不一定包含服务器未提交的逐图 PDB 改动。

**2026-10-12 实施核验：**仅在创新目录修改训练模块、追加单测，并新增独立 YAML 与启动脚本；没有创建 C0-NoES、修改 official 或启动正式训练。本机 Windows Conda 环境运行全部 `22` 项单元测试通过，Python 源文件语法和新增 Bash 脚本 `bash -n` 通过。固定输入下模拟真实 `training_step`：epoch 799 的 `L_total=0.90445673`、`abl_active=0`；epoch 800 的 `L_total=0.98341262`、`abl_active=1`，其中 `L_base=0.90445673`、`L_ABL=0.78955877`、有效系数 `0.10`。这些是本机功能检查，**不是服务器训练或 GlaS 指标**；正式运行前还需在服务器检查预训练权重、可用磁盘和 `--check-only` 预检。

**预检修正：**首次服务器 `--check-only` 暴露了 PyYAML 将 `5e-5` 等原有写法解析为字符串、而独立 runner 用浮点数直接比较的误报。已仅在新 runner 中将三个学习率转为数值后比较；YAML、实际学习率与模型代码未改。新增真实预检回归测试，修正前复现 `C1-Late configuration check failed: learning_rates`，修正后通过；本机全部 `23` 项单测和 `bash -n` 通过。仍未启动正式训练。

## 0. 任务目标和边界

本次**只实施第三个诊断问题——ABL 的时间启用策略**：

- 训练预算固定为 **1000 个 epoch，禁用 EarlyStopping**，不是修改 `patience` 为一个更大数。
- 使用零起始 epoch 计数：`0–799`（前 800 个 epoch）仅用基础损失；`800–999`（后 200 个 epoch）启用 ABL。`current_epoch == 800` 的首个训练 batch 必须是 ABL 开启点。
- 损失：`L_base = 0.6 × WeightedDice + 0.4 × WeightedBCE`；后期 `L_total = L_base + 0.10 × ActiveBoundaryLoss`。**λ=0.10 仅是此次预先冻结的诊断值，不进行 λ 搜索**。若决定沿用 0.20，需在开跑前确定并在所有对照中一致记录，不能中途切换。
- 沿用已修复的 **每张图独立 1% PDB 采样**、自适应 KL 阈值 ε、`isdetach=True`、ABL 方向与距离加权实现；不再次调整 ABL 内部算法。
- **不改变** DINOv2 ViT-L/14 reg4、`EoSegMultiQSegFusion`、Query 数、GroupNorm、输入 224×224、固定五折 manifest、seed=0、physical batch=2、accumulate=7、优化器及 LR scheduler、分模块学习率、阈值=0.5、BF1 tolerance=2px、single-view，无 TTA、checkpoint 监控指标 `metrics/val_dice`。
- **不**同时做 Fusion Gate 梯度投影/PCGrad/Decoder 修改；它们属于未来独立创新候选，不能混入 C1-Late。
- **不修改** official baseline、旧 C0/C1 YAML 与启动脚本、既有 checkpoint、`runs/` 中旧目录、`experiment_a/b/c`。
- 本轮**不建立、不训练 C0-NoES**；现有 official C0 只作历史参考，不作为同预算因果对照。
- 工作区可能含大量未提交/未跟踪文件：**严禁** `git reset --hard`、`git clean -fd`、强制 checkout、批量删除和覆盖已存在实验目录；无需 `git apply`。

## 1. 为什么要单独建立新实验组

当前 C1 源码 `training/medical_binary_segmentation.py` 在 `active_boundary_loss_enabled=True` 且 `active_boundary_loss_weight>0` 时直接计算 ABL，没有 epoch 门控。当前 C1 YAML 配置中：

```yaml
trainer:
  max_epochs: 1000
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: metrics/val_dice
        mode: max
        save_top_k: 1
        save_last: true
        save_weights_only: true
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: metrics/val_dice
        mode: max
        patience: 300
model:
  init_args:
    active_boundary_loss_enabled: true
    active_boundary_loss_weight: 0.10
```

只增加 `abl_start_epoch=800` 不够：早停可能在 800 之前发生，也可能在 800 后很快结束，造成后期训练预算不足。旧 `configs/glas/run_protocol_b.sh` 还把 `early_stopping=metrics/val_dice,mode=max,patience=300` **硬编码**进元数据，不能原封不动用于新实验后再宣称早停关闭。

旧 best checkpoint 也不能直接当作 epoch=800 的断点：`save_weights_only: true` 不包含完整优化器、scheduler、训练循环状态；best epoch 也未必是 799。**本次主方案从 DINOv2 预训练权重独立启动、训练完整 1000 epoch。**

## 2. 新实验组定义与结论边界

| 组别 | `epoch 0–799` | `epoch 800–999` | 早停 | PDB | 目的 |
| --- | --- | --- | --- | --- | --- |
| **C1-Late-NoES** | Dice+BCE | Dice+BCE+ABL | 关闭 | 每图独立 1% | 验证后期加入是否有益 |
| **C1-Always-NoES**（第二阶段建议） | Dice+BCE+ABL | Dice+BCE+ABL | 关闭 | 每图独立 1% | 单独验证“延迟启用”相对“全程启用”的作用 |

本轮只实施和验证 **C1-Late-NoES / fold0**；`C1-Always-NoES` 仅保留为将来可选的独立实验。如果准备在论文中声称**后期启用优于全程 ABL**，必须先在同一预算和逐图 PDB 口径下实施 `C1-Always-NoES`。过去五折 C1 的 batch 级 PDB 和 EarlyStopping 口径不同，不能代替它。

**特别说明：** 本轮按用户要求不做 C0-NoES。已冻结的 official C0 使用 EarlyStopping，且 PDB 不适用，不能作为 C1-Late 的同预算对照；其指标只作历史背景。完成 C1-Late 后可以检验训练稳定性、阶段切换和固定终点表现，但**不能仅凭与历史 C0 的分数差声称后期 ABL 带来因果增益**。

## 3. 推荐的源码修改：`training/medical_binary_segmentation.py`

Codex 先检查本地文件。建议只增量加入两个可选参数，确保旧 C0/C1 默认运行逻辑不变：

### 3.1 在 `__init__` 增加配置参数

在 `active_boundary_loss_weight` 附近新增：

```python
abl_schedule: str = "always",   # 老实验默认：保持原来始终启用的语义
abl_start_epoch: int = 0,       # 新 C1-Late 显式设为 800
```

初始化：

```python
self.abl_schedule = str(abl_schedule).lower()
if self.abl_schedule not in {"always", "late"}:
    raise ValueError("abl_schedule must be 'always' or 'late'")
if int(abl_start_epoch) < 0:
    raise ValueError("abl_start_epoch must be non-negative")
self.abl_start_epoch = int(abl_start_epoch)
```

要校验新实验 YAML 同时满足 `max_epochs=1000` 和 `abl_start_epoch=800`，不要在 1-epoch smoke 时把 start 改为 0 来“测试门控”；单元测试负责模拟 epoch 799/800。

### 3.2 明确门控函数

建议新增一个纯判定函数，便于单元测试；不使用 `global_step` 或验证 batch 编号控制 ABL：

```python
def _should_apply_abl(self, phase: str, epoch: int) -> bool:
    if not self.active_boundary_loss_enabled or self.active_boundary_loss_weight <= 0:
        return False
    if self.abl_schedule == "always":
        return phase in {"train", "val", "test"}  # 旧 C1 兼容
    if self.abl_schedule == "late":
        # 测试指标无需计算 ABL，测试损失仅报告 base；训练/验证
        # 在相同的后期 epoch 门控；测试行为须写明在 metadata。
        return phase in {"train", "val"} and epoch >= self.abl_start_epoch
    raise RuntimeError(f"Unexpected ABL schedule: {self.abl_schedule}")
```

在 `_shared_step` 中把原先 `apply_abl=(log_prefix in {"train", "val", "test"})` 换成：

```python
abl_active = self._should_apply_abl(log_prefix, int(self.current_epoch))
loss_base, loss_abl, loss = self._compute_loss_components(
    foreground_logits, foreground_probs, binary_targets,
    apply_abl=abl_active,
)
```

`_compute_loss_components` 内原有逻辑可保留：`apply_abl=False` 时 `loss_abl=0`，`L_total==L_base`；`apply_abl=True` 且有有效 PDB 时，`L_total=L_base+λ×L_ABL`。**不要**改动 BCE、Dice、ABL 的内部公式或使用 `detach()` 包住整个 ABL。

> 注意：`_shared_step` 是 Lightning 训练方法，`self.current_epoch` 由 Trainer 提供。若本地的直接单元测试在没有 Trainer 的情况下调用 `_shared_step`，改为单测 `_should_apply_abl()` 和 `_compute_loss_components()`，或者由 Codex 创建受控的最小测试 Trainer；不要靠人为伪造最终训练指标。

### 3.3 日志与语义

除现有 `train/loss_base`、`train/loss_abl` 外，建议新增并在每个训练 epoch 汇总：

```python
if log_prefix == "train":
    self.log(
        "train/abl_active",
        float(abl_active),
        on_step=False, on_epoch=True,
        prog_bar=False, sync_dist=True,
        batch_size=imgs.shape[0],
    )
    self.log(
        "train/abl_effective_weight",
        self.active_boundary_loss_weight if abl_active else 0.0,
        on_step=False, on_epoch=True,
        prog_bar=False, sync_dist=True,
        batch_size=imgs.shape[0],
    )
```

请 Codex 核对 Lightning 当前环境对 `self.log(float, ...)` 的接受情况；必要时使用设备上的 `torch.tensor(float(...), device=self.device)`。**不要只看 loss_abl=0 来判断阶段**：ABL 也可能因空 PDB 返回 None，必须记录明确的门控状态。验证/测试时，`metrics/val_dice`、`metrics/val_iou`、`metrics/val_boundary_f1` 等定义保持不变；Late 模式的 test loss 是基础损失，避免测试时 `current_epoch` 无意义造成隐式启用。

## 4. 独立 YAML，不覆盖旧实验

只创建一个新 YAML（复制当前本地已验证的 ViT-L C1 配置，除了下述字段以外保持不变）：

- `configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_late_abl_noes_1000.yaml`

重要配置示意；其余 `network`、`data`、`optimizer`、`lr`、`warmup`、阈值和回调 `MedicalValLogger` 等照原 YAML 完整保留：

**C1-Late-NoES：**

```yaml
trainer:
  max_epochs: 1000
  # 同样不包括 EarlyStopping 回调
model:
  init_args:
    active_boundary_loss_enabled: true
    active_boundary_loss_weight: 0.10
    abl_schedule: late
    abl_start_epoch: 800
    abl_max_n_ratio: 0.01
    abl_label_smoothing: 0.2
    abl_max_clip_dist: 20.0
```

该 YAML 的 `ModelCheckpoint` 继续使用 `monitor=metrics/val_dice`、`mode=max`、`save_top_k=1`、`save_last=true`；checkpoint 选择标准不改。设置 `save_weights_only=false`，为中断后恢复 optimizer、scheduler、epoch/global_step 保留完整训练状态；checkpoint 磁盘占用会增加，启动正式训练前须在服务器检查剩余空间。本轮没有 C0-NoES 配对训练，也不能把 checkpoint 格式差异解释为精度变化。

> `max_epochs: 1000` 与关闭早停保证完整训练预算；它们不保证最终选择的 best-Dice checkpoint 在 800 后。额外保留 `last.ckpt`（epoch 999）作为事先规定的**辅助固定终点分析**，不能把它和原 best checkpoint 混成同一指标口径。主结果仍使用 `metrics/val_dice` 选出的最佳 checkpoint；必须记录 `best_epoch` 是否 `>=800`。

## 5. 独立启动脚本与运行记录

只新增以下文件，**不要直接修改已锁定 SHA256 的 `configs/glas/run_protocol_b_c1_abl.sh`**：

- `configs/glas/run_protocol_b_c1_late_abl_noes_1000.sh`

实现方式：Codex 可复用/复制当前 `configs/glas/run_protocol_b.sh` 的固定五折路径、预训练权重检查、指标提取和源文件 SHA256 记录，但必须逐项校验以下差异：

1. 新入口使用独立配置和 `runs/glas_protocol_b_fold{fold}_vitl_c1_late800_noes1000_seed0` 目录；如果该目录已存在则拒绝启动，禁止复用旧 C0/C1 run。
2. 不加入 `lightning.pytorch.callbacks.EarlyStopping`；其他 ModelCheckpoint 与 MedicalValLogger 必须保留。
3. 新运行元数据写明 `early_stopping=disabled`、`max_epochs=1000`、`abl_schedule`、`abl_start_epoch`、`abl_effective_weight_before/after`、`pdb_sampling_scope=per_image`、`best_epoch`、`best_epoch_after_abl_start`、`last_epoch`、checkpoint 文件 SHA、loss 曲线、git/source/config hash。
4. 原通用启动脚本 `run_protocol_b.sh` 目前把 `early_stopping=...patience=300` 写死在 `effective_parameters.txt` 中。**不能产生错误元数据**；新入口采用独立 runner，不调用原通用脚本。
5. 检查原 runner 在 `fit`、`test` 命令中会用 `ABL_ENABLED`、`ABL_WEIGHT` 强制覆盖 YAML；新入口必须保证环境变量与 YAML 完全一致，防止无意间把 Late 关闭。不能直接沿用旧 C1 wrapper（其 λ、CONFIG 和 YAML hash 均锁定）。
6. 每折输出至少包括 `metrics.csv`、best/last checkpoint、best epoch、Dice/IoU/BF1 与 loss history。测试 `train/abl_active` 在阶段交界从 0 转为 1，正常跑满 1000 epoch。
7. `Test_Folder` 仍依原 Protocol B 用于每 epoch `val_dice` 选 best 与最终测试；元数据明确标记 `author_protocol_b_test_as_val` 和 `selection_bias=true`。本文档不更改五折划分。

## 6. 需要增加的单元测试：`tests/test_active_boundary_loss.py`

Codex 应在服务器已通过的 9 项测试上**追加**测试（不能覆盖旧测试），至少验证：

```python
# 判定逻辑伪示例（基于实际对象构造）
assert model._should_apply_abl("train", 0) is False
assert model._should_apply_abl("train", 799) is False
assert model._should_apply_abl("train", 800) is True
assert model._should_apply_abl("train", 999) is True
assert model._should_apply_abl("val", 799) is False
assert model._should_apply_abl("val", 800) is True
assert model._should_apply_abl("test", 999) is False
```

更重要的是**真正的 Loss 门控**，而不只是 `_should_apply_abl`：

- 取同一批 logits/probs/GT，在 `apply_abl=False` 时检验 `L_total == L_base` 且 `L_ABL==0`，在 `apply_abl=True` 且非空 PDB 时检验 `L_total ≈ L_base + 0.10*L_ABL`，并保证有限、可反传。
- 模拟 epoch 799→800 的 `training_step/_shared_step`（或受控 Lightning Trainer），确认调用门控，不在早期偷偷计算 ABL；验证 ABL/BASE/BF1 指标不会影响选模阈值。
- 原 `abl_schedule=always` 必须维持历史 C1 行为；`ABL_ENABLED=False` 的 C0 在任何 epoch 都不能计算 ABL。
- 保留逐图 PDB 不变性、`detach=True`、CUDA AMP 的已通过测试。
- 正式跑之前检查新 YAML 没有 EarlyStopping 且含 `max_epochs:1000`，新启动脚本不覆盖这些设置；验证运行元数据不是旧的 patience300。

## 7. 最少资源的验证顺序

### Gate A：代码和配置测试（无需训练）

```bash
cd /Disk/HDD/Sk/EoSeg-main/EoSe-main-innovation_point
python -u -m unittest discover -s tests -p 'test_active_boundary_loss.py' -v
python -m compileall -q training/medical_binary_segmentation.py
```

如增加了独立测试文件，则运行 `python -u -m unittest discover -s tests -v`。查看 `git diff --check`、`git diff -- training/medical_binary_segmentation.py`；由于相关文件可能未被当前 Git 跟踪，也需逐一人工检查本地实际内容。不要以空 `git diff` 证明未跟踪文件没有变化。

### Gate B：仅验证阶段切换的短程 dry-run

不能简单用 `max_epochs=2`，因为这会改变原实验 1000 epoch 的 LR scheduler 总步数，且永远到不了 epoch 800。建议由 Codex 创建**纯单元/最小 Trainer 验证**：逻辑上模拟 epoch 799 和 800，或临时的独立 smoke 配置 `abl_start_epoch=1`、`max_epochs=2`，只用于验证切换和反传，**不得作为论文训练结果**；新正式 YAML 仍必须是 800/1000。

### Gate C：fold0 完整 pilot（必须真实训练到 epoch 999）

本轮只运行 `C1-Late-NoES / fold0`：1000 epoch，前 800 epoch 使用 Dice+BCE，后 200 epoch 使用 Dice+BCE+ABL。这个完整 pilot 在代码与切换检查完成后，由用户另行启动；本次交付不启动训练。

**若将来需要形成“延迟优于全程”消融，另立 `C1-Always-NoES` 实验**，使用同样的每图采样、1000 epoch 预算与 λ=0.10。本轮不实施或训练这个可选组。

完成 Gate C 后核对：

- 训练是否真的完成 1000 epoch；训练步骤、每折样本数、有效 batch 是否一致；没有 EarlyStopping。
- C1-Late `train/abl_active`：epoch 0～799 为 0，800～999 为 1；`train/abl_effective_weight` 从 0 变成 0.10。
- 阶段切换点前后 `train/loss_base`、`train/loss_abl`、`metrics/val_dice`、`val_boundary_f1` 的变化及是否出现 NaN/Inf。
- **主结果**：C1-Late 的 best `val_dice` checkpoint 的 Dice/IoU/BF1，固定阈值 0.5，单视图。
- **辅助结果**：C1-Late 固定 epoch=999 的 `last.ckpt` 的 Dice/IoU/BF1；必须与主结果分表，不做事后择优。
- 标出 `best_epoch` 是否 `<800`：如果 C1-Late 的主结果 checkpoint 来自 800 前，那么它**实际上没有使用过 ABL 梯度**。不能把其主结果写成“后期 ABL 的有效收益”；需结合预先声明的 `last` 终点评估判断是否值得继续。
- 历史 C0/C1 可供描述背景和检查数值范围，但不能与本轮结果组成同预算对照，也不能直接据此声称延迟 ABL 的因果效果。

### Gate D：是否进入完整五折

**现在先不跑五折。** 先完成上述 Gate A/B/C，再确定是否进入五折：

- 如果 fold0 完成、阶段切换记录正确、训练稳定、BF1 与 Dice/IoU 没有明显异常，则可考虑冻结代码/λ/启用时机后跑 5 folds；由于本轮没有同预算 C0 对照，五折结果只能报告 C1-Late 的表现，不能单凭历史 C0 差值归因于延迟 ABL。
- 如果 fold0 未改善，优先检查阶段切换日志、PDB 正权重密度、C0/C1-Late 最佳 epoch 和固定终点；不要立刻反复改 λ 或阈值 0.5。
- 若要正式宣称**延迟启用优于全程启用**，需要另做同预算、同逐图 PDB 的 `C1-Always-NoES` 五折，与 C1-Late 配对比较；旧 C1 存在 batch 级采样和 EarlyStopping 混杂因素，不能代替它。本轮不安排 C0-NoES。
- 五折报告每折 Dice/IoU/BF1、跨 fold mean ± **sample std**（ddof=1）、配对各 fold 差值；相同 seed=0，单视图，无 TTA，沿用 `Test_Folder` 同时选模和测试的协议并诚实标注偏倚。不要依据单个 fold 或可视化样本挑最有利 λ/启用时机后当作未调参测试。

## 8. 为什么不推荐现在拿旧 C0 checkpoint 直接“接着训后 200 epoch”

之前使用的 C0 checkpoint 如 `epoch=772-step=7730.ckpt` 是**按历史 best Dice 选出的权重检查点**；未必携带 AdamW 的动量、LR scheduler 状态、随机状态，也不是 epoch 799 的完整训练状态。直接加载后执行 200 epoch，会混合“预训练到不同 epoch / 优化器初始化 / LR 曲线重置”等变量。

如果后期为了节省大量计算选择共享前 800 epoch 的共同起点，则必须采用专门的**完整训练状态 checkpoint**：epoch 799 精确保存 model、optimizer、scheduler、训练器进度及可恢复的随机状态，明确原训练计划总预算仍为 1000 epoch，分别从同一个状态恢复 C0/ABL 两条分支；并验证恢复后 `current_epoch`、`global_step`、LR 与 sample 顺序都正确。该加速流程是下一阶段可选工程优化，不作为本 Plan 的首轮默认方案。

## 9. Codex 执行边界和最终交付清单

**请 Codex 只修改/创建**：

1. `training/medical_binary_segmentation.py`：新增可兼容旧配置的 schedule 参数与门控、必要的日志。
2. `tests/test_active_boundary_loss.py`：追加测试；必要时新增一个纯测试文件。
3. 一个独立 YAML：C1-Late-NoES。
4. 一个独立启动脚本，准确记录 EarlyStopping 已关闭并拒绝覆盖旧 run。
5. 如有必要，另建新实验专属的结果汇总脚本；不改历史结果文件。

**先不要执行** 1000-epoch 训练或五折；先完成静态检查、单元测试和阶段切换 smoke，向用户报告结果、所有修改文件、运行命令、配置差异、是否存在显存/磁盘和 LR scheduler 风险，**由用户确认后才开始正式训练**。

交付报告至少包括：

- 改动文件及逐文件说明；对旧 C0/C1 的兼容性；逐图 PDB 和 detach 的不变性。
- 新 YAML 中：早停关闭、max_epochs=1000、ABL schedule 值、阈值 0.5、ModelCheckpoint 监控及 best/last 保存情况。
- 单元测试完整输出，以及 epoch 799/800 的 `L_total`、`abl_active` 的断言结果。
- 新启动脚本如何运行 fold0、如何扩到 5 folds、run name 样式和元数据存放位置。
- 新实验不使用历史 C0 best checkpoint 作为 800 epoch 的优化器续训状态。

---

## 10. 一段可直接发给 Codex 的执行指令

> 请阅读 `EoSeg_C1_Late_ABL_Training_Plan.md` 以及本地 AGENTS/实验协议文件。只在 `EoSe-main-innovation_point` 内实现 C1-Late-NoES 的独立配置、脚本及 ABL 的 800/1000 epoch 门控；本轮不建立 C0-NoES 实验。按本地文件实际内容增量修改，不重置/清理工作区，不覆盖现有 checkpoint。保持 0.5 评价阈值、逐图 1% PDB、detach=True、DINOv2 ViT-L 与 WeightedDice0.6+BCE0.4 不变。确保禁用 EarlyStopping，元数据不能虚报 patience300；保留 best-Dice 与 last checkpoint，并正确检查 best epoch 是否在 800 之后。先运行单元测试和阶段切换检查，不要开始真实 1000 epoch 或五折训练。把修改文件、测试输出和启动命令回报给我，等我确认后再训练。
