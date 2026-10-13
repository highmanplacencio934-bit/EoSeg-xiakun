# EoSeg / GlaS：先验证 Fixed-500 ABL，再研究 Adaptive-Start ABL

> **Codex 执行计划（两阶段，严格按顺序执行）**  
> 项目：DINOv2 ViT-L/14 reg4 + EoSeg `EoSegMultiQSegFusion`，GlaS 5-fold，作者 Protocol B。  
> 参考仓库：`https://github.com/highmanplacencio934-bit/EoSeg-xiakun`。拟定计划时核对的 `main` Git tree：`2be5ef759bd49748be3d4846521906ab69d48e77`；执行时必须以**服务器实际工作树**为准，检查差异再修改。  
> 服务器项目目录：`/Disk/HDD/Sk/EoSeg-main/EoSe-main-innovation_point`。官方 baseline 仓库只读，不作修改。  
> **本次允许执行阶段一：Fixed500 实现、测试、smoke、fold0 正式 1000-epoch 运行与只读报告。阶段二只撰写/设计诊断方案，不得直接实现自适应训练、不得启动新的五折或动态策略训练。**

## 0. 结论与问题界定（开始写代码前必须理解）

**当前需要回答的问题**：在保持模型、数据、Loss 参数和总训练预算不变的情况下，**epoch=500 开始启用 ABL**，是否比此前 epoch=800 才启动的策略更有机会改善 GlaS 腺体边界？

**现有五折事实**：C1-Late800 的 global-best 验证 Dice checkpoint epoch 为 799、729、698、842、691。4/5 折 best 位于 ABL 启用前，因此 global-best 评估大多不能体现后期边界训练的作用。但这**不能推导 epoch 500 是最优启动点**；500 只是事前定义、用于检验“更早参与是否有效”的候选时机。

**关于第二阶段的动态启用**：Train_Folder 固定样本的 `R_PDB`、训练 Dice、基础 Loss 只能评估“边界监督是否有条件开始较可靠”，**并不能单独证明某个 epoch 最优，也不保证动态启用会提升 BF1**。启用阈值、观察窗口、开始/截止区间须通过训练集诊断与后续消融研究确定，**本次不写死 0.60、200、600 为定论**。

## 1. 必须保持冻结的实验协议

| 项目 | 要求 |
|---|---|
| 数据 | 原有 `datasets/GlaS_5fold/fold{0..4}/Train_Folder` 训练，`Test_Folder` 用于按 Protocol B 的验证/测试，不改划分或 `samples.txt` |
| 模型 | `EoSegMultiQSegFusion` + `vit_large_patch14_reg4_dinov2`；同一预训练权重，不改 query 数、Decoder、Fusion Gate 或模型前向 |
| 基础损失 | `WeightedDiceBCE(dice_weight=0.6, bce_weight=0.4)`；不重新归一化 |
| ABL | 保持现有实现、`detach=True`、逐图 PDB 采样 `abl_max_n_ratio=0.01`、`label_smoothing=0.2`、`max_clip_dist=20` |
| λ | `0.10`，沿用 Late800 对照；不与 λ=0.20 梯度诊断混淆 |
| 分辨率和判别 | 输入/评价 `224×224`；预测阈值 `0.5`；无 TTA；BF1 为欧氏半径 2 px、8 邻域内边界；严格沿用 `METRICS_PROTOCOL.md` |
| 训练 | `max_epochs=1000`（0-based 为 0–999）、**关闭 EarlyStopping**、seed 0、物理 batch 2、梯度累积 7、16-mixed、clip 1.0 |
| 优化器/调度 | 原 AdamW、`lr=5e-5`、backbone `1e-5`、Decoder `1e-4`、LLRD 0.8、warmup `[500,1000]` **均为 optimizer step，不是 ABL 的启用 epoch**；Poly LR 计划按 1000-epoch 完整预算配置 |
| 全程 checkpoint 规则 | `metrics/val_dice` 最大值；保持 0–999 全部 epoch 都参与选模；不悄悄改成 BF1 选模 |
| 结果声明 | `Test_Folder` 兼作验证与测试，务必声明 `author_protocol_b_test_as_val`、`selection_bias=true`，不能宣称是完全独立测试集表现 |

**重要**：旧 C0 Baseline 可能用过 EarlyStopping，不能直接将其视为“1000 epoch 无早停”的完全匹配对照。Fixed500 vs Late800（如果其他代码和训练条件一致）可以先进行探索性比较；要确认相对 C0 的净贡献，必须再建立独立 `C0-NoES1000` 匹配实验。

## 2. 阶段一：Fixed500 实验——这次让 Codex 真正完成的工作

### 2.1 先审计当前本地工作树（不得覆盖）

1. 在服务器项目根目录执行 `git status --short`、`git rev-parse HEAD`，核对本地修改/未跟踪文件。不要假定 GitHub 的最新 commit 已经同步到服务器。
2. 阅读下列**当前服务器版**文件（不要仅使用计划中的代码片段覆盖）：
   - `training/medical_binary_segmentation.py`：`abl_schedule`、`_should_apply_abl`、`_compute_loss_components`、`_shared_step` 与实际日志。
   - `training/active_boundary_loss.py`：逐图 PDB 选择及 `isdetach`。
   - `configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_late_abl_noes_1000.yaml`。
   - `configs/glas/run_protocol_b_c1_late_abl_noes_1000.sh`。
   - `configs/glas/run_protocol_b_c1_late_epoch999_recovery.sh`（作为旧 checkpoint 事故事项的参考，不是本次训练入口）。
   - `training/two_stage_warmup_poly_schedule.py`、`METRICS_PROTOCOL.md`、`tests/test_active_boundary_loss.py`。
3. 保留现有 `runs/`、`test_correct/`、`experiment_a/b/c/`、旧 `.ckpt`、旧 JSON/CSV、官方 baseline 的所有数据。**严禁 `git reset --hard`、`git clean`、删除/重命名原 run 或复用其 logger 名称。**
4. 记录当前源码与 manifest 的 SHA256，确保新运行可复现。

### 2.2 新建 Fixed500 独立 YAML，不碰旧 C1-Late800

建议新文件：

`configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_fixed500_noes_1000.yaml`

以 Late800 YAML 复制创建，**只在实验相关字段作如下调整**：

```yaml
# 仅展示需确认的关键字段；其他训练/数据/架构配置全部沿用 Late800
trainer:
  max_epochs: 1000
  # callbacks 中必须没有 EarlyStopping
  # ModelCheckpoint 保留 monitor=metrics/val_dice, mode=max, save_top_k=1
  # save_weights_only=false（需要完整状态）；save_last=true，但见 2.5
model:
  init_args:
    active_boundary_loss_enabled: true
    active_boundary_loss_weight: 0.10
    abl_schedule: late
    abl_start_epoch: 500
    abl_max_n_ratio: 0.01
    abl_label_smoothing: 0.2
    abl_max_clip_dist: 20.0
```

- 使用既有 `late` 模式即可；**此次不新增 `adaptive` 模式、不改 ABL 数学形式**。
- epoch **499**：`L_total=L_base`；epoch **500** 开始：`L_total=L_base+0.10*ABL`，直到 999。
- `test/loss_abl=0` 在现有 `late` 调度下是正常行为：`_should_apply_abl` 只允许 `train/val`，不允许 `test`；不能依据测试 Loss=0 断言训练未使用 ABL。
- 保持验证/测试 Dice、IoU、BF1 口径；辅助 `val/loss` 因 Late 模式可能计算 ABL，分析时用 `val/loss_base` 或训练 `train/loss_base` 对齐，**不可把切换前后的 `val/loss` 直接作为同定义时间序列解释**。

### 2.3 新建独立启动脚本并修正旧脚本的硬编码假设

建议新文件：

`configs/glas/run_protocol_b_c1_fixed500_noes_1000.sh`

参考现有 Late800 脚本的防覆盖、环境离线、manifest 检查、SHA256 审计、自动测试与 BF1 导出逻辑，**不要机械复制其中写死的 epoch=799/800 开关断言、run 名或标注**。

必要功能：

1. `FOLDS` 默认仅 `0`，显式 `FOLDS="0 1 2 3 4"` 才可扩展；此次不得自动运行五折。
2. 支持 `--check-only`，只验证配置/文件/输出目录是否可用，绝不初始化模型或训练。
3. 独立运行名称（示例）：`glas_protocol_b_fold0_vitl_c1_fixed500_noes1000_seed0`；跑过的 run 已存在时直接报错，不允许静默覆盖。所有训练与报告放该 run 下。
4. 配置前置检查：epoch=500、ABL λ=0.10、1000 epoch、无早停、`save_top_k=1` + `save_last=true`、固定超参数及分辨率、manifest 存在、来源预训练权重存在。
5. 启动之前保存 YAML 副本、`git_status.txt`、Git commit、源文件哈希、预训练权重哈希、manifest 哈希、完整运行命令、运行环境与 GPU 信息。
6. 使用现有 `main.py fit -c ... --compile_disabled` 方式，从**原始 DINOv2 预训练权重初始化并完整训练 0–999**。**不要**从 Late800 的 epoch 799/999 checkpoint 或 C0 的后期 checkpoint 冷启动：那样不能检验“epoch 500 开始”的效果，也无法作匹配比较。
7. 训练后检查 `metrics.csv` 实际 gate 日志，以及 checkpoint 内部元数据，再执行 best 的 test/BF1 评价。
8. 参考启动形式（**Codex 建好并自检入口后再执行**）：

```bash
cd /Disk/HDD/Sk/EoSeg-main/EoSe-main-innovation_point
FOLDS="0" bash configs/glas/run_protocol_b_c1_fixed500_noes_1000.sh --check-only
# 先运行单元测试及 smoke；通过后执行完整 Fold0：
FOLDS="0" bash configs/glas/run_protocol_b_c1_fixed500_noes_1000.sh --compile_disabled
```

**Smoke 的要求**：不要用和正式 Fold0 相同的 run 名。单独建立 `smoke` 配置/入口或只用单元测试中的短 Trainer；确保短训练可越过 epoch=500 的 gate（可以临时将阈值设为 2，并测试 epoch 1/2/3；严禁把 smoke 的阈值带回正式配置）。

### 2.4 训练 Loss / ABL 启动核验（硬性门禁）

读取 `metrics.csv` 中 **同一个 epoch 对应的训练聚合记录**，检查：

| epoch | `train/abl_active` | `train/abl_effective_weight` | ABL 对训练的要求 |
|---|---:|---:|---|
| 0、499 | 0 | 0 | `L_total=L_base`，ABL 不参与梯度 |
| **500** | **1** | **0.10** | 已实际计算 ABL；训练分支有梯度 |
| 501、600、999 | 1 | 0.10 | 保持激活，持续训练 |

必须记录 `train/loss_base`、`train/loss_abl`、`train/loss`、`train/abl_active`、`train/abl_effective_weight`，至少验证 epoch 499/500/999。对非空 PDB 的 batch，验证 `train/loss ≈ train/loss_base + λ*train/loss_abl`（考虑不同按 batch/按 image 聚合定义，需使用一致统计口径；最好在固定 batch 的 unit test 中精确断言）。

**注意**：`train/loss_abl=0` 可能是合法的空 PDB/有效监督点缺失结果。除了平均值，建议记录启用后 `None` 返回比例、非零 ABL step 比例与 NaN/Inf 比例。若整个 fold 500–999 持续没有有效 ABL 梯度，应判为实验无效并排查，而不能继续归因于“启用过晚”。

### 2.5 Checkpoint 正确性——必须显式规避之前 best/last 相同事故

此前 Late800 各 fold 的 `best.ckpt` 和 `last.ckpt` SHA256 曾相同，虽然元数据却标记 `last_epoch=999`。用户表示已进行相关处理，**本次仍需做独立审计，而不是假定修复已生效**。

对 Fold0：

- **Global-best**：0–999 内 val Dice 最高，对应实际 best checkpoint；作为正式分析主指标。
- **Post-ABL-best（辅助）**：仅在 500–999 内选 val Dice 最高的 checkpoint；建议保存独立 `post_abl_best/` 路径，不能与 global-best 目录混淆。此结果仅用于确认边界监督阶段是否产生更好模型，不替代原始选模协议。实现可用独立且经过测试的 epoch-gated checkpoint callback，不能仅依据 1000 行日志得出 checkpoint 路径然后虚构文件。
- **Endpoint-999（辅助）**：准确保存 999 epoch 结束后的模型状态；不能把 `last.ckpt` 的**文件名**当作已验证的最终 epoch。若 Lightning 的 `save_last` 是链接/拷贝 best，需另外采用可靠保存机制，避免 `last` 指向 best。

至少输出下列数据：

```text
checkpoint_role, path, actual_epoch_from_checkpoint,
actual_global_step_from_checkpoint, sha256, saved_epoch_val_dice
```

核验文件内部 `epoch` / `global_step`（Lightning 各版本需确认零基与保存 hook 语义），并检查文件哈希：若 best epoch <999，但 best 与 endpoint-999 哈希完全一致，**必须报错暂停并诊断保存流程**。不得硬写 `last_epoch=999` 代替读取。

若第二个 callback/额外 checkpoint 增加较大内存或存储消耗，应保留主 best 和经过验证的 endpoint-999，并优先将 post-ABL-best 设为按 epoch 门控的 `save_top_k=1`；不要每 epoch 留全量 308M 参数训练状态。报告中必须注明某个 checkpoint 没有保存、无法补算，不能以错误文件冒充。

### 2.6 自动化测试和验收

新增或扩展测试文件（例如 `tests/test_c1_fixed500_schedule.py`），至少覆盖：

1. `abl_schedule=late` 且 `abl_start_epoch=500`，epoch 499/500/999 在 `train` 阶段分别 off/on/on。
2. 用同样 logits/GT 的真实 `_compute_loss_components` 检查 499/500 的总 Loss 与梯度路径；禁用期严格等于基础损失。
3. `val` 与 `test` 逻辑符合当前代码：`test/loss_abl=0` 并不表示训练阶段没调用 ABL。
4. YAML 不含 EarlyStopping、总 1000 epoch，模型和数据关键参数与 Late800 一致，差异仅限组名/启用点/必要的 checkpoint 审计。
5. 已实现的 9 项以上 PDB 批内独立采样与 CUDA AMP 单元测试通过；新增修改不破坏它们。
6. Runner `--check-only` 的真实配置核查（包括 `abl_start_epoch=500`）；重复运行同名已存在 run 时必须拒绝。
7. 真实两三个 epoch 的小训练（使用独立 smoke 阈值）显示 ABL 启用前后 train gate 按预期切换，且不改变训练/测试指标协议。
8. Checkpoint 测试：确保 global-best、post-ABL-best（若实现）、endpoint-999 文件路径与内部 epoch 能正确解析，不依赖文件名推断；恢复 full checkpoint 不丢失优化器/scheduler 状态。

**只在测试、smoke、配置检查全部通过后启动 Fold0 的完整训练。失败就停止，不能自动开始五折。**

### 2.7 Fold0 完成后生成报告（Codex 交付物）

建议报告：`reports/c1_fixed500_fold0_review.md` 和机器可读的 `reports/c1_fixed500_fold0_review.json`。报告必须包含：

- 所用 Git 状态与源码哈希、固定训练协议、是否启用了 ABL 以及真实启动 epoch。
- 499、500、501、600、700、800、900、999 的基础 Loss、ABL Loss、总 Loss、ABL gate、val Dice/IoU/BF1。
- 0–499、500–999 各自的最佳 val Dice epoch 与指标；Global-best、Post-ABL-best（如有真实模型）、Endpoint-999 的完整比较。
- 与历史 C1-Late800 五折中对应 Fold0 的走势对比（仅探索性，保留 checkpoint/训练条件差别说明）；若有完全匹配的 C0-NoES1000 Fold0，则做配对对照，没有则明确标为待补。
- 可视化：`val_dice_vs_epoch.png`、`val_bf1_vs_epoch.png`、`train_losses_vs_epoch.png`，画出 epoch=500 分界线；曲线趋势可描述但不可夸大微小波动。
- 输出差异应以 **百分点** 或原始比例写清楚，并附不确定性：单 Fold0 结果只作诊断，不可声称五折稳定改善。
- 报告 BF1 使用固定 224×224、2px 欧氏距离和阈值 0.5，不能用测试图可视化挑选有利的阈值或 checkpoint。

**阶段一停止条件**：Fold0 完成并交付报告后停止。**不要直接重训 5 folds、不要将 Fixed500 的 Test_Folder 最佳表现作为调参依据、不要开始 Adaptive 实现。** 等用户阅读并决定后续。

---

## 3. 阶段二：PDB 训练就绪度诊断——仅规划，得到阶段一报告后才启动

### 3.1 为什么三个指标只是“候选触发信号”

`R_PDB` 可以定义为“预测边界候选中距 GT 边界 ≤2px 的比例”，而不是分割性能或最佳启动时机的充分统计量。原因：

- PDB 每张图最多约 1% 的位置（膨胀前）是由当前预测分布 **挑选** 的，可能偏向少数容易的边界；`R_PDB` 较高不表示整条 GTB 都被识别。
- 训练 Dice 高、训练 Loss 下降变慢，可能是过拟合或训练停滞，而不一定代表边界已就绪。
- ABL 的真实效益还受学习率、总步数、梯度冲突、权重 λ、PDB 有效点比例影响。动态触发机制还必须通过**独立对照**证明，不允许“看了一次曲线便认定启动时机最优”。

建议把第二阶段问题分为两个层次：**(a) 何时满足可靠边界监督的必要条件？(b) 此时介入 ABL 是否确实产生更高 BF1？** 第一层用无梯度训练集诊断回答，第二层通过匹配训练预算的消融回答。

### 3.2 固定 Train_Folder 诊断子集和模型阶段

在每一 fold 的 Train_Folder 内，从冻结的 `samples.txt`/manifest 按事前确定规则选**固定**的 24–32 张样本；记录 sample ID、筛选规则/seed 和文件哈希。可包含典型易/难样本，但**不能看 Test_Folder 指标选图**；默认推荐事先固定 ID 或仅凭 manifest 做确定性抽样，避免事后选择。诊断使用：

- `model.eval()` + `torch.no_grad()`，关闭数据增强、随机 shuffle 和 TTA，固定 `224×224` 预处理与阈值 `0.5`；记录推理精度并各次保持一致。
- 只用 Train_Folder 的 GT 计算 GT 边界与 PDB 距离；严禁从 Test_Folder 拿训练触发信号。
- 在训练里每隔 25 或 50 epoch 无梯度诊断，或者使用在 epoch 0、100、200、300、400、450、500、550、600、700、800、900、999 等事前指定阶段保存的真实 checkpoint。**现有旧 C1-Late800 只保存 best/last 时，无法事后重建全部早期权重状态；不要从单条 val 曲线虚构 `R_PDB`。**
- 若通过在线诊断采集：必须恢复训练模式与 RNG 状态，不影响参数、优化器、调度器、数据顺序和训练结果；先做诊断开/关的数值一致性短测。若快照方式：定期保存仅模型权重，避免频繁保存 308M 模型的完整 optimizer 状态；限额并留足空间。

### 3.3 候选诊断指标的严格定义

令 `PDB_pre` 为 ABL **逐图采样、膨胀前**的预测边界候选集合；GTB 使用 `METRICS_PROTOCOL.md` 的 8 邻域内边界定义，距离使用像素级**欧氏距离**，`d(p,GTB)` 为候选点到最近 GT 边界的欧氏距离：

```text
R_PDB(image) = count{p in PDB_pre: d(p, GTB) <= 2px} / count(PDB_pre)
```

- 仅用于测量“当前挑选的预测边界候选是否足够靠近 GT”；**不要直接把它写成正式 BF1，也不要使用 GTB 和 PDB 的重叠线宽可视化代替距离计算**。
- 如果 PDB 为空，标 `undefined` 并单独计数，不得偷算为 1 或 0；若 GTB 为空，按照预设的空边界规则单独统计；宏平均只覆盖有效图像并报告有效率。
- 同时记录膨胀前、膨胀后 PDB 数量、有效方向点数、正距离权重点数、ABL None 比例/非零梯度批比例，以避免 `R_PDB` 高但采样严重不足。
- **增加对称覆盖指标** `R_GTB_coverage = # {g in GTB : d(g, PDB_pre) <= 2px} / #GTB` 或训练集 Boundary Recall，以防模型只找到少数容易预测的轮廓造成虚高。注意最多 1% 的采样上限对 `R_GTB_coverage` 的解释会有影响，须对比曲线而非随意比较绝对值。
- 计算固定子集 Dice（相同 0.5 阈值）和 `L_base=WeightedDiceBCE`。训练全量 `train/loss_base` 可作为补充，但其训练增强使曲线可能与固定无增强子集不同；**不要混成一条曲线**。
- 每次记录子集逐图数值和均值/中位数/分位数，保留真实 sample ID；画出随 epoch 的走势和波动带。

建议新只读分析脚本：`scripts/diagnose_abl_readiness.py`，输出到新目录：

```text
experiment_d/abl_readiness_fixed_train_subset_<run_tag>/
  manifest.json
  per_image.csv
  per_epoch.csv
  summary.json
  readiness_curves.png
```

脚本不修改权重、不进行 optimizer.step、不写入旧训练目录。

### 3.4 如何由这些诊断形成可实施的动态触发条件

此处只给**候选结构**，阈值均为待训练集校准量，不允许先验声称 0.60 或 1% 已经验证：

```python
# 伪代码。只有阶段二诊断完成、规则预注册后才能实现 adaptive。
if not abl_triggered and epoch_boundary:
    # 使用上一个完整 epoch 后的 Train_Folder 无增强固定子集诊断
    region_ready = (
        train_subset_dice_is_stable
        and smoothed_base_loss_improvement_is_small
    )
    boundary_ready = (
        r_pdb_above_calibrated_threshold
        and r_gtb_coverage_above_calibrated_floor
        and pdb_valid_fraction_above_calibrated_floor
    )
    ready_streak = (ready_streak + 1) if (region_ready and boundary_ready) else 0
    if ready_streak >= required_consecutive_checks:
        abl_triggered = True
        trigger_epoch = current_epoch
        trigger_reason = "train_readiness"
    elif current_epoch >= registered_fallback_epoch:
        abl_triggered = True
        trigger_epoch = current_epoch
        trigger_reason = "fallback"

# 一旦启动持续到结束；λ 在这一版先固定 0.10，不另加逐步增长机制
```

必须事先确定：滑动窗口长度、改善阈值、R_PDB 与 GTB 覆盖阈值、有效样本下限、连续检查次数、最早/最迟启动边界以及 fallback 策略。**所有这些参数只依据 Train_Folder 的诊断或独立内部调参决定，不直接用 Test_Folder 的 Dice/BF1 搜索。**

实际实现需要：epoch 边界原子切换（不能在某个 batch 中途乱切）、启用状态与触发原因持久化、断点恢复一致性、可选分布式同步、固定子集评估与 RNG 隔离、测试阶段不计算 ABL；每个 fold 可以触发于不同 epoch。启动 Epoch 的名字/数值必须在运行后日志中可审计。

### 3.5 动态策略要用什么证据证明有效？

待 Fixed500 报告完成并确认值得继续后，才决定是否启动以下匹配消融：

| 组别 | ABL 策略 | 意义 |
|---|---|---|
| `C0-NoES1000` | 全程 λ=0，无早停 | 匹配训练预算的真正基线 |
| `C1-Fixed500` | epoch≥500，λ=0.10 | 第一阶段本次实验 |
| `C1-Late800` | epoch≥800，λ=0.10 | 已完成历史对照，注意 checkpoint 审计 |
| `C1-Adaptive` | 训练集就绪度触发，λ=0.10 | 待验证的方法 |
| `C1-Adaptive-ablation` | 去掉边界可靠性或区域稳定性等一项 | 检验每项条件的必要性 |

同样的 seed、模型初始化、fold、数据/优化器/LR/训练步数/评价/选模规则；先小规模验证可靠性、正式参数冻结后再执行 5-fold 并报告每折差值及 mean±sample std。可从**相同的早期公共完整训练状态**分叉，但必须保证优化器/scheduler/RNG 状态一致、原始训练与初始化协议清晰，并证明分叉点之前各组训练完全一致；否则从头训练更干净。

后期启用的模型如果 Global-best 落在启动之前，主结果仍必须如实报告 Global-best；`Post-ABL-best` 和真实 Endpoint 只能列为**预先定义的辅助分析**，不能偷换 checkpoint 选择规则。还需注意 Protocol B 因 Test_Folder 参与选模而有选择偏差；如需更可信的泛化声称，应另外采用有独立测试集的扩展实验。

## 4. 阶段一完成的最低交付验收清单

- [ ] 不修改官方 baseline、旧 C0/C1/Late800、旧 `runs/` 和原有文件；新实验有独立 YAML、runner、日志目录。
- [ ] 新 YAML 除 `abl_start_epoch=500` 和组名/必须的 checkpoint 审计外，与 Late800 关键协议一致。
- [ ] `--check-only`、单元测试、smoke 通过；499=off、500=on、999=on，Loss 和梯度路径均被验证。
- [ ] Fold0 完成真实 epoch 0–999；完整记录 ABL Loss 非零/None、权重、学习率及各 epoch 的 Dice、IoU、BF1。
- [ ] 全局 best、真实 999-endpoint 不被误用同一份文件；Post-ABL-best 若实现须确实保存，所有文件可读取内部 epoch/global_step。
- [ ] Fold0 的报告 `reports/c1_fixed500_fold0_review.md` + JSON + 三张曲线和实验 SHA256 可复核。
- [ ] **在报告与测试交付后停止**：不自动运行全部五折，不自动实现自适应调度，不根据 Fold0 Test_Folder 的最好数值事后改 λ 或启动时间。

## 5. 给 Codex 的执行指令（可直接复制）

> 先完整阅读本 Markdown 和服务器项目当前代码，按**阶段一**实现独立 `C1-Fixed500-NoES1000`。保持 DINOv2/EoSeg/ABL/PDB/梯度协议完全不变，仅将 epoch 调度由 800 改为 500 并配套可靠的全程/ABL 阶段/endpoint checkpoint 审计。先对照现有 Late800 代码补齐新配置、独立 runner、单元测试、smoke；确认无误后**只运行 fold0 的完整 1000-epoch 实验**，导出曲线与 BF1 报告。严禁覆盖旧 runs、篡改官方 baseline 或因修改导致未提交工作丢失。报告中明确说明 Protocol B Test_Folder 被用于 val 选模。阶段二关于 Train_Folder 固定样本的 `R_PDB`、Dice、基础 Loss 只写成后续诊断设计，这次不要实现 Adaptive/不要运行五折。完成后汇报变更文件、测试、真实 checkpoint epoch/global_step、Fold0 结果与尚未完成的工作。

---

**论文表述边界**：目前 Fixed500 只是受历史收敛时间启发的先验候选，不是已验证的创新。将来若训练集驱动的动态时机选择确实稳定改善 BF1，且有公平消融与现有文献的新颖性比较，才有机会发展为一个可声称的方法贡献。
