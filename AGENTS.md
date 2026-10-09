# 项目全局工作规则

## 规则优先级

1. 当前用户明确指令。
2. 当前目录及其父目录中的 `AGENTS.md`。
3. `Z:\.codex\PROJECT_EXPERIMENT_PRIORITY.md` 及其引用的规则文件。
4. 历史计划、旧实验记录和聊天记录仅作参考，不得覆盖当前冻结规则。

## 主工作区和目录职责

- `Z:\` 是服务器映射盘，也是当前项目主工作区。
- `Z:\EoSeg-main-official` 是已经确定的 ViT-L Protocol B 稳定 baseline。此前按用户授权在此加入只读 Boundary-F1 评价和论文图脚本；这不是放开模型优化的先例。后续 C1、结构/损失改进、消融和训练只在 innovation_point 进行。
- `Z:\EoSe-main-innovation_point` 是当前唯一允许进行后续创新点开发、消融和训练的目录。
- `Z:\Warwick_QU_Dataset` 是固定的 GlaS 原始数据，只读，不重命名、不覆盖。
- `C:\develop\github\task_MyProfessor\26.08\EoSeg-main*` 仅作本地备份、审查和对照，不作为服务器实验主目录。

## Baseline 冻结

当前 baseline 标识为：

```text
vitl_maskonly_gn_protocol_b_seed0
```

Baseline 代码目录：

```text
Z:\EoSeg-main-official
```

Baseline 配置和启动入口：

```text
Z:\EoSeg-main-official\configs\glas\vit_query_mul_scale_fusion_protocol_b.yaml
Z:\EoSeg-main-official\configs\glas\run_protocol_b.sh
```

Baseline 变更记录：

```text
Z:\.codex\BASELINE_CHANGE_RECORD_VITL_MASKONLY_GN.md
```

未经用户新的明确授权，不得修改 official 中的模型、损失、数据划分、预处理、优化器、学习率、scheduler、seed、阈值或 checkpoint 选择规则；已有 BF1 评价及定性展示脚本只用于既定评估/作图，不得扩展为训练优化。不得把 innovation_point 的代码回写 official。

## 当前创新工作区

后续寻找和实现创新点，只能在：

```text
Z:\EoSe-main-innovation_point
```

该目录已经从 official 复制代码和配置，但不包含 official 的 `runs`。innovation_point 的训练结果、checkpoint、日志和图片必须保存在该目录自己的 `runs` 下，不得使用 official 的 runs。

每个创新实验必须：

- 建立独立 YAML 或明确的实验配置副本；
- 使用唯一的 run name / `RUN_SUFFIX`；
- 在实验记录中写明本次唯一改变的变量：C1 只能增加 ABL 损失，C2 只能修改 Decoder 融合；
- 保持 baseline 的 backbone、预训练权重、数据划分、输入尺寸、优化器、scheduler、effective batch、阈值和评估口径不变；基础 Dice/BCE 损失冻结，仅 C1/C3 可叠加 ABL；
- 如需改变冻结项，先建立新的独立实验组，不得把结果混入当前 baseline。

## 当前 baseline 固定项

```text
Backbone: vit_large_patch14_reg4_dinov2
Pretrained weight: 服务器已有 DINOv2 ViT-L 权重
Input: 224x224
Protocol: author Protocol B（Train_Folder 训练，Test_Folder 验证/选 checkpoint/最终测试）
Loss: Weighted Dice(0.6) + BCE(0.4)
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
EarlyStopping: monitor metrics/val_dice, patience 300
Main evaluation: single-view, no TTA
```

## Innovation 1（C1）：Active Boundary Loss 独立实验入口

正式计划为 `Z:\.codex\EoSeg_innov1_Dice_BCE_ABLplan.md`；进行 C1 代码、配置、训练、评估或消融前必须阅读全文。C1 只在 `Z:\EoSe-main-innovation_point` 开发，不修改 official 的模型、Loss、配置或 runs。C0 是冻结基线；C1 仅在同一最终前景预测上增加 Active Boundary Loss（ABL），不是 Boundary Dice：`L_C1=0.6 L_Dice+0.4 L_BCE+λ_ABL L_ABL`。不改 ViT-L、mask-only、GroupNorm、Decoder 三层融合、数据/预处理、分模块学习率、训练预算和指标。由一通道前景 logit `z` 构造 ABL 所需二类 logits `[0,z]`，保持 `softmax([0,z])_fg=sigmoid(z)`；空边界、设备和 AMP 路径须检查。`λ_ABL=0.10` 目前仅为计划中的首个候选值，未被确认是最终论文参数；不得用 Protocol B 的 `Test_Folder` 指标挑选 λ。

**每个创新点的入口必须相互独立、可复现消融：**C1、C2、C3 分别有独立 YAML、启动脚本/明确命令、唯一 run 名称和各自 checkpoint/日志；C1 只能启用 ABL、关闭 C2 结构，C2 只能启用 Decoder 改动、关闭 ABL，C3 才同时启用二者。通用训练代码可以共用，但开关必须由实验配置显式控制，不能靠手工改同一 YAML、覆盖同名 runs、隐式继承另一创新点或从对方 checkpoint 热启动。每次启动记录实际配置、开关、λ、fold、seed、初始化来源与代码版本。先做 `λ_ABL=0` 对 C0 的数值回归，再查 forward/backward、AMP/NaN 和 BF1/Dice/IoU，正式五折仍按 `metrics/val_dice` 选 checkpoint、single-view 测试并披露 `author_protocol_b_test_as_val`。历史 92.56% Dice/86.66% IoU 属于旧 ViT-S/Train-Test/TTA 口径，不可当作当前 ViT-L C1 五折结果。

上述 `C3` 是 I1+I2 的组合消融，**不是尚未确定的第三创新点 I3**；未来 I3 需新建独立配置、启动入口、run 和实验编号，不得复用 C3。

## 固定 seed 集合实验

当前开发阶段使用 `seed=0`。当创新点确定后，如进行固定 seed 集合实验：

- 所有对比模型必须使用完全相同的 seed 集合；
- 每个 seed 和 fold 使用独立 run name；
- 不覆盖 seed=0 的 baseline 结果；
- 主表中每个 fold 先对 seed 求均值，再对五个 fold 的均值计算 mean ± sample std；
- seed 波动另行记录，不能与 fold 标准差混写。

## 评估与 TTA

主实验使用 single-view、阈值 0.5，并标明 `author_protocol_b_test_as_val`。TTA 只能作为补充结果，必须使用四翻转并单独标记 `TTA=4-flip`，不能用于选 checkpoint、调参或定义 baseline。详细规则见 `Z:\.codex\TTA_EXPERIMENT_POLICY.md`。

## Boundary-F1 与论文定性图：两份 Plan 的现行执行规则

本节将 `Z:\EoSeg-main-official\EoSeg_BoundaryF1_Baseline_Codex_Plan.md` 与 `Z:\EoSeg-main-official\EoSeg_Paper_Qualitative_Visualization_Codex_Plan.md` 的可执行要求纳入全局规则。相关任务须先读两份原计划及 `Z:\.codex\PROJECT_EXPERIMENT_PRIORITY.md` 的对应章节。Plan 中较早的“只放 innovation_point”“尚无 BF1/仍待五折重训”等表述是历史状态；以两份 Plan 的后补章节及本节为准。后补章节仅授权当时在 official 接入 BF1/可视化，不授权未来的模型或训练优化。

- C0 为上述冻结稳定基线；C1 仅在 innovation_point 给相同基础损失加入 Active Boundary Loss。五折同 manifest、seed=0、single-view、阈值 0.5，以各折 `metrics/val_dice` 最优 checkpoint 评估。`Test_Folder` 同时选 checkpoint 和测试，必须披露 `author_protocol_b_test_as_val`，不得称独立盲测。
- BF1 是只读评价指标，不进入 loss、反向传播、训练采样、scheduler 或 checkpoint 选择。224×224 二值图上以预测概率 `>=0.5`、GT `>0.5` 提取 8 邻域内轮廓，采用欧氏距离 `<=2 px` 的双向匹配；两边界均空记 1，仅一边空记 0。逐图 BF1 后取折内宏平均，五折报 mean ± sample std (`ddof=1`)；保留每折 Dice、IoU、BF1，C1 就绪后按同折计算 `ΔBF1=C1−C0`，不可挑折或事后调阈值/容差。BF1 与区域指标不得混用不同 run 的 checkpoint。
- C0 的 `_bf1_metric_check` 五折已完成，汇总见 `Z:\EoSeg-main-official\BF1_METRIC_CHECK_5FOLD_RESULTS.md`；这是单独重训 run，不与旧 C0 run 混成同一批结果。C1 正式五折 checkpoint 尚未核实；不得预填 C1 数值或图片。
- 定性图只作辅助证据，固定 fold0 `Test_Folder` 的 `sample_id=train_3`，禁止自动挑“好看样本”。完整六面板严格为 `(a) Original`、`(b) Ground Truth`、`(c) C0 mask`、`(d) C1 mask`、`(e) C0 boundary overlay`、`(f) C1 boundary overlay`。C1 缺席时只能做 `(a),(b),(c),(e)` 的 C0-only 预览，不能把 `(e)` 改标 `(d)`。GT 边界青色 `#00DCFF`、预测橙红 `#FF4A00`、显示笔画重叠黄色 `#FFEB00`；黄色只是显示，不是 BF1 匹配。
- 局部图 C0/C1 使用同一原图 ROI `(310,210,455,455)`、同样缩放与线宽；不按 C1 效果换样本或裁剪。正式 BF1 只在 224×224 上算；原图尺寸 nearest-neighbour 放大、轮廓加粗仅供显示。图片 metadata 记录协议、折、sample ID、seed、配置/checkpoint、阈值、BF1 定义、ROI、分辨率和代码版本。
- 当前已放在 official 的 BF1/可视化代码及其结果保持原位作为基线评价材料。今后对模型、损失、训练或可视化方法的改进和 C1 对比开发默认只在 `Z:\EoSe-main-innovation_point`，使用其独立配置与 runs；official checkpoint 只读引用。Linux 推理/绘图命令由用户在服务器执行。

## 训练前必读

```text
Z:\AGENTS.md
Z:\.codex\PROJECT_EXPERIMENT_PRIORITY.md
Z:\.codex\PROTOCOL_B_SOURCE_POLICY.md
Z:\.codex\PROTOCOL_B_SELECTION_OVERRIDE.md
Z:\.codex\BASELINE_FREEZE_CHECKLIST.md
Z:\.codex\ARCHITECTURE_SINGLE_VARIABLE_RULE.md
Z:\.codex\LOSS_PROTOCOL.md
Z:\.codex\OPTIMIZER_PROTOCOL.md
Z:\.codex\LR_SCHEDULER_POLICY.md
Z:\.codex\METRICS_PROTOCOL.md
Z:\.codex\TTA_EXPERIMENT_POLICY.md
Z:\.codex\SEED_PROTOCOL.md
Z:\.codex\BASELINE_CHANGE_RECORD_VITL_MASKONLY_GN.md
Z:\.codex\EoSeg_innov1_Dice_BCE_ABLplan.md（C1/C3 与相关消融任务）
Z:\EoSeg-main-official\EoSeg_BoundaryF1_Baseline_Codex_Plan.md（BF1/论文评价相关任务）
Z:\EoSeg-main-official\EoSeg_Paper_Qualitative_Visualization_Codex_Plan.md（论文图相关任务）
```

## 有效训练记录

只有实际完成训练并生成可分析 checkpoint 和指标的运行才算有效训练。有效训练的配置、代码目录、seed、fold、checkpoint、single-view/TTA、Dice、IoU 和其他论文需要的指标，应在 `Z:\.codex` 下的 Markdown 实验记录中记录。未完成、冒烟测试和失败运行只作为诊断，不得当作论文结果。
## 服务器命令

Linux/服务器终端命令只提供给用户，由用户在服务器终端执行并反馈输出。Codex 可以直接修改 Z 盘文件，但不自动执行服务器训练、测试或删除操作。
