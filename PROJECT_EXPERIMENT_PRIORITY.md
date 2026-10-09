# 当前实验全局优先级：ViT-L Baseline 冻结与 Innovation Point 独立开发

## 1. 当前工作区切换

从本文件更新后，后续代码增删改查、模型结构实验和训练默认作用于：

```text
Z:\EoSe-main-innovation_point
```

`Z:\EoSeg-main-official` 已确定为稳定 baseline；此前用户单独授权在此加入只读 Boundary-F1 指标及论文定性图脚本。该授权已用于基线评价，不构成后续继续在 official 优化模型/训练的许可。今后 C1 及任何结构、损失、训练优化仍只在 innovation_point。

## 2. 两个目录的严格边界

| 目录 | 用途 | 是否允许修改 |
|---|---|---:|
| `Z:\EoSeg-main-official` | ViT-L Protocol B 稳定 baseline；保留已授权的 BF1/作图材料 | 模型/训练冻结；新改动须用户另行明确授权 |
| `Z:\EoSe-main-innovation_point` | 后续创新点、消融和训练 | 是 |
| `Z:\Warwick_QU_Dataset` | 原始 GlaS 数据 | 否 |
| `C:\develop\github\task_MyProfessor\26.08\EoSeg-main*` | 本地备份/对照 | 不作为主实验目录 |

innovation_point 已复制 official 的代码、配置和数据加载内容，但没有复制 runs。两边的 checkpoint、日志、图片和汇总结果必须完全分开。

## 3. 当前 baseline 作为 C0-stable

当前对创新点的公平比较基准不是原始的未稳定 clean 运行，而是已经验证可收敛的：

```text
C0-stable = DINOv2 ViT-L + mask-only + GroupNorm + gradient clipping
```

固定配置来源：

```text
Z:\EoSeg-main-official\configs\glas\vit_query_mul_scale_fusion_protocol_b.yaml
Z:\.codex\BASELINE_CHANGE_RECORD_VITL_MASKONLY_GN.md
```

固定项：

```text
loss = Weighted Dice(0.6) + BCE(0.4)
lr = 5e-5
backbone_lr = 1e-5
decoder_lr = 1e-4
llrd = 0.8
warmup_steps = [500, 1000]
gradient_clip_val = 1.0
physical batch = 2
gradient accumulation = 7
effective batch = 14
seed = 0
threshold = 0.5
max_epochs = 1000
EarlyStopping patience = 300
```

任何创新实验都必须在 innovation_point 中从这一状态开始，不能悄悄切换回旧的 1e-3 单学习率、BatchNorm 或 class-head 像素融合。

## 4. 创新变量规则

本节的结构变量限制适用于 C2；C1 是预先声明的 Loss 单变量例外（仅增加 ABL）。每次实验只能声明一个主要变量，例如：

```text
最后三层融合方式
层间差异建模
query 引导权重
边界偏置
局部/窗口注意力
```

同一实验不得同时修改：

```text
ViT-L backbone 或预训练权重
mask-only 稳定化路径
GroupNorm
loss
数据划分
输入尺寸
optimizer / scheduler / 学习率
seed
阈值和指标实现
```

如果创新本身必须改变其中一项，必须新建独立实验组并明确命名，不能将增益归因于 Decoder 创新。

## 5. 数据与评估协议

继续使用作者 Protocol B：

```text
每折 Train_Folder 训练
同折 Test_Folder 每 epoch 验证
按 val_dice 选择 checkpoint
训练结束在同一 Test_Folder 做最终 single-view test
```

结果必须标注：

```text
author_protocol_b_test_as_val
single-view
threshold=0.5
TTA=false
```

TTA 只能作为补充，不能用于 checkpoint 选择、调参或 baseline 定义。

## 6. 固定 seed 集合实验

创新点尚未最终确定前，只使用 seed=0 快速筛选。创新点确定后，才进行固定 seed 集合实验：

```text
建议集合：0, 42, 3407
```

运行规则：

```text
每个模型 × 每个 fold × 每个 seed 独立运行
每个 fold 先对多个 seed 求均值
再对五个 fold 均值计算 mean ± sample std
```

seed 必须写入 run name、metadata 和论文记录。不得覆盖此前 seed=0 的 baseline。

## 7. 结果保存和启动入口

innovation_point 的结果只能保存到：

```text
Z:\EoSe-main-innovation_point\runs
```

每个实验必须有独立配置和启动路径，例如：

```text
Z:\EoSe-main-innovation_point\configs\glas\<experiment>.yaml
Z:\EoSe-main-innovation_point\configs\glas\run_<experiment>.sh
```

当前 baseline 的原始启动路径只作为参考，不得直接用 official 路径训练创新模型。

此规则包括 C1、C2 和 C3：共用基础 Python 实现可以，但每组的配置/启动入口/run namespace 必须相互独立，并显式记录自己的启用开关；不得通过覆盖同一个 YAML、改同一个 run 名称或从其他创新组 checkpoint 热启动来制造“消融”。

## 8. 有效训练记录

每次产生正式可分析结果的训练都要在 `Z:\.codex` 下的 Markdown 文档中记录：实验名、代码目录、配置入口、fold、seed、checkpoint、checkpoint 选择规则、single-view/TTA、threshold、Dice、IoU、mean±std 和本次唯一改动。失败运行、冒烟测试和未产生可用 checkpoint 的运行不计入论文结果。
## 9. 服务器命令规则

Codex 只提供 Linux/服务器终端命令；用户在服务器终端执行。训练前必须先核对当前目录、配置路径、run suffix 和 seed，防止误写 official。

## 10. Boundary-F1 Plan 纳入全局优先级

原计划：`Z:\EoSeg-main-official\EoSeg_BoundaryF1_Baseline_Codex_Plan.md`。执行 BF1、C0/C1 边界比较或论文评价前须读原计划及本节。原计划第 8 节的本轮 official 实现覆盖更早“只放 innovation_point”的路径设想；这只是过去一次用户授权，不覆盖本文件第 1、2 节的后续创新工作区规则。原计划第 1、8 节“尚待重训”是写作时状态，不可当成当前结果。

正式口径固定为 `author_protocol_b_test_as_val`、fold0–4、每折 132 train/33 Test_Folder、seed=0、single-view、224×224、预测阈值 0.5。各折仍仅按 Test_Folder `metrics/val_dice` 选 best checkpoint；BF1 不参与 loss、梯度、scheduler、early stopping、checkpoint monitor 或超参搜索。使用现有前景概率和 GT，不另造 mask 融合、极性校正或 GT 依赖后处理。轮廓为 8 邻域内边界，双向匹配容差为**欧氏距离 ≤2 px**（不是 5×5 方形距离）；双空 BF1=1、单空 BF1=0。逐图计算、折内图像宏平均、五折 mean ± 样本标准差 `ddof=1`；保留逐图/逐折记录。Dice/IoU 与 BF1 用同一 checkpoint 和预测，任何指标回归不一致先排查，不准用 BF1 重选 checkpoint 或调阈值/容差。多 seed 时每折先对相同 seed 集合求均值，再算五折标准差。

已完成的 C0 `_bf1_metric_check` 五折是与旧 C0 分开的独立 run，汇总及每折 checkpoint 见 `Z:\EoSeg-main-official\BF1_METRIC_CHECK_5FOLD_RESULTS.md`。它的 Dice/IoU/BF1 五折均值与样本标准差分别为 `0.902661693±0.009420227`、`0.826947749±0.015472934`、`0.649110675±0.028597304`；这些数只归属该 run，不能转写成旧 run 或 C1 结果。C1 正式五折尚未核实，不预填。C1 只在 innovation_point 加 ABL，沿用相同 BF1 实现与同折配对，主表至少列每折 C0/C1 Dice、IoU、BF1 及 `ΔBF1=C1−C0`，另列五折统计；BF1 上升而 Dice/IoU 下降也须如实报告。Precision/Recall 可作为补充。

BF1 的实现/评估门禁：相同 mask 得 1；1 px 平移仍在容差内；明显超过 2 px 的错位应下降；双空/单空、图像边缘和不同 batch size 的宏平均都须测试。使用冻结 best epoch checkpoint，不以 `last.ckpt` 偷换；同 checkpoint 上原 Dice/IoU/其他指标须与旧评估数值回归一致。正式输出保留逐图 BF1 CSV、逐折宏平均、五折样本标准差、run/config/checkpoint/manifest/seed/阈值/代码版本。今后 C1/C2/C3 沿用同一实现；新增指标不得成为新训练变量。

## 11. Paper Qualitative Plan 纳入全局优先级

原计划：`Z:\EoSeg-main-official\EoSeg_Paper_Qualitative_Visualization_Codex_Plan.md`。论文证据以五折配对 BF1 为主、固定 fold0/train_3 的图为辅；不可用单图替代总体结论。计划第 9、10 节的较新执行记录覆盖第 4、8 节过去的路径和面板安排，且不扩大 official 的未来改动权限。

唯一案例为 fold0 `Test_Folder` manifest 的 `sample_id=train_3`；缺失时报错，不回退 `sample-index 0`，不按 C1 效果换图。C0 与 C1 使用同一图像/GT、预处理、阈值、确定性单次推理、展示尺度。完整六面板固定 `(a) Original`、`(b) GT`、`(c) C0 mask`、`(d) C1 mask`、`(e) C0 boundary overlay`、`(f) C1 boundary overlay`；C1 checkpoint 未核实时只生成标记 `(a),(b),(c),(e)` 的 C0-only 预览，绝不伪造 `(d),(f)` 或把 `(e)` 重编号。GT 青色 `#00DCFF`、预测橙红 `#FF4A00`、显示笔画交叠黄色 `#FFEB00`；显示交叠不等于 BF1 匹配。

局部放大图采用原图坐标 ROI `(310,210,455,455)`；C0/C1 必须同一 ROI、缩放、线宽和颜色，不得因 C1 改选。BF1 永远在 224×224 的正式预测空间计算；原图尺寸 nearest-neighbour 放大和轮廓加粗仅为展示。metadata/caption 记录协议、折、样本 ID、seed、配置、两个 checkpoint、阈值、BF1 参数、ROI、输出分辨率、代码版本；完整图优先 PNG/TIFF 600 DPI 并保留 PDF/SVG。当前仅确认源码检查，不能把尚未在服务器推理生成的图片写成已完成。

执行顺序：先核对五折 C0 评价材料及同 checkpoint 指标回归；在 innovation_point 完成 C1 的正式五折与 BF1 配对统计；最后用固定 `fold0/train_3` 的 C0/C1 checkpoint 生成六面板和同 ROI 局部放大图，并核对面板预测来源及 `train_3` 的 per-image BF1。C1 缺席时仅允许 C0-only 部分图。当前 official 下已有绘图代码/结果路径保持为历史基线展示材料，后续改进版脚本和新增创新实验结果写入 innovation_point 自己的目录，不回写 official。

## 12. 冲突处理与读法

两份 Plan 是历史累积文档：较早的“未实现/待运行/仅 innovation_point”等叙述，以较新的执行状态和本全局规则为准；**但 official 的历史例外仅限既有只读指标与定性展示，后续 baseline 优化一律转 innovation_point**。`PAPER_INNOVATIONS_MEMORY.md` 中早年的 TTA 主口径、68/17/80 split 与 ViT-L 坍塌诊断均是历史实验，不适用于当前稳定 ViT-L Protocol B。优化器实际显式学习率是 backbone `1e-5`、decoder `1e-4`；配置 `lr=5e-5` 是兼容默认值，不得误写成统一参数组学习率。若规则仍不一致，暂停该实验并核对实际配置与 checkpoint，不通过改变冻结变量来“对齐”结果。

## 13. Innovation 1（C1）计划纳入全局优先级

原计划：`Z:\.codex\EoSeg_innov1_Dice_BCE_ABLplan.md`。C1 的唯一研究变量是最终前景预测上的 Active Boundary Loss：`L_C1=0.6 L_Dice+0.4 L_BCE+λ_ABL L_ABL`。基础 `WeightedDiceBCE` 不删除、不重配权重；ABL 不是 Boundary Dice。C1 不修改 `EoSegMultiQSegFusion`、最后三层融合、mask-only 预测路径或 Decoder。单通道前景 logit `z` 转换为 ABL 的双通道 `[0,z]`，须验证前景 softmax 与原 sigmoid 等价；空边界、device-safe、AMP、loss 数值与反向传播都需单测。C1 只在 innovation_point 实现，official C0 只读。`λ_ABL=0.10` 是当前计划的首个候选，不是已经证实的最终值；若需调 λ，只在 Train_Folder 内部诊断，不通过 Fold0 Test_Folder 选值。

消融组与入口隔离规则：

| 组别 | ABL | Decoder 创新 | 独立入口与结果 |
|---|---:|---:|---|
| C0 | 关 | 关 | official 冻结 baseline；只读引用其配置/checkpoint，不覆盖 runs |
| C1 | 开 | 关 | innovation_point 专用 C1 YAML、启动脚本/命令、run 名称和 checkpoint |
| C2 | 关 | 开 | innovation_point 专用 C2 YAML、启动脚本/命令、run 名称和 checkpoint |
| C3 | 开 | 开 | C1/C2 独立验证后，另建组合组的 YAML、入口、run 名称和 checkpoint |

`C3` 只是现有 I1+I2 组合消融，不等于论文尚未确定的第三创新点 `I3`；未来 I3 及含 I3 的组合必须新建互不覆盖的配置、入口和实验组，不可复用 C3 名称。

共享 Python 代码允许通过配置显式关闭/开启组件，但不能让 C2 隐式带上 ABL、C1 隐式带上 C2，不能修改同一份 YAML 后继续复用旧组名，也不能以对方已训练 checkpoint 作为默认初始化。每次运行应保存展开后的配置、git 版本、预训练初始化来源、开关/λ、fold/seed、checkpoint 选择规则与输出目录；独立运行 C0/C1/C2/C3 的测试命令应能重建完整消融。正式训练前做 `λ_ABL=0` 对 C0 的数值回归，训练仍沿用同一五折 Protocol B、seed=0、single-view、阈值 0.5、`metrics/val_dice` 选 checkpoint，报告各折 Dice/IoU/BF1 及五折 mean ± sample std，并如实标记 Test_Folder 同时选模与测试。历史 ViT-S/Train-Test/TTA 的 92.56%/86.66% 不得混作新 C1 的 ViT-L 五折成绩。本节只确定规则和入口设计，不表示 C1 已实现或已训练。
