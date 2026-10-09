# EoSeg 五折 Protocol B 稳定基线 Boundary F1 评价计划

> 2026-09-29 修订。此文件规定后续 BF1 实施与评价；本次只修订计划，不修改代码、不运行模型。

## 1. 目标与当前事实

为现有稳定基线 C0 增加 **Boundary F1（BF1）评价指标**，定量描述腺体边界误差，并为创新点 1（C0 + Active Boundary Loss，简称 C1）提供同口径比较。BF1 只用于评价，不参与 loss、反向传播、checkpoint 选择或超参数搜索。正文主表保留 Dice、IoU，并加入 BF1；Precision、Recall 可作为补充记录。

当前 C0 已完成 `seed=0` 的 **五折**训练，五折 best checkpoint 都存在，且各折 metadata 显示 `checkpoint_selection_source=best_epoch_exact`。BF1 是对已训练预测的补充评价，默认先用这些冻结 checkpoint **离线补算五折 BF1**；不为增加一个指标而重新训练 C0。若后续确需重训，须另建独立 run、保持本节参数完全相同，并明确区分新旧结果。

本计划的永久位置虽在 `Z:\EoSeg-main-official`，后续指标代码和评价输出应放在 `Z:\EoSe-main-innovation_point`；official 的模型、训练和现有 runs 保持冻结，只读加载其 checkpoint。C1 未来也使用同一份 BF1 实现。

## 2. 实际数据协议：五折 Train_Folder / Test_Folder

每折从 165 张 GlaS 图像的固定 manifest 读取：

| Fold | Train_Folder | Test_Folder | C0 best epoch |
|---:|---:|---:|---:|
| 0 | 132 | 33 | 711 |
| 1 | 132 | 33 | 729 |
| 2 | 132 | 33 | 698 |
| 3 | 132 | 33 | 672 |
| 4 | 132 | 33 | 514 |

路径模式：

```text
datasets/GlaS_5fold/fold{0..4}/Train_Folder/samples.txt
datasets/GlaS_5fold/fold{0..4}/Test_Folder/samples.txt
source_dir = ../Warwick_QU_Dataset
```

每折 `Train_Folder` 用于训练；**同一折的 `Test_Folder` 既用于每 epoch 的 single-view `val_dice`、选择 best checkpoint，也用于训练结束后的最终 single-view test**。这里没有第三个独立 `Val_Folder`，也没有可称为独立盲测的 `Test_Folder`。结果必须标注 `author_protocol_b_test_as_val`；论文不能把这组数描述成未参与模型选择的独立测试泛化性能。

BF1 在每折已选定的 C0 best checkpoint 上评估 33 张 `Test_Folder` 图像，再汇总五折。Fold0 只作指标冒烟与回归检查，不能代替五折主结果。五折结果必须完整保留，不能选取表现最好的折。

## 3. 与冻结 C0 完全一致的参数

以下取自 official 当前 `configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml`、`configs/glas/run_protocol_b.sh`、fold0 `hparams.yaml` 和五折 run metadata；执行时再次核对配置与实际 checkpoint：

| 项目 | 冻结值 |
|---|---|
| 网络 | `EoSegMultiQSegFusion`，`num_classes=2`，`num_q=2` |
| Backbone | `vit_large_patch14_reg4_dinov2`，服务器 timm 缓存的 `dinov2_vitl14_reg4_pretrain.pth` |
| 输入与预处理 | `224×224`；图像 bilinear resize、mask nearest-neighbour resize；沿用现有归一化和训练增强 |
| 预测路径 | `mask_only_training_enabled=true`，query mask logits 求平均后 sigmoid；`decoder_norm=group` |
| C0 loss | `WeightedDiceBCE(dice_weight=0.6, bce_weight=0.4)`；ABL/Boundary Loss 关闭 |
| Optimizer | AdamW，`weight_decay=5e-4` |
| 学习率 | 配置 `lr=5e-5`；实际显式参数组 `backbone_lr=1e-5`、`decoder_lr=1e-4` |
| 层学习率 | `llrd=0.8`，`llrd_l2_enabled=true`，`lr_mult=1.0` |
| Scheduler | `TwoStageWarmupPolySchedule`，`warmup_steps=[500,1000]`，`poly_power=0.9` |
| Batch | physical batch `2`，`accumulate_grad_batches=7`，effective batch `14` |
| 数据加载 | `num_workers=8`，`ignore_idx=255`；各折沿用同一增强、归一化和样本配对规则 |
| 训练控制 | `max_epochs=1000`，`gradient_clip_val=1.0`，`precision=16-mixed`，每 epoch 验证一次，`num_sanity_val_steps=0`，EarlyStopping patience `300` |
| 随机种子 | `seed_everything=0` |
| Checkpoint | `ModelCheckpoint` 监控 `metrics/val_dice`，mode=max，`save_top_k=1`，`save_weights_only=true`，`save_last=true`；五折已选 epoch checkpoint |
| 评价 | single-view、TTA=false、`metric_threshold=0.5`，原 Dice/IoU/Precision/Recall 计算路径不变 |

`lr=5e-5` 是总体配置及兼容默认值；当前优化器显式按 backbone/decoder 分组，不能误写为“全部参数统一使用 5e-5”。启动脚本的三个学习率默认值与 YAML 一致，并在命令行显式覆盖；五折现有 metadata 也记录了这三个值。不可恢复旧的 `1e-3` 单学习率、BatchNorm 或 class-head 像素融合路径。

## 4. BF1 的唯一正式定义

在数据加载器输出的 **224×224** 原始评价空间计算，不在论文展示图放大后的尺寸上计算。直接复用现有 foreground probability 与 binary target：

```text
prediction = foreground_probability >= 0.5
GT = binary_target > 0.5
```

上述比较符与当前 `MedicalBinarySegmentation._batch_metrics` 保持一致。先为每张图从二值 mask 取 **8 邻域内边界** `B = mask & ~erosion_3x3(mask)`；图像外按背景处理。对两条边界进行对称匹配，容差冻结为 **欧氏距离 ≤2 像素**：预测边界像素落在 GT 边界的半径 2 像素圆盘邻域内，计入 Boundary Precision；GT 边界像素落在预测边界的同样邻域内，计入 Boundary Recall。实现可用固定圆盘核，不需要新依赖；普通 `5×5 max_pool` 是方形/Chebyshev 邻域，不能在文中称作欧氏 2 像素。

```text
Boundary Precision = matched predicted boundary pixels / all predicted boundary pixels
Boundary Recall    = matched GT boundary pixels / all GT boundary pixels
BF1                = 2 * Precision * Recall / (Precision + Recall)
```

特殊情况固定：两边界都为空时 BF1=1；仅一边为空时 BF1=0。其他 `Precision+Recall=0` 情况 BF1=0；结果必须在 `[0,1]` 内，且无 NaN/Inf。边界显示脚本的 3×3 加粗和黄色重叠笔画**仅用于画图**，不能作为 BF1 匹配像素。

一张图先得一个 BF1；每折 33 张图取算术平均（image-level macro）。五个 fold 的 BF1 再报 `mean ± sample std (ddof=1)`，并附五个逐折值。后来如采用多个训练 seed，每折先对相同 seed 集合求均值，再对五折均值计算 fold 标准差；seed 标准差另报。

## 5. 指标接入范围与回归检查

后续实施时在 innovation_point 的评价路径接入 `metrics/val_boundary_f1` 与 `metrics/test_boundary_f1`；复用同一个 BF1 函数用于五折 C0、C1 和后续创新模型。其输入来自现有 `_foreground_logits_and_probs` / binary target，不能另写一套 mask 融合、极性校正或 GT 依赖后处理。BF1 不进入 `criterion`、optimizer、scheduler、training loss 或 checkpoint monitor。

首先做边界单元检查：相同 mask 得 1；1 像素平移仍匹配；超过 2 像素的明显错位下降；一边空/两边空按第 4 节处理；图像边缘的轮廓不越界。另核查 batch size 改变不影响逐图 BF1 和宏平均。

再用 fold0 现有 best checkpoint 做指标回归：在相同 fold0 `Test_Folder`、相同 single-view 设置下，对照 official 旧 `protocol_b_test.txt` 中 Dice、IoU、Precision、Recall；新增 BF1 后原指标应在数值容差内一致。若出现明显差异，先排查 checkpoint 载入、mask-only、GroupNorm、预处理、阈值和指标聚合，不能继续汇总五折。不要为了使 BF1 好看而调整阈值或 2-pixel tolerance。

回归通过后依次评估 fold0–fold4 原 best checkpoint。使用各折已有 `protocol_b_metadata.txt` 和实际 epoch checkpoint；不可用 `last.ckpt` 偷换 best，也不能用 BF1 重新选 checkpoint。输出保存到 innovation_point 的独立 BF1 评价目录，并记录每折 `protocol`、manifest、样本数、代码/配置版本、checkpoint、seed、阈值、边界定义、容差、BF1、Dice、IoU、Precision 和 Recall。

## 6. 与创新点 1 及后续模型的比较

本计划先得到 C0 的五折 BF1 基准；C1 的正式训练完成后，按**相同五折 manifest、seed、best checkpoint 选择和 BF1 实现**评价 C1。逐折记录 `ΔBF1 = BF1(C1) − BF1(C0)`，连同 Dice/IoU 的逐折差值与五折统计报告。若 BF1 上升但 Dice/IoU 下降，或改善只集中在少数折，必须如实解释，不能仅根据 fold0 图得出总体结论。

BF1 一旦成为论文正式指标，创新点 2、创新点 3 及其组合也沿用同一定义、阈值和容差。主表至少含 Dice、IoU、BF1；Precision、Recall 可放补充表。Boundary F1 仍只是评价指标，不因为后续结果而改为 checkpoint monitor 或训练目标。

定性图按 `EoSeg_Paper_Qualitative_Visualization_Codex_Plan.md` 使用 fold0 `Test_Folder` 中固定的 `sample_id=train_3`，输出六面板及同 ROI 局部放大图。该图只辅助展示边界现象；总体论证以五折 BF1 与 C0/C1 配对差值为准。

## 7. 后续执行顺序与验收

1. 核对五折 C0 checkpoint、manifest、参数与当前冻结配置；确认各折 132/33、五个 Test_Folder 无重叠且并集为 165 张；确认 innovation_point 代码可严格加载 C0 权重。
2. 在 innovation_point 实现唯一 BF1 函数及评价接入，完成第 5 节的单元与 fold0 指标回归检查。
3. 离线评价已有五折 C0 best checkpoint，保存每图 BF1、每折宏平均和五折 `mean ± sample std`；不重训 C0。
4. C1 五折正式 checkpoint 就绪后，用同一 evaluator 计算五折 C1 与逐折差值，再制作固定 `train_3` 的论文图。
5. 交付完整的参数/版本/路径/指标记录。验收条件是五折均有 BF1，原 Dice/IoU 等指标未被新增评价改变，BF1 未参与训练或选 checkpoint，且结果均标注 `author_protocol_b_test_as_val`、single-view 和 threshold=0.5。

本次修订纠正了旧版“独立 train/valid/test”措辞、仅 Fold0 重训的路径以及预测阈值符号；对照当前稳定 baseline 补全了 mask-only、GroupNorm、分模块学习率、LLRD、AdamW/weight decay、scheduler、batch/梯度累积、AMP、早停和梯度裁剪。实际代码与运行留待用户后续指令。

## 8. 执行状态补充（2026-09-29）

根据本次明确指令，本轮实现范围改为 `Z:\EoSeg-main-official`，不修改 `Z:\EoSe-main-innovation_point`。这是对原第 1、5、7 节“评价代码只放 innovation_point”的本轮授权例外；后续若需恢复 official 冻结状态，以用户新的明确指令为准。

本轮增加的 BF1 是只读评价计算：仅在 validation/test 前向之后根据既有预测与 GT 计算，不加入 criterion、梯度、优化器、scheduler、训练采样或 checkpoint monitor。checkpoint 仍仅按 `metrics/val_dice` 选择。Official 的 C0 配置、模型结构和 loss 参数不因 BF1 修改。

用户明确要求在 BF1 完成后重新跑五折，因此执行顺序补充为：

1. 对已冻结的五折 C0 best checkpoint 做 test-only BF1 评估，并为每折保存 `boundary_f1_per_image.csv`；同一个 checkpoint 的 Dice、IoU、Jaccard、Precision、Recall、Accuracy、F2 和 loss 必须与既有 `protocol_b_test.txt` 回归一致。
2. 回归通过后，以新的 `_bf1_metric_check` run suffix 从头重跑五折；训练参数及 Protocol B 不变，BF1 不参与选 checkpoint；每折测试后同样保存逐图 BF1 CSV。
3. 把这次重跑和原始 C0 五折逐折比较，并单独记录新 run 的 BF1、Dice、IoU 等统计；不得覆盖原始 baseline runs 或把两次训练混为同一批结果。

本地 BF1 数学单元测试 8 项已通过。server 端 checkpoint 评估和五折重训尚待用户在 Linux 服务器终端执行；在服务器结果产生前，不填入任何新 BF1 或重跑结果。当前没有可用的 C1（C0+ABL）正式 checkpoint，因此定性脚本在未给出 `--abl-ckpt` 时只生成标注清楚的 C0 预览与 ROI 放大图，不生成虚构的 C1 面板。
