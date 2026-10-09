# 论文创新点与实验口径长期备忘录

历史记录原整理于 2026-09-17；2026-09-30 校正当前协议说明。用途：论文撰写、导师汇报和后续实验归档。历史数字不得覆盖 `Z:\.codex\PROJECT_EXPERIMENT_PRIORITY.md` 的现行规则。

历史早期 Train/Test 与 68/17/80 实验曾以 TTA=4-flip 报告；**当前论文 ViT-L 五折 Protocol B 主口径为 single-view、TTA=false、阈值=0.5**。如做四翻转 TTA，只能单独作为补充，不可与五折主表混写。

## 1. 第一创新点：Dice/BCE + Active Boundary Loss

实际损失是 WeightedDiceBCE 加 ActiveBoundaryLoss：
    L = WeightedDiceBCE + λ_ABL * ActiveBoundaryLoss
    WeightedDiceBCE = 0.6 * Dice + 0.4 * BCE

用户确认的历史论文结果：Dice=92.56%，IoU=86.66%。当前仓库没有找到一份同时直接写出 0.9256/0.8666、checkpoint 和完整命令的单独结果文件，因此定稿前应再核对原始终端输出；本数值标记为历史用户报告值。

### 已核对的历史验证方式

最初的 runs/glas_vit_query_boundary_bias_origloss/version_1/hparams.yaml 显示 train_dir=Train_Folder，val_dir=Test_Folder，test_dir=Test_Folder，dataset_dir=null，test_split=null。

因此历史流程是：Train_Folder 训练 → Test_Folder 验证/观察 → Test_Folder TTA 评估。这是“train+test”口径，不是独立的 train+val+test；Test_Folder 曾参与 checkpoint 选择或参数观察，不能无条件称为独立测试集结果。

TRAIN_TEST_DIAGNOSTIC_RESULT.md 的近似复现也使用该口径、四翻转 TTA 和阈值 0.5，得到 Dice=0.9248560555、IoU=0.8656302921，并明确标记为复现诊断。

`datasets/GlaS/research_split_seed0` 的 68/17/80 是另一项历史规范 split 实验，不能作为当前唯一活动五折 Protocol B 的数据划分。当前每折 132 train/33 Test_Folder，后者同时用于 val/checkpoint 选择和 test，必须披露该选择偏差。

## 2. 第二创新点：末层三层 Query 引导逐像素融合

原始 EoSeg 对最后三个 ViT block 使用每个 query 一组全局层权重；创新方案改为每个 query、每个像素位置一组三层权重，使边界和中心区域可以选择不同层特征。最后三层由 fusion_layer_offsets=[-3,-2,-1] 指定；ViT-L 对应 block 22、23、24。导师指出浅层边界特征不能从末三层选，因此使用独立早期 block 4，即 shallow_feature_index=3。

逐像素 QFCD 包含固定/移位窗口邻域注意力、query 条件路由和纯逐像素融合。接口、权重形状和归一化测试通过，但实际收益很小，通常接近全局融合。代表性历史结果约 Dice=0.9241–0.9244、IoU=0.8640–0.8646；修正浅层选择后的纯逐像素分支曾达到 Dice=0.925193、IoU=0.866036，仍略低于历史目标和 lambda=0.5 全局边界基线 0.925375/0.866377。

固定权重诊断：原始 query-global 验证 Dice/IoU=0.927809/0.868409；固定 [0.5,0.5,0] 为 0.927560/0.867836。GT Oracle 的 Dice≈0.957034、IoU≈0.917608 只是理论上限，不能作为模型测试结果。

## 3. 层间差异感知融合

曾加入共享投影/归一化、层间差异、query 调制和逐像素层修正。早期 difference_residual 的问题是差异通道部分线性冗余，且 gamma=0.001/0.01 把最终变化和梯度压得很小。诊断得到 mean|W_diff-W_global|≈0.38670，但 mean|W_final-W_global|≈0.003961，路由器有差异而最终混合几乎仍是全局融合。

规范 split 的 FDR-QFCD：seed=0 baseline/FDR=0.918149/0.853992 与 0.918709/0.854838；seed=1 baseline/FDR=0.915887/0.850322 与 0.916564/0.851434。增益只有约 0.056–0.068 Dice 个百分点和 0.085–0.111 IoU 个百分点，不能表述为显著结构突破；总体没有稳定超过纯逐像素融合。

## 4. 其他实验摘要

| 分支 | 结论 |
|---|---|
| 固定 Sobel boundary_bias，lambda=0.5 | 最稳定边界基线；规范 split 测试约 0.918149/0.853992 |
| learned boundary bias | 约 0.918123/0.853999，未证明稳定增益 |
| lambda clamp | 约 0.915200/0.849016，退化 |
| lambda=0.75 | train/test 口径约 0.924925/0.865763，未稳定超过 0.5 |
| boundary head-only | 0.918095/0.853911，基本中性 |
| SBEM 浅层增强 | 0.914592/0.848004，低于基线 |
| disagreement adaptive router | 约 0.925474/0.866544，提升极小 |
| 后处理 | 验证集略升、测试集下降，不采用 |
| 非连续层选择 | 更早 block 破坏 query 注入/训练路径，结果下降 |
| ViT-L/14 早期不稳定配置 | 权重加载成功，但早期运行在 epoch 10–14 全前景塌缩，最佳验证 Dice 约 0.655；这是历史失败配置，不代表当前 mask-only + GroupNorm + 分模块学习率的稳定 ViT-L baseline |

## 5. 论文写作口径

第一创新点可写为“WeightedDiceBCE 与 Active Boundary Loss 联合优化”，但 92.56%/86.66% 必须注明历史 Train_Folder→Test_Folder TTA 口径，不能与后续独立 68/17/80 split 的 test 指标混写。

第二创新点建议写为“尝试将末层三层 query-global 融合改为 query 引导逐像素跨层融合，并进一步进行层间差异感知调制”。当前只能作为结构消融和失败分析，不能宣称稳定性能提升。

当前正式五折结果记录 protocol=`author_protocol_b_test_as_val`、fold/split、single-view/TTA、threshold、checkpoint、fusion_mode、loss、Dice、IoU 和 BF1；若有 boundary_lambda 才记录该参数。Protocol B 的 Test_Folder **按作者协议确实用于 `val_dice` 选 checkpoint**，对此必须披露，不能错误记为完全未参与选择；但不许用 Test_Folder 再做阈值、lambda、结构或 BF1 容差搜索。C0 已完成的独立 `_bf1_metric_check` 五折结果见 `Z:\EoSeg-main-official\BF1_METRIC_CHECK_5FOLD_RESULTS.md`；C1 结果尚未核实。后续优化只在 innovation_point。
