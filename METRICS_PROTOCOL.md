# 二值分割指标协议（作者 Protocol B）

所有五折实验使用同一套 Dice/IoU/BF1 实现、threshold=0.5 和后处理。BF1 是评价指标，不参与训练或 checkpoint 选择；完整操作要求见 `Z:\EoSeg-main-official\EoSeg_BoundaryF1_Baseline_Codex_Plan.md` 和 `Z:\.codex\PROJECT_EXPERIMENT_PRIORITY.md` 第 10 节。

## 数据口径

- Train_Folder 用于训练。
- Test_Folder 用于每 epoch 的 single-view val 监控、checkpoint 选择和最终 test。
- 由于 Test_Folder 参与 checkpoint 选择，结果必须标记 `author_protocol_b_test_as_val`。
- 不得把该结果描述为独立测试集泛化。

## 指标

预测和真值均按前景二值化，使用固定 Dice/IoU 公式；先逐图计算，再对图像取算术平均。

主记录必须包含：

```text
protocol, fold, split, checkpoint, checkpoint_selection,
single-view/TTA, threshold, backbone, loss, Dice, IoU
```

作者 val 监控采用 single-view；四翻转 TTA 按 `.codex/TTA_EXPERIMENT_POLICY.md` 单独报告，不与监控结果混写。

## Boundary-F1 固定定义与当前状态

- 在 224×224 正式评估空间，预测前景概率 `>=0.5`、GT `>0.5`；每图取 8 邻域内边界，图像外按背景。双向匹配使用欧氏距离 `<=2 px` 的圆盘邻域，不能把 5×5 方形 max-pool 误称欧氏 2 px。
- `BF1=2×BoundaryPrecision×BoundaryRecall/(BoundaryPrecision+BoundaryRecall)`；两边界均空为 1，仅一边空为 0，其余零分母为 0。逐图 BF1 后折内宏平均，五折 mean ± sample std (`ddof=1`)。多 seed 时各折先平均 seed。
- 已完成的 C0 `_bf1_metric_check` 五折见 `Z:\EoSeg-main-official\BF1_METRIC_CHECK_5FOLD_RESULTS.md`。与旧 C0 run 分开记录；C1 必须采用相同指标定义和同折配对，保留逐折 `ΔBF1`。图片黄色重叠笔画只是显示效果，不参与 BF1 计算。

