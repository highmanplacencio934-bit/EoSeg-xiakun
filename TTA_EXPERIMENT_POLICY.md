# 项目评估口径规则

## 作者 Protocol B 主口径

- 五折主实验使用作者的 Train_Folder/Test_Folder 划分。
- 作者代码在 single-view 的 Test_Folder val_dice 上选择 checkpoint。
- 主表默认记录 single-view 结果，并明确标记 `author_protocol_b_test_as_val`。
- 二值化阈值固定为 0.5，不能通过 Test_Folder 重新搜索阈值。
- Test_Folder 参与选模，因此不能把结果称为独立测试集泛化。

## TTA 补充口径

- 如报告 TTA，必须使用四翻转：原图、水平翻转、垂直翻转、水平+垂直翻转。
- 每个增强结果先逆变换回原方向，再对前景概率求平均。
- TTA 结果必须单独标记 `TTA=4-flip`，不得与 single-view 结果混写。
- TTA 只能作为补充分析，不能用于 checkpoint、lambda、阈值或结构选择。

## 记录要求

所有论文结果必须注明 protocol、split/fold、checkpoint、checkpoint_selection、threshold、single-view/TTA、backbone、loss、fusion_mode、Dice 和 IoU。

