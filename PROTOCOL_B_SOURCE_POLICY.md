# Author Protocol B 规则

## 数据划分

每个五折目录固定为：

```text
datasets/GlaS_5fold/fold{0..4}/Train_Folder
datasets/GlaS_5fold/fold{0..4}/Test_Folder
```

每折 Train_Folder 为 132 张，Test_Folder 为 33 张。

## 训练和选择

- Train_Folder 用于训练。
- 每个 epoch 在 Test_Folder 上计算 `metrics/val_dice` 和 `metrics/val_iou`。
- ModelCheckpoint 监控 `metrics/val_dice`，保存最佳 checkpoint。
- EarlyStopping 若存在，仍以 Test_Folder 的 `val_dice` 为监控量。
- 训练结束后继续使用保存的最佳 checkpoint 在同一 Test_Folder 上执行 test。

## 评估声明

该协议忠实复现作者脚本，但 Test_Folder 同时参与 checkpoint 选择和最终评估，因此结果必须标记：

```text
protocol=author_protocol_b_test_as_val
split=Test_Folder
checkpoint_selection=Test_Folder/val_dice
```

它不等同于独立测试集泛化结果；其他五折划分不属于当前规则。

## 训练参数

- Backbone：`vit_large_patch14_reg4_dinov2`。
- 输入：224×224；threshold=0.5；train_seed=0。
- 稳定训练固定 AdamW、LLRD=0.8、warmup=[500,1000]；配置兼容值 `lr=5e-5`，实际显式参数组为 `backbone_lr=1e-5`、`decoder_lr=1e-4`。作者脚本的 `lr=1e-3` 仅作 source-reference。
- 物理 batch 与梯度累积统一为 physical=2、accumulate=7、effective=14。

## 报告

作者 B 的 val 监控是 single-view。任何 TTA 结果必须单独标记为 `TTA=4-flip`，不得与 single-view 结果混写。


