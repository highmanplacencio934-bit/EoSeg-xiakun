# Author Protocol B 选择规则

本文件是作者五折 Protocol B 的补充说明。

每折目录：

```text
datasets/GlaS_5fold/fold{0..4}/Train_Folder
datasets/GlaS_5fold/fold{0..4}/Test_Folder
```

执行顺序：

1. 使用 Train_Folder 训练。
2. 每个 epoch 在同折 Test_Folder 计算 `metrics/val_dice`。
3. 按 Test_Folder 的 `val_dice` 保存最佳 checkpoint。
4. 训练结束后用该 checkpoint 在同一 Test_Folder 执行 test。

Test_Folder 同时参与 checkpoint 选择和最终评估，因此必须标记：

```text
protocol=author_protocol_b_test_as_val
checkpoint_selection=Test_Folder/val_dice
selection_bias=true
```

不得将该结果描述为独立测试集泛化结果。当前不使用其他嵌套验证协议。

