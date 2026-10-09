# Baseline 冻结清单（作者 Protocol B）

每次训练前必须核对本清单。当前唯一活动五折协议为 `author_protocol_b_test_as_val`。

## 冻结项

| 项目 | 状态 |
|---|---|
| GlaS 数据版本、图片 ID、Train_Folder/Test_Folder manifest | 🔒 |
| 输入尺寸 224×224、图像/mask resize、增强 | 🔒 |
| Backbone 与 DINOv2 预训练权重 | 🔒 |
| Physical batch、effective batch、梯度累积 | 🔒 |
| AdamW、学习率、weight decay、scheduler、warmup、LLRD | 🔒 |
| Max epoch、EarlyStopping、验证频率 | 🔒 |
| train seed=0 和 checkpoint 选择规则 | 🔒 |
| threshold=0.5、Dice/IoU/BF1 实现、后处理 | 🔒；BF1 已作为只读评价接入，不参与 loss/选模 |
| Python/PyTorch/timm/Lightning 版本 | 🔒 |
| baseline Git commit | 🔒 |
| Loss（仅 C1/C3 可加入 ABL） | 🔒 |
| 最后三层融合结构（仅 C2/C3 可修改） | 🔒 |

## Protocol B 选择规则

```text
Train_Folder -> train
Test_Folder -> per-epoch val_dice / checkpoint selection
同一 Test_Folder -> final test
```

该选择规则存在测试集泄漏，必须在论文中如实标注。不得把 Test_Folder 结果描述为独立测试集。

## 实验白名单

- C0：不改冻结项。
- C1：只增加 ABL。
- C2：只修改最后三层 Decoder 融合。
- C3：合并已独立验证的 C1 和 C2。
- 任何其他变量必须先声明新的实验组。

已获授权在 official 添加的 BF1 和定性图只用于冻结 C0 的评价/展示，不构成允许后续在 official 优化模型的例外。C1/C2/C3 的代码、配置、训练与进一步优化只在 innovation_point；两份原计划及优先级见 `PROJECT_EXPERIMENT_PRIORITY.md` 第 10–12 节。C0 `_bf1_metric_check` 与旧 C0 run 是两个独立实验，不得混写。


