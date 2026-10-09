本篇文档记录了为了防止前景崩塌在代码文件所进行的更改，算是我的baseline


本次更改的精度是：
fold0的dice是90.87%，iou是83.85%
fold1的dice是90.26%，iou是82.55%
fold2的dice是89.02%，iou是80.71%
fold3的dice是89.76%，iou是81.8%
fold4的dice是91.43%，iou是84.57%
平均的dice是90.27%+-0.94%,iou是82.7%+-1.54%

由于更换了vit和backbone,所以导致直接对源代码进行更改会出现前景坍塌，经过核实是class head模块出现了像素级语义融合
具体来说是class head和mask的发生时机出现在一起，导致对其他没有进行mask的区域也进行了像素语义级别的融合

还加了梯度裁剪用于抑制训练初期的训练波动，以及使用groupnorm，没有使用batchnorm

以及总学习率是5e-5,ViT-L backbone_lr的学习率是1e-5，Decoder decoder_lr的学习率是1e-4
原本代码中只有一个总学习率1e-3
LLRD = 0.8
warmup = [500, 1000]


这次不算完全没有改动的baseline，后续的改进都要在这个基础上进行，可以先在这个模型上面叠加创新点，然后再看最后的论文怎么写吧




后续引进其他文档的内容：
# ViT-L 稳定化 Baseline 变更记录

状态：当前 Protocol B 五折 baseline（已冻结）

本文档记录当前 baseline 相对原始 EoSeg 配置的有效改动，以及每项改动在代码中的位置。

## 1. Mask-only 空间预测

文件：`Z:/EoSeg-main-official/training/medical_binary_segmentation.py`

- 构造函数新增 `mask_only_training_enabled` 参数（约第 107 行）。
- 当该开关为 `true` 时，`_foreground_logits_and_probs` 不再把全局 `class_logits` 融入每个像素，而是直接对 query mask logits 求平均：

  ```python
  foreground_logits = mask_logits.mean(dim=1, keepdim=True)
  foreground = torch.sigmoid(foreground_logits)
  ```

- 代码位置约第 185–200 行。
- 目的：避免 class head 的全局前景先验导致整张图预测为前景（Recall=1、Precision≈0.49）。

配置：`Z:/EoSeg-main-official/configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml:50`

```yaml
mask_only_training_enabled: true
```

## 2. Decoder 使用 GroupNorm

文件：`Z:/EoSeg-main-official/models/eoseg.py`

- `EoSegMultiQSegFusion` 构造函数新增 `decoder_norm` 参数（约第 293–308 行）。
- Decoder 中原来的 BatchNorm2d 通过 `_make_decoder_norm` 统一构造（约第 319–344 行）。
- GroupNorm 实现位于约第 373–379 行。
- 当前使用 `GroupNorm(num_groups, num_channels)`，避免物理 batch size=2 时 BatchNorm 统计不稳定。

配置：`Z:/EoSeg-main-official/configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml:56`

```yaml
decoder_norm: group
```

## 3. 梯度裁剪

配置文件：`Z:/EoSeg-main-official/configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml:6`

```yaml
gradient_clip_val: 1.0
```

作用：限制 ViT-L 训练初期的梯度尖峰，属于辅助稳定措施，不改变模型输出结构。

## 4. 分模块学习率

配置文件：`Z:/EoSeg-main-official/configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml:40–42`

```yaml
lr: 5e-5
backbone_lr: 1e-5
decoder_lr: 1e-4
```

实现文件：`Z:/EoSeg-main-official/training/lightning_module.py`

- 参数接收和默认回退：约第 55–63、76–78 行。
- backbone 参数组及 LLRD：约第 109–152 行。
- decoder/非 backbone 参数组：约第 153–155 行。
- AdamW 使用上述参数组：约第 158–163 行。

当前含义：

- ViT-L backbone 基础学习率：`1e-5`；
- Decoder 学习率：`1e-4`；
- `lr=5e-5` 作为总配置中的基础值和兼容默认值，当前显式分组值优先。

## 5. LLRD

配置文件：`Z:/EoSeg-main-official/configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml:43–45`

```yaml
llrd: 0.8
llrd_l2_enabled: true
lr_mult: 1.0
```

实现位置：`Z:/EoSeg-main-official/training/lightning_module.py:134–148`。

作用：不同 ViT block 使用逐层衰减学习率，浅层更新更保守；当前 `lr_mult=1.0`，不额外放大特殊层。

## 6. Warmup 和训练控制

配置文件：`Z:/EoSeg-main-official/configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml:7–24、48`

```yaml
warmup_steps: [500, 1000]
max_epochs: 1000
check_val_every_n_epoch: 1
EarlyStopping.monitor: metrics/val_dice
EarlyStopping.patience: 300
```

实现入口：`Z:/EoSeg-main-official/training/lightning_module.py:165–169`。

含义：backbone 和非 backbone 参数采用分阶段 warmup；每 epoch 验证一次，`val_dice` 连续 300 次无提升则早停。

## 7. 五折启动脚本中的显式覆盖

文件：`Z:/EoSeg-main-official/configs/glas/run_protocol_b.sh`

- 默认分模块学习率：第 7–9 行；
- 默认 mask-only 和 GroupNorm：第 10–11 行；
- 写入每折 metadata：第 43–47 行；
- 通过 CLI 显式传给模型：第 51–57 行；
- 五折循环：第 17 行。

这样即使服务器命令没有额外传参，每个 fold 仍然使用当前冻结 baseline 参数。

## 8. Checkpoint 选择和测试

同一脚本约第 66–110 行：

- 按 `metrics/val_dice` 选择最佳 checkpoint；
- 如果自定义 `best_metric.txt` 与实际 checkpoint 文件名不一致，使用可用 epoch checkpoint 回退；
- 训练完成后自动执行 single-view test。

## 9. 当前 baseline 固定摘要

```text
Backbone: DINOv2 ViT-L
Loss: Weighted Dice(0.6) + BCE(0.4)
Mask path: mask-only query mask average
Decoder norm: GroupNorm
lr: 5e-5
backbone_lr: 1e-5
decoder_lr: 1e-4
LLRD: 0.8
warmup_steps: [500, 1000]
gradient_clip_val: 1.0
EarlyStopping patience: 300
Protocol: author Protocol B (Test_Folder as val/test)
Evaluation: single-view, threshold=0.5, no TTA
```

以上参数在后续创新实验中全部冻结；只有预先声明的创新变量允许修改。
