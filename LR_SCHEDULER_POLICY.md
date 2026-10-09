# Protocol B 学习率与 Scheduler 锁定规则（`OPTIMIZER_PROTOCOL.md` 的索引）

统一使用 AdamW + `TwoStageWarmupPolySchedule`，interval=step。

稳定 Protocol B 的固定参数：

```yaml
lr: 5e-5
backbone_lr: 1e-5
decoder_lr: 1e-4
weight_decay: 5e-4
llrd: 0.8
llrd_l2_enabled: true
lr_mult: 1.0
warmup_steps: [500, 1000]
poly_power: 0.9
```

ViT 各 block 以 `backbone_lr=1e-5` 为基础使用 LLRD；非 ViT/Decoder 以 `decoder_lr=1e-4` 为基础。`lr=5e-5` 是兼容默认值，不代表实际统一学习率。完整参数组解释以 `OPTIMIZER_PROTOCOL.md` 和已冻结 YAML/启动脚本为准。所有 fold、C0/C1/C2/C3 和创新点实验保持一致。

作者脚本的 `lr=1e-3` 是忠实源码参考值；需要单独 source-exact 复现时才允许使用，不得与稳定结果合并。

