# Optimizer 与学习率协议（Protocol B）

## 固定实现

- Optimizer：AdamW。
- Scheduler：`training.two_stage_warmup_poly_schedule.TwoStageWarmupPolySchedule`。
- Scheduler interval：step。
- ViT backbone 使用 LLRD 参数组，当前显式基础学习率 `backbone_lr=1e-5`；Decoder/非 backbone 使用显式 `decoder_lr=1e-4`。配置 `lr=5e-5` 只作兼容默认/回退值，不是所有参数组的统一实际学习率。
- 同一组对比实验不得更换 optimizer、scheduler、weight decay、warmup 或参数分组。

## 当前稳定 Protocol B 参数

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

这些参数已在 stable baseline YAML 与 `run_protocol_b.sh` 中明确配置。作者 `run_5fold.sh` 中的 `lr=1e-3` 只记录为 source-reference，不作为当前稳定实验默认值。`LR_SCHEDULER_POLICY.md` 是本文件的索引，不得另设冲突数值。

若要复现作者原始 1e-3，必须建立独立的 source-exact 实验组，不能覆盖稳定 Protocol B 结果。


