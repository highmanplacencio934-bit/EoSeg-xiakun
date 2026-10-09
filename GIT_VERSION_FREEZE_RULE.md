# Git 版本冻结与创新点记录规则

本文件是项目 `.codex` 全局实验规则的一部分。

## 目录范围

- `Z:\EoSeg-main-official`：已确认的 ViT-L Protocol B 稳定 C0 baseline（mask-only、GroupNorm、分模块学习率）。已有用户授权的只读 BF1 评价和定性图改动应与原训练版本区分记录；未经新的明确授权不得继续修改模型、Loss 或训练配置。
- `Z:\EoSe-main-innovation_point`：创新点代码。所有后续代码和配置修改都必须纳入 Git 版本记录。

## Baseline 冻结

baseline 完成后建立独立分支并立即提交和打 tag，例如：

```text
baseline
baseline-glas-v1
```

baseline 分支和 `baseline-glas-v1` tag 只能用于复现，不得直接继续修改。任何后续改动
必须创建新分支，不能覆盖 baseline。

## Innovation Point 分支

创新代码应从冻结的 baseline tag 创建独立分支，例如：

```text
innovation1-boundary-loss
innovation2-pixel-fusion
innovation1+2
```

每个有意义的结构、损失或配置版本必须至少有一个清晰的 commit；确定为论文实验版本后，
再创建对应 tag。commit/tag 信息必须能对应到配置文件、checkpoint、seed、fold 和指标记录。

## 执行授权

- 用户明确说“执行”后，才运行 `git add`、`git commit`、`git tag`、分支创建或其他 Git
  写操作。
- 用户只要求修改代码、审查代码或分析结果时，不自动执行 Git 写操作。
- 修改完成后必须提醒用户：哪些文件发生变化、建议使用哪个分支/commit/tag 记录，以及该改动
  是否已经构成一个独立创新点实验。

## 创新点提醒

当一次修改已经形成独立的网络结构、损失函数、数据协议或可报告实验变量时，必须主动提醒：

1. 这是一个新的创新点版本或消融版本；
2. 应创建独立 branch/commit/tag；
3. 需要记录配置、checkpoint、seed/fold、评价口径、Dice 和 IoU。

不得把多个未记录的代码改动混在同一个 baseline 或创新点版本中。
