# Innovation 2 Decoder 结构单变量规则

本文件是项目 `.codex` 全局实验规则的一部分，适用于 Innovation 2 及其所有消融实验。

## 创新 2 的定义

Innovation 2 只研究 EoSeg Decoder 中最后三个 ViT Transformer block 的特征融合方式。
原始 `EoSegMultiQSegFusion` 已经从最后三个 block 获取特征并执行 query-conditioned
fusion，因此创新变量只能作用于这三个 block 的融合路径。

## 允许修改的范围

以下内容属于 Innovation 2 的合法变量：

```text
最后三层的融合权重
逐像素层权重
query-conditioned attention
局部 attention / window attention
gate
concat
weighted sum
层间差异建模
```

## 禁止顺带修改的内容

在同一个 Innovation 2 实验中，不得同时修改：

```text
num_query
feature upscale / resolution alignment
mask head
class head
ViT backbone 或预训练权重
输入尺寸
损失函数
数据增强
评价指标
```

如果确实需要修改上述内容，必须建立独立实验组，并在实验名称、配置、结果记录中明确
标记，不能把它归因于 Innovation 2 的 Decoder 融合改进。

## 公平性要求

- C0 baseline 与 Innovation 2 必须使用相同的 backbone、num_query、mask head、class head、
  输入尺寸、损失、optimizer、数据划分、seed、训练预算和评价指标。
- 每次实验只引入一个新的结构变量；消融实验一次只关闭或恢复一个变量。
- 任何性能变化必须能够归因到“最后三层 ViT 融合方式”，否则该实验不能作为 Innovation 2 的
  有效对比结果。
- 训练日志和论文记录必须注明本次实验实际修改的融合变量。
