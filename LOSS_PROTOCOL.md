# Loss 协议（作者 Protocol B）

当前 clean baseline 的基础损失固定为：

```text
Weighted Dice + Weighted BCE
Dice=0.6, BCE=0.4
```

- C0：只使用基础损失。
- C1：基础损失 + Active Boundary Loss。
- C2：只使用基础损失，不能额外加入 ABL。
- C3：基础损失 + Active Boundary Loss，并启用已验证的 Decoder 结构。

Protocol B 的 Train_Folder/Test_Folder 数据规则不改变 Loss。基础 Dice/BCE 的 `0.6/0.4` 权重在所有组和 fold 保持一致；C1/C3 只叠加预先声明的 ABL，不能把其 λ 混入基础权重。调整 ABL λ 或其他 Loss 设计必须另立有名称的诊断/实验组，不能用 Test_Folder 指标选有利 λ 后仍称原 C1。

C0/C1/C2/C3 必须有相互独立的配置、启动入口、run 名称与 checkpoint 路径；C1 启 ABL 但关 C2 Decoder 创新，C2 关 ABL，C3 才同时启用。通用训练代码可共享，但各开关需在实验配置中显式体现；不得通过覆盖同一 YAML、共用 run 或热启动另一组 checkpoint 混淆消融。完整实现门禁见 `EoSeg_innov1_Dice_BCE_ABLplan.md` 第 15 节。C3 是 I1+I2 组合，不是尚未确定的第三创新点 I3；I3 必须另立组和入口。

