# Seed 与五折统计协议（作者 Protocol B）

- 五折划分文件固定，不因训练 seed 改变。
- 当前开发和 clean baseline 使用 train_seed=0。
- 五折单 seed 结果报告五个 fold 的 mean±std，std 使用 sample standard deviation（ddof=1），只表示 fold 间波动。
- 创新点锁定后，如使用多个 seed，所有 C0/C1/C2/C3 使用相同 seed 集合；每个 fold 先对 seed 求均值，再对五个 fold 均值计算主表 mean±std。
- seed 波动作为补充统计，不能用未标注的第二个 ± 表示。
- 不得根据 Test_Folder 结果挑选 seed；Protocol B 的 Test_Folder checkpoint 选择偏差必须在记录中注明。

