# EoSeg 论文定性可视化计划：五折 BF1 主证据与 fold0/train_3 辅助图

> 2026-09-29 修订。本文件是后续执行计划；本次修订只更新计划，不修改脚本、指标、训练配置或实验结果。

## 1. 论文证据的主次关系

- **主证据：** 在同一套五折 Protocol B、seed=0、single-view 条件下，比较稳定基线 C0 与 `C0 + Active Boundary Loss`（创新点 1，记为 C1）的 Boundary F1（BF1）。同时报告 Dice 和 IoU，检查边界改善是否伴随区域指标退化。
- **辅助证据：** 固定使用 fold0 `Test_Folder` manifest 中 `sample_id=train_3` 的样本，制作一张六面板图和一张同区域局部放大对比图。单张图只展示具体的边界现象，不承担总体性能结论。
- 论证分两步：C0 的五折 BF1 和轮廓图用于描述仍存在的边界误差；C1 相对 C0 的**逐折 BF1 差值**用于检验 ABL 是否改善该误差。结果若不支持改善，论文表述必须如实反映。

## 2. 固定实验口径与当前状态

项目使用 165 张 GlaS 图像的五折划分。每折 `Train_Folder` 为 132 张，`Test_Folder` 为 33 张；每个模型都按同折 `metrics/val_dice` 选择 best checkpoint，然后在该折 `Test_Folder` 做 single-view 评估。该目录同时参与 checkpoint 选择，所有结果标记 `author_protocol_b_test_as_val`，不能称为独立盲测或独立泛化测试。

保持现有 C0 稳定基线：DINOv2 ViT-L、mask-only、GroupNorm、Weighted Dice(0.6)+BCE(0.4)、seed=0、224×224、阈值 0.5；C1 只加入既定 ABL，不借可视化任务改变模型或训练设置。两组的五折划分、预处理、checkpoint 选择、推理和指标实现必须一致。TTA 不进入本计划的主结果。

已核实 C0 的五折 run 存在。fold0 的 best checkpoint 为：

```text
Z:\EoSeg-main-official\runs\glas_protocol_b_fold0_vitl_clean_maskonly_gn_baseline_5fold\version_0\checkpoints\epoch=711-step=7120.ckpt
```

其 `best_metric.txt` 记录 `monitor=metrics/val_dice`、`best_epoch=711`。执行前仍须核对其余四折以及 C1 各折的 checkpoint 与配置。当前未核实 C1 的正式五折 checkpoint，也未在代码中发现已完成的 BF1 实现，因此不预填 BF1 数值或 C1 图片。

## 3. BF1 主指标的计算和汇总

先完成单独的 `EoSeg_BoundaryF1_Baseline_Codex_Plan.md`，并将其后续正式口径扩展到**五折 C0 与五折 C1**。该文件本次只读，不在本任务执行或修改。当前该文件的 Fold0 验证目标不足以形成五折论文主表。

本图计划只**调用同一正式指标实现**，不在绘图脚本中重写 BF1。统一使用：

```text
prediction threshold = 0.5
evaluation resolution = 224 × 224
boundary tolerance = 2 pixels
per-image BF1 → fold 内图像宏平均
single-view；TTA=false
checkpoint monitor = metrics/val_dice
```

优先对已冻结的 C0 五折 best checkpoint 离线补算 BF1；若后续 BF1 计划要求重新训练，应单列新 run，并核实训练、Dice/IoU 和 checkpoint 选择未因新增指标改变。C1 使用与 C0 同折、同 seed 的正式 best checkpoint。不要把不同 run 的指标混成一行。

每折记录 `C0 Dice / IoU / BF1`、`C1 Dice / IoU / BF1` 和 `ΔBF1 = C1 BF1 − C0 BF1`。主表报告五折 BF1 的 `mean ± sample std (ddof=1)`，并列 Dice、IoU 的五折 `mean ± sample std`；另报告五个逐折 `ΔBF1` 及其均值。不能只挑 BF1 上升的折，也不能根据结果调整 tolerance 或阈值。若以后增加多个训练 seed，每折先对相同 seed 集合求均值，再计算五折间标准差；seed 波动另列。

| Fold | C0 Dice | C0 IoU | C0 BF1 | C1 Dice | C1 IoU | C1 BF1 | ΔBF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–4 各一行 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 五折 mean ± std | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 待汇总 | 逐折汇总 |

## 4. 唯一固定定性案例：fold0/train_3

`train_3` 已在 `Z:\EoSeg-main-official\datasets\GlaS_5fold\fold0\Test_Folder\samples.txt` 中核实。主图和局部放大图都使用这一张图及其同一份 GT；不运行全 Test_Folder 的“好看案例”排名，不因 C1 结果更换 sample。

现有 `Z:\EoSe-main-innovation_point\scripts\plot_gt_prediction_boundary_overlay.py` 在没有 `--sample-id` 时会回退到 `--sample-index 0`；代码中 `available examples` 只是报错提示，不是自动挑选案例。**后续执行本计划时**，将论文案例入口固定为 `fold0/Test_Folder` + `sample_id=train_3`：默认不依赖 index、先核验 manifest 中确有该 ID，metadata 写入实际 ID/图像/GT/折号。若保留通用脚本的其他样本功能，也必须让论文图调用显式 `--sample-id train_3`，且缺失时报错，不静默回退到第一张。

后续可视化代码只放在 `Z:\EoSe-main-innovation_point`。official 的模型、训练与评估代码保持冻结；official 的 C0 checkpoint 只读加载。输出保存在 innovation_point 自己的 `runs/paper_visualizations/fold0/train_3/`，不得写入 official 的 `runs`。

## 5. 一张六面板主图

默认采用适合论文排版的 2×3 布局，六个面板使用同一张原图、GT 和固定展示尺寸：

```text
(a) Original             (b) Ground Truth       (c) C0 Baseline mask
(d) C1 Baseline + ABL mask (e) C0 boundary overlay (f) C1 boundary overlay
```

- `(a)` 原始 H&E RGB；`(b)` 二值 GT；`(c)`、`(d)` 用相同黑底白前景样式展示完整预测 mask。
- `(e)` 在原图上叠加 GT 边界与 `(c)` 的预测边界；`(f)` 在同一原图上叠加 GT 边界与 `(d)` 的预测边界。两幅图采用完全一致的线宽、亮度、裁剪与颜色：GT 青色 `#00DCFF`，预测橙红色 `#FF4A00`，显示笔画相交处黄色 `#FFEB00`。
- C0、C1 各只做一次确定性推理，同一预测结果分别用于 mask panel 和 boundary panel。禁止根据 GT 调整预测极性、阈值或后处理。
- 面板标题保持简短；fold、sample ID、阈值、checkpoint 和协议放在 caption/metadata。只有正式 BF1 算出后，才可在 `(e)`、`(f)` 标出对应 `train_3` 的 per-image BF1；不得填示例数值。

## 6. 一张局部放大对比图

在六面板之外，单独输出一张包含 **C0 边界局部图与 C1 边界局部图** 的并排放大图，使用同一矩形 ROI、同一缩放比例和同一颜色。GT 轮廓同时出现在两侧；可在六面板的 `(e)`、`(f)` 上用相同矩形标出 ROI。

ROI 由研究者在查看 GT 与 C0 边界后确定，**在查看 C1 放大效果之前**冻结；以原始图像像素坐标 `(x1, y1, x2, y2)` 记录到 metadata。先检查 `train_3` 中央细长腺体附近，但不预设未经核验的坐标。不得为 C1 单独换位置、改变 crop 或只显示改善区域。局部图是视觉展示，不在 crop 内另算论文主 BF1。

## 7. 评价图与展示图的分辨率必须分开

BF1 在 224×224 的正式评估 mask 上计算，使用未来统一的 BF1 实现及 2-pixel tolerance。论文图可将**已经二值化**的预测 mask 以 nearest-neighbour 放大到原图尺寸，再提取并加粗用于展示的轮廓。现有脚本的 8 邻域内边界和 3×3 加粗笔画可以复用，但黄色仅表示加粗后的显示笔画相交，**不等于 BF1 匹配**。不能把放大造成的锯齿直接解释为真实边界错误。

## 8. 执行顺序与交付门槛（收到执行指令后才开始）

1. 先落实后续 BF1 计划，确认同一 BF1 实现可在五折 C0/C1 checkpoint 上运行，且 Dice/IoU 与原评估回归一致。
2. 确认 C1 的五折正式结果、配置和 best checkpoint；没有 C1 checkpoint 时，只能生成 C0 的基础展示，不伪造六面板完整对比。
3. 五折逐折计算并核对 BF1、Dice、IoU，再形成主表和逐折差值。检查 BF1 定义、224×224、阈值、single-view、同折配对和 `author_protocol_b_test_as_val` 标记。
4. 在 innovation_point 实现固定 `fold0/train_3` 的论文图入口，生成六面板和一张并排局部放大图；对照原图、GT 与两个 checkpoint 检查预测来源。
5. 人工核对图上呈现的变化是否与 `train_3` per-image BF1 一致；若不一致，图文分别如实描述，不用展示笔画交叠替代指标。

交付物限定为：五折 BF1/Dice/IoU 及逐折差值表、`train_3` 六面板图、同 ROI 局部放大图、每图 metadata（协议、split、seed、配置、checkpoint、阈值、BF1 参数、ROI 坐标、输出分辨率与代码版本）。图片优先输出 PNG/TIFF 600 DPI，并保留 PDF/SVG 便于排版；这些是**后续执行要求**，不是本次已生成结果。

本次修订的实际范围仅为本计划文件。`EoSeg_BoundaryF1_Baseline_Codex_Plan.md`、任何脚本、模型、配置、数据、checkpoint 和 runs 均未因本次修订而改变。

## 9. 执行状态补充（2026-09-29）

本轮按用户明确限定，仅在 `Z:\EoSeg-main-official` 实现论文图入口；这覆盖本计划第 4、8 节原定“脚本和输出只放 innovation_point”的路径约束，但不改变模型、数据、训练结果或 checkpoints。论文案例固定为 `fold0/Test_Folder` manifest 中的 `train_3`，脚本缺少该样本时直接报错，不回退到 index 0。ROI 固定记录为原图坐标 `(310, 210, 455, 455)`，在查看 C1 结果前按原图与 GT/C0 确定，不按 C1 改选。

由于当前没有核实到 C1（C0+ABL）的正式 checkpoint，本轮只能用现有 fold0 C0 best checkpoint 生成 C0-only 预览和同 ROI 放大图；不伪造六面板的 C1 内容。待用户提供正式 C1 checkpoint 后，使用同一入口的 `--abl-ckpt` 参数生成六面板及配对局部放大图。单图仅作为辅助证据，论文总体结论仍以五折 BF1/Dice/IoU 配对结果为依据。

正式图推理和 BF1 五折评估需要服务器 Python/CUDA 与已配置预训练缓存，本地 Windows 不代替服务器执行。对应 Linux 命令在实现交付时提供；服务器生成图像和新五折指标前，均保持结果状态为“待运行”。

## 10. C0-only 部分图的面板字母（2026-09-30）

在 C1 checkpoint 尚未提供时，C0-only 预览保留完整六面板定义中的原始面板字母：可生成 `(a) Original`、`(b) Ground Truth`、`(c) C0 Baseline mask`、`(e) C0 boundary overlay`；`(d)` 和 `(f)` 暂不生成，因为它们分别需要 C1 mask 和 C1 boundary overlay。不得把 C0 boundary overlay 重编号为 `(d)`。部分预览可将现有四个面板排成 2×2，但标题仍须使用 `(a)、(b)、(c)、(e)`；metadata 记录缺失的 `(d)、(f)` 是因为 C1 checkpoint 未提供。

该面板映射已落实到 `scripts/plot_gt_prediction_boundary_overlay.py`：C0-only 预览按 2×2 排列并保留 `a,b,c,e` 字母，metadata 明确记录省略的 `d,f`；完整 C0/C1 图仍沿用原定 `a` 至 `f`。同时加入字体/PDF-SVG 可编辑文本设置及绘制前的多面板对齐门禁。语法检查和绘图源码预检均通过；尚未在服务器执行推理或生成图片。
