# Boundary F1 与 fold0/train_3 论文图执行记录

- 日期：2026-09-29
- 唯一永久修改工作区：`Z:\EoSeg-main-official`
- 未修改：`Z:\EoSe-main-innovation_point`、C 盘备份、原始数据、五折 manifest、既有 checkpoints 与既有 runs。
- 结果协议：`author_protocol_b_test_as_val`，五折、seed=0、single-view、阈值 0.5。Protocol B 的 `Test_Folder` 同时用于逐 epoch `val_dice` checkpoint 选择及最终 test；不得称为独立盲测。

## 原 C0 baseline（BF1 接入前，已有五折运行）

来源为 official 中五个既有 `protocol_b_test.txt`，checkpoint 由 `metrics/val_dice` 选择。best epoch：fold0=711、fold1=729、fold2=698、fold3=672、fold4=514。下表为原有单视图测试日志，不含 BF1。

| Fold | Dice | IoU | Precision | Recall | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.908709 | 0.838462 | 0.909370 | 0.914336 | 0.917330 |
| 1 | 0.902568 | 0.825537 | 0.888866 | 0.919829 | 0.898759 |
| 2 | 0.890227 | 0.807188 | 0.877750 | 0.908344 | 0.893198 |
| 3 | 0.897570 | 0.818090 | 0.888417 | 0.911605 | 0.899066 |
| 4 | 0.914337 | 0.845658 | 0.904963 | 0.929350 | 0.922989 |
| 五折 mean ± sample SD | 0.902682 ± 0.009400 | 0.826987 ± 0.015440 | 0.893873 ± 0.013019 | 0.916693 ± 0.008234 | 0.906268 ± 0.013049 |

fold 间 SD 使用样本标准差（`ddof=1`）。这是现有 C0 的历史结果，必须与 BF1 代码接入后的测试回归和新五折重跑分别记录，不能混成同一批实验。

## 冻结参数（不因本次加指标而改动）

- Backbone：`vit_large_patch14_reg4_dinov2`；DINOv2 ViT-L 既有服务器缓存权重。
- 输入：224×224；图像 bilinear resize、mask nearest resize；单视图，无 TTA。
- `mask_only_training_enabled=true`，decoder GroupNorm。
- Loss：Weighted Dice 0.6 + BCE 0.4；ABL/Boundary Loss 关闭（C0）。
- AdamW；`lr=5e-5`、`backbone_lr=1e-5`、`decoder_lr=1e-4`、weight decay `5e-4`、LLRD `0.8`、`llrd_l2_enabled=true`、`lr_mult=1.0`。
- `TwoStageWarmupPolySchedule`，warmup `[500,1000]`，poly power `0.9`；physical batch=2、accumulate=7、effective batch=14。
- seed=0，最大 1000 epochs，EarlyStopping patience=300，梯度裁剪 1.0，AMP 16-mixed；checkpoint 仍仅按 `metrics/val_dice` 选择。
- 指标阈值 0.5。BF1 在 224×224 的单图二值 mask 上，以 8 邻域内轮廓和 Euclidean 半径 2 像素对称匹配；先逐图求 BF1，再对图像宏平均。

## 代码变更清单

仅在 `Z:\EoSeg-main-official` 修改或新增以下文件：

| 文件 | 代码位置 | 修改内容 |
|---|---|---|
| `training/boundary_f1.py`（新增） | 行 22 `inner_boundary`、行 56 `boundary_f1_score` | 统一实现 8 邻域内轮廓、欧氏半径 2px 对称容差、空边界约定；不加第三方依赖。 |
| `training/medical_binary_segmentation.py` | import 行 9；`_batch_metrics` 行 252 起；val/test 选择行 322；记录指标行 400 起 | 仅 validation/test 计算并记录 `metrics/val_boundary_f1`、`metrics/test_boundary_f1`；原 Dice/IoU 算法和 loss 保持原样，训练阶段不计算 BF1。 |
| `training/medical_val_logger.py` | 行 68–70、92–93、126–130 | 把每 epoch BF1 加到验证历史；在 Dice 最佳 epoch 旁记 `val_boundary_f1_at_best_dice_checkpoint`，不读取 BF1 进行 checkpoint 选择。 |
| `configs/glas/run_protocol_b.sh` | 行 39–45、最终 test 后 | run metadata 记录 BF1 定义；完整五折训练/测试后调用逐图导出脚本。`main.py fit` 的训练启动参数、loss、优化器与 checkpoint 规则无变化。 |
| `scripts/evaluate_protocol_b_existing_ckpts.sh`（新增） | 逐折 exact `best_epoch` 查找及 test 流程 | 对原 C0 五折已选 checkpoint 执行 test-only 回归；严格要求 epoch 文件唯一，不回退到 `last.ckpt`，输出到独立新目录。 |
| `scripts/evaluate_boundary_f1_per_image.py`（新增） | 行 37 起 `main`；行 81 BF1 计算；行 94 起 CSV 导出 | 对同一冻结 checkpoint、相同 224×224 preprocessing、阈值 0.5 逐样本计算 BF1；不修改预测、不做 TTA，并输出每张图的 ID、BF1、checkpoint 与容差 metadata。 |
| `scripts/summarize_protocol_b.py` | 行 10–47、60 起 | 解析 BF1 和原 test 指标；按五折输出 sample SD (`ddof=1`)，并同时生成原 baseline 与新 run 的逐折差值 CSV。 |
| `scripts/plot_gt_prediction_boundary_overlay.py` | 固定 `SAMPLE_ID` 行 35、阈值行 36、ROI 行 38、主入口行 152 起 | 固定 fold0/Test_Folder/train_3；按 16-mixed CUDA 推理；可选 C1 checkpoint；无 C1 时只输出 C0 preview + 同 ROI zoom，不伪造 C1；准备 C1 后生成六面板与配对放大图。 |
| `tests/test_boundary_f1.py`（新增） | 8 个单元测试 | 覆盖同一边界、1/2px 容差、明显错位、空边界、图像边缘、尺寸不符和数值范围。 |
| `EoSeg_BoundaryF1_Baseline_Codex_Plan.md` | 第 8 节 | 追加本次用户要求的“先旧 checkpoint 回归，再独立 suffix 重训五折”的执行补充，并记录仅在 official 的授权例外。 |
| `EoSeg_Paper_Qualitative_Visualization_Codex_Plan.md` | 第 9 节 | 追加 train_3 固定入口、ROI、C1 checkpoint 缺失时的 C0-only 状态及 official 路径授权说明。 |
| 本文件 | 全文 | 记录当前 C0 结果、代码改动位置、验证状态和服务器待执行命令。 |

没有修改 `models/eoseg.py`、C0 YAML、optimizer/LR/Scheduler、loss、数据增强、manifest、阈值、checkpoint monitor 或既有 checkpoints/runs。BF1 只写 metadata，未改变训练参数。

## 验证与服务器执行状态

- BF1 数学单元测试：8 项通过（相同 mask、1/2 像素容差、明显错位、空边界、贴边 mask、形状错误、有限且 `[0,1]`）。
- Python 语法编译：通过（BF1、metric 接入、callback、汇总器、绘图脚本、逐图导出脚本）。
- 五份既有 Protocol B 日志由汇总器解析成功；其 C0 汇总即本记录首表。
- 尚未在 RTX 5090 服务器运行同 checkpoint 回归、BF1 五折离线评估、重新五折训练或 GPU 绘图；相关命令由用户在服务器终端执行后填写结果。
- 计划要求先以既有 C0 best checkpoint 做同 checkpoint test 回归，确认 Dice/IoU/Jaccard/Precision/Recall/Accuracy/F2/loss 不变；再以独立 suffix `_bf1_metric_check` 重新训练完整五折，并相对历史 C0 逐折对照。
- 当前没有可用 C1（C0+ABL）正式 checkpoint，因此只允许生成 C0-only 预览和固定 ROI 放大图；六面板 C0/C1 图待 C1 checkpoint 提供后生成。

## 服务器端复现命令（尚未执行）

以下命令需由用户在 RTX 5090 Linux 终端运行；Codex 未从 Windows/SSHFS 映射目录启动服务器训练或推理。

```bash
cd /Disk/HDD/Sk/EoSeg-main/EoSeg-main-official

HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
FOLDS="0 1 2 3 4" \
bash scripts/evaluate_protocol_b_existing_ckpts.sh

python -X utf8 scripts/summarize_protocol_b.py \
  --run-suffix _bf1_existing_ckpt_test \
  --reference-suffix _maskonly_gn_baseline_5fold \
  --output runs/boundary_f1_existing_checkpoint_regression.csv
```

只有同 checkpoint 回归的原指标（loss、Accuracy、Dice、IoU/Jaccard、Precision、Recall、F2）一致后，再按用户要求重训五折：

```bash
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
RUN_SUFFIX=_bf1_metric_check \
FOLDS="0 1 2 3 4" \
bash configs/glas/run_protocol_b.sh

python -X utf8 scripts/summarize_protocol_b.py \
  --run-suffix _bf1_metric_check \
  --reference-suffix _maskonly_gn_baseline_5fold \
  --output runs/boundary_f1_fivefold_baseline_rerun_comparison.csv
```

固定 train_3 的 C0-only 预览及同 ROI 放大图：

```bash
python -X utf8 scripts/plot_gt_prediction_boundary_overlay.py \
  --config configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml \
  --ckpt "runs/glas_protocol_b_fold0_vitl_clean_maskonly_gn_baseline_5fold/version_0/checkpoints/epoch=711-step=7120.ckpt" \
  --test-dir datasets/GlaS_5fold/fold0/Test_Folder \
  --source-dir ../Warwick_QU_Dataset \
  --output runs/paper_visualizations/fold0/train_3_c0.png \
  --device cuda
```
