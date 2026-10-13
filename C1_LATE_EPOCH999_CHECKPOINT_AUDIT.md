# C1-Late-NoES epoch-999 checkpoint audit

Date: 2026-10-13. Workspace: `EoSe-main-innovation_point` only. Original five-fold runs and the official baseline are unchanged.

## Finding

All five original runs logged epoch 999 and the ABL gate at epochs 799/800/999. However, each `last.ckpt` has the same internal `epoch` and `global_step` as the corresponding best-Dice checkpoint. The original runner wrote `last_epoch=999` as a literal string without reading the checkpoint. Therefore `protocol_b_last_test.txt` is **not** an independent epoch-999 model test and must not be reported as such.

| Fold | Actual best checkpoint | Actual `last.ckpt` epoch | Actual `last.ckpt` global step |
|---:|---|---:|---:|
| 0 | `epoch=799-step=8000.ckpt` | 799 | 8000 |
| 1 | `epoch=729-step=7300.ckpt` | 729 | 7300 |
| 2 | `epoch=698-step=6990.ckpt` | 698 | 6990 |
| 3 | `epoch=842-step=8430.ckpt` | 842 | 8430 |
| 4 | `epoch=691-step=6920.ckpt` | 691 | 6920 |

For fold 0, best and last also have exactly the same SHA-256 (`57573461cccd87524288dd09228f1812d18fa680d7fc2ca07cbf41d630cb3d6d`). Checkpoint metadata was read from the PyTorch archive's `data.pkl` without loading tensors; the original best and last files for all folds contain the same epoch/step pair. `metrics.csv` records epoch 999 in every fold, but the original epoch-999 weights are absent from the run directories.

## Original epoch-999 validation metrics, not checkpoint tests

These are the `metrics/val_*` rows at epoch 999 in each original `metrics.csv`. They were computed on the same `Test_Folder` used by Protocol B, with single-view prediction and threshold 0.5. They are **actual logged final-epoch validation results**, but not a rerun of `main.py test`; they must be labeled as validation, not substituted for a missing checkpoint-based test. In Late mode, validation loss includes ABL after epoch 800 while `test/loss` is base-only, so losses are not interchangeable.

| Fold | Validation Dice (%) | Validation IoU (%) | Validation Boundary-F1 (%) |
|---:|---:|---:|---:|
| 0 | 90.8215 | 83.7647 | 67.8493 |
| 1 | 90.2702 | 82.5690 | 64.2088 |
| 2 | 88.8750 | 80.5021 | 61.1858 |
| 3 | 89.7157 | 81.7434 | 63.3737 |
| 4 | 91.3571 | 84.5061 | 68.0208 |
| Mean ± fold sample SD | 90.2079 ± 0.9644 | 82.6171 ± 1.5910 | 64.9277 ± 2.9596 |

## Recovery protocol

The original epoch-999 weights cannot be reconstructed from logs. A **new, separately labeled continuation** can resume from each original full-state best checkpoint, train until epoch 999, save a dedicated endpoint checkpoint, verify its internal `epoch=999` and `global_step=10000`, and only then run single-view testing. This is a valid new endpoint experiment, **not an exact recovery of the original uninterrupted training trajectory**; stochastic data order and augmentation may differ after resumption.

Files added for this protocol:

- `configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_late_abl_epoch999_recovery.yaml`: same model, loss, data, optimizer, learning rates, schedule, seed, and train budget as the original C1-Late YAML; only checkpoint callback and default logger name differ. The new callback saves only at epoch 999, independent of best-Dice selection.
- `configs/glas/run_protocol_b_c1_late_epoch999_recovery.sh`: checks source checkpoint epoch/step, uses a new run name without overwriting old results, resumes full training state, verifies the saved endpoint's internal epoch/step, then tests and computes per-image Boundary-F1.

Before running, verify the new script with `bash -n` and `FOLDS=0 bash ... --check-only` on the server. Run fold 0 first; inspect its endpoint and test log before extending to folds 1–4. The new results must be cited as `C1-Late epoch999 recovery continuation`, and original best-Dice and original final-epoch validation results remain separate.
