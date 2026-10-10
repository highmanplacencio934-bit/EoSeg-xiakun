#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate the frozen C0 best checkpoints with the added BF1 metric.
# This is a test-only pass: it does not fit, alter checkpoints, or select by BF1.
DEVICES="${DEVICES:-1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
CONFIG="${CONFIG:-configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml}"
SOURCE_DIR="${SOURCE_DIR:-../Warwick_QU_Dataset}"
BASELINE_SUFFIX="${BASELINE_SUFFIX:-_maskonly_gn_baseline_5fold}"
RUN_SUFFIX="${RUN_SUFFIX:-_bf1_existing_ckpt_test}"

for fold in ${FOLDS}; do
  baseline_run="runs/glas_protocol_b_fold${fold}_vitl_clean${BASELINE_SUFFIX}"
  version_dir="$(find "${baseline_run}" -mindepth 1 -maxdepth 1 -type d \
    -name 'version_*' -print | sort -V | tail -n 1)"
  test -n "${version_dir}" || { echo "Missing logger version under ${baseline_run}" >&2; exit 1; }

  best_file="${version_dir}/best_metric.txt"
  test -f "${best_file}" || { echo "Missing ${best_file}" >&2; exit 1; }
  best_epoch="$(awk -F= '/^best_epoch=/{print $2; exit}' "${best_file}")"
  [[ "${best_epoch}" =~ ^[0-9]+$ ]] || { echo "Invalid best_epoch in ${best_file}" >&2; exit 1; }

  ckpt_dir="${version_dir}/checkpoints"
  mapfile -t checkpoints < <(find "${ckpt_dir}" -maxdepth 1 -type f \
    -name "epoch=${best_epoch}-*.ckpt" -print | sort -V)
  [[ "${#checkpoints[@]}" -eq 1 ]] || {
    echo "Expected one exact best-epoch checkpoint for fold${fold}; found ${#checkpoints[@]}" >&2
    exit 1
  }
  ckpt="${checkpoints[0]}"

  train_dir="datasets/GlaS_5fold/fold${fold}/Train_Folder"
  test_dir="datasets/GlaS_5fold/fold${fold}/Test_Folder"
  test_run_name="glas_protocol_b_fold${fold}_vitl_clean${RUN_SUFFIX}"
  test_run_dir="runs/${test_run_name}"
  test_log="${test_run_dir}/protocol_b_test.txt"
  if [[ -e "${test_log}" ]]; then
    echo "Refusing to overwrite existing evaluation log: ${test_log}" >&2
    exit 1
  fi
  mkdir -p "${test_run_dir}"
  {
    echo "protocol=author_protocol_b_test_as_val"
    echo "purpose=metric_regression_same_frozen_checkpoint"
    echo "fold=${fold}"
    echo "checkpoint=${ckpt}"
    echo "checkpoint_selection=original_best_epoch_by_metrics/val_dice"
    echo "single_view=true"
    echo "tta=false"
    echo "threshold=0.5"
    echo "boundary_metric=per_image_macro_boundary_f1"
    echo "boundary_edge=8-neighbour_inner_contour"
    echo "boundary_tolerance=2px_euclidean_disk"
    echo "boundary_metric_affects_training=false"
    echo "boundary_metric_affects_checkpoint_selection=false"
  } > "${test_run_dir}/protocol_b_metadata.txt"

  echo "Evaluating frozen fold${fold} checkpoint: ${ckpt}"
  python -X utf8 main.py test -c "${CONFIG}" \
    --model.init_args.ckpt_path "${ckpt}" \
    --trainer.devices "${DEVICES}" \
    --trainer.logger.init_args.name "${test_run_name}" \
    --trainer.default_root_dir "${test_run_dir}" \
    --data.init_args.train_dir "${train_dir}" \
    --data.init_args.val_dir "${test_dir}" \
    --data.init_args.test_dir "${test_dir}" \
    --data.init_args.source_dir "${SOURCE_DIR}" \
    | tee "${test_log}"

  python -X utf8 scripts/evaluate_boundary_f1_per_image.py \
    --config "${CONFIG}" \
    --ckpt "${ckpt}" \
    --data-dir "${test_dir}" \
    --source-dir "${SOURCE_DIR}" \
    --output "${test_run_dir}/boundary_f1_per_image.csv" \
    --device cuda
done
