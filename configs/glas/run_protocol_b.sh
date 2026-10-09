#!/usr/bin/env bash
set -euo pipefail
DEVICES="${DEVICES:-1}"
FOLDS="${FOLDS:-0 1 2 3 4}"
CONFIG="${CONFIG:-configs/glas/vit_query_mul_scale_fusion_protocol_b.yaml}"
SOURCE_DIR="${SOURCE_DIR:-../Warwick_QU_Dataset}"
FIXED_LR="${FIXED_LR:-5e-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
DECODER_LR="${DECODER_LR:-1e-4}"
MASK_ONLY_TRAINING_ENABLED="${MASK_ONLY_TRAINING_ENABLED:-true}"
DECODER_NORM="${DECODER_NORM:-group}"
MODEL_TAG="${MODEL_TAG:-clean}"
ABL_ENABLED="${ABL_ENABLED:-false}"
ABL_WEIGHT="${ABL_WEIGHT:-0.0}"
LOSS_DESCRIPTION="${LOSS_DESCRIPTION:-WeightedDiceBCE(dice=0.6,bce=0.4)}"
PRETRAINED_WEIGHT_PATH="${PRETRAINED_WEIGHT_PATH:-/Disk/HDD/Sk/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
[[ -n "${FOLDS//[[:space:]]/}" ]] || { echo "FOLDS must select at least one fold (0-4)." >&2; exit 2; }

sha256_or_missing() {
  if [[ -f "$1" ]]; then
    sha256sum "$1" | awk '{print $1}'
  else
    printf 'missing'
  fi
}

latest_version_dir() {
  find "$1" -mindepth 1 -maxdepth 1 -type d -name 'version_*' -print | sort -V | tail -n 1
}
for fold in ${FOLDS}; do
  [[ "${fold}" =~ ^[0-4]$ ]] || { echo "Invalid Protocol B fold '${fold}'; expected 0-4." >&2; exit 2; }
  run_name="glas_protocol_b_fold${fold}_vitl_${MODEL_TAG}${RUN_SUFFIX}"
  fold_root="datasets/GlaS_5fold/fold${fold}"
  train_dir="${fold_root}/Train_Folder"
  test_dir="${fold_root}/Test_Folder"
  run_dir="runs/${run_name}"
  execution_id="$(date -u +'%Y%m%dT%H%M%SZ')_${BASHPID}"
  execution_dir="${run_dir}/executions/${execution_id}"
  metadata_file="${execution_dir}/protocol_b_metadata.txt"
  [[ -f "${PRETRAINED_WEIGHT_PATH}" ]] || {
    echo "Configured local DINOv2 ViT-L weights do not exist: ${PRETRAINED_WEIGHT_PATH}" >&2
    exit 1
  }
  for required_file in \
    "datasets/GlaS_5fold/sample_manifest.csv" \
    "datasets/GlaS_5fold/fold_assignments.csv" \
    "${train_dir}/samples.txt" \
    "${test_dir}/samples.txt"; do
    [[ -f "${required_file}" ]] || { echo "Missing frozen Protocol B data manifest: ${required_file}" >&2; exit 1; }
  done
  mkdir -p "${execution_dir}"
  cp -- "${CONFIG}" "${execution_dir}/config.yaml"
  git_commit="$(git rev-parse HEAD 2>/dev/null || printf 'unavailable')"
  git status --short > "${execution_dir}/git_status.txt" 2>&1 || printf 'unavailable\n' > "${execution_dir}/git_status.txt"
  {
    echo "execution_id=${execution_id}"
    echo "run_name=${run_name}"
    echo "protocol=author_protocol_b_test_as_val"
    echo "fold=${fold}"
    echo "train_seed=0"
    echo "train_split=${train_dir}"
    echo "val_split=${test_dir}"
    echo "test_split=${test_dir}"
    echo "checkpoint_selection=Test_Folder/val_dice"
    echo "single_view=true"
    echo "tta=false"
    echo "threshold=0.5"
    echo "backbone=vit_large_patch14_reg4_dinov2"
    echo "model_tag=${MODEL_TAG}"
    echo "pretrained_weight=${PRETRAINED_WEIGHT_PATH}"
    echo "pretrained_weight_sha256=$(sha256_or_missing "${PRETRAINED_WEIGHT_PATH}")"
    echo "pretrained_weight_mode=timm_cached"
    echo "loss=${LOSS_DESCRIPTION}"
    echo "active_boundary_loss_enabled=${ABL_ENABLED}"
    echo "active_boundary_loss_weight=${ABL_WEIGHT}"
    echo "boundary_metric=boundary_f1"
    echo "boundary_tolerance_pixels=2"
    echo "boundary_tolerance_geometry=euclidean_disk"
    echo "boundary_edge_definition=8-neighbour_inner_contour"
    echo "boundary_metric_resolution=224x224"
    echo "boundary_metric_affects_loss=false"
    echo "boundary_metric_affects_checkpoint_selection=false"
    echo "physical_batch_size=2"
    echo "accumulate_grad_batches=7"
    echo "effective_batch_size=14"
    echo "lr=${FIXED_LR}"
    echo "backbone_lr=${BACKBONE_LR}"
    echo "decoder_lr=${DECODER_LR}"
    echo "mask_only_training_enabled=${MASK_ONLY_TRAINING_ENABLED}"
    echo "decoder_norm=${DECODER_NORM}"
    echo "source_dir=${SOURCE_DIR}"
    echo "config_source=${CONFIG}"
    echo "config_copy=${execution_dir}/config.yaml"
    echo "config_sha256=$(sha256_or_missing "${execution_dir}/config.yaml")"
    echo "git_commit=${git_commit}"
    echo "git_status_file=${execution_dir}/git_status.txt"
    printf 'extra_cli_args='
    printf '%q ' "$@"
    printf '\n'
  } > "${metadata_file}"
  {
    echo "created_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "hostname=$(hostname 2>/dev/null || printf 'unavailable')"
    echo "python=$(python --version 2>&1 | tr -d '\r')"
    python -c 'import importlib.metadata as m
for n in ("torch", "torchvision", "timm", "lightning"):
    try:
        v = m.version(n)
    except m.PackageNotFoundError:
        v = "not-installed"
    print("{}={}".format(n, v))' 2>&1 || true
    python -c 'import torch; print("torch_cuda_version={}".format(torch.version.cuda)); print("cuda_available={}".format(torch.cuda.is_available()))' 2>&1 || true
  } > "${execution_dir}/runtime_environment.txt"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
      >> "${execution_dir}/runtime_environment.txt" 2>&1 || true
  else
    echo "nvidia-smi=unavailable" >> "${execution_dir}/runtime_environment.txt"
  fi
  {
    for manifest in \
      "datasets/GlaS_5fold/sample_manifest.csv" \
      "datasets/GlaS_5fold/fold_assignments.csv" \
      "${train_dir}/samples.txt" \
      "${test_dir}/samples.txt"; do
      printf '%s  %s\n' "$(sha256_or_missing "${manifest}")" "${manifest}"
    done
  } > "${execution_dir}/manifest_sha256.txt"
  {
    for source_file in main.py configs/glas/run_protocol_b.sh \
      configs/glas/run_protocol_b_c1_abl.sh "${CONFIG}" \
      scripts/evaluate_boundary_f1_per_image.py scripts/summarize_protocol_b.py \
      tests/test_active_boundary_loss.py requirements.txt; do
      printf '%s  %s\n' "$(sha256_or_missing "${source_file}")" "${source_file}"
    done
    find models training datasets -type f -name '*.py' -print 2>/dev/null | sort -u |
      while IFS= read -r source_file; do
        printf '%s  %s\n' "$(sha256_or_missing "${source_file}")" "${source_file}"
      done
  } > "${execution_dir}/source_sha256.txt"
  effective_max_epochs=1000
  effective_num_workers=8
  effective_limit_train_batches=all
  effective_limit_val_batches=all
  runtime_args=("$@")
  for ((arg_index = 0; arg_index < ${#runtime_args[@]}; arg_index++)); do
    runtime_arg="${runtime_args[${arg_index}]}"
    case "${runtime_arg}" in
      --trainer.max_epochs)
        ((arg_index += 1))
        effective_max_epochs="${runtime_args[${arg_index}]}"
        ;;
      --trainer.max_epochs=*)
        effective_max_epochs="${runtime_arg#*=}"
        ;;
      --data.init_args.num_workers)
        ((arg_index += 1))
        effective_num_workers="${runtime_args[${arg_index}]}"
        ;;
      --data.init_args.num_workers=*)
        effective_num_workers="${runtime_arg#*=}"
        ;;
      --trainer.limit_train_batches)
        ((arg_index += 1))
        effective_limit_train_batches="${runtime_args[${arg_index}]}"
        ;;
      --trainer.limit_train_batches=*)
        effective_limit_train_batches="${runtime_arg#*=}"
        ;;
      --trainer.limit_val_batches)
        ((arg_index += 1))
        effective_limit_val_batches="${runtime_args[${arg_index}]}"
        ;;
      --trainer.limit_val_batches=*)
        effective_limit_val_batches="${runtime_arg#*=}"
        ;;
    esac
  done
  {
    echo "config=${CONFIG}"
    echo "config_sha256=$(sha256_or_missing "${execution_dir}/config.yaml")"
    echo "backbone=vit_large_patch14_reg4_dinov2"
    echo "pretrained_weight=${PRETRAINED_WEIGHT_PATH}"
    echo "pretrained_weight_sha256=$(sha256_or_missing "${PRETRAINED_WEIGHT_PATH}")"
    echo "loss=${LOSS_DESCRIPTION}"
    echo "active_boundary_loss_enabled=${ABL_ENABLED}"
    echo "active_boundary_loss_weight=${ABL_WEIGHT}"
    echo "fold=${fold}"
    echo "seed=0"
    echo "input_size=224x224"
    echo "threshold=0.5"
    echo "physical_batch_size=2"
    echo "accumulate_grad_batches=7"
    echo "effective_batch_size=14"
    echo "optimizer=AdamW"
    echo "dice_weight=0.6"
    echo "bce_weight=0.4"
    echo "abl_max_n_ratio=0.01"
    echo "abl_label_smoothing=0.2"
    echo "abl_max_clip_dist=20.0"
    echo "lr=${FIXED_LR}"
    echo "backbone_lr=${BACKBONE_LR}"
    echo "decoder_lr=${DECODER_LR}"
    echo "weight_decay=5e-4"
    echo "llrd=0.8"
    echo "llrd_l2_enabled=true"
    echo "warmup_steps=500,1000"
    echo "poly_power=0.9"
    echo "gradient_clip_val=1.0"
    echo "precision=16-mixed"
    echo "mask_only_training_enabled=${MASK_ONLY_TRAINING_ENABLED}"
    echo "decoder_norm=${DECODER_NORM}"
    echo "max_epochs=${effective_max_epochs}"
    echo "num_workers=${effective_num_workers}"
    echo "limit_train_batches=${effective_limit_train_batches}"
    echo "limit_val_batches=${effective_limit_val_batches}"
    echo "early_stopping=metrics/val_dice,mode=max,patience=300"
    echo "checkpoint_selection=metrics/val_dice,mode=max,save_top_k=1"
    echo "early_stopping_validation_split=${test_dir}"
    echo "checkpoint_metric=metrics/val_dice"
    echo "tta=false"
    printf 'runtime_overrides='
    printf '%q ' "$@"
    printf '\n'
  } > "${execution_dir}/effective_parameters.txt"
  echo "Author Protocol B | fold=${fold} | test-as-val=${test_dir} | lr=${FIXED_LR} | backbone_lr=${BACKBONE_LR} | decoder_lr=${DECODER_LR} | mask_only=${MASK_ONLY_TRAINING_ENABLED} | decoder_norm=${DECODER_NORM} | abl=${ABL_ENABLED}:${ABL_WEIGHT} | suffix=${RUN_SUFFIX}"

  fit_cmd=(python -X utf8 main.py fit -c "${CONFIG}"
    --trainer.devices "${DEVICES}" \
    --model.init_args.lr "${FIXED_LR}" \
    --model.init_args.backbone_lr "${BACKBONE_LR}" \
    --model.init_args.decoder_lr "${DECODER_LR}" \
    --model.init_args.mask_only_training_enabled "${MASK_ONLY_TRAINING_ENABLED}" \
    --model.init_args.active_boundary_loss_enabled "${ABL_ENABLED}" \
    --model.init_args.active_boundary_loss_weight "${ABL_WEIGHT}" \
    --model.init_args.network.init_args.decoder_norm "${DECODER_NORM}" \
    --trainer.logger.init_args.name "${run_name}" \
    --trainer.default_root_dir "${run_dir}" \
    --data.init_args.train_dir "${train_dir}" \
    --data.init_args.val_dir "${test_dir}" \
    --data.init_args.test_dir "${test_dir}" \
    --data.init_args.source_dir "${SOURCE_DIR}" \
    "$@")
  {
    printf 'fit_command='
    printf '%q ' "${fit_cmd[@]}"
    printf '\n'
  } >> "${metadata_file}"
  "${fit_cmd[@]}"

  version_dir="$(latest_version_dir "${run_dir}")"
  test -n "${version_dir}" || { echo "Missing logger version under ${run_dir}" >&2; exit 1; }
  echo "selected_version=${version_dir}"
  echo "logger_version=${version_dir}" >> "${metadata_file}"
  best_file="${version_dir}/best_metric.txt"
  test -f "${best_file}" || { echo "Missing ${best_file}" >&2; exit 1; }
  cp -- "${best_file}" "${execution_dir}/best_metric.txt"
  best_epoch="$(sed -n 's/^best_epoch=//p' "${best_file}" | head -n 1)"
  ckpt=""
  checkpoint_selection_source="best_epoch_exact"
  if [[ "${best_epoch}" =~ ^[0-9]+$ ]]; then
    ckpt="$(find "${version_dir}/checkpoints" -maxdepth 1 -type f -name "epoch=${best_epoch}-*.ckpt" -print | sort -V | head -n 1 || true)"
  fi

  # The custom best_metric.txt logger can record a later epoch on a rounded
  # metric tie, while Lightning's ModelCheckpoint keeps only the checkpoint
  # it actually selected. Prefer that materialized checkpoint over failing
  # the run when the two records disagree.
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(find "${version_dir}/checkpoints" -maxdepth 1 -type f \
      -name 'epoch=*.ckpt' ! -name 'last.ckpt' -print | sort -V | tail -n 1 || true)"
    if [[ -n "${ckpt}" ]]; then
      checkpoint_selection_source="materialized_modelcheckpoint_fallback"
      echo "Warning: best_epoch=${best_epoch} has no matching file; using materialized checkpoint ${ckpt}" >&2
    fi
  fi

  test -n "${ckpt}" || { echo "No usable checkpoint under ${version_dir}/checkpoints" >&2; exit 1; }
  echo "checkpoint_selection_source=${checkpoint_selection_source}" >> "${metadata_file}"
  echo "checkpoint=${ckpt}" >> "${metadata_file}"
  echo "checkpoint_sha256=$(sha256_or_missing "${ckpt}")" >> "${metadata_file}"

  test_cmd=(python -X utf8 main.py test -c "${CONFIG}"
    --model.init_args.ckpt_path "${ckpt}" \
    --model.init_args.active_boundary_loss_enabled "${ABL_ENABLED}" \
    --model.init_args.active_boundary_loss_weight "${ABL_WEIGHT}" \
    --trainer.devices "${DEVICES}" \
    --trainer.logger.init_args.name "${run_name}_test" \
    --trainer.default_root_dir "${run_dir}_test" \
    --data.init_args.train_dir "${train_dir}" \
    --data.init_args.val_dir "${test_dir}" \
    --data.init_args.test_dir "${test_dir}" \
    --data.init_args.source_dir "${SOURCE_DIR}" \
    "$@")
  {
    printf 'test_command='
    printf '%q ' "${test_cmd[@]}"
    printf '\n'
  } >> "${metadata_file}"
  # Keep the unique per-execution log and refresh the legacy run-root view
  # used by the existing Protocol B summary scripts.
  "${test_cmd[@]}" | tee "${execution_dir}/protocol_b_test.txt" "${run_dir}/protocol_b_test.txt"

  python -X utf8 scripts/evaluate_boundary_f1_per_image.py \
    --config "${CONFIG}" \
    --ckpt "${ckpt}" \
    --data-dir "${test_dir}" \
    --source-dir "${SOURCE_DIR}" \
    --output "${execution_dir}/boundary_f1_per_image.csv" \
    --device cuda
  echo "test_log=${execution_dir}/protocol_b_test.txt" >> "${metadata_file}"
  echo "boundary_f1_csv=${execution_dir}/boundary_f1_per_image.csv" >> "${metadata_file}"
done
