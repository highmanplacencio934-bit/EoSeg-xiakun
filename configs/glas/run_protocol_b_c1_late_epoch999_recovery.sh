#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_late_abl_epoch999_recovery.yaml"
SOURCE_DIR="../Warwick_QU_Dataset"
PRETRAINED_WEIGHT="/Disk/HDD/Sk/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
FOLDS="${FOLDS:-0}"
RUN_TAG="${RUN_TAG:-epoch999_recovery}"
CHECK_ONLY=false
for arg in "$@"; do
  case "${arg}" in
    --check-only) CHECK_ONLY=true ;;
    *) echo "Unsupported argument: ${arg}" >&2; exit 2 ;;
  esac
done
[[ "${RUN_TAG}" =~ ^[a-z0-9_]+$ ]] || { echo "Invalid RUN_TAG: ${RUN_TAG}" >&2; exit 2; }
[[ -f "${CONFIG}" && -d "${SOURCE_DIR}" && -f "${PRETRAINED_WEIGHT}" ]] || {
  echo "Missing recovery config, GlaS source, or ViT-L pretrained weight." >&2
  exit 1
}

export HF_HUB_OFFLINE=1
export TORCH_HOME="/Disk/HDD/Sk/.cache/torch"
export TIMM_USE_OLD_CACHE=1

read -r -a SELECTED_FOLDS <<< "${FOLDS}"
((${#SELECTED_FOLDS[@]})) || { echo "FOLDS is empty." >&2; exit 2; }
declare -A SEEN_FOLDS=() SOURCE_CKPTS=() SOURCE_EPOCHS=() SOURCE_STEPS=() RUN_NAMES=()

# Validate every fold before starting any training. Never overwrite original runs.
for fold in "${SELECTED_FOLDS[@]}"; do
  [[ "${fold}" =~ ^[0-4]$ && ! -v SEEN_FOLDS[${fold}] ]] || {
    echo "Invalid or duplicate fold: ${fold}" >&2; exit 2
  }
  SEEN_FOLDS[${fold}]=1
  fold_root="datasets/GlaS_5fold/fold${fold}"
  [[ -f "${fold_root}/Train_Folder/samples.txt" && -f "${fold_root}/Test_Folder/samples.txt" ]] || {
    echo "Missing frozen fold manifests: ${fold_root}" >&2; exit 1
  }
  source_dir="runs/glas_protocol_b_fold${fold}_vitl_c1_late800_noes1000_seed0/version_0/checkpoints"
  [[ -d "${source_dir}" ]] || { echo "Missing source checkpoint directory: ${source_dir}" >&2; exit 1; }
  mapfile -t candidates < <(find "${source_dir}" -maxdepth 1 -type f -name 'epoch=*.ckpt' -print)
  ((${#candidates[@]} == 1)) || {
    echo "Expected one best checkpoint in ${source_dir}; found ${#candidates[@]}" >&2; exit 1
  }
  source_ckpt="${candidates[0]}"
  source_state="$(python - "${source_ckpt}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
print(int(checkpoint["epoch"]), int(checkpoint["global_step"]))
PY
)"
  read -r source_epoch source_step <<< "${source_state}"
  ((source_epoch < 999)) || { echo "Source is already at epoch ${source_epoch}" >&2; exit 1; }
  run_name="glas_protocol_b_fold${fold}_vitl_c1_late800_noes1000_seed0_${RUN_TAG}"
  [[ ! -e "runs/${run_name}" && ! -e "runs/${run_name}_test" ]] || {
    echo "Recovery run already exists; refusing to overwrite: ${run_name}" >&2; exit 1
  }
  SOURCE_CKPTS[${fold}]="${source_ckpt}"
  SOURCE_EPOCHS[${fold}]="${source_epoch}"
  SOURCE_STEPS[${fold}]="${source_step}"
  RUN_NAMES[${fold}]="${run_name}"
  echo "READY fold=${fold} source_epoch=${source_epoch} source_step=${source_step} run=${run_name}"
done

if [[ "${CHECK_ONLY}" == true ]]; then
  exit 0
fi

for fold in "${SELECTED_FOLDS[@]}"; do
  fold_root="datasets/GlaS_5fold/fold${fold}"
  train_dir="${fold_root}/Train_Folder"
  test_dir="${fold_root}/Test_Folder"
  run_name="${RUN_NAMES[${fold}]}"
  run_dir="runs/${run_name}"
  test_run_dir="runs/${run_name}_test"
  source_ckpt="${SOURCE_CKPTS[${fold}]}"
  mkdir "${run_dir}"
  cp -- "${CONFIG}" "${run_dir}/config.yaml"
  {
    echo "protocol=author_protocol_b_test_as_val"
    echo "fold=${fold}"
    echo "seed=0"
    echo "recovery_exact_original=false"
    echo "recovery_method=resume_full_training_state_from_best_checkpoint"
    echo "source_checkpoint=${source_ckpt}"
    echo "source_epoch=${SOURCE_EPOCHS[${fold}]}"
    echo "source_global_step=${SOURCE_STEPS[${fold}]}"
    echo "source_checkpoint_sha256=$(sha256sum "${source_ckpt}" | awk '{print $1}')"
    echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
    echo "train_split=${train_dir}"
    echo "val_split=${test_dir}"
    echo "test_split=${test_dir}"
    echo "max_epochs=1000"
    echo "abl_start_epoch=800"
    echo "checkpoint_selection=fixed_epoch_999"
    echo "single_view=true"
    echo "tta=false"
    echo "threshold=0.5"
    echo "created_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || printf unavailable)"
  } > "${run_dir}/experiment_metadata.txt"

  fit_cmd=(python -X utf8 main.py fit -c "${CONFIG}"
    --ckpt_path "${source_ckpt}"
    --trainer.devices 1
    --trainer.logger.init_args.name "${run_name}"
    --trainer.default_root_dir "${run_dir}"
    --data.init_args.train_dir "${train_dir}"
    --data.init_args.val_dir "${test_dir}"
    --data.init_args.test_dir "${test_dir}"
    --data.init_args.source_dir "${SOURCE_DIR}"
    --compile_disabled)
  printf 'fit_command=' >> "${run_dir}/experiment_metadata.txt"
  printf '%q ' "${fit_cmd[@]}" >> "${run_dir}/experiment_metadata.txt"
  printf '\n' >> "${run_dir}/experiment_metadata.txt"
  "${fit_cmd[@]}" 2>&1 | tee "${run_dir}/fit_console.log"

  endpoint_dir="${run_dir}/version_0/checkpoints"
  [[ -d "${endpoint_dir}" ]] || { echo "Missing endpoint directory: ${endpoint_dir}" >&2; exit 1; }
  mapfile -t endpoints < <(find "${endpoint_dir}" -maxdepth 1 -type f \
    -name 'recovered-endpoint-epoch=999-step=*.ckpt' -print)
  ((${#endpoints[@]} == 1)) || {
    echo "Expected one explicit epoch-999 checkpoint; found ${#endpoints[@]}" >&2; exit 1
  }
  endpoint="${endpoints[0]}"
  endpoint_state="$(python - "${endpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
epoch = int(checkpoint["epoch"])
global_step = int(checkpoint["global_step"])
if (epoch, global_step) != (999, 10000):
    raise SystemExit(f"Wrong endpoint state: epoch={epoch}, global_step={global_step}")
print(f"epoch={epoch} global_step={global_step}")
PY
)"
  {
    echo "endpoint_checkpoint=${endpoint}"
    echo "endpoint_checkpoint_sha256=$(sha256sum "${endpoint}" | awk '{print $1}')"
    echo "endpoint_state=${endpoint_state}"
  } >> "${run_dir}/experiment_metadata.txt"
  echo "VERIFIED fold=${fold} ${endpoint_state} checkpoint=${endpoint}"

  test_cmd=(python -X utf8 main.py test -c "${CONFIG}"
    --model.init_args.ckpt_path "${endpoint}"
    --trainer.devices 1
    --trainer.logger.init_args.name "${run_name}_test"
    --trainer.default_root_dir "${test_run_dir}"
    --data.init_args.train_dir "${train_dir}"
    --data.init_args.val_dir "${test_dir}"
    --data.init_args.test_dir "${test_dir}"
    --data.init_args.source_dir "${SOURCE_DIR}"
    --compile_disabled)
  printf 'test_command=' >> "${run_dir}/experiment_metadata.txt"
  printf '%q ' "${test_cmd[@]}" >> "${run_dir}/experiment_metadata.txt"
  printf '\n' >> "${run_dir}/experiment_metadata.txt"
  "${test_cmd[@]}" 2>&1 | tee "${run_dir}/protocol_b_endpoint_test.txt"
  python -X utf8 scripts/evaluate_boundary_f1_per_image.py \
    --config "${CONFIG}" --ckpt "${endpoint}" --data-dir "${test_dir}" \
    --source-dir "${SOURCE_DIR}" --output "${run_dir}/boundary_f1_endpoint_per_image.csv" \
    --device cuda
done
