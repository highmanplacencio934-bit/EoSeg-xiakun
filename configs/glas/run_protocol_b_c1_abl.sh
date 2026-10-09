#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

assert_fixed_env() {
  local name="$1"
  local expected="$2"
  if [[ -v "${name}" ]]; then
    local actual="${!name}"
    if [[ "${actual}" != "${expected}" ]]; then
      echo "${name} is locked to '${expected}' for the C1 experiment; got '${actual}'." >&2
      exit 2
    fi
  fi
}

# C1 is a controlled ablation: only fold selection, the one-GPU runtime, and
# explicitly marked smoke-test limits may vary. Core experiment parameters
# are fixed here instead of inheriting shell state silently.
assert_fixed_env ABL_ENABLED true
if [[ -v ABL_WEIGHT ]]; then
  if [[ "${ABL_WEIGHT}" != "0.10" && "${ABL_WEIGHT}" != "0.1" ]]; then
    echo "ABL_WEIGHT is locked to 0.10 for C1; got '${ABL_WEIGHT}'." >&2
    exit 2
  fi
fi
assert_fixed_env FIXED_LR 5e-5
assert_fixed_env BACKBONE_LR 1e-5
assert_fixed_env DECODER_LR 1e-4
assert_fixed_env MASK_ONLY_TRAINING_ENABLED true
assert_fixed_env DECODER_NORM group
assert_fixed_env SOURCE_DIR ../Warwick_QU_Dataset
assert_fixed_env DEVICES 1
assert_fixed_env CONFIG configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_abl.yaml
assert_fixed_env PRETRAINED_WEIGHT_PATH /Disk/HDD/Sk/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
assert_fixed_env MODEL_TAG c1_abl

ABL_WEIGHT="0.10"
expected_loss_description="WeightedDiceBCE(dice=0.6,bce=0.4)+${ABL_WEIGHT}*ActiveBoundaryLoss"
assert_fixed_env LOSS_DESCRIPTION "${expected_loss_description}"
if [[ -v RUN_SUFFIX ]]; then
  echo "RUN_SUFFIX is managed by the C1 wrapper; use C1_RUN_SUFFIX=smoke only for a smoke run." >&2
  exit 2
fi
WEIGHT_TAG="${ABL_WEIGHT//./p}"
WEIGHT_TAG="${WEIGHT_TAG//-/m}"
C1_RUN_SUFFIX="${C1_RUN_SUFFIX:-}"
if [[ -n "${C1_RUN_SUFFIX}" && ! "${C1_RUN_SUFFIX}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "C1_RUN_SUFFIX may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi

smoke_mode=false
if [[ "${C1_RUN_SUFFIX}" == "smoke" ]]; then
  smoke_mode=true
fi

validate_cli_overrides() {
  local arg value saw_epochs=false saw_train_limit=false saw_val_limit=false
  while (($#)); do
    arg="$1"
    shift
    case "${arg}" in
      --compile_disabled)
        ;;
      --data.init_args.num_workers)
        (($#)) || { echo "Missing value for ${arg}." >&2; exit 2; }
        value="$1"
        shift
        if [[ "${smoke_mode}" != true || ! "${value}" =~ ^[0-9]+$ ]]; then
          echo "num_workers may only be changed to a non-negative integer in C1 smoke mode." >&2
          exit 2
        fi
        ;;
      --trainer.max_epochs)
        (($#)) || { echo "Missing value for ${arg}." >&2; exit 2; }
        value="$1"
        shift
        [[ "${smoke_mode}" == true && "${value}" == "1" ]] || {
          echo "${arg} may only be set to 1 when C1_RUN_SUFFIX=smoke." >&2
          exit 2
        }
        saw_epochs=true
        ;;
      --trainer.limit_train_batches)
        (($#)) || { echo "Missing value for ${arg}." >&2; exit 2; }
        value="$1"
        shift
        [[ "${smoke_mode}" == true && "${value}" == "1" ]] || {
          echo "${arg} may only be set to 1 when C1_RUN_SUFFIX=smoke." >&2
          exit 2
        }
        saw_train_limit=true
        ;;
      --trainer.limit_val_batches)
        (($#)) || { echo "Missing value for ${arg}." >&2; exit 2; }
        value="$1"
        shift
        [[ "${smoke_mode}" == true && "${value}" == "1" ]] || {
          echo "${arg} may only be set to 1 when C1_RUN_SUFFIX=smoke." >&2
          exit 2
        }
        saw_val_limit=true
        ;;
      --data.init_args.num_workers=*)
        value="${arg#*=}"
        [[ "${smoke_mode}" == true && "${value}" =~ ^[0-9]+$ ]] || {
          echo "num_workers may only be changed to a non-negative integer in C1 smoke mode." >&2
          exit 2
        }
        ;;
      --trainer.max_epochs=1)
        [[ "${smoke_mode}" == true ]] || {
          echo "Training limits may only be set when C1_RUN_SUFFIX=smoke." >&2
          exit 2
        }
        saw_epochs=true
        ;;
      --trainer.limit_train_batches=1)
        [[ "${smoke_mode}" == true ]] || {
          echo "Training limits may only be set when C1_RUN_SUFFIX=smoke." >&2
          exit 2
        }
        saw_train_limit=true
        ;;
      --trainer.limit_val_batches=1)
        [[ "${smoke_mode}" == true ]] || {
          echo "Training limits may only be set when C1_RUN_SUFFIX=smoke." >&2
          exit 2
        }
        saw_val_limit=true
        ;;
      *)
        echo "Unapproved C1 CLI override '${arg}'. Core C1 settings are frozen; use only documented runtime/smoke flags." >&2
        exit 2
        ;;
    esac
  done
  if [[ "${smoke_mode}" == true && ( "${saw_epochs}" != true || "${saw_train_limit}" != true || "${saw_val_limit}" != true ) ]]; then
    echo "C1 smoke mode requires --trainer.max_epochs 1 --trainer.limit_train_batches 1 --trainer.limit_val_batches 1." >&2
    exit 2
  fi
}
validate_cli_overrides "$@"

export CONFIG="configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_abl.yaml"
export SOURCE_DIR="../Warwick_QU_Dataset"
export FIXED_LR="5e-5"
export BACKBONE_LR="1e-5"
export DECODER_LR="1e-4"
export MASK_ONLY_TRAINING_ENABLED="true"
export DECODER_NORM="group"
export DEVICES="1"
export PRETRAINED_WEIGHT_PATH="/Disk/HDD/Sk/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
export ABL_ENABLED="true"
export ABL_WEIGHT
export LOSS_DESCRIPTION="${expected_loss_description}"
export MODEL_TAG="c1_abl"
export RUN_SUFFIX="_w${WEIGHT_TAG}${C1_RUN_SUFFIX:+_${C1_RUN_SUFFIX}}"

expected_config_sha256="a152008a59d7d35bb91a2275b229886de40cc62596416d543bb02f208e17a263"
actual_config_sha256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
if [[ "${actual_config_sha256}" != "${expected_config_sha256}" ]]; then
  echo "The locked C1 YAML hash changed. Review/update this wrapper deliberately before training." >&2
  exit 2
fi

exec bash "${PROJECT_ROOT}/configs/glas/run_protocol_b.sh" "$@"
