#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="configs/glas/vit_query_mul_scale_fusion_protocol_b_c1_late_abl_noes_1000.yaml"
PRETRAINED_WEIGHT="/Disk/HDD/Sk/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
SOURCE_DIR="../Warwick_QU_Dataset"
FOLDS="${FOLDS:-0}"
CHECK_ONLY=false
RUNTIME_FLAGS=()

for arg in "$@"; do
  case "${arg}" in
    --check-only) CHECK_ONLY=true ;;
    --compile_disabled) RUNTIME_FLAGS+=("${arg}") ;;
    *) echo "Unsupported override: ${arg}" >&2; exit 2 ;;
  esac
done

python - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
trainer = config["trainer"]
model = config["model"]["init_args"]
data = config["data"]["init_args"]
callbacks = {entry["class_path"].rsplit(".", 1)[-1]: entry.get("init_args", {})
             for entry in trainer["callbacks"]}
checkpoint = callbacks.get("ModelCheckpoint")
expected = {
    "max_epochs": trainer["max_epochs"] == 1000,
    "no_early_stopping": "EarlyStopping" not in callbacks,
    "checkpoint": checkpoint is not None and checkpoint.get("monitor") == "metrics/val_dice"
                  and checkpoint.get("mode") == "max" and checkpoint.get("save_top_k") == 1
                  and checkpoint.get("save_last") is True
                  and checkpoint.get("save_weights_only") is False,
    "val_logger": "MedicalValLogger" in callbacks,
    "schedule": model["active_boundary_loss_enabled"] is True
                and model["active_boundary_loss_weight"] == 0.10
                and model["abl_schedule"] == "late" and model["abl_start_epoch"] == 800,
    "pdb": model["abl_max_n_ratio"] == 0.01,
    "threshold": model["metric_threshold"] == 0.5,
    "training": config["seed_everything"] == 0 and data["batch_size"] == 2
                and trainer["accumulate_grad_batches"] == 7
                and trainer["gradient_clip_val"] == 1.0,
    # PyYAML reads the existing 5e-5/1e-5/1e-4 YAML scalars as strings.
    # Compare their numeric values; LightningCLI converts them to floats.
    "learning_rates": float(model["lr"]) == 5e-5
                      and float(model["backbone_lr"]) == 1e-5
                      and float(model["decoder_lr"]) == 1e-4
                      and float(model["llrd"]) == 0.8
                      and model["warmup_steps"] == [500, 1000],
    "backbone": model["network"]["init_args"]["encoder"]["init_args"]["backbone_name"]
                == "vit_large_patch14_reg4_dinov2",
}
failed = [name for name, passed in expected.items() if not passed]
if failed:
    raise SystemExit("C1-Late configuration check failed: " + ", ".join(failed))
print("C1-Late configuration check passed: 1000 epochs, no EarlyStopping, ABL starts at 800")
PY

[[ -f "${PRETRAINED_WEIGHT}" ]] || {
  echo "Missing DINOv2 ViT-L pretrained weight: ${PRETRAINED_WEIGHT}" >&2
  exit 1
}
[[ -d "${SOURCE_DIR}" ]] || { echo "Missing GlaS source directory: ${SOURCE_DIR}" >&2; exit 1; }
for manifest in datasets/GlaS_5fold/sample_manifest.csv \
  datasets/GlaS_5fold/fold_assignments.csv; do
  [[ -f "${manifest}" ]] || { echo "Missing frozen manifest: ${manifest}" >&2; exit 1; }
done
read -r -a SELECTED_FOLDS <<< "${FOLDS}"
((${#SELECTED_FOLDS[@]})) || { echo "FOLDS is empty." >&2; exit 2; }
declare -A SEEN_FOLDS=()
for fold in "${SELECTED_FOLDS[@]}"; do
  [[ "${fold}" =~ ^[0-4]$ ]] || { echo "Invalid fold: ${fold}" >&2; exit 2; }
  [[ ! -v SEEN_FOLDS[${fold}] ]] || { echo "Duplicate fold: ${fold}" >&2; exit 2; }
  SEEN_FOLDS[${fold}]=1
  fold_root="datasets/GlaS_5fold/fold${fold}"
  [[ -f "${fold_root}/Train_Folder/samples.txt" && -f "${fold_root}/Test_Folder/samples.txt" ]] || {
    echo "Missing frozen fold manifests under ${fold_root}" >&2; exit 1;
  }
  run_name="glas_protocol_b_fold${fold}_vitl_c1_late800_noes1000_seed0"
  [[ ! -e "runs/${run_name}" && ! -e "runs/${run_name}_test" \
    && ! -e "runs/${run_name}_last_test" ]] || {
    echo "Run already exists; refusing to overwrite runs/${run_name}" >&2; exit 1;
  }
done

if [[ "${CHECK_ONLY}" == true ]]; then
  for fold in "${SELECTED_FOLDS[@]}"; do
    echo "READY fold=${fold} run=runs/glas_protocol_b_fold${fold}_vitl_c1_late800_noes1000_seed0"
  done
  exit 0
fi

export HF_HUB_OFFLINE=1
export TORCH_HOME="/Disk/HDD/Sk/.cache/torch"
export TIMM_USE_OLD_CACHE=1
pretrained_sha="$(sha256sum "${PRETRAINED_WEIGHT}" | awk '{print $1}')"

for fold in "${SELECTED_FOLDS[@]}"; do
  fold_root="datasets/GlaS_5fold/fold${fold}"
  train_dir="${fold_root}/Train_Folder"
  test_dir="${fold_root}/Test_Folder"
  run_name="glas_protocol_b_fold${fold}_vitl_c1_late800_noes1000_seed0"
  run_dir="runs/${run_name}"
  test_run_dir="runs/${run_name}_test"
  last_test_run_dir="runs/${run_name}_last_test"
  mkdir "${run_dir}"
  cp -- "${CONFIG}" "${run_dir}/config.yaml"
  {
    echo "protocol=author_protocol_b_test_as_val"
    echo "selection_bias=true"
    echo "fold=${fold}"
    echo "seed=0"
    echo "train_split=${train_dir}"
    echo "val_split=${test_dir}"
    echo "test_split=${test_dir}"
    echo "early_stopping=disabled"
    echo "max_epochs=1000"
    echo "abl_schedule=late"
    echo "abl_start_epoch=800"
    echo "abl_effective_weight_before=0.0"
    echo "abl_effective_weight_after=0.10"
    echo "pdb_sampling_scope=per_image"
    echo "pdb_max_n_ratio=0.01"
    echo "test_loss=base_only"
    echo "checkpoint_selection=metrics/val_dice,mode=max,save_top_k=1"
    echo "checkpoint_format=full_training_state"
    echo "single_view=true"
    echo "tta=false"
    echo "threshold=0.5"
    echo "pretrained_weight=${PRETRAINED_WEIGHT}"
    echo "pretrained_weight_sha256=${pretrained_sha}"
    echo "config_sha256=$(sha256sum "${CONFIG}" | awk '{print $1}')"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || printf unavailable)"
    echo "created_at_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "python=$(python --version 2>&1)"
    echo "disk_before=$(df -Pk "${run_dir}" | tail -n 1)"
  } > "${run_dir}/experiment_metadata.txt"
  git status --short > "${run_dir}/git_status.txt" 2>&1 || true
  {
    for source in "${CONFIG}" "configs/glas/run_protocol_b_c1_late_abl_noes_1000.sh" \
      training/medical_binary_segmentation.py training/active_boundary_loss.py \
      models/eoseg.py models/vit.py main.py tests/test_active_boundary_loss.py; do
      sha256sum "${source}"
    done
  } > "${run_dir}/source_sha256.txt"
  {
    for manifest in datasets/GlaS_5fold/sample_manifest.csv \
      datasets/GlaS_5fold/fold_assignments.csv \
      "${train_dir}/samples.txt" "${test_dir}/samples.txt"; do
      sha256sum "${manifest}"
    done
  } > "${run_dir}/manifest_sha256.txt"

  fit_cmd=(python -X utf8 main.py fit -c "${CONFIG}"
    --trainer.devices 1
    --trainer.logger.init_args.name "${run_name}"
    --trainer.default_root_dir "${run_dir}"
    --data.init_args.train_dir "${train_dir}"
    --data.init_args.val_dir "${test_dir}"
    --data.init_args.test_dir "${test_dir}"
    --data.init_args.source_dir "${SOURCE_DIR}"
    "${RUNTIME_FLAGS[@]}")
  printf 'fit_command=' >> "${run_dir}/experiment_metadata.txt"
  printf '%q ' "${fit_cmd[@]}" >> "${run_dir}/experiment_metadata.txt"
  printf '\n' >> "${run_dir}/experiment_metadata.txt"
  "${fit_cmd[@]}" 2>&1 | tee "${run_dir}/fit_console.log"

  version_dir="${run_dir}/version_0"
  [[ -f "${version_dir}/metrics.csv" ]] || { echo "Missing metrics.csv: ${version_dir}" >&2; exit 1; }
  python - "${version_dir}/metrics.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
by_epoch = {}
for row in rows:
    if row.get("train/abl_active") not in (None, ""):
        by_epoch[int(float(row["epoch"]))] = row
for epoch, expected_active, expected_weight in ((799, 0.0, 0.0), (800, 1.0, 0.10), (999, 1.0, 0.10)):
    row = by_epoch.get(epoch)
    if row is None:
        raise SystemExit(f"Missing epoch {epoch} ABL gate log; training budget is incomplete")
    active = float(row["train/abl_active"])
    weight = float(row["train/abl_effective_weight"])
    if abs(active - expected_active) > 1e-6 or abs(weight - expected_weight) > 1e-5:
        raise SystemExit(f"Wrong ABL gate at epoch {epoch}: active={active}, weight={weight}")
print("Confirmed complete epoch 999 and ABL gate 799=off, 800=on")
PY

  mapfile -t best_candidates < <(find "${version_dir}/checkpoints" -maxdepth 1 -type f \
    -name 'epoch=*.ckpt' -print)
  ((${#best_candidates[@]} == 1)) || {
    echo "Expected exactly one best checkpoint; found ${#best_candidates[@]}" >&2; exit 1;
  }
  best_ckpt="${best_candidates[0]}"
  last_ckpt="${version_dir}/checkpoints/last.ckpt"
  [[ -f "${last_ckpt}" ]] || { echo "Missing last checkpoint: ${last_ckpt}" >&2; exit 1; }
  best_filename="$(basename -- "${best_ckpt}")"
  [[ "${best_filename}" =~ ^epoch=([0-9]+)-step= ]] || {
    echo "Cannot parse best checkpoint epoch: ${best_filename}" >&2; exit 1;
  }
  best_epoch="${BASH_REMATCH[1]}"
  {
    echo "best_checkpoint=${best_ckpt}"
    echo "best_checkpoint_sha256=$(sha256sum "${best_ckpt}" | awk '{print $1}')"
    echo "best_epoch=${best_epoch}"
    if ((best_epoch >= 800)); then echo "best_epoch_after_abl_start=true";
    else echo "best_epoch_after_abl_start=false"; fi
    echo "last_checkpoint=${last_ckpt}"
    echo "last_checkpoint_sha256=$(sha256sum "${last_ckpt}" | awk '{print $1}')"
    echo "last_epoch=999"
    echo "loss_history=${version_dir}/metrics.csv"
  } >> "${run_dir}/experiment_metadata.txt"

  test_cmd=(python -X utf8 main.py test -c "${CONFIG}"
    --model.init_args.ckpt_path "${best_ckpt}"
    --trainer.devices 1
    --trainer.logger.init_args.name "${run_name}_test"
    --trainer.default_root_dir "${test_run_dir}"
    --data.init_args.train_dir "${train_dir}"
    --data.init_args.val_dir "${test_dir}"
    --data.init_args.test_dir "${test_dir}"
    --data.init_args.source_dir "${SOURCE_DIR}"
    "${RUNTIME_FLAGS[@]}")
  printf 'test_command=' >> "${run_dir}/experiment_metadata.txt"
  printf '%q ' "${test_cmd[@]}" >> "${run_dir}/experiment_metadata.txt"
  printf '\n' >> "${run_dir}/experiment_metadata.txt"
  "${test_cmd[@]}" 2>&1 | tee "${run_dir}/protocol_b_test.txt"
  python -X utf8 scripts/evaluate_boundary_f1_per_image.py \
    --config "${CONFIG}" --ckpt "${best_ckpt}" --data-dir "${test_dir}" \
    --source-dir "${SOURCE_DIR}" --output "${run_dir}/boundary_f1_per_image.csv" --device cuda

  # Fixed epoch-999 endpoint is supplementary; keep its metrics separate from best.
  last_test_cmd=(python -X utf8 main.py test -c "${CONFIG}"
    --model.init_args.ckpt_path "${last_ckpt}"
    --trainer.devices 1
    --trainer.logger.init_args.name "${run_name}_last_test"
    --trainer.default_root_dir "${last_test_run_dir}"
    --data.init_args.train_dir "${train_dir}"
    --data.init_args.val_dir "${test_dir}"
    --data.init_args.test_dir "${test_dir}"
    --data.init_args.source_dir "${SOURCE_DIR}"
    "${RUNTIME_FLAGS[@]}")
  printf 'last_test_command=' >> "${run_dir}/experiment_metadata.txt"
  printf '%q ' "${last_test_cmd[@]}" >> "${run_dir}/experiment_metadata.txt"
  printf '\n' >> "${run_dir}/experiment_metadata.txt"
  "${last_test_cmd[@]}" 2>&1 | tee "${run_dir}/protocol_b_last_test.txt"
  python -X utf8 scripts/evaluate_boundary_f1_per_image.py \
    --config "${CONFIG}" --ckpt "${last_ckpt}" --data-dir "${test_dir}" \
    --source-dir "${SOURCE_DIR}" --output "${run_dir}/boundary_f1_last_per_image.csv" --device cuda
done
