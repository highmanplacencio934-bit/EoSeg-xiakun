"""Read-only ABL boundary overlay for the fixed fold0/Test_Folder/train_3 case.

The checkpoint is never trained or modified. The selected ABL mask is taken
from ActiveBoundaryLoss._predicted_boundary and checked against experiment A.
This diagnostic image is not the six-panel manuscript figure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.medical_binary_semantic import MedicalBinaryFolderDataset
from diagnose_abl_sampling import OFFSETS, audit_batch
from training.active_boundary_loss import ActiveBoundaryLoss, binary_logits_to_two_class
from visualize_binary_medical_results_flexible import load_config, load_model


SAMPLE_ID = "train_3"
PREDICTION_THRESHOLD = 0.5
ROI_NATIVE = (310, 210, 455, 455)
GT_COLOR = np.array([0, 220, 255], dtype=np.uint8)
PRED_COLOR = np.array([255, 74, 0], dtype=np.uint8)
BOUNDARY_OVERLAP_COLOR = np.array([255, 235, 0], dtype=np.uint8)
ABL_SELECTED_COLOR = np.array([255, 0, 184], dtype=np.uint8)
ABL_POSITIVE_WEIGHT_COLOR = np.array([60, 255, 85], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT, prediction and ABL-selected pixels on train_3.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partner-id", default=None,
                        help="Optional fixed second sample; changes the current ABL batch-level selection cap.")
    parser.add_argument("--abl-weight-label", type=float, required=True,
                        help="Training lambda for provenance only; does not change sampling or the checkpoint.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    """8-neighbour inner contour, as used by the existing BF1 display script."""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = np.ones((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + height, dx:dx + width]
    return binary & ~eroded


def boundary_overlay(image: np.ndarray, gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    output = np.clip(image.astype(np.float32) * 0.60, 0, 255).astype(np.uint8)
    output[gt & ~prediction] = GT_COLOR
    output[prediction & ~gt] = PRED_COLOR
    output[gt & prediction] = BOUNDARY_OVERLAP_COLOR
    return output


def add_transparent_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    result = image.copy()
    if bool(mask.any()):
        mixed = (1.0 - alpha) * result[mask].astype(np.float32) + alpha * color.astype(np.float32)
        result[mask] = np.rint(mixed).clip(0, 255).astype(np.uint8)
    return result


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def direction_and_weight_maps(
    abl: ActiveBoundaryLoss,
    selected: torch.Tensor,
    gt_distance: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the ABL direction target and center-distance weight on one image."""
    rows, cols = torch.nonzero(selected, as_tuple=True)
    valid_map = torch.zeros_like(selected, dtype=torch.bool)
    weight_map = torch.zeros_like(gt_distance, dtype=torch.float32)
    if rows.numel() == 0:
        return valid_map.cpu().numpy(), weight_map.cpu().numpy()

    distance_pad = F.pad(gt_distance[None, None], (1, 1, 1, 1), value=1e5)[0, 0]
    neighbour_distances = torch.stack([
        distance_pad[rows + dr + 1, cols + dc + 1] for dr, dc in OFFSETS
    ], dim=1)
    valid = neighbour_distances.argmin(dim=1) != 8
    valid_rows, valid_cols = rows[valid], cols[valid]
    valid_map[valid_rows, valid_cols] = True
    weight_map[valid_rows, valid_cols] = gt_distance[valid_rows, valid_cols].clamp(
        min=0.0, max=abl.max_clip_dist
    ) / abl.max_clip_dist
    return valid_map.cpu().numpy(), weight_map.cpu().numpy()


def mapped_roi(native_size: tuple[int, int], eval_size: tuple[int, int]) -> tuple[int, int, int, int]:
    native_width, native_height = native_size
    eval_height, eval_width = eval_size
    x1, y1, x2, y2 = ROI_NATIVE
    if not (0 <= x1 < x2 <= native_width and 0 <= y1 < y2 <= native_height):
        raise ValueError(f"Frozen native ROI {ROI_NATIVE} does not fit image {native_size}")
    return (
        max(0, math.floor(x1 * eval_width / native_width)),
        max(0, math.floor(y1 * eval_height / native_height)),
        min(eval_width, math.ceil(x2 * eval_width / native_width)),
        min(eval_height, math.ceil(y2 * eval_height / native_height)),
    )


def main() -> None:
    args = parse_args()
    for path in (args.config, args.ckpt, args.test_dir, args.source_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    parts = {part.lower() for part in args.test_dir.parts}
    if "fold0" not in parts or "test_folder" not in parts:
        raise ValueError("Experiment C is fixed to fold0/Test_Folder/train_3")
    if args.partner_id == SAMPLE_ID:
        raise ValueError("The batch partner must differ from train_3")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")

    config = load_config(args.config)
    size = tuple(int(value) for value in config["data"]["init_args"]["img_size"])
    if size != (224, 224):
        raise ValueError(f"Expected frozen 224x224 preprocessing, got {size}")
    dataset = MedicalBinaryFolderDataset(args.test_dir, size, False, args.source_dir)
    index_by_id = {image_path.stem: index for index, (image_path, _) in enumerate(dataset.samples)}
    sample_ids = [SAMPLE_ID] + ([args.partner_id] if args.partner_id else [])
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Missing fixed sample(s) from fold0/Test_Folder: {missing}")
    samples = [dataset[index_by_id[sample_id]] for sample_id in sample_ids]
    image_path, mask_path = dataset.samples[index_by_id[SAMPLE_ID]]
    image_tensor = torch.stack([image for image, _ in samples])
    target = torch.stack([entry["masks"].any(dim=0).long() for _, entry in samples])
    input_rgb = np.rint(image_tensor[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    native_size = Image.open(image_path).size
    roi = mapped_roi(native_size, size)

    device = torch.device(args.device)
    model = load_model(config, args.ckpt, device)
    abl = model.active_boundary_loss
    if not isinstance(abl, ActiveBoundaryLoss) or not model.mask_only_training_enabled:
        raise RuntimeError("Expected the existing mask-only C1 model with ABL enabled")
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), amp:
        foreground_logits, foreground_probs = model._foreground_logits_and_probs(image_tensor.to(device))
        target = target.to(device)
        if foreground_probs.shape[-2:] != size:
            raise ValueError("Checkpoint prediction resolution differs from 224x224 GT")
        if not bool(torch.isfinite(foreground_logits).all()):
            raise ValueError("Non-finite foreground logits")
        abl_logits = binary_logits_to_two_class(foreground_logits)
        image_rows, batch_row, _ = audit_batch(abl, abl_logits, target, sample_ids, 0)
        selected = abl._predicted_boundary(abl_logits.float())[0]
        gt_distance = abl._gt_distance_maps(target)[0]
        valid_map, weight_map = direction_and_weight_maps(abl, selected, gt_distance)

    gt_mask = target[0].cpu().numpy().astype(bool)
    prediction_mask = (foreground_probs[0, 0] >= PREDICTION_THRESHOLD).cpu().numpy().astype(bool)
    selected_mask = selected.cpu().numpy().astype(bool)
    positive_weight_mask = weight_map > 0
    gt_boundary = inner_boundary(gt_mask)
    prediction_boundary = inner_boundary(prediction_mask)
    if int(selected_mask.sum()) != image_rows[0]["selected_after_dilation"]:
        raise RuntimeError("Selected-mask count differs from experiment A")
    if int(valid_map.sum()) != image_rows[0]["valid_direction_count"]:
        raise RuntimeError("Valid-direction count differs from experiment A")
    if int(positive_weight_mask.sum()) != image_rows[0]["positive_weight_count"]:
        raise RuntimeError("Positive-weight count differs from experiment A")

    boundaries = boundary_overlay(input_rgb, gt_boundary, prediction_boundary)
    selected_overlay = add_transparent_mask(boundaries, selected_mask, ABL_SELECTED_COLOR, 0.55)
    effective_overlay = add_transparent_mask(boundaries, positive_weight_mask, ABL_POSITIVE_WEIGHT_COLOR, 0.85)
    x1, y1, x2, y2 = roi
    roi_overlay = Image.fromarray(selected_overlay[y1:y2, x1:x2]).resize(
        ((x2 - x1) * 6, (y2 - y1) * 6), resample=Image.Resampling.NEAREST
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(input_rgb).save(args.output_dir / "input_224.png")
    Image.fromarray(boundaries).save(args.output_dir / "gt_prediction_boundaries.png")
    Image.fromarray(selected_overlay).save(args.output_dir / "gt_prediction_abl_selected.png")
    Image.fromarray(effective_overlay).save(args.output_dir / "gt_prediction_abl_positive_weight.png")
    roi_overlay.save(args.output_dir / "selected_overlay_fixed_roi_zoom.png")
    for filename, mask in (
        ("gt_mask.png", gt_mask),
        ("prediction_mask.png", prediction_mask),
        ("gt_boundary.png", gt_boundary),
        ("prediction_boundary.png", prediction_boundary),
        ("abl_selected_after_dilation.png", selected_mask),
        ("abl_valid_direction.png", valid_map),
        ("abl_positive_weight.png", positive_weight_mask),
    ):
        save_binary_mask(args.output_dir / filename, mask)
    np.savez_compressed(
        args.output_dir / "pixel_maps_224.npz",
        gt_mask=gt_mask,
        prediction_mask=prediction_mask,
        gt_boundary=gt_boundary,
        prediction_boundary=prediction_boundary,
        abl_selected_after_dilation=selected_mask,
        abl_valid_direction=valid_map,
        abl_weight=weight_map.astype(np.float32),
    )
    metadata = {
        "experiment": "C: fixed-sample ABL boundary visualization; read-only",
        "sample_id": SAMPLE_ID,
        "partner_id": args.partner_id,
        "batch_size": len(sample_ids),
        "batch_context": "deterministic Test_Folder batch; not an observed training batch",
        "protocol": "author_protocol_b_test_as_val",
        "split": "fold0/Test_Folder",
        "checkpoint": str(args.ckpt.resolve()),
        "config": str(args.config.resolve()),
        "training_abl_weight_label_only": args.abl_weight_label,
        "lambda_changes_sampling": False,
        "source_image": str(image_path.resolve()),
        "gt_image": str(mask_path.resolve()),
        "input_size": size,
        "native_size_wh": native_size,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "tta": False,
        "augmentation": False,
        "precision": "16-mixed" if device.type == "cuda" else "32-fp",
        "boundary_definition": "8-neighbour inner contour at 224x224; no display dilation",
        "abl_selection": "ActiveBoundaryLoss._predicted_boundary after 3x3 dilation, batch-level cap",
        "abl_max_n_ratio": abl.max_n_ratio,
        "abl_max_clip_dist": abl.max_clip_dist,
        "roi_native_xyxy": ROI_NATIVE,
        "roi_224_xyxy": roi,
        "gt_boundary_pixels": int(gt_boundary.sum()),
        "prediction_boundary_pixels": int(prediction_boundary.sum()),
        "selected_pixels": int(selected_mask.sum()),
        "valid_direction_pixels": int(valid_map.sum()),
        "positive_weight_pixels": int(positive_weight_mask.sum()),
        "weight_sum": float(weight_map.sum()),
        "abl_image_sampling_row": image_rows[0],
        "abl_batch_sampling_row": batch_row,
        "display_colors": {
            "gt_boundary": "#00DCFF",
            "prediction_boundary": "#FF4A00",
            "gt_prediction_overlap": "#FFEB00",
            "abl_selected_overlay": "#FF00B8 at 55% opacity",
            "abl_positive_weight_overlay": "#3CFF55 at 85% opacity",
        },
        "display_note": "Colors blend where masks overlap; inspect individual PNGs or NPZ for exact membership. ROI is display-only.",
        "caveat": "train_3 is in fold0 Test_Folder and did not contribute to fold0 training; this cannot show actual training-time ABL pixel sampling.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: metadata[key] for key in (
            "sample_id", "partner_id", "batch_size", "selected_pixels",
            "valid_direction_pixels", "positive_weight_pixels", "weight_sum",
        )
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"saved={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
