"""Read-only sampling audit for the existing Active Boundary Loss.

This script does not train, backpropagate, or change the ABL implementation.
It reproduces the current batch-level threshold and verifies the resulting
boundary mask against ActiveBoundaryLoss._predicted_boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.medical_binary_semantic import MedicalBinaryFolderDataset
from training.active_boundary_loss import ActiveBoundaryLoss, binary_logits_to_two_class
from visualize_binary_medical_results_flexible import load_config, load_model


OFFSETS = (
    (1, 0), (-1, 0), (0, -1), (0, 1),
    (-1, 1), (1, 1), (-1, -1), (1, -1), (0, 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ABL sampling on a frozen checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: None for key in ("min", "mean", "median", "p90", "p95", "max")}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_batch(
    abl: ActiveBoundaryLoss,
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_ids: list[str],
    batch_index: int,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    """Mirror the ABL selector; assert exact agreement with the real selector."""
    logits = logits.float()
    _, _, height, width = logits.shape
    if height < 2 or width < 2:
        raise ValueError("ABL sampling audit requires spatial dimensions >= 2.")

    vertical = abl._kl_divergence(logits[:, :, 1:, :], logits[:, :, :-1, :]).sum(1, keepdim=True)
    horizontal = abl._kl_divergence(logits[:, :, :, 1:], logits[:, :, :, :-1]).sum(1, keepdim=True)
    kl_map = F.pad(vertical, (0, 0, 0, 1)) + F.pad(horizontal, (0, 1, 0, 0))
    threshold = 1e-5
    cap = height * width * abl.max_n_ratio  # Current code applies this ONCE per batch.
    selected = kl_map > threshold
    initial_counts = selected[:, 0].sum(dim=(1, 2)).tolist()
    threshold_updates = 0
    for _ in range(256):
        if int(selected.sum().item()) <= cap:
            break
        threshold *= 1.2
        threshold_updates += 1
        selected = kl_map > threshold
    else:
        selected = torch.zeros_like(selected)
    iteration_limit_hit = threshold_updates == 256
    before = selected[:, 0]
    after = (F.max_pool2d(selected.float(), kernel_size=3, stride=1, padding=1) > 0)[:, 0]
    if not torch.equal(after, abl._predicted_boundary(logits)):
        raise RuntimeError("Diagnostic selector differs from ActiveBoundaryLoss._predicted_boundary.")

    gt_distance = abl._gt_distance_maps(target)
    batch_ids, rows, cols = torch.nonzero(after, as_tuple=True)
    if batch_ids.numel():
        distance_pad = F.pad(gt_distance.unsqueeze(1), (1, 1, 1, 1), value=1e5)[:, 0]
        neighbour_distances = torch.stack(
            [distance_pad[batch_ids, rows + dr + 1, cols + dc + 1] for dr, dc in OFFSETS],
            dim=1,
        )
        valid = neighbour_distances.argmin(dim=1) != 8
        valid_batch_ids = batch_ids[valid]
        valid_rows = rows[valid]
        valid_cols = cols[valid]
        weights = gt_distance[valid_batch_ids, valid_rows, valid_cols].clamp(
            min=0.0, max=abl.max_clip_dist
        ) / abl.max_clip_dist
    else:
        valid_batch_ids = batch_ids
        weights = gt_distance.new_empty((0,))

    actual_loss = abl(logits, target)
    batch_none = actual_loss is None
    if batch_none != (valid_batch_ids.numel() == 0):
        raise RuntimeError("Diagnostic valid-direction count disagrees with ABL None return.")

    all_selected = batch_ids.cpu().numpy()
    all_valid = valid_batch_ids.cpu().numpy()
    all_weights = weights.cpu().numpy()
    image_rows: list[dict[str, object]] = []
    for image_index, sample_id in enumerate(sample_ids):
        selected_count = int((all_selected == image_index).sum())
        own_weights = all_weights[all_valid == image_index]
        valid_count = int(own_weights.size)
        stats = distribution(own_weights)
        image_rows.append({
            "batch_index": batch_index,
            "sample_id": sample_id,
            "initial_kl_candidate_count": int(initial_counts[image_index]),
            "selected_before_dilation": int(before[image_index].sum().item()),
            "selected_after_dilation": selected_count,
            "valid_direction_count": valid_count,
            "invalid_direction_count": selected_count - valid_count,
            "valid_fraction_after_dilation": valid_count / selected_count if selected_count else None,
            "positive_weight_count": int(np.count_nonzero(own_weights > 0)),
            "zero_weight_count": int(np.count_nonzero(own_weights == 0)),
            "zero_weight_fraction": float(np.mean(own_weights == 0)) if valid_count else None,
            "weight_sum": float(own_weights.sum()),
            "weight_min": stats["min"],
            "weight_mean": stats["mean"],
            "weight_median": stats["median"],
            "weight_p90": stats["p90"],
            "weight_p95": stats["p95"],
            "weight_max": stats["max"],
            "image_no_valid_direction": valid_count == 0,
            "batch_abl_returns_none": batch_none,
        })

    batch_row: dict[str, object] = {
        "batch_index": batch_index,
        "sample_ids": ",".join(sample_ids),
        "batch_size": len(sample_ids),
        "cap_for_whole_batch": cap,
        "initial_kl_candidate_count": int(sum(initial_counts)),
        "cap_exceeded_initially": sum(initial_counts) > cap,
        "threshold_updates": threshold_updates,
        "final_kl_threshold": threshold,
        "iteration_limit_hit": iteration_limit_hit,
        "selected_before_dilation": int(before.sum().item()),
        "selected_after_dilation": int(after.sum().item()),
        "valid_direction_count": int(valid_batch_ids.numel()),
        "positive_weight_count": int(np.count_nonzero(all_weights > 0)),
        "abl_returns_none": batch_none,
        "abl_loss_unweighted": None if batch_none else float(actual_loss.item()),
    }
    return image_rows, batch_row, all_weights


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for path in (args.config, args.ckpt, args.data_dir, args.source_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")

    config = load_config(args.config)
    img_size = tuple(int(v) for v in config["data"]["init_args"]["img_size"])
    dataset = MedicalBinaryFolderDataset(args.data_dir, img_size, False, args.source_dir)
    device = torch.device(args.device)
    model = load_model(config, args.ckpt, device)
    abl = model.active_boundary_loss
    if not isinstance(abl, ActiveBoundaryLoss):
        raise RuntimeError("Loaded model has no ActiveBoundaryLoss; use the C1 config/checkpoint.")
    if not model.mask_only_training_enabled:
        raise RuntimeError("Expected the frozen mask-only C1 prediction path.")

    image_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    pooled_weights: list[np.ndarray] = []
    for start in range(0, len(dataset), args.batch_size):
        stop = min(start + args.batch_size, len(dataset))
        samples = [dataset[index] for index in range(start, stop)]
        image_tensor = torch.stack([image for image, _ in samples]).to(device)
        target = torch.stack([
            entry["masks"].any(dim=0).long() for _, entry in samples
        ]).to(device)
        sample_ids = [dataset.samples[index][0].stem for index in range(start, stop)]
        amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with torch.inference_mode(), amp:
            foreground_logits, _ = model._foreground_logits_and_probs(image_tensor)
            if foreground_logits.shape[-2:] != target.shape[-2:]:
                raise ValueError(f"Prediction/GT shape mismatch: {sample_ids}")
            if not bool(torch.isfinite(foreground_logits).all()):
                raise ValueError(f"Non-finite foreground logits: {sample_ids}")
            abl_logits = binary_logits_to_two_class(foreground_logits)
            own_rows, batch_row, weights = audit_batch(
                abl, abl_logits, target, sample_ids, len(batch_rows)
            )
        image_rows.extend(own_rows)
        batch_rows.append(batch_row)
        pooled_weights.append(weights)
        print(f"batch={len(batch_rows)}/{(len(dataset) + args.batch_size - 1) // args.batch_size} "
              f"selected={batch_row['selected_before_dilation']} "
              f"dilated={batch_row['selected_after_dilation']} "
              f"valid={batch_row['valid_direction_count']} "
              f"none={batch_row['abl_returns_none']}", flush=True)

    weight_values = np.concatenate(pooled_weights)
    all_valid = sum(int(row["valid_direction_count"]) for row in image_rows)
    summary: dict[str, object] = {
        "diagnostic": "ABL sampling, frozen-checkpoint inference; no training/backprop",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.ckpt.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "source_dir": str(args.source_dir.resolve()),
        "augmentation": "disabled",
        "sample_order": "manifest order, no shuffle",
        "inference_precision": "16-mixed" if device.type == "cuda" else "32-fp",
        "physical_batch_size": args.batch_size,
        "image_count": len(image_rows),
        "batch_count": len(batch_rows),
        "max_n_ratio": abl.max_n_ratio,
        "max_clip_dist": abl.max_clip_dist,
        "initial_candidate_count": sum(int(row["initial_kl_candidate_count"]) for row in image_rows),
        "selected_before_dilation": sum(int(row["selected_before_dilation"]) for row in image_rows),
        "selected_after_dilation": sum(int(row["selected_after_dilation"]) for row in image_rows),
        "valid_direction_count": all_valid,
        "positive_weight_count": int(np.count_nonzero(weight_values > 0)),
        "zero_weight_count": int(np.count_nonzero(weight_values == 0)),
        "zero_weight_fraction": float(np.mean(weight_values == 0)) if all_valid else None,
        "weight_distribution": distribution(weight_values),
        "image_no_valid_count": sum(bool(row["image_no_valid_direction"]) for row in image_rows),
        "image_no_valid_fraction": sum(bool(row["image_no_valid_direction"]) for row in image_rows) / len(image_rows),
        "batch_none_count": sum(bool(row["abl_returns_none"]) for row in batch_rows),
        "batch_none_fraction": sum(bool(row["abl_returns_none"]) for row in batch_rows) / len(batch_rows),
        "batch_cap_triggered_count": sum(bool(row["cap_exceeded_initially"]) for row in batch_rows),
        "batch_cap_triggered_fraction": sum(bool(row["cap_exceeded_initially"]) for row in batch_rows) / len(batch_rows),
        "selected_before_dilation_per_image": distribution(np.array([
            row["selected_before_dilation"] for row in image_rows
        ])),
        "selected_after_dilation_per_image": distribution(np.array([
            row["selected_after_dilation"] for row in image_rows
        ])),
        "valid_direction_per_image": distribution(np.array([
            row["valid_direction_count"] for row in image_rows
        ])),
        "caveat": (
            "The batch-wide cap is reproduced exactly for this deterministic sample order. "
            "No augmentation or shuffle is used, so None fractions are diagnostic "
            "checkpoint-inference values, not measured training-step frequencies."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_image.csv", image_rows)
    write_csv(args.output_dir / "per_batch.csv", batch_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: summary[key] for key in (
            "image_count", "batch_count", "batch_cap_triggered_fraction",
            "selected_before_dilation", "selected_after_dilation", "valid_direction_count",
            "zero_weight_fraction", "image_no_valid_fraction", "batch_none_fraction",
        )
    }, ensure_ascii=False, indent=2))
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
