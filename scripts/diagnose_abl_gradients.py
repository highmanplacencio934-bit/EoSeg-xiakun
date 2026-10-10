"""Compare base-loss and ABL gradients on one frozen C1 batch.

No optimizer, backward accumulation, parameter update, or training run occurs.
The two losses share one forward pass and exactly the same decoder parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.medical_binary_semantic import MedicalBinaryFolderDataset
from training.active_boundary_loss import binary_logits_to_two_class
from visualize_binary_medical_results_flexible import load_config, load_model


DECODER_PREFIXES = {
    "feature_upscale": "network.feature_upscale.",
    "mask_head": "network.mask_head.",
    "layer_gate": "network.layer_gate.",
}
DECODER_SCALARS = {
    "network.query_embed": "query_embed",
    "network.layer_fusion_bias": "layer_fusion_bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only C1 base-vs-ABL gradient diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", nargs="+", default=["train_1", "train_2"])
    parser.add_argument("--abl-weight", type=float, required=True,
                        help="ABL coefficient used for contribution ratios; supply the run's actual lambda.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def decoder_part(name: str) -> str | None:
    if name in DECODER_SCALARS:
        return DECODER_SCALARS[name]
    for part, prefix in DECODER_PREFIXES.items():
        if name.startswith(prefix):
            return part
    return None


def grad_sums(base: torch.Tensor | None, abl: torch.Tensor | None) -> tuple[float, float, float]:
    """Return ||base||², ||abl||², dot product without flattening large tensors."""
    base_sq = 0.0 if base is None else float(torch.sum(base.float().square(), dtype=torch.float64).item())
    abl_sq = 0.0 if abl is None else float(torch.sum(abl.float().square(), dtype=torch.float64).item())
    dot = 0.0
    if base is not None and abl is not None:
        dot = float(torch.sum(base.float() * abl.float(), dtype=torch.float64).item())
    return base_sq, abl_sq, dot


def measures(base_sq: float, abl_sq: float, dot: float, abl_weight: float) -> dict[str, float | None]:
    base_norm = base_sq ** 0.5
    abl_norm = abl_sq ** 0.5
    weighted_abl_norm = abl_weight * abl_norm
    combined_sq = base_sq + abl_weight ** 2 * abl_sq + 2 * abl_weight * dot
    return {
        "base_grad_norm": base_norm,
        "abl_grad_norm_unweighted": abl_norm,
        "abl_grad_norm_weighted": weighted_abl_norm,
        "weighted_abl_to_base_ratio": weighted_abl_norm / base_norm if base_norm else None,
        "cosine_base_vs_abl": dot / (base_norm * abl_norm) if base_norm and abl_norm else None,
        "weighted_abl_projection_on_base_ratio": abl_weight * dot / base_sq if base_sq else None,
        "combined_grad_norm": max(combined_sq, 0.0) ** 0.5,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not (args.abl_weight > 0):
        raise ValueError("--abl-weight must be positive")
    if len(args.sample_ids) != 2 or len(set(args.sample_ids)) != 2:
        raise ValueError("Provide exactly two distinct --sample-ids for the physical batch size of 2")
    for path in (args.config, args.ckpt, args.data_dir, args.source_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output_dir}")

    config = load_config(args.config)
    image_size = tuple(int(v) for v in config["data"]["init_args"]["img_size"])
    dataset = MedicalBinaryFolderDataset(args.data_dir, image_size, False, args.source_dir)
    index_by_id = {image_path.stem: index for index, (image_path, _) in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in args.sample_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Samples not in fixed training manifest: {missing}")
    selected_samples = [dataset[index_by_id[sample_id]] for sample_id in args.sample_ids]
    device = torch.device(args.device)
    model = load_model(config, args.ckpt, device)
    if not model.mask_only_training_enabled or model.active_boundary_loss is None:
        raise RuntimeError("Expected mask-only C1 with Active Boundary Loss enabled")

    selected_named = [
        (name, parameter, decoder_part(name))
        for name, parameter in model.named_parameters()
        if decoder_part(name) is not None
    ]
    if not selected_named:
        raise RuntimeError("No shared decoder parameters matched the expected module names")
    selected_names = {name for name, _, _ in selected_named}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected_names)
    params = [parameter for _, parameter, _ in selected_named]

    images = torch.stack([image for image, _ in selected_samples]).to(device)
    targets = model._targets_to_binary_masks([target for _, target in selected_samples]).to(device)
    # FP32 avoids mistaking unscaled AMP underflow for a weak ABL gradient.
    foreground_logits, foreground_probs = model._foreground_logits_and_probs(images)
    if foreground_logits.shape != targets.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {foreground_logits.shape} vs {targets.shape}")
    loss_base = model.criterion(foreground_logits, foreground_probs, targets)
    abl_logits = binary_logits_to_two_class(foreground_logits)
    loss_abl = model.active_boundary_loss(abl_logits, targets[:, 0].long())

    if loss_abl is None:
        raise RuntimeError("ABL returned None on this fixed batch; choose a documented nonempty batch from experiment A")
    if not bool(torch.isfinite(loss_base)) or not bool(torch.isfinite(loss_abl)):
        raise FloatingPointError("Non-finite base or ABL loss")
    if not loss_base.requires_grad or not loss_abl.requires_grad:
        raise RuntimeError("One loss has no gradient path to the selected decoder parameters")

    # The first call retains the SAME forward graph for the second loss.
    # autograd.grad does not write parameter .grad or invoke an optimizer.
    base_grads = torch.autograd.grad(loss_base, params, retain_graph=True, allow_unused=True)
    abl_grads = torch.autograd.grad(loss_abl, params, allow_unused=True)
    if any(parameter.grad is not None for parameter in params):
        raise RuntimeError("Unexpected accumulated parameter gradients")

    per_param: list[dict[str, object]] = []
    grouped: dict[str, dict[str, float | int]] = {}
    for (name, parameter, part), base_grad, abl_grad in zip(selected_named, base_grads, abl_grads):
        base_sq, abl_sq, dot = grad_sums(base_grad, abl_grad)
        row: dict[str, object] = {
            "parameter": name,
            "decoder_part": part,
            "parameter_count": parameter.numel(),
            "base_grad_present": base_grad is not None,
            "abl_grad_present": abl_grad is not None,
            **measures(base_sq, abl_sq, dot, args.abl_weight),
        }
        per_param.append(row)
        totals = grouped.setdefault(part, {
            "parameter_count": 0, "base_sq": 0.0, "abl_sq": 0.0, "dot": 0.0,
            "base_grad_present_count": 0, "abl_grad_present_count": 0,
        })
        totals["parameter_count"] += parameter.numel()
        totals["base_sq"] += base_sq
        totals["abl_sq"] += abl_sq
        totals["dot"] += dot
        totals["base_grad_present_count"] += int(base_grad is not None)
        totals["abl_grad_present_count"] += int(abl_grad is not None)

    per_group: list[dict[str, object]] = []
    for part, totals in grouped.items():
        per_group.append({
            "decoder_part": part,
            "parameter_count": totals["parameter_count"],
            "base_grad_present_count": totals["base_grad_present_count"],
            "abl_grad_present_count": totals["abl_grad_present_count"],
            **measures(totals["base_sq"], totals["abl_sq"], totals["dot"], args.abl_weight),
        })
    whole_base_sq = sum(float(totals["base_sq"]) for totals in grouped.values())
    whole_abl_sq = sum(float(totals["abl_sq"]) for totals in grouped.values())
    whole_dot = sum(float(totals["dot"]) for totals in grouped.values())
    if not all(math.isfinite(value) for value in (whole_base_sq, whole_abl_sq, whole_dot)):
        raise FloatingPointError("Non-finite decoder gradient norm or dot product")
    overall = measures(whole_base_sq, whole_abl_sq, whole_dot, args.abl_weight)
    if overall["base_grad_norm"] == 0 or overall["abl_grad_norm_unweighted"] == 0:
        raise RuntimeError("A selected loss has zero decoder gradient; inspect per-parameter values")

    summary: dict[str, object] = {
        "diagnostic": "one fixed batch, same decoder parameters, two losses; no parameter update",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.ckpt.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "sample_ids": args.sample_ids,
        "batch_size": len(args.sample_ids),
        "augmentation": "disabled",
        "model_mode": "eval with autograd enabled",
        "precision": "32-fp",
        "base_loss": float(loss_base.detach().item()),
        "abl_loss_unweighted": float(loss_abl.detach().item()),
        "abl_weight_for_comparison": args.abl_weight,
        "weighted_abl_loss": float(args.abl_weight * loss_abl.detach().item()),
        "selected_group": "shared mask-only decoder (feature_upscale, mask_head, layer_gate, query_embed, layer_fusion_bias)",
        "selected_parameter_count": sum(parameter.numel() for _, parameter, _ in selected_named),
        "selected_tensor_count": len(selected_named),
        "overall": overall,
        "caveats": [
            "One deterministic batch at one checkpoint is diagnostic, not an across-training estimate.",
            "FP32 gradient computation avoids AMP underflow but is not identical to the 16-mixed training step.",
            "Raw gradient ratio/cosine do not equal AdamW update ratio, especially with optimizer state and clipping.",
            "Parameters unused by mask-only prediction (such as class_head) and the ViT backbone are excluded.",
            "The explicit --abl-weight is used because the YAML default may differ from this checkpoint's CLI training override.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_parameter.csv", per_param)
    write_csv(args.output_dir / "per_decoder_part.csv", per_group)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "sample_ids": args.sample_ids,
        "base_loss": summary["base_loss"],
        "abl_loss_unweighted": summary["abl_loss_unweighted"],
        "abl_weight_for_comparison": args.abl_weight,
        "overall": overall,
    }, ensure_ascii=False, indent=2))
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
