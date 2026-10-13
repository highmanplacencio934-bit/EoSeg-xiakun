"""Create a fixed fold0/train_3 baseline or baseline-versus-ABL figure.

When both checkpoints are supplied, the output is a six-panel comparison plus
a paired zoom. Without an ABL checkpoint, the script emits an explicitly
baseline-only preview and never fabricates ABL panels.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from datasets.medical_binary_semantic import MedicalBinaryFolderDataset
from training.boundary_f1 import boundary_f1_score
from visualize_binary_medical_results_flexible import infer_binary_mask, load_config, load_model
from audit_panel_alignment import require_matplotlib_panel_alignment


SAMPLE_ID = "train_3"
THRESHOLD = 0.5
# Frozen after inspecting GT and C0 only; source train_3 is 775x522.
ROI = (310, 210, 455, 455)
GT_COLOR = np.array([0, 220, 255], dtype=np.uint8)
PRED_COLOR = np.array([255, 74, 0], dtype=np.uint8)
OVERLAP_COLOR = np.array([255, 235, 0], dtype=np.uint8)
HALO_COLOR = np.array([12, 12, 16], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the fixed fold0/Test_Folder/train_3 case."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", "--baseline-ckpt", dest="baseline_ckpt", type=Path, required=True)
    parser.add_argument("--abl-ckpt", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/paper_visualizations/fold0/train_3_paper_figure.png"),
        help="Output figure path; PNG/TIFF/PDF/SVG and metadata share this stem.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def dilate3(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    result = np.zeros((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = np.ones((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + height, dx : dx + width]
    return binary & ~eroded


def make_overlay(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    gt_stroke = dilate3(inner_boundary(gt_mask))
    pred_stroke = dilate3(inner_boundary(pred_mask))
    halo = dilate3(gt_stroke | pred_stroke)
    overlay = np.asarray(image_rgb, dtype=np.uint8).copy()
    overlay[halo] = HALO_COLOR
    overlay[gt_stroke & ~pred_stroke] = GT_COLOR
    overlay[pred_stroke & ~gt_stroke] = PRED_COLOR
    overlay[gt_stroke & pred_stroke] = OVERLAP_COLOR
    return overlay


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    resized = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    resized = resized.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def overlap_dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    intersection = int(np.logical_and(pred, gt).sum())
    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    dice = (2.0 * intersection + 1e-15) / (pred_count + gt_count + 1e-15)
    union = pred_count + gt_count - intersection
    iou = (intersection + 1e-15) / (union + 1e-15)
    return dice, iou


def save_figure_formats(
    fig: plt.Figure,
    output_stem: Path,
    panel_ids: list[str] | None = None,
) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if len(fig.axes) > 1:
        if panel_ids is None:
            raise ValueError("Panel IDs are required for multi-panel figure alignment QA.")
        require_matplotlib_panel_alignment(
            fig,
            axes=fig.axes,
            panel_ids=panel_ids,
            json_out=output_stem.with_name(output_stem.name + "_alignment.json"),
            overlay_svg=output_stem.with_name(output_stem.name + "_alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    outputs = [
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".tiff"),
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".svg"),
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[3], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def add_roi_box(ax: plt.Axes) -> None:
    x1, y1, x2, y2 = ROI
    ax.add_patch(
        Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1.5)
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    from matplotlib.transforms import ScaledTranslation

    offset = ScaledTranslation(-4 / 72, 3 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        label,
        transform=ax.transAxes + offset,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def add_boundary_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], color=tuple(GT_COLOR / 255.0), linewidth=3, label="GT boundary"),
        Line2D([0], [0], color=tuple(PRED_COLOR / 255.0), linewidth=3, label="Prediction boundary"),
        Line2D(
            [0], [0], color=tuple(OVERLAP_COLOR / 255.0), linewidth=3,
            label="Intersecting display strokes",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
        }
    )
    for path in (args.config, args.baseline_ckpt, args.test_dir, args.source_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.abl_ckpt is not None and not args.abl_ckpt.is_file():
        raise FileNotFoundError(args.abl_ckpt)

    path_parts = {part.lower() for part in args.test_dir.parts}
    if "fold0" not in path_parts or "test_folder" not in path_parts:
        raise ValueError("The paper case is fixed to fold0/Test_Folder/train_3.")

    config = load_config(args.config)
    data_init = config["data"]["init_args"]
    img_size = tuple(int(value) for value in data_init["img_size"])
    if img_size != (224, 224):
        raise ValueError(f"Expected the frozen 224x224 evaluation input, got {img_size}.")
    dataset = MedicalBinaryFolderDataset(
        dataset_dir=args.test_dir,
        img_size=img_size,
        train=False,
        source_dir=args.source_dir,
    )
    sample_ids = [image_path.stem for image_path, _ in dataset.samples]
    if SAMPLE_ID not in sample_ids:
        raise ValueError(f"Fixed sample_id={SAMPLE_ID} is missing from {args.test_dir}.")
    sample_index = sample_ids.index(SAMPLE_ID)
    image_path, mask_path = dataset.samples[sample_index]
    image_tensor, target = dataset[sample_index]
    gt_eval = target["masks"].any(dim=0).cpu().numpy().astype(bool)
    original_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    gt_native = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    if gt_eval.shape != img_size:
        raise ValueError(f"Expected 224x224 GT for BF1, got {gt_eval.shape}.")
    if gt_native.shape != original_rgb.shape[:2]:
        raise ValueError("Native GT mask and source image dimensions differ.")
    x1, y1, x2, y2 = ROI
    height, width = gt_native.shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Frozen ROI {ROI} does not fit source image size {(width, height)}.")

    device = torch.device(args.device)
    baseline_model = load_model(config, args.baseline_ckpt, device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        baseline_eval = infer_binary_mask(
            model=baseline_model,
            image_tensor=image_tensor,
            output_size=img_size,
            threshold=THRESHOLD,
            device=device,
        ).astype(bool)
    del baseline_model
    baseline_native = resize_mask(baseline_eval, gt_native.shape)
    baseline_scores = (
        *overlap_dice_iou(baseline_eval, gt_eval),
        boundary_f1_score(baseline_eval, gt_eval),
    )
    baseline_overlay = make_overlay(original_rgb, gt_native, baseline_native)

    abl_eval = None
    abl_native = None
    abl_scores = None
    abl_overlay = None
    if args.abl_ckpt is not None:
        abl_model = load_model(config, args.abl_ckpt, device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            abl_eval = infer_binary_mask(
                model=abl_model,
                image_tensor=image_tensor,
                output_size=img_size,
                threshold=THRESHOLD,
                device=device,
            ).astype(bool)
        del abl_model
        abl_native = resize_mask(abl_eval, gt_native.shape)
        abl_scores = (
            *overlap_dice_iou(abl_eval, gt_eval),
            boundary_f1_score(abl_eval, gt_eval),
        )
        abl_overlay = make_overlay(original_rgb, gt_native, abl_native)

    output_stem = args.output.with_suffix("") if args.output.suffix else args.output
    if abl_eval is None:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
        panels = [
            ("a", original_rgb, "Original"),
            ("b", gt_native, "Ground Truth"),
            ("c", baseline_native, "C0 Baseline mask"),
            ("e", baseline_overlay, f"C0 boundary overlay | train_3 BF1={baseline_scores[2]:.3f}"),
        ]
        for ax, (label, panel, title) in zip(axes.flat, panels):
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None, interpolation="nearest")
            ax.set_title(title, fontsize=9, pad=5)
            ax.axis("off")
            add_panel_label(ax, label)
        add_roi_box(axes[1, 1])
        add_boundary_legend(fig)
        fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.98))
        figure_outputs = save_figure_formats(
            fig,
            output_stem.with_name(output_stem.name + "_baseline_preview"),
            panel_ids=["a", "b", "c", "e"],
        )
        zoom_stem = output_stem.with_name(output_stem.name + "_baseline_zoom")
        crop = baseline_overlay[y1:y2, x1:x2]
        fig_zoom, ax_zoom = plt.subplots(figsize=(5.5, 5.5))
        ax_zoom.imshow(crop, interpolation="nearest")
        ax_zoom.set_title("fold0/train_3 | C0 boundary zoom", fontsize=10)
        ax_zoom.axis("off")
        figure_outputs.extend(save_figure_formats(fig_zoom, zoom_stem))
        mode = "baseline_only_preview; ABL checkpoint not supplied"
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.2))
        panels = [
            ("a", original_rgb, "Original"),
            ("b", gt_native, "Ground Truth"),
            ("c", baseline_native, "C0 Baseline mask"),
            ("d", abl_native, "C0 + ABL mask"),
            ("e", baseline_overlay, f"C0 boundary overlay | train_3 BF1={baseline_scores[2]:.3f}"),
            ("f", abl_overlay, f"C0 + ABL boundary overlay | train_3 BF1={abl_scores[2]:.3f}"),
        ]
        for ax, (label, panel, title) in zip(axes.flat, panels):
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None, interpolation="nearest")
            ax.set_title(title, fontsize=9, pad=5)
            ax.axis("off")
            add_panel_label(ax, label)
        add_roi_box(axes[1, 1])
        add_roi_box(axes[1, 2])
        add_boundary_legend(fig)
        fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.98))
        figure_outputs = save_figure_formats(
            fig,
            output_stem.with_name(output_stem.name + "_six_panel"),
            panel_ids=["a", "b", "c", "d", "e", "f"],
        )

        zoom_stem = output_stem.with_name(output_stem.name + "_paired_zoom")
        fig_zoom, zoom_axes = plt.subplots(1, 2, figsize=(9.0, 5.0))
        for ax, label, overlay, title in zip(
            zoom_axes,
            ("a", "b"),
            (baseline_overlay, abl_overlay),
            ("C0 Baseline", "C0 + ABL"),
        ):
            ax.imshow(overlay[y1:y2, x1:x2], interpolation="nearest")
            ax.set_title(title, fontsize=9, pad=5)
            ax.axis("off")
            add_panel_label(ax, label)
        fig_zoom.tight_layout()
        figure_outputs.extend(save_figure_formats(fig_zoom, zoom_stem, panel_ids=["a", "b"]))
        mode = "six_panel_C0_vs_C1"

    note_path = output_stem.with_name(output_stem.name + "_metadata.txt")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    notes = [
        f"sample_id={SAMPLE_ID}",
        "protocol=author_protocol_b_test_as_val",
        "split=fold0/Test_Folder",
        f"source_image={image_path}",
        f"gt_mask={mask_path}",
        f"baseline_checkpoint={args.baseline_ckpt}",
        f"abl_checkpoint={args.abl_ckpt if args.abl_ckpt else 'not_supplied'}",
        f"config={args.config}",
        f"threshold={THRESHOLD}",
        "inference=single-view; TTA=false; no GT-dependent prediction correction",
        "metric_resolution=224x224; boundary_tolerance=2px_euclidean_disk",
        "boundary_metric=8-neighbour inner contour; per-image symmetric Boundary F1",
        f"C0_dice={baseline_scores[0]:.8f}",
        f"C0_iou={baseline_scores[1]:.8f}",
        f"C0_boundary_f1={baseline_scores[2]:.8f}",
        f"ROI_original_pixels={ROI}",
        "display=thresholded 224x224 mask resized with nearest-neighbour to source resolution",
        "display_boundary=8-neighbour inner contour with 3x3 stroke",
        "colors=GT cyan #00DCFF; prediction orange-red #FF4A00; intersecting display strokes yellow #FFEB00",
        "yellow means thickened display strokes intersect; it is not the BF1 tolerance match map",
        f"figure_mode={mode}",
    ]
    if abl_scores is not None:
        notes.extend(
            [
                f"C1_dice={abl_scores[0]:.8f}",
                f"C1_iou={abl_scores[1]:.8f}",
                f"C1_boundary_f1={abl_scores[2]:.8f}",
                f"delta_boundary_f1={abl_scores[2] - baseline_scores[2]:+.8f}",
            ]
        )
    else:
        notes.append("available_six_panel_labels=a,b,c,e; omitted_panels=d,f (C1 checkpoint not supplied)")
    note_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"sample_id={SAMPLE_ID}")
    print(f"figure_mode={mode}")
    print(f"C0 Dice={baseline_scores[0]:.6f} IoU={baseline_scores[1]:.6f} BF1={baseline_scores[2]:.6f}")
    if abl_scores is not None:
        print(f"C1 Dice={abl_scores[0]:.6f} IoU={abl_scores[1]:.6f} BF1={abl_scores[2]:.6f}")
        print(f"delta_BF1={abl_scores[2] - baseline_scores[2]:+.6f}")
    for output in figure_outputs:
        print(f"saved={output}")
    print(f"saved_metadata={note_path}")


if __name__ == "__main__":
    main()
