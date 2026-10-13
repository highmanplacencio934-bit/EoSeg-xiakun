"""Export image-level Boundary F1 for one frozen Protocol B checkpoint."""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch

from datasets.medical_binary_semantic import MedicalBinaryFolderDataset
from training.boundary_f1 import BOUNDARY_TOLERANCE_PIXELS, boundary_f1_score
from visualize_binary_medical_results_flexible import infer_binary_mask, load_config, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write per-image BF1 for a fixed, already-selected checkpoint."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.config, args.ckpt, args.data_dir, args.source_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    config = load_config(args.config)
    img_size = tuple(int(v) for v in config["data"]["init_args"]["img_size"])
    dataset = MedicalBinaryFolderDataset(
        dataset_dir=args.data_dir,
        img_size=img_size,
        train=False,
        source_dir=args.source_dir,
    )
    if not dataset.samples:
        raise RuntimeError(f"No samples found in {args.data_dir}")

    device = torch.device(args.device)
    model = load_model(config, args.ckpt, device)
    rows: list[dict[str, object]] = []
    for index, (image_path, _) in enumerate(dataset.samples):
        image_tensor, target = dataset[index]
        target_mask = target["masks"].any(dim=0).cpu().numpy().astype(bool)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            prediction = infer_binary_mask(
                model=model,
                image_tensor=image_tensor,
                output_size=img_size,
                threshold=0.5,
                device=device,
            ).astype(bool)
        if prediction.shape != target_mask.shape:
            raise ValueError(
                f"Mask shape mismatch for {image_path.stem}: "
                f"{prediction.shape} vs {target_mask.shape}"
            )
        rows.append(
            {
                "sample_id": image_path.stem,
                "boundary_f1": boundary_f1_score(prediction, target_mask),
                "protocol": "author_protocol_b_test_as_val",
                "split": str(args.data_dir),
                "checkpoint": str(args.ckpt),
                "seed": 0,
                "tta": False,
                "threshold": 0.5,
                "inference_precision": "16-mixed" if device.type == "cuda" else "32-fp",
                "metric_size": f"{img_size[0]}x{img_size[1]}",
                "boundary_tolerance_pixels": BOUNDARY_TOLERANCE_PIXELS,
                "boundary_tolerance_geometry": "euclidean_disk",
                "boundary_edge_definition": "8-neighbour_inner_contour",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    macro = float(np.mean([float(row["boundary_f1"]) for row in rows]))
    print(f"protocol=author_protocol_b_test_as_val")
    print(f"split={args.data_dir}")
    print(f"sample_count={len(rows)}")
    print(f"image_macro_boundary_f1={macro:.8f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
