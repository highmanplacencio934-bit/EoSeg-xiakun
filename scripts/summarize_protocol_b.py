"""Summarize five-fold Protocol B runs with optional metric-regression comparison."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

METRICS = {
    "loss": "test/loss",
    "accuracy": "metrics/test_acc",
    "dice": "metrics/test_dice",
    "iou": "metrics/test_iou",
    "jaccard": "metrics/test_jaccard",
    "precision": "metrics/test_precision",
    "recall": "metrics/test_recall",
    "f2": "metrics/test_f2",
    "boundary_f1": "metrics/test_boundary_f1",
}


def extract(path: Path, require_boundary: bool) -> dict[str, float]:
    found: dict[str, float] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        cells = [cell.strip() for cell in re.split(r"[|│]", line)]
        if len(cells) < 2:
            continue
        for index, label in enumerate(cells[:-1]):
            for key, expected_label in METRICS.items():
                if label != expected_label:
                    continue
                try:
                    found[key] = float(cells[index + 1])
                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid {expected_label} value in {path}: {cells[index + 1]!r}"
                    ) from exc
    required = set(METRICS) if require_boundary else set(METRICS) - {"boundary_f1"}
    missing = required - found.keys()
    if missing:
        raise RuntimeError(f"Missing {sorted(missing)} from Protocol B test log: {path}")
    return found


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("Sample standard deviation needs at least two folds.")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def run_log(runs_root: Path, prefix: str, fold: int, suffix: str) -> Path:
    run_name = f"{prefix}{fold}_vitl_clean{suffix}"
    return runs_root / run_name / "protocol_b_test.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Protocol B folds and compare old/new metric-only runs."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="glas_protocol_b_fold")
    parser.add_argument("--run-suffix", default="_bf1_metric_check")
    parser.add_argument("--reference-suffix", default="_maskonly_gn_baseline_5fold")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/boundary_f1_fivefold_baseline_comparison.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    current_by_fold: list[dict[str, float]] = []
    reference_by_fold: list[dict[str, float]] = []
    for fold in range(5):
        current_path = run_log(args.runs_root, args.prefix, fold, args.run_suffix)
        reference_path = run_log(args.runs_root, args.prefix, fold, args.reference_suffix)
        current = extract(current_path, require_boundary=True)
        reference = extract(reference_path, require_boundary=False)
        current_by_fold.append(current)
        reference_by_fold.append(reference)
        row: dict[str, object] = {"fold": fold, "protocol": "author_protocol_b_test_as_val"}
        for key in METRICS:
            row[f"current_{key}"] = current[key]
        for key in METRICS:
            if key == "boundary_f1":
                continue
            row[f"reference_{key}"] = reference[key]
            row[f"delta_{key}"] = current[key] - reference[key]
        rows.append(row)
        print(
            f"fold{fold}: current Dice={current['dice']:.6f} "
            f"IoU={current['iou']:.6f} BF1={current['boundary_f1']:.6f}; "
            f"reference Dice={reference['dice']:.6f} IoU={reference['iou']:.6f}; "
            f"delta Dice={current['dice'] - reference['dice']:+.6f} "
            f"IoU={current['iou'] - reference['iou']:+.6f} "
            f"Precision={current['precision'] - reference['precision']:+.6f} "
            f"Recall={current['recall'] - reference['recall']:+.6f} "
            f"Acc={current['accuracy'] - reference['accuracy']:+.6f}"
        )

    summary: dict[str, object] = {"fold": "mean", "protocol": "author_protocol_b_test_as_val"}
    for key in METRICS:
        mean, sd = mean_std([values[key] for values in current_by_fold])
        summary[f"current_{key}"] = mean
        summary[f"current_{key}_sample_sd"] = sd
        print(f"current {key}: {mean:.6f} ± {sd:.6f}")
    for key in METRICS:
        if key == "boundary_f1":
            continue
        mean, sd = mean_std([values[key] for values in reference_by_fold])
        summary[f"reference_{key}"] = mean
        summary[f"reference_{key}_sample_sd"] = sd
        delta_mean, delta_sd = mean_std(
            [cur[key] - ref[key] for cur, ref in zip(current_by_fold, reference_by_fold)]
        )
        summary[f"delta_{key}"] = delta_mean
        summary[f"delta_{key}_sample_sd"] = delta_sd
        print(f"delta {key}: {delta_mean:+.6f} ± {delta_sd:.6f}")
    rows.append(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary_csv={args.output}")


if __name__ == "__main__":
    main()
