"""Summarize five verified endpoint tests from the C1-Late recovery runs."""
from __future__ import annotations

import argparse
import csv
import pickletools
import statistics
import zipfile
from pathlib import Path

from summarize_protocol_b import extract


def checkpoint_state(path: Path) -> tuple[int, int]:
    """Read scalar checkpoint metadata without materializing 3.7-GB tensors."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one data.pkl in {path}; found {len(names)}")
        operations = list(pickletools.genops(archive.read(names[0])))
    fields: dict[str, int] = {}
    for index, (_, value, _) in enumerate(operations[:-2]):
        if value in {"epoch", "global_step"} and value not in fields:
            number = operations[index + 2][1]
            if not isinstance(number, int):
                raise RuntimeError(f"Non-integer checkpoint field {value!r} in {path}")
            fields[value] = number
    if set(fields) != {"epoch", "global_step"}:
        raise RuntimeError(f"Missing checkpoint epoch/step in {path}")
    return fields["epoch"], fields["global_step"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-tag", default="epoch999_recovery")
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/c1_late_epoch999_recovery_summary.csv"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for fold in range(5):
        run_name = (
            f"glas_protocol_b_fold{fold}_vitl_c1_late800_noes1000_seed0_{args.run_tag}"
        )
        run_dir = args.runs_root / run_name
        candidates = list((run_dir / "version_0" / "checkpoints").glob(
            "recovered-endpoint-epoch=999-step=*.ckpt"
        ))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one endpoint for fold {fold}; found {len(candidates)}")
        checkpoint = candidates[0]
        epoch, global_step = checkpoint_state(checkpoint)
        if (epoch, global_step) != (999, 10000):
            raise RuntimeError(
                f"Invalid fold {fold} endpoint: epoch={epoch}, global_step={global_step}"
            )
        metrics = extract(run_dir / "protocol_b_endpoint_test.txt", require_boundary=True)
        row: dict[str, object] = {
            "fold": fold,
            "protocol": "author_protocol_b_test_as_val",
            "run_kind": "recovery_continuation_not_original_uninterrupted_endpoint",
            "checkpoint": str(checkpoint),
            "epoch": epoch,
            "global_step": global_step,
            "single_view": True,
            "threshold": 0.5,
            **metrics,
        }
        rows.append(row)
        print(
            f"fold{fold}: Dice={metrics['dice']:.6f} IoU={metrics['iou']:.6f} "
            f"Boundary-F1={metrics['boundary_f1']:.6f}"
        )
    summary: dict[str, object] = {"fold": "mean", "protocol": "author_protocol_b_test_as_val"}
    for key in ("dice", "iou", "boundary_f1"):
        values = [float(row[key]) for row in rows]
        summary[key] = statistics.mean(values)
        summary[f"{key}_fold_sample_sd"] = statistics.stdev(values)
        print(f"{key}: {summary[key]:.6f} ± {summary[f'{key}_fold_sample_sd']:.6f}")
    rows.append(summary)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing summary: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary_csv={args.output}")


if __name__ == "__main__":
    main()
