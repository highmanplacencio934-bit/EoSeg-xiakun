"""Create the frozen author-Protocol-B GlaS five-fold manifests."""
from __future__ import annotations
import argparse, csv, json, random, re
from pathlib import Path

GROUP_ORDER = {"train": 0, "testA": 1, "testB": 2}
SAMPLE_RE = re.compile(r"^(train|testA|testB)_(\d+)$")


def sample_key(sample_id: str) -> tuple[int, int]:
    match = SAMPLE_RE.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"Unexpected GlaS sample id: {sample_id}")
    return GROUP_ORDER[match.group(1)], int(match.group(2))


def discover_samples(source_dir: Path) -> list[str]:
    samples = []
    for image_path in source_dir.glob("*.bmp"):
        if image_path.stem.endswith("_anno"):
            continue
        sample_id = image_path.stem
        if not SAMPLE_RE.fullmatch(sample_id):
            continue
        if not (source_dir / f"{sample_id}_anno.bmp").is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        samples.append(sample_id)
    samples = sorted(set(samples), key=sample_key)
    expected = {"train": 85, "testA": 60, "testB": 20}
    counts = {g: sum(s.startswith(f"{g}_") for s in samples) for g in expected}
    if counts != expected or len(samples) != 165:
        raise RuntimeError(f"Expected {expected} and 165 samples, got {counts}/{len(samples)}")
    return samples


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/GlaS_5fold"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source_dir, output_root = args.source_dir.resolve(), args.output_root.resolve()
    samples = discover_samples(source_dir)
    shuffled = list(samples)
    random.Random(args.seed).shuffle(shuffled)
    fold_tests = [shuffled[i * 33 : (i + 1) * 33] for i in range(5)]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sample_manifest.csv").write_text(
        "sample_id,source_image,source_mask\n"
        + "".join(f"{s},{s}.bmp,{s}_anno.bmp\n" for s in samples), encoding="utf-8"
    )
    with (output_root / "fold_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "split", "sample_id"])
        for fold, fold_test in enumerate(fold_tests):
            test_set = set(fold_test)
            train_ids = [s for s in samples if s not in test_set]
            test_ids = sorted(fold_test, key=sample_key)
            for split, ids in (("train", train_ids), ("test", test_ids)):
                for sample_id in ids:
                    writer.writerow([fold, split, sample_id])
            root = output_root / f"fold{fold}"
            write_lines(root / "Train_Folder" / "samples.txt", train_ids)
            write_lines(root / "Test_Folder" / "samples.txt", test_ids)
            (root / "metadata.json").write_text(json.dumps({
                "protocol": "author_protocol_b_test_as_val", "fold": fold,
                "n_splits": 5, "split_seed": args.seed,
                "source_dir": str(source_dir), "train_count": len(train_ids),
                "test_count": len(test_ids),
            }, indent=2) + "\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        f"# GlaS author Protocol B five-fold manifests\n\nGenerated from {source_dir}; split_seed={args.seed}.\n"
        "Each fold has Train_Folder/samples.txt (132 IDs) and Test_Folder/samples.txt (33 IDs). "
        "The raw BMP files are not copied. Test_Folder is used for per-epoch val_dice checkpoint selection "
        "and final test; label this author_protocol_b_test_as_val.\n", encoding="utf-8")
    print(f"Generated five Protocol B folds under {output_root}")


if __name__ == "__main__":
    main()
