"""Validate frozen author-Protocol-B GlaS five-fold manifests."""
from __future__ import annotations
import argparse, re
from pathlib import Path
SAMPLE_RE = re.compile(r"^(train|testA|testB)_\d+$")


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=Path("datasets/GlaS_5fold"))
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    held_out = set()
    for fold in range(5):
        root = args.manifest_root / f"fold{fold}"
        train, test = read_ids(root / "Train_Folder/samples.txt"), read_ids(root / "Test_Folder/samples.txt")
        if len(train) != 132 or len(test) != 33 or set(train) & set(test) or len(set(train) | set(test)) != 165:
            raise RuntimeError(f"{root}: invalid 132/33 disjoint split")
        for sid in set(train) | set(test):
            if not SAMPLE_RE.fullmatch(sid):
                raise RuntimeError(f"Invalid sample ID: {sid}")
            for suffix in (".bmp", "_anno.bmp"):
                if not (args.source_dir / f"{sid}{suffix}").is_file():
                    raise FileNotFoundError(args.source_dir / f"{sid}{suffix}")
        held_out.update(test)
        print(f"fold{fold}: train={len(train)} test={len(test)}")
    if len(held_out) != 165:
        raise RuntimeError("Held-out folds do not cover each of the 165 samples once")
    print("Protocol B manifest validation passed.")


if __name__ == "__main__":
    main()
