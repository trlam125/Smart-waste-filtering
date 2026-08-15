from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.class_schema import WASTE_CLASS_KEYS  # noqa: E402
from training.dataset_utils import DEFAULT_DATASET_SOURCE, prepare_dataset  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the 11-class SmartWaste dataset layout.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_SOURCE,
        help="Dataset ZIP or extracted root. Default: data/dataset/dataset.zip",
    )
    args = parser.parse_args()
    root = prepare_dataset(args.data)
    print(f"Dataset root: {root}\n")

    grand_total = 0
    for split in ("train", "val", "test"):
        split_root = root / split
        if not split_root.is_dir():
            raise SystemExit(f"Missing split: {split_root}")
        present = {p.name for p in split_root.iterdir() if p.is_dir()}
        if present != set(WASTE_CLASS_KEYS):
            raise SystemExit(
                f"{split} class mismatch. Missing={sorted(set(WASTE_CLASS_KEYS)-present)} "
                f"Extra={sorted(present-set(WASTE_CLASS_KEYS))}"
            )
        split_total = 0
        empty_classes: list[str] = []
        print(f"[{split}]")
        for key in WASTE_CLASS_KEYS:
            count = sum(
                1 for p in (split_root / key).rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if count == 0:
                empty_classes.append(key)
            split_total += count
            print(f"  {key:14s} {count:6d}")
        grand_total += split_total
        print(f"  {'TOTAL':14s} {split_total:6d}\n")
        if empty_classes:
            raise SystemExit(
                f"{split} contains class directories with no supported images: {empty_classes}"
            )
    print(f"Grand total: {grand_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
