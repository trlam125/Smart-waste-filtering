from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classifier import WasteClassifier  # noqa: E402
from app.database import DB_PATH  # noqa: E402
from app.main import COLLECTED_DATA_DIR, COLLECTED_PENDING_DIR, SCAN_THUMBNAIL_DIR  # noqa: E402
from app.paths import is_google_colab, runtime_root, torch_home, training_output_dir  # noqa: E402
from training.dataset_utils import DEFAULT_DATASET_SOURCE, DEFAULT_EXTRACT_DIR  # noqa: E402


def _inside(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def main() -> int:
    classifier = WasteClassifier()
    colab = is_google_colab()
    runtime = runtime_root()

    persistent = {
        "project": PROJECT_ROOT,
        "dataset_zip": DEFAULT_DATASET_SOURCE,
        "database": DB_PATH,
        "scan_thumbnails": SCAN_THUMBNAIL_DIR,
        "collected_data": COLLECTED_DATA_DIR,
        "model": classifier.checkpoint_path,
        "ood_reference": classifier.ood_reference_path,
    }
    disposable = {
        "dataset_extract": DEFAULT_EXTRACT_DIR,
        "torch_home": torch_home(),
        "collection_pending": COLLECTED_PENDING_DIR,
        "training_output": training_output_dir("efficientnet_b0"),
    }

    print(f"PROJECT_ROOT : {PROJECT_ROOT.resolve()}")
    print(f"COLAB        : {colab}")
    print(f"RUNTIME_ROOT : {runtime.resolve()}")

    errors: list[tuple[str, Path]] = []
    print("\nPersistent paths (must stay inside the Drive project):")
    for name, path in persistent.items():
        resolved = Path(path).resolve()
        ok = _inside(resolved, PROJECT_ROOT)
        print(f"[{'DRIVE OK' if ok else 'ERROR':8s}] {name:20s} {resolved}")
        if not ok:
            errors.append((name, resolved))

    print("\nDisposable/runtime paths:")
    for name, path in disposable.items():
        resolved = Path(path).resolve()
        expected = _inside(resolved, runtime)
        status = "RUNTIME" if expected else "CUSTOM"
        print(f"[{status:8s}] {name:20s} {resolved}")

    if colab:
        legacy_paths = (
            PROJECT_ROOT / "data" / "dataset" / "_extracted",
            PROJECT_ROOT / "data" / ".torch",
            PROJECT_ROOT / "data" / "collected" / "_pending",
            PROJECT_ROOT / "runs",
            Path("/content/smart_waste_scanner_dataset"),
        )
        existing = [path for path in legacy_paths if path.exists()]
        if existing:
            print("\nOld caches/artifacts detected. The new Colab defaults do not use these:")
            for path in existing:
                print(f"  - {path}")
            print("You can delete them after confirming the paths above.")

    if errors:
        print("\nERROR: persistent project data is configured outside PROJECT_ROOT:")
        for name, path in errors:
            print(f"  - {name}: {path}")
        print("Use relative .env paths such as data/... or models/best_model.pt.")
        return 2

    print("\nPersistent paths are project-contained; disposable paths are separated correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
