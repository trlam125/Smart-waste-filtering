from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_SOURCE = PROJECT_ROOT / "data" / "dataset" / "dataset.zip"


def _running_in_colab() -> bool:
    """Return True when the code is running inside a Google Colab runtime."""
    return (
        "COLAB_RELEASE_TAG" in os.environ
        or "COLAB_GPU" in os.environ
        or (Path("/content").is_dir() and Path("/content/drive").exists())
    )


def _default_extract_dir() -> Path:
    """
    Keep extracted images off Google Drive when running in Colab.

    - Colab: /content/smart_waste_scanner_dataset
    - Local:  <project>/data/dataset/_extracted

    SMARTWASTE_DATASET_EXTRACT_DIR can override both locations.
    """
    override = os.environ.get("SMARTWASTE_DATASET_EXTRACT_DIR")
    if override:
        return Path(override).expanduser()
    if _running_in_colab():
        return Path("/content/smart_waste_scanner_dataset")
    return PROJECT_ROOT / "data" / "dataset" / "_extracted"


DEFAULT_EXTRACT_DIR = _default_extract_dir()
_MARKER_NAME = ".dataset_source.json"
_SPLITS = ("train", "val", "test")


def _has_splits(path: Path) -> bool:
    return path.is_dir() and all((path / split).is_dir() for split in _SPLITS)


def locate_dataset_root(path: Path) -> Path:
    """Return the directory that directly contains train/, val/ and test/."""
    path = path.expanduser().resolve()
    if _has_splits(path):
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    # The distributed SmartWaste ZIP has one top-level dataset directory.
    candidates = [child for child in path.iterdir() if child.is_dir() and _has_splits(child)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple dataset roots found under {path}: {candidates}")

    # Also tolerate an extra wrapper directory, but do not scan the image tree deeply.
    candidates = []
    for first in path.iterdir():
        if not first.is_dir():
            continue
        for second in first.iterdir():
            if second.is_dir() and _has_splits(second):
                candidates.append(second)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple dataset roots found under {path}: {candidates}")

    raise FileNotFoundError(f"Could not find train/, val/, test/ under {path}")


def _source_signature(source: Path) -> dict[str, int | str]:
    stat = source.stat()
    return {
        "source": str(source.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_marker(extract_dir: Path) -> dict[str, object] | None:
    marker = extract_dir / _MARKER_NAME
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _validate_zip_members(archive: zipfile.ZipFile) -> None:
    """Reject absolute/path-traversal members before extracting."""
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        member = PurePosixPath(normalized)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unsafe path in dataset ZIP: {info.filename}")


def extract_dataset_zip(source: Path, extract_dir: Path = DEFAULT_EXTRACT_DIR) -> Path:
    source = source.expanduser().resolve()
    extract_dir = extract_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {source}")
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip dataset archive, got: {source}")

    signature = _source_signature(source)
    cached = _read_marker(extract_dir)
    if cached == signature:
        try:
            return locate_dataset_root(extract_dir)
        except (FileNotFoundError, RuntimeError):
            pass

    if extract_dir.exists():
        print(f"Dataset ZIP changed or cache is incomplete. Rebuilding: {extract_dir}")
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting dataset ZIP: {source}")
    print(f"Extraction cache:      {extract_dir}")
    print("This is normally done only on the first run, or when dataset.zip changes.")
    try:
        with zipfile.ZipFile(source, "r") as archive:
            _validate_zip_members(archive)
            archive.extractall(extract_dir)
        root = locate_dataset_root(extract_dir)
        (extract_dir / _MARKER_NAME).write_text(
            json.dumps(signature, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return root
    except Exception:
        # Never leave a half-extracted cache that could be mistaken for valid data.
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise


def prepare_dataset(source: Path | str = DEFAULT_DATASET_SOURCE) -> Path:
    """Accept either dataset.zip or an already-extracted dataset directory."""
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() == ".zip":
            return extract_dataset_zip(path)
        raise ValueError(f"Unsupported dataset file: {path}. Expected a .zip archive.")
    if path.is_dir():
        return locate_dataset_root(path)
    raise FileNotFoundError(
        f"Dataset source not found: {path}\n"
        f"Place the archive at: {DEFAULT_DATASET_SOURCE}"
    )
