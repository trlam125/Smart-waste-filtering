from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from app.paths import PROJECT_ROOT, detector_dataset_extract_dir, resolve_project_path

DEFAULT_DETECTOR_DATASET_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "detector_class"
    / "ObjectDetector_dataset.zip"
)
DEFAULT_DETECTOR_EXTRACT_DIR = detector_dataset_extract_dir()
_MARKER_NAME = ".detector_dataset_source.json"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_SPLITS = ("train", "val", "test")

def _project_path(value: Path | str) -> Path:
    return resolve_project_path(value)

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
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        member = PurePosixPath(normalized)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unsafe path in detector ZIP: {info.filename}")

def _is_detector_root(path: Path) -> bool:
    return path.is_dir() and all(
        (path / "images" / split).is_dir() and (path / "labels" / split).is_dir()
        for split in _SPLITS
    )

def locate_detector_root(path: Path) -> Path:
    path = _project_path(path)
    if _is_detector_root(path):
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Detector dataset directory not found: {path}")

    candidates = [child for child in path.iterdir() if child.is_dir() and _is_detector_root(child)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple detector dataset roots found under {path}: {candidates}")

    candidates = []
    for first in path.iterdir():
        if not first.is_dir():
            continue
        for second in first.iterdir():
            if second.is_dir() and _is_detector_root(second):
                candidates.append(second)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple detector dataset roots found under {path}: {candidates}")

    raise FileNotFoundError(
        f"Could not find images/train|val|test and labels/train|val|test under {path}"
    )

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _validate_label_file(path: Path) -> int:
    count = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise RuntimeError(f"Invalid YOLO label at {path}:{line_number}: expected 5 fields")
        try:
            class_id = int(parts[0])
            x, y, w, h = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise RuntimeError(f"Invalid numeric YOLO label at {path}:{line_number}") from exc
        if class_id != 0:
            raise RuntimeError(
                f"Detector dataset must be class-agnostic class 0 only; got {class_id} at "
                f"{path}:{line_number}"
            )
        values = (x, y, w, h)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite YOLO coordinates at {path}:{line_number}")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise RuntimeError(f"Out-of-range YOLO coordinates at {path}:{line_number}: {values}")
        count += 1
    return count

def validate_detector_dataset(root: Path) -> dict[str, dict[str, int]]:
    root = locate_detector_root(root)
    summary: dict[str, dict[str, int]] = {}
    seen_image_hashes: dict[str, tuple[str, Path]] = {}

    for split in _SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        labels = sorted(path for path in label_dir.rglob("*.txt") if path.is_file())
        if not images:
            raise RuntimeError(f"No detector images found in {image_dir}")

        for image_path in images:
            content_sha256 = _sha256_file(image_path)
            previous = seen_image_hashes.get(content_sha256)
            if previous is not None and previous[0] != split:
                previous_split, previous_path = previous
                raise RuntimeError(
                    "Exact-image data leakage across detector splits: "
                    f"{previous_path} ({previous_split}) and {image_path} ({split}) "
                    "have identical file content. Remove one copy before training."
                )
            if previous is None:
                seen_image_hashes[content_sha256] = (split, image_path)

        image_stems = {path.relative_to(image_dir).with_suffix("") for path in images}
        label_stems = {path.relative_to(label_dir).with_suffix("") for path in labels}
        extra_labels = sorted(label_stems - image_stems)
        if extra_labels:
            raise RuntimeError(
                f"Detector split {split} has label files without matching images: "
                f"extra_labels={len(extra_labels)}"
            )

        label_box_counts = {
            path.relative_to(label_dir).with_suffix(""): _validate_label_file(path)
            for path in labels
        }
        boxes = sum(label_box_counts.values())
        empty_label_backgrounds = sum(
            1 for count in label_box_counts.values() if count == 0
        )
        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "backgrounds": len(image_stems - label_stems) + empty_label_backgrounds,
            "boxes": boxes,
        }

    return summary

def write_runtime_yaml(root: Path) -> Path:
    root = locate_detector_root(root)
    yaml_path = root / "smartwaste_runtime.yaml"
    root_string = json.dumps(root.resolve().as_posix(), ensure_ascii=False)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root_string}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: waste_object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path

def extract_detector_dataset_zip(
    source: Path,
    extract_dir: Path = DEFAULT_DETECTOR_EXTRACT_DIR,
) -> Path:
    source = _project_path(source)
    extract_dir = _project_path(extract_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Detector dataset ZIP not found: {source}")
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Expected a .zip detector dataset archive, got: {source}")

    signature = _source_signature(source)
    cached = _read_marker(extract_dir)
    if cached == signature:
        try:
            root = locate_detector_root(extract_dir)
            validate_detector_dataset(root)
            write_runtime_yaml(root)
            return root
        except (FileNotFoundError, RuntimeError, OSError):
            pass

    if extract_dir.exists():
        print(f"Detector ZIP changed or cache is incomplete. Rebuilding: {extract_dir}")
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting detector ZIP: {source}")
    print(f"Extraction cache:       {extract_dir}")
    print("The extracted copy is disposable and is deleted after successful training.")
    try:
        with zipfile.ZipFile(source, "r") as archive:
            _validate_zip_members(archive)
            archive.extractall(extract_dir)
        root = locate_detector_root(extract_dir)
        validate_detector_dataset(root)
        write_runtime_yaml(root)
        (extract_dir / _MARKER_NAME).write_text(
            json.dumps(signature, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return root
    except Exception:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise

def prepare_detector_dataset(
    source: Path | str = DEFAULT_DETECTOR_DATASET_SOURCE,
) -> tuple[Path, Path, dict[str, dict[str, int]]]:
    path = _project_path(source)
    if path.is_file():
        if path.suffix.lower() != ".zip":
            raise ValueError(f"Unsupported detector dataset file: {path}. Expected a .zip archive.")
        root = extract_detector_dataset_zip(path)
    elif path.is_dir():
        root = locate_detector_root(path)
    else:
        raise FileNotFoundError(
            f"Detector dataset source not found: {path}\n"
            f"Place the detector archive at: {DEFAULT_DETECTOR_DATASET_SOURCE}"
        )

    summary = validate_detector_dataset(root)
    yaml_path = write_runtime_yaml(root)
    return root, yaml_path, summary
