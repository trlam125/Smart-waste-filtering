from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.paths import resolve_project_path

CONFIG_VERSION = 1
CONFIG_MARKER_NAME = ".training_config.json"
_GENERATED_DATASET_FILES = {
    "smartwaste_runtime.yaml",
    ".detector_dataset_source.json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.name not in _GENERATED_DATASET_FILES
    )
    for item in files:
        stat = item.stat()
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_fingerprint(source: Path | str) -> dict[str, Any]:
    path = resolve_project_path(source)
    if path.is_file():
        stat = path.stat()
        return {
            "kind": "file",
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "sha256": _sha256_file(path),
        }
    if path.is_dir():
        return {
            "kind": "directory",
            "path": str(path.resolve()),
            "fingerprint": _directory_fingerprint(path),
        }
    raise FileNotFoundError(f"Detector dataset source not found: {path}")


def build_resume_config(
    data_source: Path | str,
    *,
    model: str,
    imgsz: int,
    batch: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "model": str(model),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "seed": int(seed),
        "dataset": dataset_fingerprint(data_source),
    }


def read_resume_config(marker: Path | str) -> dict[str, Any] | None:
    path = resolve_project_path(marker)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_resume_config(marker: Path | str, config: dict[str, Any]) -> Path:
    path = resolve_project_path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def resume_config_differences(
    saved: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    if saved is None:
        return ["resume metadata is missing or unreadable"]

    differences: list[str] = []
    for key in ("version", "model", "imgsz", "batch", "seed"):
        if saved.get(key) != current.get(key):
            differences.append(f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}")

    saved_dataset = saved.get("dataset")
    current_dataset = current.get("dataset")

    def _identity(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("kind") == "file":
            return {
                "kind": "file",
                "size": value.get("size"),
                "sha256": value.get("sha256"),
            }
        if value.get("kind") == "directory":
            return {
                "kind": "directory",
                "fingerprint": value.get("fingerprint"),
            }
        return value

    if _identity(saved_dataset) != _identity(current_dataset):
        differences.append("dataset fingerprint changed")
    return differences


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage SmartWaste detector resume metadata.")
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    current = build_resume_config(
        args.data,
        model=args.model,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
    )
    if args.action == "write":
        marker = write_resume_config(args.marker, current)
        print(marker)
        return 0

    saved = read_resume_config(args.marker)
    differences = resume_config_differences(saved, current)
    if differences:
        print("Detector resume configuration mismatch:")
        for item in differences:
            print(f"  - {item}")
        return 3
    print("Detector resume configuration matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
