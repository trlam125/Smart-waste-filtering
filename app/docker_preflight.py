from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from .class_schema import WASTE_CLASS_KEYS


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false, got {raw!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_from_env(name: str, default: str) -> Path:
    raw = os.getenv(name, default).strip() or default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path("/app") / path
    return path.resolve()


def validate_deploy_assets() -> None:
    checkpoint_path = _path_from_env("WASTE_MODEL_CHECKPOINT", "/app/models/best_model.pt")
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Missing model checkpoint: {checkpoint_path}. "
            "Mount/copy models/best_model.pt before starting the container."
        )

    if not _bool_env("OOD_DETECTION_ENABLED", True):
        print(f"[docker-preflight] checkpoint OK: {checkpoint_path}")
        print("[docker-preflight] OOD detection disabled; skipping OOD reference validation.")
        return

    if _bool_env("OOD_AUTO_BUILD", False):
        raise RuntimeError(
            "OOD_AUTO_BUILD=true is not supported in the Docker deployment. "
            "The models directory is mounted read-only and the training dataset/tools are not "
            "part of the runtime image. Build models/ood_reference.npz offline for the current "
            "best_model.pt, then restart with OOD_AUTO_BUILD=false."
        )

    ood_path = _path_from_env("OOD_REFERENCE_PATH", "/app/models/ood_reference.npz")
    if not ood_path.is_file():
        raise RuntimeError(
            f"Missing OOD reference: {ood_path}. "
            "Run training/build_ood_reference.py for the deployed checkpoint before starting Docker."
        )

    checkpoint_hash = _sha256(checkpoint_path)
    try:
        with np.load(ood_path, allow_pickle=False) as reference:
            stored_hash = str(reference["checkpoint_sha256"].item())
            stored_classes = tuple(str(item) for item in reference["class_names"].tolist())
    except Exception as exc:
        raise RuntimeError(f"Unreadable/invalid OOD reference: {ood_path}: {exc}") from exc

    if stored_hash != checkpoint_hash:
        raise RuntimeError(
            "OOD reference does not match the deployed checkpoint. "
            f"checkpoint_sha256={checkpoint_hash}, ood_checkpoint_sha256={stored_hash}. "
            "Rebuild models/ood_reference.npz from this exact best_model.pt before deployment."
        )

    if stored_classes != WASTE_CLASS_KEYS:
        raise RuntimeError(
            "OOD reference class_names do not match the project's 11-class schema. "
            f"Expected {WASTE_CLASS_KEYS}, got {stored_classes}."
        )

    print(f"[docker-preflight] checkpoint OK: {checkpoint_path}")
    print(f"[docker-preflight] OOD reference OK and hash-matched: {ood_path}")


def main() -> int:
    try:
        validate_deploy_assets()
    except Exception as exc:
        print(f"[docker-preflight] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
