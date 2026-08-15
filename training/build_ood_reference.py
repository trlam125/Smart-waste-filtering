from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.class_schema import WASTE_CLASS_KEYS  # noqa: E402
from app.model_factory import build_eval_transform, create_model, get_final_classifier_layer  # noqa: E402
from app.paths import ood_reference_path, resolve_project_path  # noqa: E402
from training.dataset_utils import DEFAULT_DATASET_SOURCE, prepare_dataset  # noqa: E402


def _project_path(value: Path | str) -> Path:
    return resolve_project_path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(preference: str) -> torch.device:
    preference = preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if preference == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
    return torch.device(preference)


def _image_paths(root: Path, split: str, class_name: str) -> list[Path]:
    directory = root / split / class_name
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset class directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def build_ood_reference(
    dataset_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    references_per_class: int = 50,
    validation_quantile: float = 0.01,
    batch_size: int = 32,
    seed: int = 20260815,
    device_preference: str = "auto",
) -> dict[str, Any]:
    """Build a checkpoint-specific nearest-neighbor bank for OOD rejection."""
    if references_per_class < 1:
        raise ValueError("references_per_class must be >= 1")
    if not 0.0 <= validation_quantile <= 0.20:
        raise ValueError("validation_quantile must be between 0 and 0.20")

    dataset_root = _project_path(dataset_root)
    checkpoint_path = _project_path(checkpoint_path)
    output_path = _project_path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names = tuple(str(x) for x in checkpoint.get("class_names", ()))
    if class_names != WASTE_CLASS_KEYS:
        raise RuntimeError("Checkpoint class_names do not match the canonical 11-class schema")

    device = _resolve_device(device_preference)
    model = create_model(str(checkpoint["arch"]), len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    transform = build_eval_transform(
        int(checkpoint.get("image_size", 224)),
        checkpoint.get("mean", (0.485, 0.456, 0.406)),
        checkpoint.get("std", (0.229, 0.224, 0.225)),
    )
    feature_layer = get_final_classifier_layer(model, str(checkpoint["arch"]))
    captured: dict[str, torch.Tensor] = {}

    def capture_features(_module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs:
            captured["features"] = inputs[0].detach()

    hook = feature_layer.register_forward_pre_hook(capture_features)

    def embed(paths: list[Path]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            with torch.inference_mode():
                model(batch)
            features = captured.get("features")
            if features is None:
                raise RuntimeError("Could not capture classifier input features")
            features = torch.nn.functional.normalize(features.float().flatten(1), dim=1)
            batches.append(features.cpu().numpy().astype(np.float32, copy=False))
        if not batches:
            raise RuntimeError("No images available for OOD reference generation")
        return np.concatenate(batches, axis=0)

    rng = random.Random(seed)
    reference_vectors: list[np.ndarray] = []
    reference_labels: list[int] = []
    try:
        for class_index, class_name in enumerate(class_names):
            candidates = _image_paths(dataset_root, "train", class_name)
            rng.shuffle(candidates)
            selected = candidates[: min(references_per_class, len(candidates))]
            vectors = embed(selected)
            reference_vectors.append(vectors)
            reference_labels.extend([class_index] * len(vectors))

        references = np.ascontiguousarray(np.concatenate(reference_vectors, axis=0), dtype=np.float32)
        labels = np.asarray(reference_labels, dtype=np.int16)

        nearest_scores: list[np.ndarray] = []
        reference_tensor = torch.from_numpy(references).to(device)
        for class_name in class_names:
            validation_paths = _image_paths(dataset_root, "val", class_name)
            validation_vectors = embed(validation_paths)
            for start in range(0, len(validation_vectors), 256):
                values = torch.from_numpy(validation_vectors[start : start + 256]).to(device)
                nearest = (values @ reference_tensor.T).max(dim=1).values
                nearest_scores.append(nearest.cpu().numpy())

        calibration_scores = np.concatenate(nearest_scores)
        threshold = float(np.quantile(calibration_scores, validation_quantile))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            embeddings=references,
            labels=labels,
            class_names=np.asarray(class_names),
            checkpoint_sha256=np.asarray(_sha256(checkpoint_path)),
            threshold=np.float32(threshold),
            validation_quantile=np.float32(validation_quantile),
            reference_count=np.int32(len(references)),
        )
    finally:
        hook.remove()

    summary = {
        "output": str(output_path),
        "reference_count": int(len(references)),
        "feature_dimension": int(references.shape[1]),
        "validation_samples": int(len(calibration_scores)),
        "validation_quantile": float(validation_quantile),
        "threshold": threshold,
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    print(
        "OOD reference built: "
        f"{summary['reference_count']} refs, threshold={threshold:.4f}, output={output_path}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build checkpoint-specific OOD reference embeddings.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "best_model.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=ood_reference_path(),
        help="OOD bank output. Defaults to <PROJECT_ROOT>/models/ood_reference.npz on all platforms.",
    )
    parser.add_argument("--references-per-class", type=int, default=50)
    parser.add_argument("--validation-quantile", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    root = prepare_dataset(args.data)
    build_ood_reference(
        root,
        args.checkpoint,
        args.output,
        references_per_class=args.references_per_class,
        validation_quantile=args.validation_quantile,
        batch_size=args.batch_size,
        device_preference=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
