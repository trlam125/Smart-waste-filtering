from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from typing import Any

import numpy as np
from PIL import Image

from .class_schema import WASTE_CLASS_KEYS
from .model_factory import build_eval_transform, create_model, get_final_classifier_layer
from .paths import PROJECT_ROOT, ood_reference_path, resolve_project_path
from .waste_rules import RULE_BY_KEY

logger = logging.getLogger(__name__)
load_dotenv(PROJECT_ROOT / ".env", override=False)
DEFAULT_CHECKPOINT = "models/best_model.pt"


class ModelUnavailableError(RuntimeError):
    """Raised when the trained classifier checkpoint cannot be loaded or executed."""


def _finite_float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite, got {raw!r}")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{name} must be <= {maximum}, got {value}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false, got: {raw!r}")


def _checkpoint_path(raw: str) -> Path:
    return resolve_project_path(raw)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ClassificationResult:
    key: str
    confidence: float
    alternatives: list[dict[str, Any]]
    uncertain: bool = False
    analysis: dict[str, Any] = field(default_factory=dict)
    score_map: dict[str, float] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None
    embedding_kind: str | None = None


@dataclass
class _LoadedModel:
    model: Any
    transform: Any
    device: Any
    architecture: str
    image_size: int
    class_names: tuple[str, ...]
    checkpoint_hash: str
    temperature: float
    feature_layer: Any
    ood_embeddings: np.ndarray | None
    ood_labels: np.ndarray | None
    ood_threshold: float | None
    ood_reference_path: str | None


class WasteClassifier:
    """Supervised 11-class image classifier matching the dataset class schema exactly."""

    def __init__(self) -> None:
        self.checkpoint_path = _checkpoint_path(
            os.getenv("WASTE_MODEL_CHECKPOINT", DEFAULT_CHECKPOINT).strip() or DEFAULT_CHECKPOINT
        )
        self.device_preference = os.getenv("WASTE_DEVICE", "auto").strip().lower() or "auto"
        self.unknown_threshold = _finite_float_env(
            "UNKNOWN_THRESHOLD", 0.60, minimum=0.0, maximum=1.0
        )
        self.uncertainty_margin = _finite_float_env(
            "UNCERTAINTY_MARGIN", 0.10, minimum=0.0, maximum=1.0
        )
        self.retry_seconds = _finite_float_env("MODEL_RETRY_SECONDS", 10.0, minimum=0.0)
        self.device_recovery_seconds = _finite_float_env(
            "MODEL_DEVICE_RECOVERY_SECONDS", 60.0, minimum=0.0
        )
        self.auto_reload = _bool_env("MODEL_AUTO_RELOAD", True)
        self.reload_check_seconds = _finite_float_env(
            "MODEL_RELOAD_CHECK_SECONDS", 2.0, minimum=0.0
        )
        self.framing_rescue_enabled = _bool_env("FRAMING_RESCUE_ENABLED", False)
        self.framing_rescue_min_confidence = _finite_float_env(
            "FRAMING_RESCUE_MIN_CONFIDENCE", 0.75, minimum=0.0, maximum=1.0
        )
        self.ood_enabled = _bool_env("OOD_DETECTION_ENABLED", True)
        self.ood_auto_build = _bool_env("OOD_AUTO_BUILD", True)
        self.ood_class_mismatch_enabled = _bool_env("OOD_CLASS_MISMATCH_ENABLED", True)
        self.ood_class_mismatch_min_similarity = _finite_float_env(
            "OOD_CLASS_MISMATCH_MIN_SIMILARITY", 0.60, minimum=-1.0, maximum=1.0
        )
        self.ood_class_mismatch_min_gap = _finite_float_env(
            "OOD_CLASS_MISMATCH_MIN_GAP", 0.08, minimum=0.0, maximum=2.0
        )
        self.ood_reference_path = ood_reference_path()
        raw_ood_threshold = os.getenv("OOD_MIN_SIMILARITY", "").strip()
        self.ood_threshold_override = (
            _finite_float_env("OOD_MIN_SIMILARITY", 0.0, minimum=-1.0, maximum=1.0)
            if raw_ood_threshold
            else None
        )

        self._loaded: _LoadedModel | None = None
        self._loading = False
        self._load_error: str | None = None
        self._load_error_at: float | None = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._known_checkpoint_hash: str | None = None
        self._loaded_checkpoint_signature: tuple[int, int] | None = None
        self._last_checkpoint_check = 0.0
        self._runtime_device_override: str | None = None
        self._runtime_device_override_at: float | None = None

    def _retry_allowed(self) -> bool:
        if self._load_error is None or self._load_error_at is None:
            return True
        return (time.monotonic() - self._load_error_at) >= self.retry_seconds

    def _checkpoint_signature(self) -> tuple[int, int]:
        stat = self.checkpoint_path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)

    def _device_recovery_retry_in_seconds(self) -> float:
        if (
            self.device_preference != "auto"
            or self._runtime_device_override is None
            or self._runtime_device_override_at is None
        ):
            return 0.0
        return max(
            0.0,
            self.device_recovery_seconds
            - (time.monotonic() - self._runtime_device_override_at),
        )

    def _recover_auto_device_if_due(self) -> None:
        """Retry automatic accelerator selection after a temporary CPU fallback."""
        if self._device_recovery_retry_in_seconds() > 0.0:
            return
        if self.device_preference != "auto" or self._runtime_device_override is None:
            return

        with self._lock:
            if self._device_recovery_retry_in_seconds() > 0.0:
                return
            if self.device_preference != "auto" or self._runtime_device_override is None:
                return

            logger.info(
                "CPU device fallback cooldown expired; retrying automatic device selection."
            )
            self._runtime_device_override = None
            self._runtime_device_override_at = None
            # Drop the cached CPU runtime so the next load resolves auto again.
            self._loaded = None
            self._load_error = None
            self._load_error_at = None

    def _invalidate_if_checkpoint_changed(self) -> None:
        """Drop the cached runtime when best_model.pt is replaced after training."""
        if not self.auto_reload or self._loaded is None:
            return

        now = time.monotonic()
        if (
            self.reload_check_seconds > 0.0
            and (now - self._last_checkpoint_check) < self.reload_check_seconds
        ):
            return
        self._last_checkpoint_check = now

        try:
            current_signature = self._checkpoint_signature()
        except OSError:
            # Keep serving the already-loaded model if the file is only temporarily
            # unavailable while a trainer is replacing it. A later request retries.
            return

        if current_signature == self._loaded_checkpoint_signature:
            return

        with self._lock:
            if self._loaded is None:
                return
            if current_signature == self._loaded_checkpoint_signature:
                return
            logger.info(
                "Checkpoint changed on disk; invalidating cached model for hot reload: %s",
                self.checkpoint_path,
            )
            self._loaded = None
            self._known_checkpoint_hash = None
            self._loaded_checkpoint_signature = None
            self._runtime_device_override = None
            self._runtime_device_override_at = None
            self._load_error = None
            self._load_error_at = None

    @staticmethod
    def _resolve_device(torch_module: Any, preference: str) -> Any:
        pref = preference.lower()
        if pref == "auto":
            if torch_module.cuda.is_available():
                return torch_module.device("cuda")
            mps = getattr(torch_module.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch_module.device("mps")
            return torch_module.device("cpu")
        if pref == "cuda":
            if not torch_module.cuda.is_available():
                raise ModelUnavailableError("WASTE_DEVICE=cuda, but CUDA is not available.")
            return torch_module.device("cuda")
        if pref == "mps":
            mps = getattr(torch_module.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ModelUnavailableError("WASTE_DEVICE=mps, but Apple MPS is not available.")
            return torch_module.device("mps")
        if pref == "cpu":
            return torch_module.device("cpu")
        raise ModelUnavailableError("WASTE_DEVICE must be auto, cuda, mps, or cpu.")

    def _ood_reference_matches_checkpoint(self, checkpoint_hash: str) -> bool:
        if not self.ood_reference_path.is_file():
            return False
        try:
            with np.load(self.ood_reference_path, allow_pickle=False) as reference:
                return str(reference["checkpoint_sha256"].item()) == checkpoint_hash
        except Exception:
            logger.warning(
                "OOD reference is unreadable and will be rebuilt if auto-build is enabled: %s",
                self.ood_reference_path,
                exc_info=True,
            )
            return False

    def _ensure_ood_reference(self, checkpoint_hash: str) -> None:
        if not self.ood_enabled or self._ood_reference_matches_checkpoint(checkpoint_hash):
            return
        if not self.ood_auto_build:
            raise FileNotFoundError(
                f"No valid OOD reference found: {self.ood_reference_path}. "
                "Run training/build_ood_reference.py or enable OOD_AUTO_BUILD=true."
            )

        # Import lazily so normal app startup does not pull training utilities unless
        # the persistent OOD bank is missing/stale. On Colab, dataset extraction still
        # happens in /content while the small rebuilt OOD bank is saved on Drive.
        from training.build_ood_reference import build_ood_reference
        from training.dataset_utils import DEFAULT_DATASET_SOURCE, prepare_dataset

        logger.info(
            "Building persistent OOD reference for current checkpoint: %s",
            self.ood_reference_path,
        )
        dataset_root = prepare_dataset(DEFAULT_DATASET_SOURCE)
        self.ood_reference_path.parent.mkdir(parents=True, exist_ok=True)
        build_ood_reference(
            dataset_root,
            self.checkpoint_path,
            self.ood_reference_path,
            device_preference=self._runtime_device_override or self.device_preference,
        )
        if not self._ood_reference_matches_checkpoint(checkpoint_hash):
            raise RuntimeError("The newly created OOD reference does not match the current checkpoint.")

    @staticmethod
    def _safe_torch_load(torch_module: Any, path: Path) -> dict[str, Any]:
        try:
            payload = torch_module.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # compatibility with older supported torch releases
            payload = torch_module.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must be a dictionary created by training/train.py")
        return payload

    def _load(self, *, allow_device_recovery: bool = True) -> _LoadedModel:
        if allow_device_recovery:
            self._recover_auto_device_if_due()
        self._invalidate_if_checkpoint_changed()
        if self._loaded is not None:
            return self._loaded
        if self._load_error is not None and not self._retry_allowed():
            raise ModelUnavailableError(self._load_error)

        with self._lock:
            if self._loaded is not None:
                return self._loaded
            if self._load_error is not None and not self._retry_allowed():
                raise ModelUnavailableError(self._load_error)

            self._loading = True
            self._load_error = None
            self._load_error_at = None
            try:
                if not self.checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Checkpoint not found: {self.checkpoint_path}. "
                        "Train with training/train.py, then copy best_model.pt into the models/ directory."
                    )

                import torch

                signature_before = self._checkpoint_signature()
                checkpoint = self._safe_torch_load(torch, self.checkpoint_path)
                architecture = str(checkpoint.get("arch", "")).strip().lower()
                image_size = int(checkpoint.get("image_size", 224))
                class_names = tuple(str(x) for x in checkpoint.get("class_names", ()))
                if class_names != WASTE_CLASS_KEYS:
                    raise ValueError(
                        "Checkpoint class_names do not match the project's 11-class schema. "
                        f"Expected {WASTE_CLASS_KEYS}, got {class_names}."
                    )
                if image_size < 64 or image_size > 1024:
                    raise ValueError(f"Invalid image_size in checkpoint: {image_size}")

                state_dict = checkpoint.get("model_state_dict")
                if not isinstance(state_dict, dict):
                    raise ValueError("Checkpoint is missing model_state_dict")

                checkpoint_hash = _file_sha256(self.checkpoint_path)
                signature_after = self._checkpoint_signature()
                if signature_after != signature_before:
                    raise RuntimeError(
                        "Checkpoint changed while it was being loaded; retry the request."
                    )
                self._ensure_ood_reference(checkpoint_hash)

                effective_device_preference = self._runtime_device_override or self.device_preference
                device = self._resolve_device(torch, effective_device_preference)
                model = create_model(architecture, len(class_names), pretrained=False)
                model.load_state_dict(state_dict, strict=True)
                model.to(device)
                model.eval()
                feature_layer = get_final_classifier_layer(model, architecture)

                mean = tuple(float(x) for x in checkpoint.get("mean", (0.485, 0.456, 0.406)))
                std = tuple(float(x) for x in checkpoint.get("std", (0.229, 0.224, 0.225)))
                if len(mean) != 3 or len(std) != 3:
                    raise ValueError("Checkpoint mean/std must each contain 3 values")
                transform = build_eval_transform(image_size, mean, std)
                temperature = float(checkpoint.get("temperature", 1.0))
                if not math.isfinite(temperature) or temperature <= 0.0:
                    temperature = 1.0

                ood_embeddings: np.ndarray | None = None
                ood_labels: np.ndarray | None = None
                ood_threshold: float | None = None
                ood_reference_path: str | None = None
                if self.ood_enabled:
                    if not self.ood_reference_path.is_file():
                        raise FileNotFoundError(
                            f"OOD reference not found: {self.ood_reference_path}. "
                            "Run training/build_ood_reference.py for the current checkpoint."
                        )
                    with np.load(self.ood_reference_path, allow_pickle=False) as reference:
                        stored_hash = str(reference["checkpoint_sha256"].item())
                        stored_classes = tuple(str(x) for x in reference["class_names"].tolist())
                        embeddings = np.asarray(reference["embeddings"], dtype=np.float32)
                        labels = np.asarray(reference["labels"], dtype=np.int16)
                        stored_threshold = float(reference["threshold"].item())

                    if stored_hash != checkpoint_hash:
                        raise ValueError(
                            "OOD reference does not match the current checkpoint. "
                            "Rebuild the OOD reference for the current checkpoint."
                        )
                    if stored_classes != class_names:
                        raise ValueError("OOD reference class_names do not match the model schema.")
                    if embeddings.ndim != 2 or embeddings.shape[0] <= 0:
                        raise ValueError("OOD reference embeddings are invalid.")
                    if embeddings.shape[1] != int(feature_layer.in_features):
                        raise ValueError(
                            "OOD reference feature dimension does not match the classifier head."
                        )
                    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
                        raise ValueError("OOD reference labels do not match the embeddings.")
                    if np.any(labels < 0) or np.any(labels >= len(class_names)):
                        raise ValueError("OOD reference contains an invalid class index.")
                    if not np.isfinite(embeddings).all():
                        raise ValueError("OOD reference contains a non-finite embedding.")
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    if np.any(norms <= 1e-12):
                        raise ValueError("OOD reference contains a zero-norm embedding.")
                    embeddings = np.ascontiguousarray(embeddings / norms, dtype=np.float32)
                    threshold = (
                        self.ood_threshold_override
                        if self.ood_threshold_override is not None
                        else stored_threshold
                    )
                    if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
                        raise ValueError(f"Invalid OOD threshold: {threshold}")
                    ood_embeddings = embeddings
                    ood_labels = labels
                    ood_threshold = float(threshold)
                    ood_reference_path = str(self.ood_reference_path)

                self._known_checkpoint_hash = checkpoint_hash
                self._loaded_checkpoint_signature = signature_after
                self._loaded = _LoadedModel(
                    model=model,
                    transform=transform,
                    device=device,
                    architecture=architecture,
                    image_size=image_size,
                    class_names=class_names,
                    checkpoint_hash=checkpoint_hash,
                    temperature=temperature,
                    feature_layer=feature_layer,
                    ood_embeddings=ood_embeddings,
                    ood_labels=ood_labels,
                    ood_threshold=ood_threshold,
                    ood_reference_path=ood_reference_path,
                )
                logger.info(
                    "Loaded supervised waste model: arch=%s classes=%s device=%s checkpoint=%s",
                    architecture,
                    len(class_names),
                    device,
                    self.checkpoint_path,
                )
                return self._loaded
            except ModelUnavailableError as exc:
                self._load_error = str(exc)
                self._load_error_at = time.monotonic()
                raise
            except Exception as exc:
                self._load_error = f"Failed to load the trained model. Details: {exc}"
                self._load_error_at = time.monotonic()
                logger.exception("Could not load trained waste classifier")
                raise ModelUnavailableError(self._load_error) from exc
            finally:
                self._loading = False

    def warmup(self) -> None:
        self._load()

    def _embedding_kind(self) -> str:
        digest = self._known_checkpoint_hash
        if digest is None and self.checkpoint_path.is_file():
            try:
                digest = _file_sha256(self.checkpoint_path)
                self._known_checkpoint_hash = digest
            except OSError:
                digest = None
        return f"feature-v1:{(digest or 'missing')[:12]}"

    def learning_embedding_kinds(self) -> tuple[str, ...]:
        return (self._embedding_kind(),)

    @property
    def status(self) -> dict[str, Any]:
        retry_in_seconds = 0.0
        if self._load_error and self._load_error_at is not None:
            retry_in_seconds = max(0.0, self.retry_seconds - (time.monotonic() - self._load_error_at))

        if self._loaded is not None:
            state = "ready"
        elif self._loading:
            state = "loading"
        elif self._load_error and retry_in_seconds > 0:
            state = "error"
        elif self._load_error:
            state = "retry_available"
        else:
            state = "not_loaded"

        loaded = self._loaded
        device_fallback_retry_in_seconds = self._device_recovery_retry_in_seconds()
        return {
            "state": state,
            "model_type": "supervised-image-classifier",
            "checkpoint": str(self.checkpoint_path),
            "architecture": loaded.architecture if loaded else None,
            "device": str(loaded.device) if loaded else (self._runtime_device_override or self.device_preference),
            "device_fallback_active": bool(self._runtime_device_override),
            "device_recovery_seconds": self.device_recovery_seconds,
            "device_fallback_retry_in_seconds": round(device_fallback_retry_in_seconds, 1),
            "image_size": loaded.image_size if loaded else None,
            "num_classes": len(WASTE_CLASS_KEYS),
            "class_names": list(WASTE_CLASS_KEYS),
            "error": self._load_error,
            "retry_in_seconds": round(retry_in_seconds, 1),
            "model_auto_reload": self.auto_reload,
            "model_reload_check_seconds": self.reload_check_seconds,
            "framing_rescue_enabled": self.framing_rescue_enabled,
            "framing_rescue_min_confidence": self.framing_rescue_min_confidence,
            "unknown_threshold": self.unknown_threshold,
            "uncertainty_margin": self.uncertainty_margin,
            "ood_detection_enabled": self.ood_enabled,
            "ood_class_mismatch_enabled": self.ood_class_mismatch_enabled,
            "ood_class_mismatch_min_similarity": self.ood_class_mismatch_min_similarity,
            "ood_class_mismatch_min_gap": self.ood_class_mismatch_min_gap,
            "ood_reference": loaded.ood_reference_path if loaded else str(self.ood_reference_path),
            "ood_min_similarity": loaded.ood_threshold if loaded else self.ood_threshold_override,
        }

    def _run_inference(self, loaded: _LoadedModel, image: Image.Image) -> tuple[list[float], tuple[float, ...]]:
        import torch

        tensor = loaded.transform(image).unsqueeze(0).to(loaded.device)
        captured: dict[str, Any] = {}

        def capture_features(_module: Any, inputs: tuple[Any, ...]) -> None:
            if inputs:
                captured["features"] = inputs[0].detach()

        with self._inference_lock, torch.inference_mode():
            hook = loaded.feature_layer.register_forward_pre_hook(capture_features)
            try:
                logits = loaded.model(tensor)
            finally:
                hook.remove()
            probabilities = torch.softmax(logits / loaded.temperature, dim=1)[0]
            scores = probabilities.detach().to("cpu", dtype=torch.float32).tolist()

        feature_tensor = captured.get("features")
        if feature_tensor is None or feature_tensor.ndim < 2 or feature_tensor.size(0) != 1:
            raise RuntimeError("Could not capture the model feature vector before the classifier head")
        feature_vector = feature_tensor[0].detach().to("cpu", dtype=torch.float32).flatten()
        if feature_vector.numel() <= 0 or feature_vector.numel() > 4096:
            raise RuntimeError(f"Invalid feedback feature dimension: {feature_vector.numel()}")
        if not bool(torch.isfinite(feature_vector).all()):
            raise RuntimeError("Feedback feature vector contains non-finite values")
        norm = float(torch.linalg.vector_norm(feature_vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("Feedback feature vector has zero/invalid norm")
        feature_vector = feature_vector / norm
        embedding = tuple(float(value) for value in feature_vector.tolist())
        return scores, embedding

    @staticmethod
    def _center_crop_fraction(image: Image.Image, fraction: float) -> Image.Image:
        width, height = image.size
        crop_width = max(2, min(width, int(round(width * fraction))))
        crop_height = max(2, min(height, int(round(height * fraction))))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))

    def _apply_framing_rescue(
        self,
        loaded: _LoadedModel,
        image: Image.Image,
        scores: list[float],
        embedding: tuple[float, ...],
    ) -> tuple[list[float], tuple[float, ...], dict[str, Any]]:
        """Legacy full-frame rescue for direct classifier use only.

        Rescue small electronic accessories that a background dominates.

        The current classifier is trained on single-object crops. In a phone camera
        photo, a small white charger on a large table can be classified as paper or
        cardboard because the background occupies most of the frame. We only run
        this extra path for that narrow failure mode, and require two independent
        center zooms to agree on ``electronic`` before overriding the full-frame
        prediction. This avoids applying aggressive multi-crop voting to every class.
        """
        base_index = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        base_key = loaded.class_names[base_index]
        info: dict[str, Any] = {
            "enabled": self.framing_rescue_enabled,
            "allowed": True,
            "applied": False,
            "base_key": base_key,
            "base_confidence": round(float(scores[base_index]), 4),
            "reason": "not_needed",
            "views": [],
        }

        if not self.framing_rescue_enabled:
            info["reason"] = "disabled"
            return scores, embedding, info
        if base_key not in {"paper", "cardboard"}:
            return scores, embedding, info
        if "electronic" not in loaded.class_names:
            return scores, embedding, info

        electronic_index = loaded.class_names.index("electronic")
        candidate_results: list[tuple[float, list[float], tuple[float, ...]]] = []
        for fraction in (0.70, 0.55):
            crop = self._center_crop_fraction(image, fraction)
            crop_scores, crop_embedding = self._run_inference(loaded, crop)
            crop_top_index = int(np.argmax(np.asarray(crop_scores, dtype=np.float32)))
            crop_top_key = loaded.class_names[crop_top_index]
            electronic_score = float(crop_scores[electronic_index])
            info["views"].append(
                {
                    "center_fraction": fraction,
                    "top1_key": crop_top_key,
                    "top1_confidence": round(float(crop_scores[crop_top_index]), 4),
                    "electronic_confidence": round(electronic_score, 4),
                }
            )
            candidate_results.append((electronic_score, crop_scores, crop_embedding))

        both_electronic = all(
            view["top1_key"] == "electronic" for view in info["views"]
        )
        mean_electronic_confidence = float(
            sum(result[0] for result in candidate_results) / len(candidate_results)
        )
        info["mean_electronic_confidence"] = round(mean_electronic_confidence, 4)

        if (
            both_electronic
            and mean_electronic_confidence >= self.framing_rescue_min_confidence
        ):
            best_candidate = max(candidate_results, key=lambda item: item[0])
            info["applied"] = True
            info["reason"] = "two_zoom_views_agree_on_electronic"
            info["selected_key"] = "electronic"
            info["selected_confidence"] = round(best_candidate[0], 4)
            return best_candidate[1], best_candidate[2], info

        info["reason"] = "zoom_views_not_strong_enough"
        return scores, embedding, info

    def _handle_inference_failure(self, loaded: _LoadedModel, exc: Exception) -> bool:
        """Invalidate a broken runtime and tell the caller whether to retry once on CPU."""
        message = f"Model inference failed on {loaded.device}: {exc}"
        retry_on_cpu = False
        with self._lock:
            if self._loaded is loaded:
                self._loaded = None
            if self.device_preference == "auto" and getattr(loaded.device, "type", "") != "cpu":
                self._runtime_device_override = "cpu"
                self._runtime_device_override_at = time.monotonic()
                self._load_error = None
                self._load_error_at = None
                retry_on_cpu = True
            else:
                self._load_error = message
                self._load_error_at = time.monotonic()
        if retry_on_cpu:
            logger.warning("%s; retrying once on CPU", message)
        else:
            logger.error("%s", message)
        return retry_on_cpu

    def classify(
        self,
        image: Image.Image,
        *,
        allow_framing_rescue: bool = True,
    ) -> ClassificationResult:
        loaded = self._load()
        image = image.convert("RGB")

        def maybe_apply_framing_rescue(
            active_loaded: _LoadedModel,
            active_scores: list[float],
            active_embedding: tuple[float, ...],
        ) -> tuple[list[float], tuple[float, ...], dict[str, Any]]:
            if allow_framing_rescue:
                return self._apply_framing_rescue(
                    active_loaded, image, active_scores, active_embedding
                )
            base_index = int(np.argmax(np.asarray(active_scores, dtype=np.float32)))
            return active_scores, active_embedding, {
                "enabled": self.framing_rescue_enabled,
                "allowed": False,
                "applied": False,
                "base_key": active_loaded.class_names[base_index],
                "base_confidence": round(float(active_scores[base_index]), 4),
                "reason": "disabled_for_detector_crop",
                "views": [],
            }

        framing_rescue: dict[str, Any] = {}
        try:
            scores, embedding = self._run_inference(loaded, image)
            scores, embedding, framing_rescue = maybe_apply_framing_rescue(
                loaded, scores, embedding
            )
        except Exception as exc:
            logger.exception("Trained-model inference failed")
            if self._handle_inference_failure(loaded, exc):
                try:
                    fallback_loaded = self._load(allow_device_recovery=False)
                except Exception as retry_exc:
                    # _load() already recorded the load failure for health/status. Do
                    # not clear it by treating the old GPU runtime as the failed retry.
                    logger.exception("Could not load CPU fallback model")
                    raise ModelUnavailableError(f"Model inference failed: {retry_exc}") from retry_exc
                try:
                    scores, embedding = self._run_inference(fallback_loaded, image)
                    scores, embedding, framing_rescue = maybe_apply_framing_rescue(
                        fallback_loaded, scores, embedding
                    )
                    loaded = fallback_loaded
                except Exception as retry_exc:
                    logger.exception("CPU fallback inference failed")
                    self._handle_inference_failure(fallback_loaded, retry_exc)
                    raise ModelUnavailableError(f"Model inference failed: {retry_exc}") from retry_exc
            else:
                raise ModelUnavailableError(f"Model inference failed: {exc}") from exc

        score_map = {
            key: float(scores[index]) for index, key in enumerate(loaded.class_names)
        }
        ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            raise ModelUnavailableError("Model returned no class scores")

        best_key, confidence = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = confidence - runner_up_score
        ood_similarity: float | None = None
        ood_nearest_key: str | None = None
        ood_predicted_class_similarity: float | None = None
        ood_class_similarity_gap: float | None = None
        ood_detected = False
        ood_class_mismatch = False
        if (
            self.ood_enabled
            and loaded.ood_embeddings is not None
            and loaded.ood_labels is not None
            and loaded.ood_threshold is not None
        ):
            query = np.asarray(embedding, dtype=np.float32)
            similarities = loaded.ood_embeddings @ query
            nearest_index = int(np.argmax(similarities))
            ood_similarity = float(similarities[nearest_index])
            nearest_label = int(loaded.ood_labels[nearest_index])
            ood_nearest_key = loaded.class_names[nearest_label]
            ood_detected = ood_similarity < loaded.ood_threshold

            predicted_label = loaded.class_names.index(best_key)
            predicted_mask = loaded.ood_labels == predicted_label
            if bool(np.any(predicted_mask)):
                ood_predicted_class_similarity = float(np.max(similarities[predicted_mask]))
                ood_class_similarity_gap = ood_similarity - ood_predicted_class_similarity
                mismatch_similarity_floor = max(
                    float(loaded.ood_threshold),
                    self.ood_class_mismatch_min_similarity,
                )
                ood_class_mismatch = bool(
                    self.ood_class_mismatch_enabled
                    and not ood_detected
                    and ood_nearest_key != best_key
                    and ood_similarity >= mismatch_similarity_floor
                    and ood_class_similarity_gap >= self.ood_class_mismatch_min_gap
                )

        low_confidence = confidence < self.unknown_threshold
        low_margin = margin < self.uncertainty_margin
        uncertain = low_confidence or low_margin or ood_detected or ood_class_mismatch
        uncertainty_reasons: list[str] = []
        if low_confidence:
            uncertainty_reasons.append("low_confidence")
        if low_margin:
            uncertainty_reasons.append("low_margin")
        if ood_detected:
            uncertainty_reasons.append("out_of_distribution")
        if ood_class_mismatch:
            uncertainty_reasons.append("embedding_class_mismatch")

        alternatives = [
            {
                "key": key,
                "display_name": RULE_BY_KEY[key].display_name,
                "confidence": round(score, 4),
            }
            for key, score in ranked[1:4]
        ]

        analysis = {
            "model_type": "supervised",
            "architecture": loaded.architecture,
            "device": str(loaded.device),
            "image_size": loaded.image_size,
            "temperature": round(loaded.temperature, 6),
            "top1_key": best_key,
            "top1_confidence": round(confidence, 4),
            "runner_up_key": ranked[1][0] if len(ranked) > 1 else None,
            "runner_up_confidence": round(runner_up_score, 4),
            "margin": round(margin, 4),
            "uncertainty_reasons": uncertainty_reasons,
            "ood": {
                "enabled": self.ood_enabled,
                "detected": ood_detected,
                "similarity": round(ood_similarity, 4) if ood_similarity is not None else None,
                "threshold": round(loaded.ood_threshold, 4)
                if loaded.ood_threshold is not None
                else None,
                "nearest_reference_key": ood_nearest_key,
                "predicted_class_similarity": round(ood_predicted_class_similarity, 4)
                if ood_predicted_class_similarity is not None
                else None,
                "class_similarity_gap": round(ood_class_similarity_gap, 4)
                if ood_class_similarity_gap is not None
                else None,
                "class_mismatch": ood_class_mismatch,
                "class_mismatch_min_similarity": self.ood_class_mismatch_min_similarity,
                "class_mismatch_min_gap": self.ood_class_mismatch_min_gap,
            },
            "checkpoint_id": loaded.checkpoint_hash[:12],
            "feature_dimension": len(embedding),
            "framing_rescue": framing_rescue,
        }
        return ClassificationResult(
            key=best_key,
            confidence=round(confidence, 4),
            alternatives=alternatives,
            uncertain=uncertain,
            analysis=analysis,
            score_map=score_map,
            embedding=embedding,
            embedding_kind=self._embedding_kind(),
        )

