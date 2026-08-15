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
        self.ood_enabled = _bool_env("OOD_DETECTION_ENABLED", True)
        self.ood_auto_build = _bool_env("OOD_AUTO_BUILD", True)
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
        self._runtime_device_override: str | None = None

    def _retry_allowed(self) -> bool:
        if self._load_error is None or self._load_error_at is None:
            return True
        return (time.monotonic() - self._load_error_at) >= self.retry_seconds

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
                raise ModelUnavailableError("WASTE_DEVICE=cuda nhưng CUDA không khả dụng.")
            return torch_module.device("cuda")
        if pref == "mps":
            mps = getattr(torch_module.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ModelUnavailableError("WASTE_DEVICE=mps nhưng Apple MPS không khả dụng.")
            return torch_module.device("mps")
        if pref == "cpu":
            return torch_module.device("cpu")
        raise ModelUnavailableError("WASTE_DEVICE phải là auto, cuda, mps hoặc cpu.")

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
                f"Không tìm thấy OOD reference hợp lệ: {self.ood_reference_path}. "
                "Hãy chạy training/build_ood_reference.py hoặc bật OOD_AUTO_BUILD=true."
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
            raise RuntimeError("OOD reference vừa tạo không khớp checkpoint hiện tại.")

    @staticmethod
    def _safe_torch_load(torch_module: Any, path: Path) -> dict[str, Any]:
        try:
            payload = torch_module.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # compatibility with older supported torch releases
            payload = torch_module.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must be a dictionary created by training/train.py")
        return payload

    def _load(self) -> _LoadedModel:
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
                        f"Không tìm thấy checkpoint: {self.checkpoint_path}. "
                        "Hãy train bằng training/train.py rồi chép best_model.pt vào thư mục models/."
                    )

                import torch

                checkpoint = self._safe_torch_load(torch, self.checkpoint_path)
                architecture = str(checkpoint.get("arch", "")).strip().lower()
                image_size = int(checkpoint.get("image_size", 224))
                class_names = tuple(str(x) for x in checkpoint.get("class_names", ()))
                if class_names != WASTE_CLASS_KEYS:
                    raise ValueError(
                        "Checkpoint class_names không khớp schema 11 lớp của project. "
                        f"Expected {WASTE_CLASS_KEYS}, got {class_names}."
                    )
                if image_size < 64 or image_size > 1024:
                    raise ValueError(f"image_size trong checkpoint không hợp lệ: {image_size}")

                state_dict = checkpoint.get("model_state_dict")
                if not isinstance(state_dict, dict):
                    raise ValueError("Checkpoint thiếu model_state_dict")

                checkpoint_hash = _file_sha256(self.checkpoint_path)
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
                            f"Không tìm thấy OOD reference: {self.ood_reference_path}. "
                            "Hãy chạy training/build_ood_reference.py cho checkpoint hiện tại."
                        )
                    with np.load(self.ood_reference_path, allow_pickle=False) as reference:
                        stored_hash = str(reference["checkpoint_sha256"].item())
                        stored_classes = tuple(str(x) for x in reference["class_names"].tolist())
                        embeddings = np.asarray(reference["embeddings"], dtype=np.float32)
                        labels = np.asarray(reference["labels"], dtype=np.int16)
                        stored_threshold = float(reference["threshold"].item())

                    if stored_hash != checkpoint_hash:
                        raise ValueError(
                            "OOD reference không khớp checkpoint hiện tại. "
                            "Hãy tạo lại OOD reference cho checkpoint hiện tại."
                        )
                    if stored_classes != class_names:
                        raise ValueError("OOD reference class_names không khớp schema model.")
                    if embeddings.ndim != 2 or embeddings.shape[0] <= 0:
                        raise ValueError("OOD reference embeddings không hợp lệ.")
                    if embeddings.shape[1] != int(feature_layer.in_features):
                        raise ValueError(
                            "OOD reference feature dimension không khớp classifier head."
                        )
                    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
                        raise ValueError("OOD reference labels không khớp embeddings.")
                    if np.any(labels < 0) or np.any(labels >= len(class_names)):
                        raise ValueError("OOD reference chứa class index không hợp lệ.")
                    if not np.isfinite(embeddings).all():
                        raise ValueError("OOD reference chứa embedding không hữu hạn.")
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    if np.any(norms <= 1e-12):
                        raise ValueError("OOD reference chứa embedding zero-norm.")
                    embeddings = np.ascontiguousarray(embeddings / norms, dtype=np.float32)
                    threshold = (
                        self.ood_threshold_override
                        if self.ood_threshold_override is not None
                        else stored_threshold
                    )
                    if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
                        raise ValueError(f"OOD threshold không hợp lệ: {threshold}")
                    ood_embeddings = embeddings
                    ood_labels = labels
                    ood_threshold = float(threshold)
                    ood_reference_path = str(self.ood_reference_path)

                self._known_checkpoint_hash = checkpoint_hash
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
                self._load_error = f"Không thể load model đã train. Chi tiết: {exc}"
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
        return {
            "state": state,
            "model_type": "supervised-image-classifier",
            "checkpoint": str(self.checkpoint_path),
            "architecture": loaded.architecture if loaded else None,
            "device": str(loaded.device) if loaded else (self._runtime_device_override or self.device_preference),
            "device_fallback_active": bool(self._runtime_device_override),
            "image_size": loaded.image_size if loaded else None,
            "num_classes": len(WASTE_CLASS_KEYS),
            "class_names": list(WASTE_CLASS_KEYS),
            "error": self._load_error,
            "retry_in_seconds": round(retry_in_seconds, 1),
            "unknown_threshold": self.unknown_threshold,
            "uncertainty_margin": self.uncertainty_margin,
            "ood_detection_enabled": self.ood_enabled,
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

    def _handle_inference_failure(self, loaded: _LoadedModel, exc: Exception) -> bool:
        """Invalidate a broken runtime and tell the caller whether to retry once on CPU."""
        message = f"Model inference failed on {loaded.device}: {exc}"
        retry_on_cpu = False
        with self._lock:
            if self._loaded is loaded:
                self._loaded = None
            if self.device_preference == "auto" and getattr(loaded.device, "type", "") != "cpu":
                self._runtime_device_override = "cpu"
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

    def classify(self, image: Image.Image) -> ClassificationResult:
        loaded = self._load()
        image = image.convert("RGB")

        try:
            scores, embedding = self._run_inference(loaded, image)
        except Exception as exc:
            logger.exception("Trained-model inference failed")
            if self._handle_inference_failure(loaded, exc):
                try:
                    fallback_loaded = self._load()
                except Exception as retry_exc:
                    # _load() already recorded the load failure for health/status. Do
                    # not clear it by treating the old GPU runtime as the failed retry.
                    logger.exception("Could not load CPU fallback model")
                    raise ModelUnavailableError(f"Model inference failed: {retry_exc}") from retry_exc
                try:
                    scores, embedding = self._run_inference(fallback_loaded, image)
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
        ood_detected = False
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
            ood_nearest_key = loaded.class_names[int(loaded.ood_labels[nearest_index])]
            ood_detected = ood_similarity < loaded.ood_threshold

        low_confidence = confidence < self.unknown_threshold
        low_margin = margin < self.uncertainty_margin
        uncertain = low_confidence or low_margin or ood_detected
        uncertainty_reasons: list[str] = []
        if low_confidence:
            uncertainty_reasons.append("low_confidence")
        if low_margin:
            uncertainty_reasons.append("low_margin")
        if ood_detected:
            uncertainty_reasons.append("out_of_distribution")

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
            },
            "checkpoint_id": loaded.checkpoint_hash[:12],
            "feature_dimension": len(embedding),
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

