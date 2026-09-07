from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image

from .paths import PROJECT_ROOT, resolve_project_path

logger = logging.getLogger(__name__)
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_CHECKPOINT = "models/best_detector.pt"


class DetectorUnavailableError(RuntimeError):
    """Raised when the trained waste-object detector cannot be loaded or executed."""


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


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got {value}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false, got {raw!r}")


@dataclass(frozen=True)
class DetectionResult:
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]


@dataclass
class _LoadedDetector:
    model: Any
    device: str
    class_names: dict[int, str]
    checkpoint_signature: tuple[int, int]


class WasteDetector:
    """Class-agnostic YOLO detector used to isolate each waste object before classification."""

    def __init__(self) -> None:
        raw_checkpoint = os.getenv("WASTE_DETECTOR_CHECKPOINT", DEFAULT_CHECKPOINT).strip()
        self.checkpoint_path = resolve_project_path(raw_checkpoint or DEFAULT_CHECKPOINT)
        self.device_preference = os.getenv("WASTE_DETECTOR_DEVICE", "auto").strip() or "auto"
        self.confidence_threshold = _finite_float_env(
            "DETECTOR_CONFIDENCE", 0.45, minimum=0.0, maximum=1.0
        )
        self.iou_threshold = _finite_float_env(
            "DETECTOR_IOU", 0.45, minimum=0.0, maximum=1.0
        )
        self.image_size = _positive_int_env("DETECTOR_IMAGE_SIZE", 640)
        self.max_detections = _positive_int_env("DETECTOR_MAX_DETECTIONS", 8)
        self.min_box_area_ratio = _finite_float_env(
            "DETECTOR_MIN_BOX_AREA_RATIO", 0.015, minimum=0.0, maximum=1.0
        )

        # Multi-pass rescue: when the full-frame detector returns only a few
        # objects, run additional overlapping crops along the image's long
        # axis. This makes smaller/partially obscured objects occupy more
        # pixels at YOLO input resolution without changing the trained model.
        self.multi_pass_enabled = _bool_env("DETECTOR_MULTIPASS_ENABLED", True)
        self.multi_pass_trigger_count = _positive_int_env(
            "DETECTOR_MULTIPASS_TRIGGER_COUNT", 4
        )
        self.multi_pass_confidence = _finite_float_env(
            "DETECTOR_MULTIPASS_CONFIDENCE",
            min(self.confidence_threshold, 0.05),
            minimum=0.0,
            maximum=1.0,
        )
        self.multi_pass_splits = _positive_int_env("DETECTOR_MULTIPASS_SPLITS", 2)
        if self.multi_pass_splits > 4:
            raise RuntimeError(
                "DETECTOR_MULTIPASS_SPLITS must be <= 4 to avoid excessive inference cost"
            )
        self.multi_pass_overlap = _finite_float_env(
            "DETECTOR_MULTIPASS_OVERLAP", 0.25, minimum=0.0, maximum=0.49
        )
        self.multi_pass_edge_margin = _finite_float_env(
            "DETECTOR_MULTIPASS_EDGE_MARGIN", 0.015, minimum=0.0, maximum=0.10
        )
        self.merge_iou_threshold = _finite_float_env(
            "DETECTOR_MERGE_IOU", 0.50, minimum=0.0, maximum=1.0
        )
        self.merge_containment_threshold = _finite_float_env(
            "DETECTOR_MERGE_CONTAINMENT", 0.85, minimum=0.0, maximum=1.0
        )

        self.crop_padding = _finite_float_env(
            "DETECTOR_CROP_PADDING", 0.08, minimum=0.0, maximum=0.50
        )
        self.retry_seconds = _finite_float_env("DETECTOR_RETRY_SECONDS", 10.0, minimum=0.0)
        self.auto_reload = _bool_env("DETECTOR_AUTO_RELOAD", True)
        self.reload_check_seconds = _finite_float_env(
            "DETECTOR_RELOAD_CHECK_SECONDS", 2.0, minimum=0.0
        )

        self._loaded: _LoadedDetector | None = None
        self._loading = False
        self._load_error: str | None = None
        self._load_error_at: float | None = None
        self._last_checkpoint_check = 0.0
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _retry_allowed(self) -> bool:
        if self._load_error is None or self._load_error_at is None:
            return True
        return (time.monotonic() - self._load_error_at) >= self.retry_seconds

    def _checkpoint_signature(self) -> tuple[int, int]:
        stat = self.checkpoint_path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)

    def _invalidate_if_checkpoint_changed(self) -> None:
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
            return
        if current_signature == self._loaded.checkpoint_signature:
            return
        with self._lock:
            if self._loaded is None or current_signature == self._loaded.checkpoint_signature:
                return
            logger.info(
                "Detector checkpoint changed on disk; invalidating cached detector: %s",
                self.checkpoint_path,
            )
            self._loaded = None
            self._load_error = None
            self._load_error_at = None

    def _device_argument(self) -> Any:
        value = self.device_preference.strip().lower()
        if value in {"", "auto"}:
            return None
        if "," in value:
            parts = [part.strip() for part in value.split(",") if part.strip()]
            try:
                return [int(part) for part in parts]
            except ValueError:
                return self.device_preference
        try:
            return int(value)
        except ValueError:
            return self.device_preference

    @staticmethod
    def _runtime_device(model: Any, fallback: str) -> str:
        """Return the device actually used by Ultralytics after prediction."""
        try:
            predictor = getattr(model, "predictor", None)
            predictor_device = getattr(predictor, "device", None)
            if predictor_device is not None:
                return str(predictor_device)
        except Exception:
            pass

        try:
            torch_model = getattr(model, "model", None)
            if torch_model is not None:
                parameter = next(torch_model.parameters())
                return str(parameter.device)
        except Exception:
            pass
        return fallback

    def _load(self) -> _LoadedDetector:
        self._invalidate_if_checkpoint_changed()
        if self._loaded is not None:
            return self._loaded
        if not self._retry_allowed():
            raise DetectorUnavailableError(self._load_error or "Detector is temporarily unavailable.")

        with self._lock:
            self._invalidate_if_checkpoint_changed()
            if self._loaded is not None:
                return self._loaded
            if not self._retry_allowed():
                raise DetectorUnavailableError(self._load_error or "Detector is temporarily unavailable.")

            self._loading = True
            try:
                if not self.checkpoint_path.is_file():
                    raise DetectorUnavailableError(
                        f"Detector checkpoint not found: {self.checkpoint_path}"
                    )
                try:
                    from ultralytics import YOLO
                except ImportError as exc:
                    raise DetectorUnavailableError(
                        "Ultralytics is not installed. Run pip install -r requirements.txt."
                    ) from exc

                signature = self._checkpoint_signature()
                model = YOLO(str(self.checkpoint_path))
                names_raw = getattr(model, "names", {}) or {}
                if isinstance(names_raw, (list, tuple)):
                    class_names = {index: str(name) for index, name in enumerate(names_raw)}
                else:
                    class_names = {int(key): str(value) for key, value in dict(names_raw).items()}
                normalized_names = {
                    index: name.strip().lower()
                    for index, name in class_names.items()
                }
                if normalized_names != {0: "waste_object"}:
                    raise DetectorUnavailableError(
                        "The detector checkpoint must contain exactly 1 class: 0 waste_object; "
                        f"the current checkpoint has classes={class_names}."
                    )

                device = self._runtime_device(model, self.device_preference)

                self._loaded = _LoadedDetector(
                    model=model,
                    device=device,
                    class_names=class_names,
                    checkpoint_signature=signature,
                )
                self._load_error = None
                self._load_error_at = None
                logger.info(
                    "Loaded waste-object detector: checkpoint=%s classes=%s",
                    self.checkpoint_path,
                    class_names,
                )
                return self._loaded
            except DetectorUnavailableError as exc:
                self._load_error = str(exc)
                self._load_error_at = time.monotonic()
                raise
            except Exception as exc:
                self._load_error = f"Failed to load the trained detector. Details: {exc}"
                self._load_error_at = time.monotonic()
                logger.exception("Could not load trained waste detector")
                raise DetectorUnavailableError(self._load_error) from exc
            finally:
                self._loading = False

    def warmup(self) -> None:
        self._load()

    @staticmethod
    def _axis_windows(length: int, splits: int, overlap: float) -> list[tuple[int, int]]:
        """Build overlapping windows that cover one image axis end-to-end."""
        if length <= 1 or splits <= 1:
            return [(0, length)]

        effective_parts = splits - (splits - 1) * overlap
        window_size = max(1, min(length, int(math.ceil(length / effective_parts))))
        if window_size >= length:
            return [(0, length)]

        max_start = length - window_size
        if splits == 2:
            starts = [0, max_start]
        else:
            starts = [
                int(round(index * max_start / (splits - 1)))
                for index in range(splits)
            ]

        windows: list[tuple[int, int]] = []
        for start in starts:
            start = max(0, min(max_start, start))
            end = min(length, start + window_size)
            candidate = (start, end)
            if candidate not in windows:
                windows.append(candidate)
        return windows

    @staticmethod
    def _intersection_metrics(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """Return IoU and intersection-over-smaller-area for two xyxy boxes."""
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if intersection <= 0.0:
            return 0.0, 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        iou = intersection / union if union > 0.0 else 0.0
        smaller = min(area_a, area_b)
        containment = intersection / smaller if smaller > 0.0 else 0.0
        return iou, containment

    def _merge_detections(
        self,
        detections: list[DetectionResult],
    ) -> list[DetectionResult]:
        """Deduplicate full-frame and rescue-pass detections."""
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept: list[DetectionResult] = []
        for candidate in ordered:
            duplicate = False
            for existing in kept:
                iou, containment = self._intersection_metrics(
                    candidate.bbox_xyxy,
                    existing.bbox_xyxy,
                )
                candidate_area = (
                    max(0.0, candidate.bbox_xyxy[2] - candidate.bbox_xyxy[0])
                    * max(0.0, candidate.bbox_xyxy[3] - candidate.bbox_xyxy[1])
                )
                existing_area = (
                    max(0.0, existing.bbox_xyxy[2] - existing.bbox_xyxy[0])
                    * max(0.0, existing.bbox_xyxy[3] - existing.bbox_xyxy[1])
                )
                larger_area = max(candidate_area, existing_area)
                size_similarity = (
                    min(candidate_area, existing_area) / larger_area
                    if larger_area > 0.0
                    else 0.0
                )
                if iou >= self.merge_iou_threshold or (
                    containment >= self.merge_containment_threshold
                    and size_similarity >= 0.50
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
            if len(kept) >= self.max_detections:
                break
        return kept

    @staticmethod
    def _boxes_from_result(results: Any) -> tuple[list[list[float]], list[float]]:
        if not results:
            return [], []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], []
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        return xyxy, confidences

    def _convert_boxes(
        self,
        *,
        xyxy: list[list[float]],
        confidences: list[float],
        full_width: int,
        full_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
        crop_width: int | None = None,
        crop_height: int | None = None,
        reject_internal_edges: bool = False,
    ) -> list[DetectionResult]:
        converted: list[DetectionResult] = []
        local_width = crop_width if crop_width is not None else full_width
        local_height = crop_height if crop_height is not None else full_height
        edge_margin_x = max(1.0, local_width * self.multi_pass_edge_margin)
        edge_margin_y = max(1.0, local_height * self.multi_pass_edge_margin)

        for coordinates, confidence in zip(xyxy, confidences):
            if len(coordinates) != 4:
                continue
            local_x1, local_y1, local_x2, local_y2 = (
                float(value) for value in coordinates
            )

            if reject_internal_edges:
                touches_left_cut = offset_x > 0 and local_x1 <= edge_margin_x
                touches_top_cut = offset_y > 0 and local_y1 <= edge_margin_y
                touches_right_cut = (
                    offset_x + local_width < full_width
                    and local_x2 >= local_width - edge_margin_x
                )
                touches_bottom_cut = (
                    offset_y + local_height < full_height
                    and local_y2 >= local_height - edge_margin_y
                )
                if touches_left_cut or touches_top_cut or touches_right_cut or touches_bottom_cut:
                    # The object is probably clipped by an artificial rescue-crop
                    # boundary. An overlapping neighboring crop should see it whole.
                    continue

            x1 = max(0.0, min(float(full_width), local_x1 + offset_x))
            y1 = max(0.0, min(float(full_height), local_y1 + offset_y))
            x2 = max(0.0, min(float(full_width), local_x2 + offset_x))
            y2 = max(0.0, min(float(full_height), local_y2 + offset_y))
            if x2 <= x1 or y2 <= y1:
                continue

            area_ratio = ((x2 - x1) * (y2 - y1)) / float(full_width * full_height)
            if area_ratio < self.min_box_area_ratio:
                continue

            converted.append(
                DetectionResult(
                    confidence=round(float(confidence), 4),
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_normalized=(
                        x1 / full_width,
                        y1 / full_height,
                        x2 / full_width,
                        y2 / full_height,
                    ),
                )
            )
        return converted

    def detect(self, image: Image.Image) -> list[DetectionResult]:
        loaded = self._load()
        source = image.convert("RGB")
        width, height = source.size
        if width <= 0 or height <= 0:
            return []

        # Ask YOLO for more candidates than the public limit because tiny boxes
        # are filtered by area below. Otherwise small high-confidence boxes can
        # consume max_det slots and hide valid objects ranked just after them.
        raw_max_detections = max(
            self.max_detections,
            min(300, self.max_detections * 3),
        )

        detections: list[DetectionResult] = []
        try:
            with self._inference_lock:
                full_results = loaded.model.predict(
                    source=source,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.image_size,
                    max_det=raw_max_detections,
                    agnostic_nms=True,
                    device=self._device_argument(),
                    verbose=False,
                )
                full_xyxy, full_confidences = self._boxes_from_result(full_results)
                detections.extend(
                    self._convert_boxes(
                        xyxy=full_xyxy,
                        confidences=full_confidences,
                        full_width=width,
                        full_height=height,
                    )
                )

                # Rescue low-count full-frame predictions with overlapping crops
                # along the long image axis. A portrait camera frame is split into
                # horizontal bands; a landscape frame into vertical bands. This
                # avoids cutting large objects across both axes while giving small
                # objects substantially more pixels at YOLO input resolution.
                if (
                    self.multi_pass_enabled
                    and len(detections) < self.multi_pass_trigger_count
                    and self.multi_pass_splits > 1
                ):
                    if height >= width:
                        windows = self._axis_windows(
                            height,
                            self.multi_pass_splits,
                            self.multi_pass_overlap,
                        )
                        rescue_regions = [(0, start, width, end) for start, end in windows]
                    else:
                        windows = self._axis_windows(
                            width,
                            self.multi_pass_splits,
                            self.multi_pass_overlap,
                        )
                        rescue_regions = [(start, 0, end, height) for start, end in windows]

                    for left, top, right, bottom in rescue_regions:
                        if left == 0 and top == 0 and right == width and bottom == height:
                            continue
                        crop = source.crop((left, top, right, bottom))
                        crop_width, crop_height = crop.size
                        rescue_results = loaded.model.predict(
                            source=crop,
                            conf=self.multi_pass_confidence,
                            iou=self.iou_threshold,
                            imgsz=self.image_size,
                            max_det=raw_max_detections,
                            agnostic_nms=True,
                            device=self._device_argument(),
                            verbose=False,
                        )
                        rescue_xyxy, rescue_confidences = self._boxes_from_result(
                            rescue_results
                        )
                        detections.extend(
                            self._convert_boxes(
                                xyxy=rescue_xyxy,
                                confidences=rescue_confidences,
                                full_width=width,
                                full_height=height,
                                offset_x=left,
                                offset_y=top,
                                crop_width=crop_width,
                                crop_height=crop_height,
                                reject_internal_edges=True,
                            )
                        )

            actual_device = self._runtime_device(loaded.model, loaded.device)
            with self._lock:
                if self._loaded is loaded:
                    loaded.device = actual_device
        except Exception as exc:
            logger.exception("Waste detector inference failed")
            with self._lock:
                if self._loaded is loaded:
                    self._loaded = None
                self._load_error = f"Detector inference failed: {exc}"
                self._load_error_at = time.monotonic()
            raise DetectorUnavailableError(self._load_error) from exc

        if not detections:
            return []
        return self._merge_detections(detections)

    @property
    def status(self) -> dict[str, Any]:
        retry_in_seconds = 0.0
        if self._load_error and self._load_error_at is not None:
            retry_in_seconds = max(
                0.0,
                self.retry_seconds - (time.monotonic() - self._load_error_at),
            )

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
            "model_type": "yolo-class-agnostic-detector",
            "checkpoint": str(self.checkpoint_path),
            "device": loaded.device if loaded else self.device_preference,
            "class_names": loaded.class_names if loaded else {0: "waste_object"},
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "image_size": self.image_size,
            "max_detections": self.max_detections,
            "min_box_area_ratio": self.min_box_area_ratio,
            "multi_pass_enabled": self.multi_pass_enabled,
            "multi_pass_trigger_count": self.multi_pass_trigger_count,
            "multi_pass_confidence": self.multi_pass_confidence,
            "multi_pass_splits": self.multi_pass_splits,
            "multi_pass_overlap": self.multi_pass_overlap,
            "multi_pass_edge_margin": self.multi_pass_edge_margin,
            "merge_iou_threshold": self.merge_iou_threshold,
            "merge_containment_threshold": self.merge_containment_threshold,
            "crop_padding": self.crop_padding,
            "error": self._load_error,
            "retry_in_seconds": round(retry_in_seconds, 1),
            "model_auto_reload": self.auto_reload,
            "model_reload_check_seconds": self.reload_check_seconds,
        }
