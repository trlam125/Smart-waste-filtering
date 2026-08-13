from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from PIL import Image

from .waste_rules import ALL_PROMPTS, PROMPT_TO_KEY, RULE_BY_KEY

logger = logging.getLogger(__name__)


# Stage 1 is intentionally shape/object heavy. It answers a different question
# from the normal material prompts: "what kind of object is in the focus area?".
# The prompts still map to the final waste keys, but they emphasize geometry such
# as a bottle neck/cap, can pull tab, box folds, circuit board, etc. This makes a
# printed shrink sleeve much less likely to turn a plastic bottle into "paper".
OBJECT_SHAPE_PROMPTS_BY_KEY: dict[str, tuple[str, ...]] = {
    "plastic": (
        "an upright rigid plastic beverage bottle with rounded shoulders and a screw cap",
        "an opaque white plastic drink bottle wrapped in a printed shrink sleeve",
        "a clear PET water bottle with a plastic cap and narrow bottle neck",
        "a molded rigid plastic cup food tub or plastic container",
        "a hard plastic household bottle or container that keeps its shape",
    ),
    "nylon": (
        "a crumpled thin plastic shopping bag with handles",
        "a transparent flexible plastic bag lying flat with wrinkles",
        "a soft plastic pouch or sachet with heat sealed flexible edges",
        "a snack wrapper or flexible plastic packaging packet",
        "a loose sheet of plastic film cling wrap or stretch wrap",
    ),
    "paper": (
        "a rectangular cardboard box with straight folded edges and flat panels",
        "a flat sheet of paper newspaper or magazine page",
        "a brown kraft paper bag with folded paper sides",
        "a corrugated cardboard sheet with visible fold lines",
        "a rectangular paperboard carton with folded top seams",
    ),
    "metal": (
        "an aluminum beverage can with a metal pull tab",
        "a cylindrical steel food can with a metal rim",
        "a shallow aluminum food tray with rigid metallic edges",
        "a small rigid household object made of bare metal",
        "an empty metal tin can prepared for recycling",
    ),
    "glass": (
        "a transparent glass bottle with a narrow neck and rigid glass body",
        "a glass food jar with a thick transparent rim",
        "a colored wine or beverage bottle made of glass",
        "a clear rigid glass container with reflective highlights",
        "a household bottle or jar visibly made from glass",
    ),
    "organic": (
        "a banana peel or fruit peel discarded as food waste",
        "loose vegetable scraps from food preparation",
        "leftover cooked food without its packaging",
        "spoiled fruit or vegetables ready for composting",
        "a pile of biodegradable kitchen food scraps",
    ),
    "hazardous": (
        "a loose household battery requiring special waste collection",
        "a compact fluorescent lamp or fluorescent tube for special disposal",
        "a household chemical container with hazard warning symbols",
        "a pesticide or solvent container requiring hazardous waste collection",
        "a paint or toxic chemical container for special disposal",
    ),
    "electronic": (
        "a discarded mobile phone or handheld electronic device",
        "a charger power adapter or electronic cable with connectors",
        "a printed circuit board with electronic components",
        "a broken computer accessory or small electrical appliance",
        "a broken keyboard computer mouse or small electronic accessory for e-waste",
    ),
}
OBJECT_PROMPT_TO_KEY = {
    prompt: key
    for key, prompts in OBJECT_SHAPE_PROMPTS_BY_KEY.items()
    for prompt in prompts
}
ALL_OBJECT_PROMPTS: tuple[str, ...] = tuple(OBJECT_PROMPT_TO_KEY)


def _finite_float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric, got: {raw_value!r}") from exc

    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite, got: {raw_value!r}")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got: {value}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{name} must be <= {maximum}, got: {value}")
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got: {raw_value!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got: {value}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "true" if default else "false").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false, got: {raw_value!r}")


DEFAULT_WASTE_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_EMBEDDING_DIMENSION = 512


class ModelUnavailableError(RuntimeError):
    """Raised when the AI model cannot be loaded or executed."""


def _model_embedding_kind(base_kind: str, model_name: str) -> str:
    """Namespace learned vectors by representation schema and model identity.

    Embeddings produced by different CLIP checkpoints are not comparable even
    when they happen to have the same dimension. Keep the key short because the
    database intentionally caps ``embedding_kind`` at 32 characters.
    """
    model_digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    return f"{base_kind}:{model_digest}"


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


@dataclass(frozen=True)
class _FocusCandidate:
    name: str
    image: Image.Image
    rotation_degrees: int


@dataclass(frozen=True)
class _ScoredFocusCandidate:
    view: _FocusCandidate
    scores: dict[str, float]
    quality: float


class WasteClassifier:
    """Two-stage CLIP classifier with orientation and object-shape evidence.

    Stage 1 scores focused crops using shape/object prompts and simultaneously
    chooses the most plausible orientation. Stage 2 scores material/category
    prompts on the selected focus view plus a normalized full frame and a second
    crop. The two stages are fused, with targeted guards for paper versus hard/soft plastic and for hard versus soft plastic.

    The broad "other/general waste" class never competes directly in CLIP. It is
    only used when no usable score exists; low-confidence predictions are exposed
    to the UI as "unknown" through the ``uncertain`` flag.
    """

    def __init__(self) -> None:
        self.model_name = os.getenv("WASTE_MODEL", DEFAULT_WASTE_MODEL)
        self.unknown_threshold = _finite_float_env(
            "UNKNOWN_THRESHOLD", 0.24, minimum=0.0, maximum=1.0
        )
        self.uncertainty_margin = _finite_float_env(
            "UNCERTAINTY_MARGIN", 0.05, minimum=0.0, maximum=1.0
        )
        self.stage_disagreement_margin = _finite_float_env(
            "STAGE_DISAGREEMENT_MARGIN", 0.08, minimum=0.0, maximum=1.0
        )
        self.retry_seconds = _finite_float_env(
            "MODEL_RETRY_SECONDS", 30.0, minimum=0.0
        )

        self.center_crop_enabled = _bool_env("CENTER_CROP_ENABLED", True)
        self.center_crop_width_ratio = _finite_float_env(
            "CENTER_CROP_WIDTH_RATIO", 0.72, minimum=0.35, maximum=1.0
        )
        self.center_crop_height_ratio = _finite_float_env(
            "CENTER_CROP_HEIGHT_RATIO", 0.86, minimum=0.35, maximum=1.0
        )
        self.wide_crop_enabled = _bool_env("WIDE_CROP_ENABLED", True)
        self.wide_crop_width_ratio = _finite_float_env(
            "WIDE_CROP_WIDTH_RATIO", 0.96, minimum=0.50, maximum=1.0
        )
        self.wide_crop_height_ratio = _finite_float_env(
            "WIDE_CROP_HEIGHT_RATIO", 0.62, minimum=0.30, maximum=0.95
        )
        self.secondary_crop_width_ratio = _finite_float_env(
            "SECONDARY_CROP_WIDTH_RATIO", 0.88, minimum=0.50, maximum=1.0
        )
        self.secondary_crop_height_ratio = _finite_float_env(
            "SECONDARY_CROP_HEIGHT_RATIO", 0.74, minimum=0.40, maximum=1.0
        )

        self.rotation_ensemble_enabled = _bool_env("ROTATION_ENSEMBLE_ENABLED", True)
        self.object_view_top_k = _positive_int_env("OBJECT_VIEW_TOP_K", 2)
        self.object_prompt_top_k = _positive_int_env("OBJECT_PROMPT_TOP_K", 2)
        self.prompt_top_k = _positive_int_env("PROMPT_SCORE_TOP_K", 2)

        self.object_prior_weight = _finite_float_env(
            "OBJECT_PRIOR_WEIGHT", 0.55, minimum=0.0, maximum=1.0
        )
        self.material_primary_weight = _finite_float_env(
            "MATERIAL_PRIMARY_WEIGHT", 0.60, minimum=0.0
        )
        self.material_secondary_weight = _finite_float_env(
            "MATERIAL_SECONDARY_WEIGHT", 0.25, minimum=0.0
        )
        self.material_full_weight = _finite_float_env(
            "MATERIAL_FULL_WEIGHT", 0.15, minimum=0.0
        )

        self.paper_plastic_guard_enabled = _bool_env(
            "PAPER_PLASTIC_GUARD_ENABLED", True
        )
        self.paper_plastic_object_weight = _finite_float_env(
            "PAPER_PLASTIC_OBJECT_WEIGHT", 0.72, minimum=0.0, maximum=1.0
        )
        self.paper_plastic_object_margin = _finite_float_env(
            "PAPER_PLASTIC_OBJECT_MARGIN", 0.08, minimum=0.0, maximum=1.0
        )

        self._pipeline: Any | None = None
        self._loading = False
        self._load_error: str | None = None
        self._load_error_at: float | None = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def status(self) -> dict[str, Any]:
        retry_in_seconds = 0.0
        if self._load_error and self._load_error_at is not None:
            elapsed = time.monotonic() - self._load_error_at
            retry_in_seconds = max(0.0, self.retry_seconds - elapsed)

        if self._pipeline is not None:
            state = "ready"
        elif self._loading:
            state = "loading"
        elif self._load_error and retry_in_seconds > 0:
            state = "error"
        elif self._load_error:
            state = "retry_available"
        else:
            state = "not_loaded"

        return {
            "state": state,
            "model": self.model_name,
            "error": self._load_error,
            "retry_in_seconds": round(retry_in_seconds, 1),
            "unknown_threshold": self.unknown_threshold,
            "uncertainty_margin": self.uncertainty_margin,
            "stage_disagreement_margin": self.stage_disagreement_margin,
            "direct_categories": len({PROMPT_TO_KEY[prompt] for prompt in ALL_PROMPTS}),
            "prompt_count": len(ALL_PROMPTS),
            "material_prompt_count": len(ALL_PROMPTS),
            "object_prompt_count": len(ALL_OBJECT_PROMPTS),
            "center_crop_enabled": self.center_crop_enabled,
            "center_crop_width_ratio": self.center_crop_width_ratio,
            "center_crop_height_ratio": self.center_crop_height_ratio,
            "wide_crop_enabled": self.wide_crop_enabled,
            "wide_crop_width_ratio": self.wide_crop_width_ratio,
            "wide_crop_height_ratio": self.wide_crop_height_ratio,
            "secondary_crop_width_ratio": self.secondary_crop_width_ratio,
            "secondary_crop_height_ratio": self.secondary_crop_height_ratio,
            "rotation_ensemble_enabled": self.rotation_ensemble_enabled,
            "object_view_top_k": self.object_view_top_k,
            "object_prompt_top_k": self.object_prompt_top_k,
            "prompt_score_top_k": self.prompt_top_k,
            "object_prior_weight": self.object_prior_weight,
            "material_primary_weight": self.material_primary_weight,
            "material_secondary_weight": self.material_secondary_weight,
            "material_full_weight": self.material_full_weight,
            "paper_plastic_guard_enabled": self.paper_plastic_guard_enabled,
            "paper_plastic_object_weight": self.paper_plastic_object_weight,
            "paper_plastic_object_margin": self.paper_plastic_object_margin,
        }

    def _retry_allowed(self) -> bool:
        if self._load_error is None or self._load_error_at is None:
            return True
        return (time.monotonic() - self._load_error_at) >= self.retry_seconds

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if self._load_error is not None and not self._retry_allowed():
            raise ModelUnavailableError(self._load_error)

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            if self._load_error is not None and not self._retry_allowed():
                raise ModelUnavailableError(self._load_error)

            self._load_error = None
            self._load_error_at = None
            self._loading = True
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    task="zero-shot-image-classification",
                    model=self.model_name,
                )
                logger.info("Loaded waste classification model: %s", self.model_name)
                return self._pipeline
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                self._load_error = (
                    "Could not load the AI model. Check Internet access on the first run "
                    "and verify torch/transformers are installed. Details: " + str(exc)
                )
                self._load_error_at = time.monotonic()
                logger.exception("Could not load AI model")
                raise ModelUnavailableError(self._load_error) from exc
            finally:
                self._loading = False

    def warmup(self) -> None:
        self._load()

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for key, value in scores.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric) and numeric >= 0.0:
                clean[key] = numeric
        total = sum(clean.values())
        if total <= 0.0:
            return {}
        return {key: value / total for key, value in clean.items()}

    @staticmethod
    def _rank(scores: dict[str, float]) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def _crop_center(
        image: Image.Image,
        width_ratio: float,
        height_ratio: float,
    ) -> Image.Image | None:
        width, height = image.size
        crop_width = max(1, round(width * width_ratio))
        crop_height = max(1, round(height * height_ratio))
        if crop_width >= width and crop_height >= height:
            return None

        left = max(0, (width - crop_width) // 2)
        top = max(0, (height - crop_height) // 2)
        right = min(width, left + crop_width)
        bottom = min(height, top + crop_height)
        if right - left < 2 or bottom - top < 2:
            return None
        return image.crop((left, top, right, bottom))

    def _category_scores(
        self,
        raw_predictions: list[dict[str, Any]],
        *,
        prompt_to_key: dict[str, str],
        prompt_top_k: int,
    ) -> dict[str, float]:
        scores_by_key: dict[str, list[float]] = defaultdict(list)
        for prediction in raw_predictions:
            prompt = str(prediction.get("label", ""))
            key = prompt_to_key.get(prompt)
            if not key:
                continue
            try:
                score = float(prediction["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(score) and score >= 0.0:
                scores_by_key[key].append(score)

        category_scores: dict[str, float] = {}
        for key, values in scores_by_key.items():
            strongest = sorted(values, reverse=True)[: max(1, prompt_top_k)]
            category_scores[key] = sum(strongest) / len(strongest)
        return self._normalize_scores(category_scores)

    def _run_view(
        self,
        classifier: Any,
        image: Image.Image,
        *,
        candidate_labels: Iterable[str],
        prompt_to_key: dict[str, str],
        prompt_top_k: int,
    ) -> dict[str, float]:
        raw_predictions = classifier(image, candidate_labels=list(candidate_labels))
        if not isinstance(raw_predictions, list):
            return {}
        return self._category_scores(
            raw_predictions,
            prompt_to_key=prompt_to_key,
            prompt_top_k=prompt_top_k,
        )

    def _run_material_view(self, classifier: Any, image: Image.Image) -> dict[str, float]:
        return self._run_view(
            classifier,
            image,
            candidate_labels=ALL_PROMPTS,
            prompt_to_key=PROMPT_TO_KEY,
            prompt_top_k=self.prompt_top_k,
        )

    def _run_object_view(self, classifier: Any, image: Image.Image) -> dict[str, float]:
        return self._run_view(
            classifier,
            image,
            candidate_labels=ALL_OBJECT_PROMPTS,
            prompt_to_key=OBJECT_PROMPT_TO_KEY,
            prompt_top_k=self.object_prompt_top_k,
        )

    def _weighted_merge(
        self,
        score_maps: Iterable[tuple[dict[str, float], float]],
    ) -> dict[str, float]:
        available = [
            (scores, weight)
            for scores, weight in score_maps
            if scores and weight > 0.0 and math.isfinite(weight)
        ]
        if not available:
            return {}

        total_weight = sum(weight for _, weight in available)
        if total_weight <= 0.0:
            return {}

        combined: dict[str, float] = defaultdict(float)
        for scores, weight in available:
            normalized_weight = weight / total_weight
            for key, score in scores.items():
                combined[key] += normalized_weight * score
        return self._normalize_scores(dict(combined))

    @staticmethod
    def _view_quality(scores: dict[str, float]) -> float:
        ranked = sorted(scores.values(), reverse=True)
        if not ranked:
            return 0.0
        top = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else 0.0
        # A concentrated object-stage distribution is more useful for selecting
        # orientation than a nearly uniform one.
        return top + 0.35 * max(0.0, top - runner_up)

    def _build_focus_candidates(self, image: Image.Image) -> list[_FocusCandidate]:
        portrait = (
            self._crop_center(
                image,
                self.center_crop_width_ratio,
                self.center_crop_height_ratio,
            )
            if self.center_crop_enabled
            else None
        ) or image
        wide = (
            self._crop_center(
                image,
                self.wide_crop_width_ratio,
                self.wide_crop_height_ratio,
            )
            if self.wide_crop_enabled
            else None
        ) or image

        candidates = [
            _FocusCandidate("portrait-0", portrait, 0),
            _FocusCandidate("wide-0", wide, 0),
        ]
        if self.rotation_ensemble_enabled:
            candidates.extend(
                [
                    _FocusCandidate("portrait-180", portrait.rotate(180, expand=True), 180),
                    _FocusCandidate("wide-90", wide.rotate(90, expand=True), 90),
                    _FocusCandidate("wide-270", wide.rotate(270, expand=True), 270),
                ]
            )
        return candidates

    def _score_focus_candidates(
        self,
        classifier: Any,
        image: Image.Image,
    ) -> list[_ScoredFocusCandidate]:
        scored: list[_ScoredFocusCandidate] = []
        for view in self._build_focus_candidates(image):
            scores = self._run_object_view(classifier, view.image)
            scored.append(
                _ScoredFocusCandidate(
                    view=view,
                    scores=scores,
                    quality=self._view_quality(scores),
                )
            )
        return sorted(scored, key=lambda item: item.quality, reverse=True)

    def _pool_object_evidence(
        self,
        candidates: list[_ScoredFocusCandidate],
    ) -> dict[str, float]:
        usable = [candidate for candidate in candidates if candidate.scores]
        if not usable:
            return {}
        selected = usable[: min(self.object_view_top_k, len(usable))]
        weighted = []
        for candidate in selected:
            # Keep a tiny positive weight for valid but flat distributions.
            weighted.append((candidate.scores, max(candidate.quality, 1e-6)))
        return self._weighted_merge(weighted)

    def _material_evidence(
        self,
        classifier: Any,
        original_image: Image.Image,
        best_focus: _FocusCandidate,
    ) -> dict[str, float]:
        angle = best_focus.rotation_degrees
        normalized_full = (
            original_image.rotate(angle, expand=True) if angle else original_image
        )
        secondary = self._crop_center(
            normalized_full,
            self.secondary_crop_width_ratio,
            self.secondary_crop_height_ratio,
        )

        primary_scores = self._run_material_view(classifier, best_focus.image)
        secondary_scores = (
            self._run_material_view(classifier, secondary) if secondary is not None else {}
        )
        full_scores = self._run_material_view(classifier, normalized_full)

        return self._weighted_merge(
            [
                (primary_scores, self.material_primary_weight),
                (secondary_scores, self.material_secondary_weight),
                (full_scores, self.material_full_weight),
            ]
        )

    def _fuse_stages(
        self,
        material_scores: dict[str, float],
        object_scores: dict[str, float],
    ) -> dict[str, float]:
        if material_scores and object_scores:
            return self._weighted_merge(
                [
                    (material_scores, 1.0 - self.object_prior_weight),
                    (object_scores, self.object_prior_weight),
                ]
            )
        return material_scores or object_scores

    @staticmethod
    def _score_fingerprint(
        combined: dict[str, float],
        object_scores: dict[str, float],
        material_scores: dict[str, float],
    ) -> tuple[float, ...] | None:
        keys = tuple(OBJECT_SHAPE_PROMPTS_BY_KEY)
        values = [
            float(scores.get(key, 0.0))
            for scores in (combined, object_scores, material_scores)
            for key in keys
        ]
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0.0:
            return None
        return tuple(value / norm for value in values)

    @staticmethod
    def _extract_clip_embedding(
        classifier: Any,
        image: Image.Image,
    ) -> tuple[float, ...] | None:
        """Return a normalized CLIP image vector when the pipeline exposes it.

        Classification must still work if a future Transformers pipeline changes
        its internals, so embedding extraction is intentionally best-effort.
        """
        try:
            model = getattr(classifier, "model", None)
            image_processor = getattr(classifier, "image_processor", None)
            if image_processor is None:
                image_processor = getattr(classifier, "feature_extractor", None)
            if model is None or image_processor is None or not hasattr(model, "get_image_features"):
                return None

            import torch

            processed = image_processor(images=image, return_tensors="pt")
            pixel_values = processed.get("pixel_values")
            if pixel_values is None:
                return None
            device = getattr(model, "device", None)
            if device is not None:
                pixel_values = pixel_values.to(device)
            with torch.no_grad():
                features = model.get_image_features(pixel_values=pixel_values)
            if features is None or len(features) == 0:
                return None
            vector = features[0].detach().float().cpu().tolist()
            clean = [float(value) for value in vector if math.isfinite(float(value))]
            if len(clean) != len(vector):
                return None
            norm = math.sqrt(sum(value * value for value in clean))
            if norm <= 0.0:
                return None
            return tuple(value / norm for value in clean)
        except Exception as exc:  # pragma: no cover - depends on transformers backend
            logger.warning("Could not extract CLIP feedback embedding: %s", exc)
            return None

    def _apply_pair_object_guard(
        self,
        combined: dict[str, float],
        material_scores: dict[str, float],
        object_scores: dict[str, float],
        left_key: str,
        right_key: str,
        *,
        object_weight: float | None = None,
        object_margin: float | None = None,
    ) -> dict[str, float]:
        """Use shape evidence only when two easily-confused classes are top competitors.

        The guard never creates a global bias toward either class. It only
        redistributes the probability mass already assigned to the pair.
        """
        if not combined or not object_scores:
            return combined

        combined_top2 = {key for key, _ in self._rank(combined)[:2]}
        material_top2 = {key for key, _ in self._rank(material_scores)[:2]} if material_scores else set()
        pair = {left_key, right_key}
        if not (pair.issubset(combined_top2) or pair.issubset(material_top2)):
            return combined

        left_object = object_scores.get(left_key, 0.0)
        right_object = object_scores.get(right_key, 0.0)
        margin = self.paper_plastic_object_margin if object_margin is None else object_margin
        if abs(left_object - right_object) < margin:
            return combined

        pair_total = combined.get(left_key, 0.0) + combined.get(right_key, 0.0)
        object_pair_total = left_object + right_object
        if pair_total <= 0.0 or object_pair_total <= 0.0:
            return combined

        base_left_ratio = combined.get(left_key, 0.0) / pair_total
        object_left_ratio = left_object / object_pair_total
        weight = self.paper_plastic_object_weight if object_weight is None else object_weight
        revised_left_ratio = (1.0 - weight) * base_left_ratio + weight * object_left_ratio

        revised = dict(combined)
        revised[left_key] = pair_total * revised_left_ratio
        revised[right_key] = pair_total * (1.0 - revised_left_ratio)
        return self._normalize_scores(revised)

    def _apply_paper_plastic_guard(
        self,
        combined: dict[str, float],
        material_scores: dict[str, float],
        object_scores: dict[str, float],
    ) -> dict[str, float]:
        if not self.paper_plastic_guard_enabled:
            return combined

        revised = combined
        # Printed labels can make bottles and plastic film look paper-like. Use
        # object geometry to arbitrate only when Paper is actually a top-two
        # competitor with the relevant plastic form.
        revised = self._apply_pair_object_guard(
            revised, material_scores, object_scores, "paper", "plastic"
        )
        revised = self._apply_pair_object_guard(
            revised, material_scores, object_scores, "paper", "nylon"
        )
        # Hard and flexible plastics share material semantics, so the material
        # stage may be nearly tied. Shape evidence (rigid bottle/container vs
        # wrinkled bag/film/pouch) is the more useful discriminator here.
        revised = self._apply_pair_object_guard(
            revised,
            material_scores,
            object_scores,
            "plastic",
            "nylon",
            object_weight=max(self.paper_plastic_object_weight, 0.78),
            object_margin=min(self.paper_plastic_object_margin, 0.06),
        )
        return revised

    @staticmethod
    def _inference_failure_requires_reload(exc: Exception) -> bool:
        """Return True only for failures that strongly indicate a broken backend/device."""
        if isinstance(exc, MemoryError):
            return True
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc).casefold()
        fatal_markers = (
            "cuda",
            "cudnn",
            "cublas",
            "device-side assert",
            "out of memory",
            "mps backend",
            "xpu",
            "hip error",
            "accelerator",
            "expected all tensors to be on the same device",
        )
        return any(marker in message for marker in fatal_markers)

    def learning_embedding_kinds(self) -> tuple[str, str]:
        """Return representation kinds that this configured model can consume."""
        return (
            _model_embedding_kind("clip-v1", self.model_name),
            _model_embedding_kind("score-v2", self.model_name),
        )

    def legacy_embedding_migrations(self) -> tuple[tuple[str, str, int], ...]:
        """Describe legacy embeddings that can be upgraded without guessing.

        v1.8-v1.9 stored default-model CLIP vectors under the generic ``clip``
        kind.  That name carried no model identity, so it is only safe to
        migrate automatically when the app still uses the historical default
        checkpoint and the expected 512-dimensional representation.
        """
        if self.model_name != DEFAULT_WASTE_MODEL:
            return ()
        return ((
            "clip",
            _model_embedding_kind("clip-v1", self.model_name),
            DEFAULT_CLIP_EMBEDDING_DIMENSION,
        ),)

    def classify(self, image: Image.Image) -> ClassificationResult:
        image = image.convert("RGB")
        combined: dict[str, float] = {}
        object_scores: dict[str, float] = {}
        material_scores: dict[str, float] = {}
        best_focus: _FocusCandidate | None = None
        embedding: tuple[float, ...] | None = None
        embedding_kind: str | None = None

        with self._inference_lock:
            classifier = self._load()
            try:
                scored_candidates = self._score_focus_candidates(classifier, image)
                if scored_candidates:
                    best_focus = scored_candidates[0].view
                    object_scores = self._pool_object_evidence(scored_candidates)
                    material_scores = self._material_evidence(
                        classifier,
                        image,
                        best_focus,
                    )
                    combined = self._fuse_stages(material_scores, object_scores)
                    combined = self._apply_paper_plastic_guard(
                        combined,
                        material_scores,
                        object_scores,
                    )
                    embedding = self._extract_clip_embedding(classifier, best_focus.image)
                    if embedding is not None:
                        embedding_kind = _model_embedding_kind("clip-v1", self.model_name)
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                error_message = f"AI model could not process the image: {exc}"
                requires_reload = self._inference_failure_requires_reload(exc)
                if requires_reload:
                    with self._lock:
                        self._pipeline = None
                        self._load_error = error_message
                        self._load_error_at = time.monotonic()
                    logger.exception(
                        "Waste model inference failed; backend/device will be reloaded after retry delay"
                    )
                else:
                    # Keep the already-loaded model available. A malformed/request-specific
                    # failure must not block every user for MODEL_RETRY_SECONDS.
                    logger.exception(
                        "Waste model inference failed for this request; keeping loaded pipeline"
                    )
                raise ModelUnavailableError(error_message) from exc

        if embedding is None:
            embedding = self._score_fingerprint(combined, object_scores, material_scores)
            if embedding is not None:
                embedding_kind = _model_embedding_kind("score-v2", self.model_name)

        if not combined:
            return ClassificationResult(
                key="other",
                confidence=0.0,
                alternatives=[],
                uncertain=True,
                analysis={
                    "orientation_degrees": 0,
                    "object_stage": None,
                    "material_stage": None,
                },
                score_map={},
                embedding=embedding,
                embedding_kind=embedding_kind,
            )

        ranked = self._rank(combined)
        best_key, confidence = ranked[0]
        runner_up_confidence = ranked[1][1] if len(ranked) > 1 else 0.0
        score_margin = confidence - runner_up_confidence

        object_ranked = self._rank(object_scores)
        material_ranked = self._rank(material_scores)
        object_top = object_ranked[0] if object_ranked else None
        material_top = material_ranked[0] if material_ranked else None
        stage_disagreement = bool(
            object_top
            and material_top
            and object_top[0] != material_top[0]
            and score_margin < self.stage_disagreement_margin
        )

        uncertain = (
            confidence < self.unknown_threshold
            or score_margin < self.uncertainty_margin
            or stage_disagreement
        )

        alternatives = [
            {
                "key": key,
                "display_name": RULE_BY_KEY[key].display_name,
                "confidence": round(score, 4),
            }
            for key, score in ranked[1:4]
        ]

        return ClassificationResult(
            key=best_key,
            confidence=round(confidence, 4),
            alternatives=alternatives,
            uncertain=uncertain,
            analysis={
                "orientation_degrees": best_focus.rotation_degrees if best_focus else 0,
                "focus_view": best_focus.name if best_focus else None,
                "object_stage": (
                    {
                        "key": object_top[0],
                        "display_name": RULE_BY_KEY[object_top[0]].display_name,
                        "confidence": round(object_top[1], 4),
                    }
                    if object_top
                    else None
                ),
                "material_stage": (
                    {
                        "key": material_top[0],
                        "display_name": RULE_BY_KEY[material_top[0]].display_name,
                        "confidence": round(material_top[1], 4),
                    }
                    if material_top
                    else None
                ),
                "stage_disagreement": stage_disagreement,
            },
            score_map=combined,
            embedding=embedding,
            embedding_kind=embedding_kind,
        )
