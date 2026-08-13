from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable

from .classifier import ClassificationResult
from .waste_rules import LEARNABLE_RULE_KEYS, RULE_BY_KEY


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false, got: {raw!r}")


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric, got: {raw!r}") from exc
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}, got: {raw!r}")
    return value


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got: {raw!r}") from exc
    if value < minimum or value > maximum:
        raise RuntimeError(f"{name} must be between {minimum} and {maximum}, got: {value}")
    return value


LEARNING_ENABLED = _bool_env("FEEDBACK_LEARNING_ENABLED", True)
LEARNING_TOP_K = _int_env("LEARNING_TOP_K", 5, 1, 50)
LEARNING_MAX_EXAMPLES = _int_env("LEARNING_MAX_EXAMPLES", 500, 1, 5000)
LEARNING_MIN_SIMILARITY = _float_env("LEARNING_MIN_SIMILARITY", 0.84, 0.0, 0.9999)
LEARNING_SCORE_MIN_SIMILARITY = _float_env(
    "LEARNING_SCORE_MIN_SIMILARITY", 0.96, 0.0, 0.9999
)
LEARNING_MAX_WEIGHT = _float_env("LEARNING_MAX_WEIGHT", 0.60, 0.0, 0.95)
LEARNING_SINGLE_EXAMPLE_FACTOR = _float_env(
    "LEARNING_SINGLE_EXAMPLE_FACTOR", 0.78, 0.0, 1.0
)
LEARNING_DISAGREEMENT_OVERRIDE_WEIGHT = _float_env(
    "LEARNING_DISAGREEMENT_OVERRIDE_WEIGHT", 0.20, 0.0, 0.95
)


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    total = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        a_value = float(a)
        b_value = float(b)
        total += a_value * b_value
        left_norm += a_value * a_value
        right_norm += b_value * b_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return total / math.sqrt(left_norm * right_norm)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    clean = {
        key: float(value)
        for key, value in scores.items()
        if key in RULE_BY_KEY and math.isfinite(float(value)) and float(value) >= 0.0
    }
    total = sum(clean.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in clean.items()}


def _rank(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def apply_feedback_memory(
    result: ClassificationResult,
    *,
    embedding: tuple[float, ...] | None,
    embedding_kind: str | None,
    examples: list[dict[str, Any]],
    unknown_threshold: float,
    uncertainty_margin: float,
) -> ClassificationResult:
    """Blend explicit user feedback into a CLIP prediction using shared k-NN memory.

    The memory is deliberately conservative: the caller supplies only examples with
    the same embedding representation, while examples may come from any device using
    this scanner database. A single correction can influence a very similar future
    scan; weaker or conflicting examples are down-weighted instead of becoming hard
    rules.
    """
    if not LEARNING_ENABLED or not embedding or not embedding_kind or not examples:
        analysis = dict(result.analysis)
        analysis["learning_memory"] = {
            "enabled": LEARNING_ENABLED,
            "applied": False,
            "matched_examples": 0,
        }
        return replace(result, analysis=analysis)

    minimum_similarity = (
        LEARNING_SCORE_MIN_SIMILARITY
        if embedding_kind.startswith("score-")
        else LEARNING_MIN_SIMILARITY
    )

    neighbors: list[tuple[float, dict[str, Any]]] = []
    for example in examples:
        if example.get("embedding_kind") != embedding_kind:
            continue
        # ``other`` is intentionally a fallback category, not a visual class.
        # Never let feedback for it enter the k-NN neighborhood: filtering only
        # after TOP-K selection would allow many ``other`` examples to crowd out
        # valid direct-class examples and disable learning for the current scan.
        corrected_key = str(example.get("corrected_key", ""))
        if corrected_key not in LEARNABLE_RULE_KEYS:
            continue
        vector = example.get("embedding")
        if not vector or len(vector) != len(embedding):
            continue
        similarity = _cosine(embedding, vector)
        if math.isfinite(similarity) and similarity >= minimum_similarity:
            neighbors.append((similarity, example))

    neighbors.sort(key=lambda item: item[0], reverse=True)
    neighbors = neighbors[:LEARNING_TOP_K]
    if not neighbors:
        analysis = dict(result.analysis)
        analysis["learning_memory"] = {
            "enabled": True,
            "applied": False,
            "matched_examples": 0,
            "embedding_kind": embedding_kind,
        }
        return replace(result, analysis=analysis)

    votes: dict[str, float] = defaultdict(float)
    count_by_key: dict[str, int] = defaultdict(int)
    for similarity, example in neighbors:
        key = str(example.get("corrected_key", ""))
        if key not in LEARNABLE_RULE_KEYS:
            continue
        # Similarities close to the threshold contribute little; near-identical
        # samples contribute strongly. Squaring sharpens that distinction.
        proximity = max(
            0.0,
            min(1.0, (similarity - minimum_similarity) / (1.0 - minimum_similarity)),
        )
        vote_weight = max(1e-6, proximity * proximity)
        votes[key] += vote_weight
        count_by_key[key] += 1

    memory_distribution = _normalize_scores(dict(votes))
    if not memory_distribution:
        return result

    best_similarity = neighbors[0][0]
    proximity = max(
        0.0,
        min(1.0, (best_similarity - minimum_similarity) / (1.0 - minimum_similarity)),
    )
    evidence_factor = LEARNING_SINGLE_EXAMPLE_FACTOR + (
        1.0 - LEARNING_SINGLE_EXAMPLE_FACTOR
    ) * min(1.0, max(0, len(neighbors) - 1) / 3.0)
    memory_weight = min(
        LEARNING_MAX_WEIGHT,
        LEARNING_MAX_WEIGHT * proximity * evidence_factor,
    )

    base_scores = _normalize_scores(dict(result.score_map))
    if not base_scores or memory_weight <= 0.0:
        return result

    fused: dict[str, float] = defaultdict(float)
    for key, score in base_scores.items():
        fused[key] += (1.0 - memory_weight) * score
    for key, score in memory_distribution.items():
        fused[key] += memory_weight * score
    fused_scores = _normalize_scores(dict(fused))
    ranked = _rank(fused_scores)
    if not ranked:
        return result

    best_key, confidence = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = confidence - runner_up

    # A disagreement between the object/material stages is an independent
    # uncertainty signal.  Do not accidentally erase it just because even a
    # tiny feedback-memory blend nudged confidence over the numeric thresholds.
    stage_disagreement = bool(result.analysis.get("stage_disagreement", False))
    disagreement_overridden = bool(
        stage_disagreement
        and memory_weight >= LEARNING_DISAGREEMENT_OVERRIDE_WEIGHT
    )
    uncertain = (
        confidence < unknown_threshold
        or margin < uncertainty_margin
        or (stage_disagreement and not disagreement_overridden)
    )

    alternatives = [
        {
            "key": key,
            "display_name": RULE_BY_KEY[key].display_name,
            "confidence": round(score, 4),
        }
        for key, score in ranked[1:4]
    ]

    memory_top = _rank(memory_distribution)[0]
    analysis = dict(result.analysis)
    analysis["learning_memory"] = {
        "enabled": True,
        "applied": True,
        "embedding_kind": embedding_kind,
        "matched_examples": len(neighbors),
        "best_similarity": round(best_similarity, 4),
        "weight": round(memory_weight, 4),
        "suggested_key": memory_top[0],
        "suggested_display_name": RULE_BY_KEY[memory_top[0]].display_name,
        "agreement_count": count_by_key.get(memory_top[0], 0),
        "stage_disagreement_preserved": bool(
            stage_disagreement and not disagreement_overridden
        ),
        "stage_disagreement_overridden": disagreement_overridden,
    }

    return replace(
        result,
        key=best_key,
        confidence=round(confidence, 4),
        alternatives=alternatives,
        uncertain=uncertain,
        analysis=analysis,
        score_map=fused_scores,
    )
