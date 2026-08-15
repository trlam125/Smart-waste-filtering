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
LEARNING_MAX_EXAMPLES = _int_env(
    "LEARNING_MAX_EXAMPLES", 500, len(LEARNABLE_RULE_KEYS), 5000
)
LEARNING_MIN_SIMILARITY = _float_env("LEARNING_MIN_SIMILARITY", 0.84, 0.0, 0.9999)
LEARNING_MAX_WEIGHT = _float_env("LEARNING_MAX_WEIGHT", 0.60, 0.0, 0.95)
LEARNING_SINGLE_EXAMPLE_FACTOR = _float_env(
    "LEARNING_SINGLE_EXAMPLE_FACTOR", 0.78, 0.0, 1.0
)
# A fused k-NN score is a ranking signal, not a calibrated probability.  Memory
# therefore needs independent evidence before it may turn an uncertain model
# prediction into a certain result (or confidently replace the model's top class).
LEARNING_CERTAINTY_MIN_EXAMPLES = _int_env(
    "LEARNING_CERTAINTY_MIN_EXAMPLES", 3, 2, 50
)
LEARNING_CERTAINTY_MIN_AGREEMENT = _float_env(
    "LEARNING_CERTAINTY_MIN_AGREEMENT", 0.75, 0.5, 1.0
)
LEARNING_CERTAINTY_MIN_SIMILARITY = _float_env(
    "LEARNING_CERTAINTY_MIN_SIMILARITY", 0.92, 0.0, 0.9999
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
    """Blend explicit user feedback into the supervised prediction using shared k-NN memory.

    Examples are compared with checkpoint-specific semantic feature vectors captured
    immediately before the model's final classifier layer. A single correction can
    influence a very similar future scan, while weaker/conflicting examples are blended
    conservatively rather than becoming hard rules.
    """
    if not LEARNING_ENABLED or not embedding or not embedding_kind or not examples:
        analysis = dict(result.analysis)
        analysis["learning_memory"] = {
            "enabled": LEARNING_ENABLED,
            "applied": False,
            "matched_examples": 0,
        }
        return replace(result, analysis=analysis)

    neighbors: list[tuple[float, dict[str, Any]]] = []
    for example in examples:
        if example.get("embedding_kind") != embedding_kind:
            continue
        corrected_key = str(example.get("corrected_key", ""))
        if corrected_key not in LEARNABLE_RULE_KEYS:
            continue
        vector = example.get("embedding")
        if not vector or len(vector) != len(embedding):
            continue
        similarity = _cosine(embedding, vector)
        if math.isfinite(similarity) and similarity >= LEARNING_MIN_SIMILARITY:
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
        proximity = max(
            0.0,
            min(
                1.0,
                (similarity - LEARNING_MIN_SIMILARITY) / (1.0 - LEARNING_MIN_SIMILARITY),
            ),
        )
        vote_weight = max(1e-6, proximity * proximity)
        votes[key] += vote_weight
        count_by_key[key] += 1

    memory_distribution = _normalize_scores(dict(votes))
    if not memory_distribution:
        return result

    memory_top = _rank(memory_distribution)[0]
    memory_key = memory_top[0]
    agreement_count = count_by_key.get(memory_key, 0)
    agreement_ratio = agreement_count / max(1, len(neighbors))

    # Only close examples that support the winning memory label may strengthen
    # the memory blend. Weak or conflicting neighbors must not increase its weight.
    certainty_neighbors = [
        (similarity, example)
        for similarity, example in neighbors
        if similarity >= LEARNING_CERTAINTY_MIN_SIMILARITY
    ]
    certainty_count_by_key: dict[str, int] = defaultdict(int)
    for _similarity, example in certainty_neighbors:
        key = str(example.get("corrected_key", ""))
        if key in LEARNABLE_RULE_KEYS:
            certainty_count_by_key[key] += 1
    certainty_agreement_count = certainty_count_by_key.get(memory_key, 0)
    certainty_agreement_ratio = certainty_agreement_count / max(1, len(certainty_neighbors))

    best_similarity = neighbors[0][0]
    proximity = max(
        0.0,
        min(
            1.0,
            (best_similarity - LEARNING_MIN_SIMILARITY) / (1.0 - LEARNING_MIN_SIMILARITY),
        ),
    )
    evidence_factor = LEARNING_SINGLE_EXAMPLE_FACTOR + (
        1.0 - LEARNING_SINGLE_EXAMPLE_FACTOR
    ) * min(1.0, max(0, certainty_agreement_count - 1) / 3.0)
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

    best_key, effective_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    fused_margin = effective_score - runner_up

    # Do not compare the fused score against UNKNOWN_THRESHOLD.  The model score
    # is calibrated; the blended model+k-NN score is not.  Memory may resolve an
    # uncertain prediction only when there are several close, agreeing examples.
    certainty_evidence = bool(
        certainty_agreement_count >= LEARNING_CERTAINTY_MIN_EXAMPLES
        and certainty_agreement_ratio >= LEARNING_CERTAINTY_MIN_AGREEMENT
        and fused_margin >= uncertainty_margin
        and best_key == memory_key
    )

    # Preserve the model's uncertainty unless memory has enough independent
    # evidence.  If memory changes a previously-certain class without that
    # evidence, mark the effective result uncertain rather than presenting the
    # ranking score as calibrated confidence.
    # The fused ranking must also satisfy the same top-1/top-2 margin rule.
    # Memory can narrow a previously-certain model prediction even when the
    # winning class does not change; in that case the effective result must be
    # marked uncertain instead of inheriting the model's old certainty.
    if fused_margin < uncertainty_margin:
        uncertain = True
    elif result.uncertain:
        uncertain = not certainty_evidence
    elif best_key != result.key:
        uncertain = not certainty_evidence
    else:
        uncertain = False

    alternatives = [
        {
            "key": key,
            "display_name": RULE_BY_KEY[key].display_name,
            "confidence": round(score, 4),
        }
        for key, score in ranked[1:4]
    ]

    analysis = dict(result.analysis)
    analysis["learning_memory"] = {
        "enabled": True,
        "applied": True,
        "embedding_kind": embedding_kind,
        "matched_examples": len(neighbors),
        "best_similarity": round(best_similarity, 4),
        "weight": round(memory_weight, 4),
        "suggested_key": memory_key,
        "suggested_display_name": RULE_BY_KEY[memory_key].display_name,
        "agreement_count": agreement_count,
        "agreement_ratio": round(agreement_ratio, 4),
        "certainty_matched_examples": len(certainty_neighbors),
        "certainty_agreement_count": certainty_agreement_count,
        "certainty_agreement_ratio": round(certainty_agreement_ratio, 4),
        "certainty_evidence": certainty_evidence,
        "effective_score": round(effective_score, 4),
        "fused_margin": round(fused_margin, 4),
    }

    return replace(
        result,
        key=best_key,
        confidence=round(effective_score, 4),
        alternatives=alternatives,
        uncertain=uncertain,
        analysis=analysis,
        score_map=fused_scores,
    )
