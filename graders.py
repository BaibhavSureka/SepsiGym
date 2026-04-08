from __future__ import annotations

from typing import Any

from tasks import TaskConfig


SCORE_EPS = 1e-3
SCORE_MARGIN = 1e-6


def _clamp(value: float, low: float = SCORE_EPS, high: float = 1.0 - SCORE_EPS) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = low
    if numeric_value <= low:
        return low + SCORE_MARGIN
    if numeric_value >= high:
        return high - SCORE_MARGIN
    return numeric_value


def _strict_score(value: float) -> float:
    return _clamp(value, SCORE_EPS, 1.0 - SCORE_EPS)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(float(weight) for weight in weights.values())
    if total <= 0:
        return weights
    return {metric_name: float(weight) / total for metric_name, weight in weights.items()}


def _format_metric(value: float) -> float:
    return float(f"{_clamp(value):.6f}")


def grade_episode(task: TaskConfig, metrics: dict[str, Any]) -> float:
    weights = _normalize_weights(task.score_weights)
    score = sum(weight * _clamp(metrics.get(metric_name, 0.0)) for metric_name, weight in weights.items())
    return float(f"{_strict_score(score):.6f}")


def summarize_episode(total_reward: float, state_history: list[dict[str, Any]], terminal_outcome: str) -> dict[str, Any]:
    step_count = max(len(state_history), 1)
    safety_violations = sum(1 for item in state_history if item.get("unsafe", False))
    lab_steps = [item for item in state_history if item.get("action_type") == "request_lab"]
    treatment_steps = [item for item in state_history if item.get("action_type") == "request_treatment"]
    early_window = state_history[: min(3, len(state_history))] or state_history

    detection = max((item.get("detection_credit", 0.0) for item in early_window), default=0.0)
    lab_workup = (
        sum(item.get("lab_score", 0.0) for item in lab_steps) / len(lab_steps)
        if lab_steps
        else 0.0
    )
    treatment = (
        sum(item.get("treatment_score", 0.0) for item in treatment_steps) / len(treatment_steps)
        if treatment_steps
        else 0.0
    )
    first_meaningful_step = next(
        (
            idx
            for idx, item in enumerate(state_history)
            if item.get("detection_credit", 0.0) > 0.0 or item.get("treatment_score", 0.0) > 0.0
        ),
        step_count,
    )
    timeliness = _clamp(1.0 - (first_meaningful_step / step_count))
    stability = _clamp(sum(item.get("stability_score", 0.0) for item in state_history) / step_count)
    safety = _clamp(1.0 - (safety_violations / step_count))
    outcome = _format_metric(1.0 if terminal_outcome == "survived" else 0.0)
    return {
        "steps": step_count,
        "avg_reward": _format_metric(total_reward / step_count),
        "detection": _format_metric(detection),
        "lab_workup": _format_metric(lab_workup),
        "treatment": _format_metric(treatment),
        "timeliness": _format_metric(timeliness),
        "stability": _format_metric(stability),
        "safety": _format_metric(safety),
        "safety_violation_rate": _format_metric(safety_violations / step_count),
        "outcome": outcome,
    }
