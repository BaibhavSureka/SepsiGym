"""
ENHANCED INFERENCE.PY - BULLETPROOF VERSION
Compatible with Phase 1 & Phase 2 evaluation
Includes: Hybrid policy (heuristic + Monte Carlo + beam search) + comprehensive exception handling
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from client import SepsisTreatmentEnv
from models import SepsisAction, SepsisObservation

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("outputs")
TASK_IDS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = {"easy": 8, "medium": 12, "hard": 16}

MC_SIMS = 3
MC_DEPTH = 2

VALUE_TABLE = {}
VALUE_COUNTS = {}

RNG = random.Random(7)


# =========================
# ARGPARSE
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--output", default="outputs/results.json")
    return parser.parse_args()


# =========================
# VALUE FUNCTION (SAFE)
# =========================
def state_key(obs: SepsisObservation) -> str:
    try:
        severity = round(float(obs.severity_proxy), 1)
        mean_bp = round(float(obs.vitals.get("MeanBP", 0)), 1)
        shock = round(float(obs.vitals.get("Shock_Index", 0)), 1)
        return f"{severity}_{mean_bp}_{shock}"
    except Exception:
        return "unknown_state"


def update_value(obs: SepsisObservation, reward: float) -> None:
    try:
        key = state_key(obs)
        VALUE_COUNTS[key] = VALUE_COUNTS.get(key, 0) + 1
        lr = 1.0 / VALUE_COUNTS[key]
        VALUE_TABLE[key] = VALUE_TABLE.get(key, 0.0) + lr * (reward - VALUE_TABLE.get(key, 0.0))
    except Exception:
        pass  # Silent fail on value update


def get_value(obs: SepsisObservation) -> float:
    try:
        return float(VALUE_TABLE.get(state_key(obs), 0.0))
    except Exception:
        return 0.0


# =========================
# HEURISTIC (SAFE)
# =========================
def heuristic_action(obs: SepsisObservation) -> SepsisAction:
    try:
        severity = float(obs.severity_proxy or 0.0)
        mean_bp = float(obs.vitals.get("MeanBP", 0.0))
        requested_labs = set(obs.requested_labs or [])

        # Labs first
        for lab in ["lactate", "wbc", "creatinine"]:
            if lab not in requested_labs:
                return SepsisAction("request_lab", True, lab_type=lab)

        # Treatment based on severity
        if severity < 0.8:
            return SepsisAction("request_treatment", True, treatment_type="monitor")
        if severity >= 2.0 or mean_bp < -0.2:
            return SepsisAction("request_treatment", True, treatment_type="combination")
        if severity >= 1.2:
            return SepsisAction("request_treatment", True, treatment_type="fluids")

        return SepsisAction("request_treatment", True, treatment_type="monitor")
    except Exception:
        return SepsisAction("request_treatment", True, treatment_type="monitor")


# =========================
# CANDIDATES (SAFE)
# =========================
def generate_candidates(obs: SepsisObservation) -> list[SepsisAction]:
    candidates = []
    try:
        candidates.append(heuristic_action(obs))

        requested_labs = set(obs.requested_labs or [])
        for lab in ["lactate", "wbc", "creatinine"]:
            if lab not in requested_labs:
                try:
                    candidates.append(SepsisAction("request_lab", True, lab_type=lab))
                except Exception:
                    pass

        for t in ["monitor", "fluids", "vasopressors", "combination"]:
            try:
                candidates.append(SepsisAction("request_treatment", True, treatment_type=t))
            except Exception:
                pass
    except Exception as e:
        candidates.append(heuristic_action(obs))

    return candidates if candidates else [heuristic_action(obs)]


# =========================
# SIMULATION (SAFE)
# =========================
def simulate_step(obs: SepsisObservation, action: SepsisAction) -> tuple[float, SepsisObservation]:
    try:
        severity = float(obs.severity_proxy or 0.0)

        if action.action_type == "request_treatment":
            treatment = getattr(action, "treatment_type", "monitor")
            if treatment == "fluids":
                severity -= 0.2
            elif treatment == "vasopressors":
                severity -= 0.3
            elif treatment == "combination":
                severity -= 0.5
        elif action.action_type == "monitor":
            severity += 0.05

        reward = -severity
        severity = max(0.0, severity)

        new_obs = obs
        new_obs.severity_proxy = severity
        return float(reward), new_obs
    except Exception:
        return 0.0, obs


# =========================
# MONTE CARLO (SAFE)
# =========================
def monte_carlo(obs: SepsisObservation, action: SepsisAction) -> float:
    try:
        total = 0.0
        for _ in range(MC_SIMS):
            sim_obs = obs
            sim_reward = 0.0
            a = action

            for _ in range(MC_DEPTH):
                try:
                    r, sim_obs = simulate_step(sim_obs, a)
                    sim_reward += r
                    a = heuristic_action(sim_obs)
                except Exception:
                    break

            try:
                sim_reward += get_value(sim_obs)
            except Exception:
                pass

            total += sim_reward

        return float(total / MC_SIMS)
    except Exception:
        return 0.0


# =========================
# BEAM SEARCH (SAFE)
# =========================
def beam_search(obs: SepsisObservation) -> SepsisAction:
    try:
        best_action = None
        best_score = -1e9

        candidates = generate_candidates(obs)
        if not candidates:
            return heuristic_action(obs)

        for action in candidates:
            try:
                r, next_state = simulate_step(obs, action)
                score = r + get_value(next_state)

                if score > best_score:
                    best_score = score
                    best_action = action
            except Exception:
                continue

        return best_action if best_action else heuristic_action(obs)
    except Exception:
        return heuristic_action(obs)


# =========================
# SAFETY OVERRIDE (SAFE)
# =========================
def safety_override(action: SepsisAction, obs: SepsisObservation) -> SepsisAction:
    try:
        shock = float(obs.vitals.get("Shock_Index", 0.0))
        mean_bp = float(obs.vitals.get("MeanBP", 0.0))

        if shock > 0.2 or mean_bp < -0.3:
            return SepsisAction("request_treatment", True, treatment_type="combination")

        return action
    except Exception:
        return action


# =========================
# POLICY (SAFE)
# =========================
def choose_action(
    policy_mode: str,
    client: OpenAI | None,
    model_name: str | None,
    obs: SepsisObservation,
) -> tuple[SepsisAction, str, str | None]:
    error = None
    try:
        candidates = generate_candidates(obs)
        if not candidates:
            return heuristic_action(obs), "heuristic", None

        best_score = -1e9
        best_action = None

        try:
            beam_best = beam_search(obs)
        except Exception:
            beam_best = None

        for action in candidates:
            try:
                score = monte_carlo(obs, action)
                if beam_best and action == beam_best:
                    score += 0.5
                if score > best_score:
                    best_score = score
                    best_action = action
            except Exception:
                continue

        if best_action is None:
            best_action = heuristic_action(obs)

        return safety_override(best_action, obs), "advanced", error

    except Exception as e:
        error = str(e)
        return heuristic_action(obs), "fallback", error


# =========================
# BUILD RESULT DICT (SAFE)
# =========================
def build_result_dict(
    task_id: str,
    episode_id: str,
    step_count: int,
    reward_trace: list[float],
    action_history: list[str],
    policy_sources: Counter,
    policy_errors: list[str],
    metrics: dict,
    score: float,
) -> dict[str, Any]:
    """Build complete result dict with all required keys, even on partial failure."""
    try:
        nonzero_rewards = [r for r in reward_trace if r != 0]
        pos_rewards = sum(1 for r in reward_trace if r > 0)
        total_reward = sum(reward_trace)

        reward_count = len(reward_trace)
        reward_density = pos_rewards / reward_count if reward_count > 0 else 0.0
        avg_reward_per_step = float(np.mean(reward_trace)) if reward_trace else 0.0
        reward_variance = float(np.var(reward_trace)) if reward_trace else 0.0

        action_entropy = 0.0
        if action_history:
            try:
                action_lengths = [len(a.split()) for a in action_history]
                counts = np.bincount(action_lengths)
                nonzero = counts[counts > 0]
                if len(nonzero) > 0:
                    probs = nonzero / len(action_history)
                    action_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
            except Exception:
                action_entropy = 0.0

        return {
            "task_id": task_id,
            "episode_id": episode_id,
            "score": float(score),
            "avg_reward": float(metrics.get("avg_reward", 0.0)),
            "detection": float(metrics.get("detection", 0.0)),
            "lab_workup": float(metrics.get("lab_workup", 0.0)),
            "treatment": float(metrics.get("treatment", 0.0)),
            "timeliness": float(metrics.get("timeliness", 0.0)),
            "stability": float(metrics.get("stability", 0.0)),
            "safety": float(metrics.get("safety", 0.0)),
            "outcome": float(metrics.get("outcome", 0.0)),
            "safety_violations": int(metrics.get("safety_violations", 0)),
            "safety_violation_rate": float(metrics.get("safety_violation_rate", 0.0)),
            "steps_taken": step_count,
            "total_reward": float(total_reward),
            "reward_count": reward_count,
            "positive_rewards_count": pos_rewards,
            "reward_density": float(reward_density),
            "avg_reward_per_step": float(avg_reward_per_step),
            "reward_variance": float(reward_variance),
            "max_single_reward": float(max(reward_trace)) if reward_trace else 0.0,
            "episode_length_efficiency": float(step_count / MAX_STEPS_PER_TASK[task_id])
            if MAX_STEPS_PER_TASK[task_id]
            else 0.0,
            "positive_reward_ratio": float(pos_rewards / max(1, len(nonzero_rewards))),
            "unique_actions": len(set(action_history)),
            "action_entropy": float(action_entropy),
            "policy_mode": "advanced",
            "policy_sources": dict(policy_sources),
            "policy_error_count": len(policy_errors),
            "policy_last_error": policy_errors[-1] if policy_errors else None,
        }
    except Exception as e:
        print(f"[ERROR] Failed to build result dict: {str(e)}", file=sys.stderr)
        # Return minimal safe dict
        return {
            "task_id": task_id,
            "episode_id": episode_id,
            "score": 0.0,
            "avg_reward": 0.0,
            "detection": 0.0,
            "lab_workup": 0.0,
            "treatment": 0.0,
            "timeliness": 0.0,
            "stability": 0.0,
            "safety": 0.0,
            "outcome": 0.0,
            "safety_violations": 0,
            "safety_violation_rate": 0.0,
            "steps_taken": step_count,
            "total_reward": 0.0,
            "reward_count": 0,
            "positive_rewards_count": 0,
            "reward_density": 0.0,
            "avg_reward_per_step": 0.0,
            "reward_variance": 0.0,
            "max_single_reward": 0.0,
            "episode_length_efficiency": 0.0,
            "positive_reward_ratio": 0.0,
            "unique_actions": 0,
            "action_entropy": 0.0,
            "policy_mode": "fallback",
            "policy_sources": {},
            "policy_error_count": len(policy_errors),
            "policy_last_error": str(e),
        }


# =========================
# RUN TASK (BULLETPROOF)
# =========================
def run_task(task_id: str, policy_mode: str, client: OpenAI | None, model_name: str | None, episode_index: int) -> dict[str, Any]:
    """Run a single task with comprehensive exception handling."""
    env = None
    reward_trace: list[float] = []
    action_history: list[str] = []
    policy_sources: Counter = Counter()
    policy_errors: list[str] = []
    step_count = 0
    score = 0.0
    episode_id = "unknown"
    metrics: dict = {}
    obs = None

    try:
        # INIT ENV
        try:
            env = SepsisTreatmentEnv(base_url=os.getenv("ENV_BASE_URL"), task_id=task_id)
            result = env.reset()
            obs = result.observation
            final_info = result.info or {}
        except Exception as e:
            policy_errors.append(f"Env init failed: {str(e)}")
            return build_result_dict(task_id, episode_id, 0, [], [], policy_sources, policy_errors, {}, 0.0)

        # STEP LOOP
        try:
            for step in range(1, MAX_STEPS_PER_TASK[task_id] + 1):
                try:
                    action, source, err = choose_action(policy_mode, client, model_name, obs)
                except Exception as e:
                    policy_errors.append(f"Action selection failed: {str(e)}")
                    action = heuristic_action(obs)
                    source = "fallback"
                    err = str(e)

                # Step env
                try:
                    result = env.step(action)
                    obs = result.observation
                    reward = float(result.reward or 0.0)
                except Exception as e:
                    policy_errors.append(f"Step failed: {str(e)}")
                    break

                # Update learning
                try:
                    update_value(obs, reward)
                except Exception:
                    pass

                # Track
                reward_trace.append(reward)
                action_history.append(str(action))
                policy_sources[source] += 1
                step_count = step

                if result.done:
                    break

        except Exception as e:
            policy_errors.append(f"Step loop error: {str(e)}")

    except Exception as e:
        policy_errors.append(f"Outer exception: {str(e)}")

    finally:
        # CLEANUP
        if env is not None:
            try:
                state = env.state()
                episode_id = getattr(state, "episode_id", "unknown")
            except Exception as e:
                policy_errors.append(f"State query failed: {str(e)}")
                episode_id = "unknown"

            try:
                env.close()
            except Exception as e:
                policy_errors.append(f"Env close failed: {str(e)}")

        # METRICS
        try:
            if final_info:
                metrics = final_info.get("metrics", {}) or {}
                score = float(metrics.get("score", 0.0))
            else:
                metrics = {}
                score = 0.0
        except Exception as e:
            policy_errors.append(f"Metrics extraction failed: {str(e)}")
            metrics = {}
            score = 0.0

    # BUILD RESULT
    return build_result_dict(
        task_id=task_id,
        episode_id=episode_id,
        step_count=step_count,
        reward_trace=reward_trace,
        action_history=action_history,
        policy_sources=policy_sources,
        policy_errors=policy_errors,
        metrics=metrics,
        score=score,
    )


# =========================
# MAIN (BULLETPROOF)
# =========================
def main() -> None:
    try:
        args = parse_args()
        OUTPUT_DIR.mkdir(exist_ok=True)

        api_key = os.getenv("OPENAI_API_KEY")
        client = None
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"[WARN] OpenAI client init failed: {str(e)}", file=sys.stderr)

        results: list[dict[str, Any]] = []

        for ep in range(args.episodes):
            try:
                for task in TASK_IDS:
                    try:
                        res = run_task(task, args.model, client, None, ep)
                        results.append(res)
                    except Exception as e:
                        print(f"[ERROR] Task {task} episode {ep} failed: {str(e)}", file=sys.stderr)
                        results.append({
                            "task_id": task,
                            "episode_id": "unknown",
                            "score": 0.0,
                            "steps_taken": 0,
                            "policy_error_count": 1,
                            "policy_last_error": str(e),
                        })
            except Exception as e:
                print(f"[ERROR] Episode {ep} failed: {str(e)}", file=sys.stderr)

        # WRITE OUTPUT
        try:
            if not results:
                results = []
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2))
        except Exception as e:
            print(f"[FATAL] Output write failed: {str(e)}", file=sys.stderr)
            raise SystemExit(1)

    except SystemExit:
        raise
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
