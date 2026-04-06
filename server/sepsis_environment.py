from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graders import grade_episode, summarize_episode
from models import SepsisAction, SepsisObservation, SepsisState
from openenv_compat import Environment
from tasks import TaskConfig, build_task_catalog


ROOT = Path(__file__).resolve().parent.parent
ENV_DATA_DIR = ROOT / "env_data"
DATASET_PATH = ENV_DATA_DIR / "processed_demo_dataset.pkl"
FEATURES_PATH = ENV_DATA_DIR / "selected_features.json"

LAB_FIELDS = {
    "lactate": "Arterial_lactate",
    "wbc": "WBC_count",
    "creatinine": "Creatinine",
    "bicarbonate": "Bicarbonate",
    "platelets": "Platelets_count",
    "bilirubin": "Total_bili",
}
LAB_OPTIONS = list(LAB_FIELDS.keys())
TREATMENT_OPTIONS = ["monitor", "fluids", "vasopressors", "combination"]
VITAL_FIELDS = ["HR", "MeanBP", "RR", "Temp_C", "SpO2", "Shock_Index"]
DEMOGRAPHIC_FIELDS = ["age", "is_male"]


def load_processed_assets() -> tuple[pd.DataFrame, list[str]]:
    if DATASET_PATH.exists() and FEATURES_PATH.exists():
        dataset = pd.read_pickle(DATASET_PATH)
        with open(FEATURES_PATH, "r", encoding="utf-8") as handle:
            selected_features = json.load(handle)
        return dataset, selected_features
    raise FileNotFoundError(
        "Missing env_data assets. Expected env_data/processed_demo_dataset.pkl "
        "and env_data/selected_features.json."
    )


def build_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        dataset.groupby("icustay_id")
        .agg(
            length=("bin_idx", "count"),
            mean_severity=("severity_proxy", "mean"),
            max_severity=("severity_proxy", "max"),
            mortality=("mortality", "max"),
        )
        .reset_index()
        .sort_values("icustay_id")
        .reset_index(drop=True)
    )
    return grouped


class SepsisTreatmentEnvironment(Environment):
    def __init__(self, task_id: str = "easy"):
        self.dataset, self.selected_features = load_processed_assets()
        self.summary = build_summary(self.dataset)
        self.task_catalog = build_task_catalog(self.summary)
        self._task_cycle = {task_key: 0 for task_key in self.task_catalog}
        self.default_task_id = task_id if task_id in self.task_catalog else "easy"
        self._episode = pd.DataFrame()
        self._cursor = 0
        self._task: TaskConfig = self.task_catalog[self.default_task_id]
        self._metrics: dict[str, Any] = {}
        self._visited_state_actions: set[tuple[str, str]] = set()
        self._state = SepsisState(
            episode_id=str(uuid.uuid4()),
            task_id=self.default_task_id,
            current_stay_id=-1,
            step_count=0,
            max_steps=0,
            cumulative_reward=0.0,
            safety_violations=0,
            terminal_outcome="ongoing",
            requested_labs=[],
            visible_labs={},
            history=[],
        )

    def available_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "id": task.task_id,
                "title": task.title,
                "description": task.description,
                "min_steps": task.min_steps,
                "max_steps": task.max_steps,
                "num_episodes": len(task.preferred_stay_ids),
            }
            for task in self.task_catalog.values()
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "sepsi-gym",
            "description": "A real-world clinical RL environment for sepsis diagnosis and treatment management using MIMIC-III trajectories.",
            "tasks": self.available_tasks(),
            "selected_features": self.selected_features,
            "action_space": {
                "action_type": ["request_lab", "request_treatment", "monitor"],
                "lab_type": LAB_OPTIONS,
                "treatment_type": TREATMENT_OPTIONS,
                "suspect_sepsis": "boolean",
            },
            "observation_space": {
                "vitals": VITAL_FIELDS,
                "visible_labs": LAB_OPTIONS,
                "demographics": DEMOGRAPHIC_FIELDS,
            },
        }

    def _select_episode(self, task: TaskConfig) -> pd.DataFrame:
        stay_ids = task.preferred_stay_ids
        if not stay_ids:
            stay_ids = tuple(int(stay_id) for stay_id in self.summary["icustay_id"].tolist())
        index = self._task_cycle[task.task_id] % len(stay_ids)
        self._task_cycle[task.task_id] += 1
        stay_id = stay_ids[index]
        episode = self.dataset[self.dataset["icustay_id"] == stay_id].sort_values("bin_idx").reset_index(drop=True)
        episode = episode.head(task.max_steps).copy()
        if len(episode) < task.min_steps:
            episode = self.dataset[self.dataset["icustay_id"] == stay_id].sort_values("bin_idx").reset_index(drop=True)
        return episode

    def _row_float(self, row: pd.Series, key: str) -> float:
        value = row.get(key, 0.0)
        if pd.isna(value):
            return 0.0
        return float(value)

    def _make_observation(self, row: pd.Series, *, reward: float, done: bool) -> SepsisObservation:
        hidden_lab_columns = set(LAB_FIELDS.values())
        vitals = {name: self._row_float(row, name) for name in VITAL_FIELDS}
        demographics = {name: self._row_float(row, name) for name in DEMOGRAPHIC_FIELDS}
        context_features = {
            name: self._row_float(row, name)
            for name in self.selected_features
            if name not in hidden_lab_columns and name not in VITAL_FIELDS
        }
        return SepsisObservation(
            episode_id=self._state.episode_id,
            task_id=self._task.task_id,
            task_description=self._task.description,
            patient_id=int(row["icustay_id"]),
            step_index=int(self._cursor),
            max_steps=int(len(self._episode)),
            severity_proxy=self._row_float(row, "severity_proxy"),
            mortality_risk_flag=int(row["mortality"]),
            demographics=demographics,
            vitals=vitals,
            context_features=context_features,
            visible_labs=dict(self._state.visible_labs),
            requested_labs=list(self._state.requested_labs),
            available_lab_options=LAB_OPTIONS,
            available_treatment_options=TREATMENT_OPTIONS,
            cumulative_reward=float(self._state.cumulative_reward),
            last_reward=float(reward),
            done=done,
            reward=float(reward),
        )

    def _priority_labs(self, row: pd.Series) -> set[str]:
        priority: set[str] = set()
        severity = self._row_float(row, "severity_proxy")
        shock = self._row_float(row, "Shock_Index")
        mean_bp = self._row_float(row, "MeanBP")
        lactate = self._row_float(row, "Arterial_lactate")
        wbc = self._row_float(row, "WBC_count")
        creatinine = self._row_float(row, "Creatinine")
        bicarbonate = self._row_float(row, "Bicarbonate")
        platelets = self._row_float(row, "Platelets_count")
        bilirubin = self._row_float(row, "Total_bili")

        if severity >= 1.0 or shock > 0.1 or mean_bp < 0.0 or lactate > 0.25:
            priority.update(["lactate", "wbc"])
        if creatinine > 0.15 or self._row_float(row, "BUN_Creatinine_Ratio") > 0.2:
            priority.add("creatinine")
        if bicarbonate < -0.15 or self._row_float(row, "Arterial_pH") < -0.15:
            priority.add("bicarbonate")
        if platelets < -0.2:
            priority.add("platelets")
        if bilirubin > 0.15:
            priority.add("bilirubin")
        if not priority:
            priority.add("lactate")
        return priority

    def _sepsis_signal(self, row: pd.Series) -> bool:
        return bool(
            self._row_float(row, "severity_proxy") >= 1.0
            or self._row_float(row, "Shock_Index") > 0.1
            or self._row_float(row, "MeanBP") < 0.0
            or self._row_float(row, "Arterial_lactate") > 0.25
            or self._row_float(row, "WBC_count") > 0.2
        )

    def _target_treatment(self, row: pd.Series) -> str:
        clinician_fluid = int(row["fluid_bin"])
        clinician_pressor = int(row["pressor_bin"])
        severity = self._row_float(row, "severity_proxy")
        if clinician_pressor >= 2 and clinician_fluid >= 2:
            return "combination"
        if clinician_pressor >= 2:
            return "vasopressors"
        if clinician_fluid >= 2 or severity >= 1.25 or self._row_float(row, "Shock_Index") > 0.15:
            return "fluids"
        return "monitor"

    def _treatment_match(self, chosen: str, target: str) -> float:
        if chosen == target:
            return 1.0
        close_pairs = {
            ("fluids", "combination"),
            ("combination", "fluids"),
            ("vasopressors", "combination"),
            ("combination", "vasopressors"),
            ("monitor", "fluids"),
            ("fluids", "monitor"),
        }
        if (chosen, target) in close_pairs:
            return 0.5
        return 0.0

    def _compute_reward(
        self,
        row: pd.Series,
        next_row: pd.Series | None,
        action: SepsisAction,
        is_terminal: bool,
    ) -> tuple[float, dict[str, Any]]:
        reward = 0.0
        unsafe = False
        sepsis_signal = self._sepsis_signal(row)
        priority_labs = self._priority_labs(row)
        target_treatment = self._target_treatment(row)
        severity_now = self._row_float(row, "severity_proxy")
        severity_next = self._row_float(next_row, "severity_proxy") if next_row is not None else severity_now
        progress = float(0.15 * np.tanh(severity_now - severity_next))
        stability_score = 1.0 if severity_next <= severity_now else max(0.0, 1.0 - min(severity_next - severity_now, 1.0))

        detection_credit = 0.0
        lab_score = 0.0
        treatment_score = 0.0
        inefficiency_penalty = 0.0
        safety_penalty = 0.0
        novelty_bonus = 0.0

        if action.suspect_sepsis:
            detection_credit = 1.0 if sepsis_signal else 0.25
            reward += 0.15 if sepsis_signal else -0.05
        elif sepsis_signal and self._cursor <= 1:
            reward -= 0.05

        if action.action_type == "request_lab":
            if action.lab_type in LAB_FIELDS:
                duplicate_lab = action.lab_type in self._state.requested_labs
                revealed_value = self._row_float(row, LAB_FIELDS[action.lab_type])
                self._state.visible_labs[action.lab_type] = revealed_value
                self._state.requested_labs.append(action.lab_type)
                if duplicate_lab:
                    inefficiency_penalty += 0.08
                    reward -= 0.08
                elif action.lab_type in priority_labs:
                    lab_score = 1.0
                    reward += 0.20
                else:
                    lab_score = 0.4
                    reward += 0.05
                if action.lab_type in {"lactate", "wbc"} and sepsis_signal:
                    detection_credit = max(detection_credit, 0.8)
            else:
                inefficiency_penalty += 0.05
                reward -= 0.05
        elif action.action_type == "request_treatment":
            if action.treatment_type in TREATMENT_OPTIONS:
                treatment_score = self._treatment_match(action.treatment_type, target_treatment)
                reward += 0.30 * treatment_score
                if severity_now <= 0.8 and action.treatment_type in {"vasopressors", "combination"}:
                    safety_penalty += 0.25
                    unsafe = True
                if severity_now >= 2.0 and action.treatment_type == "monitor":
                    safety_penalty += 0.30
                    unsafe = True
                if self._row_float(row, "MeanBP") < -0.2 and action.treatment_type == "monitor":
                    safety_penalty += 0.10
                    unsafe = True
            else:
                inefficiency_penalty += 0.05
                reward -= 0.05
        else:
            if severity_now >= 1.75:
                inefficiency_penalty += 0.05
                reward -= 0.05

        state_id = f"{int(row['icustay_id'])}_{self._cursor}"
        action_id = f"{action.action_type}_{action.lab_type}_{action.treatment_type}"
        state_action_key = (state_id, action_id)
        if state_action_key not in self._visited_state_actions:
            novelty_bonus = 0.03
            reward += novelty_bonus
            self._visited_state_actions.add(state_action_key)
        else:
            inefficiency_penalty += 0.01
            reward -= 0.01

        if sepsis_signal and self._cursor <= 2 and action.suspect_sepsis:
            detection_credit = max(detection_credit, 1.0)
            reward += 0.05

        if severity_now > 1.5 and action.action_type == "monitor":
            safety_penalty += 0.05
            reward -= 0.05

        reward += progress
        reward -= safety_penalty
        reward -= inefficiency_penalty

        if is_terminal:
            reward += 0.30 if int(row["mortality"]) == 0 else -0.30

        reward = float(np.clip(reward, -1.0, 1.0))
        return reward, {
            "detection_credit": round(detection_credit, 4),
            "lab_score": round(lab_score, 4),
            "treatment_score": round(treatment_score, 4),
            "stability_score": round(stability_score, 4),
            "progress_score": round(progress, 4),
            "novelty_bonus": round(novelty_bonus, 4),
            "safety_penalty": round(safety_penalty, 4),
            "inefficiency_penalty": round(inefficiency_penalty, 4),
            "target_treatment": target_treatment,
            "priority_labs": sorted(priority_labs),
            "unsafe": unsafe,
        }

    def reset(self, task_id: str | None = None) -> SepsisObservation:
        chosen_task = task_id or self.default_task_id
        if chosen_task not in self.task_catalog:
            chosen_task = self.default_task_id
        self._task = self.task_catalog[chosen_task]
        self._episode = self._select_episode(self._task)
        self._cursor = 0
        current_row = self._episode.iloc[self._cursor]
        self._state = SepsisState(
            episode_id=str(uuid.uuid4()),
            task_id=self._task.task_id,
            current_stay_id=int(current_row["icustay_id"]),
            step_count=0,
            max_steps=int(len(self._episode)),
            cumulative_reward=0.0,
            safety_violations=0,
            terminal_outcome="ongoing",
            requested_labs=[],
            visible_labs={},
            history=[],
        )
        self._metrics = {}
        self._visited_state_actions = set()
        return self._make_observation(current_row, reward=0.0, done=False)

    def step(self, action: SepsisAction) -> SepsisObservation:
        if self._episode.empty:
            return self.reset(self.default_task_id)

        row = self._episode.iloc[self._cursor]
        next_index = self._cursor + 1
        next_row = self._episode.iloc[next_index] if next_index < len(self._episode) else None
        done = next_row is None

        reward, details = self._compute_reward(row, next_row, action, done)
        self._state.step_count += 1
        self._state.cumulative_reward += reward
        if details["unsafe"]:
            self._state.safety_violations += 1
        if done:
            self._state.terminal_outcome = "survived" if int(row["mortality"]) == 0 else "died"

        history_row = {
            "step": int(self._cursor),
            "action_index": action.action_index,
            "action_type": action.action_type,
            "suspect_sepsis": action.suspect_sepsis,
            "lab_type": action.lab_type,
            "treatment_type": action.treatment_type,
            "detection_credit": details["detection_credit"],
            "lab_score": details["lab_score"],
            "treatment_score": details["treatment_score"],
            "stability_score": details["stability_score"],
            "progress_score": details["progress_score"],
            "novelty_bonus": details["novelty_bonus"],
            "safety_penalty": details["safety_penalty"],
            "inefficiency_penalty": details["inefficiency_penalty"],
            "unsafe": details["unsafe"],
            "severity_proxy": round(self._row_float(row, "severity_proxy"), 4),
            "target_treatment": details["target_treatment"],
            "priority_labs": details["priority_labs"],
            "reward": round(reward, 4),
        }
        self._state.history.append(history_row)

        if not done:
            self._cursor = next_index
            observation_row = self._episode.iloc[self._cursor]
        else:
            observation_row = row

        episode_metrics = summarize_episode(
            total_reward=float(self._state.cumulative_reward),
            state_history=self._state.history,
            terminal_outcome=self._state.terminal_outcome,
        )
        episode_metrics["score"] = grade_episode(self._task, episode_metrics)
        self._metrics = episode_metrics
        return self._make_observation(observation_row, reward=reward, done=done)

    def current_metrics(self) -> dict[str, Any]:
        return {
            **self._metrics,
            "task_id": self._task.task_id,
            "current_stay_id": self._state.current_stay_id,
            "cumulative_reward": round(float(self._state.cumulative_reward), 4),
            "requested_labs": list(self._state.requested_labs),
            "safety_violations": int(self._state.safety_violations),
        }

    @property
    def state(self) -> SepsisState:
        return self._state
