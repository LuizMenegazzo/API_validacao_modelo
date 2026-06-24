from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


DEFAULT_PROJECT_GLOBAL_SETTINGS: dict[str, Any] = {
    "score_mode": "continuous",
    "window_aggregation_mode": "average",
    "outlier_strategy": "none",
    "min_runs_for_outlier_filter": 5,
    "test_type_weights": {
        "steady_state": 1.0,
        "step_test": 1.0,
        "fault_test": 1.0,
        "ramp_test": 1.0,
    },
    "score_category_weights": {
        "waveform_fidelity": 1.0,
        "spectral_fidelity": 1.0,
        "steady_state_operating": 1.0,
        "steady_state_variability": 1.0,
        "transient_magnitude": 1.0,
        "transient_timing": 1.0,
        "transient_similarity": 1.0,
    },
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _copied_default_settings() -> dict[str, Any]:
    return {
        "score_mode": DEFAULT_PROJECT_GLOBAL_SETTINGS["score_mode"],
        "window_aggregation_mode": DEFAULT_PROJECT_GLOBAL_SETTINGS["window_aggregation_mode"],
        "outlier_strategy": DEFAULT_PROJECT_GLOBAL_SETTINGS["outlier_strategy"],
        "min_runs_for_outlier_filter": DEFAULT_PROJECT_GLOBAL_SETTINGS["min_runs_for_outlier_filter"],
        "test_type_weights": dict(DEFAULT_PROJECT_GLOBAL_SETTINGS["test_type_weights"]),
        "score_category_weights": dict(DEFAULT_PROJECT_GLOBAL_SETTINGS["score_category_weights"]),
    }


@dataclass(slots=True)
class AssessmentProject:
    project_id: str
    project_name: str
    model_name: str
    model_version: str = ""
    developer: str = ""
    date_created: str = field(default_factory=_utc_now_iso)
    date_updated: str = field(default_factory=_utc_now_iso)
    notes: str = ""
    global_settings: dict[str, Any] = field(default_factory=_copied_default_settings)
    test_cases: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.project_id = _normalized_text(self.project_id) or str(uuid4())
        self.project_name = _normalized_text(self.project_name)
        self.model_name = _normalized_text(self.model_name)
        self.model_version = _normalized_text(self.model_version)
        self.developer = _normalized_text(self.developer)
        self.notes = str(self.notes or "").strip()
        self.date_created = _normalized_text(self.date_created) or _utc_now_iso()
        self.date_updated = _normalized_text(self.date_updated) or self.date_created
        self.global_settings = self._merge_global_settings(self.global_settings)
        self.test_cases = list(self.test_cases or [])

    @classmethod
    def create(
        cls,
        project_name: str,
        model_name: str,
        *,
        model_version: str = "",
        developer: str = "",
        notes: str = "",
        global_settings: dict[str, Any] | None = None,
    ) -> "AssessmentProject":
        return cls(
            project_id=str(uuid4()),
            project_name=project_name,
            model_name=model_name,
            model_version=model_version,
            developer=developer,
            notes=notes,
            global_settings=global_settings or _copied_default_settings(),
        )

    @staticmethod
    def _merge_global_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
        merged = _copied_default_settings()
        if not isinstance(settings, dict):
            return merged

        for key in ("score_mode", "window_aggregation_mode", "outlier_strategy"):
            if key in settings and settings[key] not in (None, ""):
                merged[key] = settings[key]

        min_runs_value = settings.get("min_runs_for_outlier_filter")
        try:
            min_runs_number = int(min_runs_value)
        except (TypeError, ValueError):
            min_runs_number = int(merged["min_runs_for_outlier_filter"])
        merged["min_runs_for_outlier_filter"] = max(min_runs_number, 1)

        for weight_key in ("test_type_weights", "score_category_weights"):
            source_weights = settings.get(weight_key)
            if not isinstance(source_weights, dict):
                continue
            target_weights = dict(merged[weight_key])
            for item_key, item_value in source_weights.items():
                try:
                    numeric_value = float(item_value)
                except (TypeError, ValueError):
                    continue
                if numeric_value > 0:
                    target_weights[str(item_key)] = numeric_value
            merged[weight_key] = target_weights

        return merged

    def touch(self) -> None:
        self.date_updated = _utc_now_iso()

    def register_test_case(self, test_case_payload: dict[str, Any]) -> None:
        self.test_cases.append(dict(test_case_payload))
        self.touch()

    @property
    def test_case_count(self) -> int:
        return len(self.test_cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "developer": self.developer,
            "date_created": self.date_created,
            "date_updated": self.date_updated,
            "notes": self.notes,
            "global_settings": self._merge_global_settings(self.global_settings),
            "test_cases": [dict(test_case) for test_case in self.test_cases],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AssessmentProject":
        if not isinstance(payload, dict):
            raise TypeError("AssessmentProject payload must be a dictionary.")
        return cls(
            project_id=str(payload.get("project_id", "")),
            project_name=str(payload.get("project_name", "")),
            model_name=str(payload.get("model_name", "")),
            model_version=str(payload.get("model_version", "")),
            developer=str(payload.get("developer", "")),
            date_created=str(payload.get("date_created", "")),
            date_updated=str(payload.get("date_updated", "")),
            notes=str(payload.get("notes", "")),
            global_settings=payload.get("global_settings"),
            test_cases=list(payload.get("test_cases", [])),
        )
