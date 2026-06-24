from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from app.project_models import AssessmentProject


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_STORAGE_DIR = PROJECT_ROOT / "storage" / "assessment_projects"
FALLBACK_PROJECT_STORAGE_DIR = Path(tempfile.gettempdir()) / "ModelValidationProjects"


def _storage_dirs() -> list[Path]:
    return [PROJECT_STORAGE_DIR, FALLBACK_PROJECT_STORAGE_DIR]


def _payload_has_processed_results(payload: dict[str, Any]) -> bool:
    processed_report = payload.get("global_settings", {}).get("processed_report")
    if isinstance(processed_report, dict) and processed_report:
        return True
    for case in payload.get("test_cases", []):
        if not isinstance(case, dict):
            continue
        runs = case.get("runs")
        if not isinstance(runs, list) or not runs:
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            score_sections = run.get("score_sections")
            if isinstance(score_sections, list) and score_sections:
                return True
    return False


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _sanitize_for_json(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def assessment_project_path(project_id: str) -> Path:
    safe_id = str(project_id).strip() or "project"
    for directory in _storage_dirs():
        candidate = directory / f"{safe_id}.json"
        if candidate.exists():
            return candidate
    return PROJECT_STORAGE_DIR / f"{safe_id}.json"


def save_assessment_project(project: AssessmentProject) -> Path:
    payload = _sanitize_for_json(project.to_dict())
    serialized_payload = json.dumps(payload, ensure_ascii=True, indent=2)
    last_error: OSError | None = None
    for directory in _storage_dirs():
        target_path = directory / f"{str(project.project_id).strip() or 'project'}.json"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target_path.write_text(serialized_payload, encoding="utf-8")
            return target_path
        except OSError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise OSError("No model validation project storage directory is available.")


def load_assessment_project(project_id: str) -> AssessmentProject:
    source_path = assessment_project_path(project_id)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return AssessmentProject.from_dict(payload)


def list_saved_assessment_projects() -> list[dict[str, Any]]:
    projects_by_id: dict[str, dict[str, Any]] = {}
    candidate_paths: list[Path] = []
    for directory in _storage_dirs():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            candidate_paths.extend(directory.glob("*.json"))
        except OSError:
            continue

    for path in sorted(candidate_paths, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        project_id = str(payload.get("project_id", path.stem))
        if project_id in projects_by_id:
            continue
        projects_by_id[project_id] = (
            {
                "project_id": project_id,
                "model_name": str(payload.get("model_name", "")).strip() or path.stem,
                "model_version": str(payload.get("model_version", "")).strip(),
                "date_updated": str(payload.get("date_updated", "")).strip(),
                "path": path,
                "has_processed_report": _payload_has_processed_results(payload),
            }
        )
    return list(projects_by_id.values())
