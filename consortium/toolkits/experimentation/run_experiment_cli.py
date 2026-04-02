"""CLI wrapper for canonical experiment execution and bookkeeping."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


STAGE_SUMMARY_FILES = {
    2: "baseline_summary.json",
    3: "research_summary.json",
    4: "ablation_summary.json",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _slugify(text: str) -> str:
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "experiment"


def _extract_experiment_spec(spec_payload: Any, experiment_id: Optional[str]) -> Dict[str, Any]:
    if isinstance(spec_payload, dict) and "experiments" in spec_payload:
        experiments = spec_payload.get("experiments") or []
        if experiment_id:
            for spec in experiments:
                if spec.get("experiment_id") == experiment_id:
                    return spec
            raise ValueError(f"experiment_id '{experiment_id}' not found in experiments list")
        if len(experiments) == 1:
            return experiments[0]
        raise ValueError("spec file contains multiple experiments; pass --experiment-id")

    if isinstance(spec_payload, dict):
        return spec_payload

    raise ValueError("experiment spec must be a JSON object or experiment_design.json payload")


def _validate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not spec.get("experiment_id"):
        raise ValueError("experiment spec missing required field 'experiment_id'")
    if not spec.get("title"):
        spec["title"] = str(spec["experiment_id"])
    if not spec.get("hypothesis"):
        raise ValueError("experiment spec missing required field 'hypothesis'")
    if not spec.get("goal_id") and not spec.get("goal_ids"):
        spec["goal_id"] = None
    end_stage = int(spec.get("end_stage", 4))
    if end_stage not in {1, 2, 3, 4}:
        raise ValueError("end_stage must be one of 1, 2, 3, 4")
    spec["end_stage"] = end_stage
    return spec


def _build_ai_scientist_idea(spec: Dict[str, Any]) -> Dict[str, Any]:
    experiment_lines = [
        f"Experiment ID: {spec['experiment_id']}",
        f"Goal ID: {spec.get('goal_id') or ', '.join(spec.get('goal_ids', [])) or 'unassigned'}",
        f"Target questions: {', '.join(spec.get('addresses_questions', [])) or 'not specified'}",
        f"Model: {spec.get('model', 'unspecified')}",
        f"Dataset: {spec.get('dataset', 'unspecified')}",
        f"Baselines: {', '.join(spec.get('baselines', [])) or 'none listed'}",
        f"Metrics: {', '.join(spec.get('metrics', [])) or 'none listed'}",
        f"Ablations: {', '.join(spec.get('ablations', [])) or 'none listed'}",
        f"Success criteria: {spec.get('success_criteria', 'not specified')}",
        f"Estimated runtime (hours): {spec.get('estimated_runtime_hours', 'unknown')}",
        f"Requested end stage: {spec['end_stage']}",
        f"End-stage rationale: {spec.get('end_stage_rationale', 'not provided')}",
    ]

    return {
        "Name": _slugify(str(spec["experiment_id"])),
        "Title": str(spec["title"]),
        "Short Hypothesis": str(spec["hypothesis"]),
        "Abstract": (
            "Execute the experiment exactly as specified in consortium experiment_design.json. "
            "Focus on producing reproducible artifacts for downstream verification."
        ),
        "Experiments": {
            "Primary Experiment": "\n".join(experiment_lines),
        },
        "Risk Factors and Limitations": (
            spec.get("open_decisions")
            or spec.get("risks")
            or "Do not fabricate unavailable datasets, metrics, or baselines."
        ),
    }


def _find_experiment_dir(run_root: Path) -> Optional[Path]:
    experiments_dir = run_root / "experiments"
    if not experiments_dir.exists():
        return None
    candidates = [p for p in experiments_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _infer_end_stage_executed(exp_dir: Optional[Path]) -> int:
    if exp_dir is None:
        return 0
    log_root = exp_dir / "logs" / "0-run"
    if not log_root.exists():
        return 1
    for stage, filename in sorted(STAGE_SUMMARY_FILES.items(), reverse=True):
        if (log_root / filename).exists():
            return stage
    return 1


def _walk_numeric_metrics(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_numeric_metrics(child, child_prefix)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            yield from _walk_numeric_metrics(child, child_prefix)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix or "value", float(value)


def _extract_primary_metric(exp_dir: Optional[Path], preferred_name: Optional[str]) -> Dict[str, Any]:
    if exp_dir is None:
        return {"name": preferred_name or "metric", "value": None}

    log_root = exp_dir / "logs" / "0-run"
    summary_candidates = [
        log_root / "ablation_summary.json",
        log_root / "research_summary.json",
        log_root / "baseline_summary.json",
    ]
    numeric_hits: list[tuple[str, float]] = []
    for path in summary_candidates:
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        numeric_hits.extend(_walk_numeric_metrics(payload))
        if numeric_hits:
            break

    if preferred_name:
        for name, value in numeric_hits:
            if preferred_name.lower() in name.lower():
                return {"name": preferred_name, "value": value}

    if numeric_hits:
        name, value = numeric_hits[0]
        return {"name": preferred_name or name, "value": value}
    return {"name": preferred_name or "metric", "value": None}


def _append_execution_log(workspace_dir: Path, entry: Dict[str, Any]) -> None:
    log_path = workspace_dir / "experiment_workspace" / "execution_log.json"
    if log_path.exists():
        try:
            payload = _load_json(log_path)
        except Exception:
            payload = {"runs": []}
    else:
        payload = {"runs": []}
    runs = payload.setdefault("runs", [])
    runs.append(entry)
    _write_json(log_path, payload)


def _append_partial_notes(workspace_dir: Path, entry: Dict[str, Any]) -> None:
    notes_path = workspace_dir / "experiment_workspace" / "partial_run_notes.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## {entry['experiment_id']}",
        f"- status: {entry['status']}",
        f"- end_stage_requested: {entry['end_stage_requested']}",
        f"- end_stage_executed: {entry['end_stage_executed']}",
        f"- run_dir: {entry['run_dir']}",
        f"- failure_reason: {entry['failure_reason'] or 'n/a'}",
        "",
    ]
    with notes_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _get_python_executable() -> str:
    current = sys.executable
    return current if current and os.path.exists(current) else "python3"


def _run_mock(run_root: Path, spec: Dict[str, Any]) -> tuple[int, Optional[Path], Optional[str]]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = run_root / "experiments" / f"{timestamp}_{_slugify(spec['experiment_id'])}"
    log_dir = exp_dir / "logs" / "0-run"
    log_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "research_idea.md").write_text(spec["hypothesis"] + "\n", encoding="utf-8")
    (exp_dir / "best_code.py").write_text("print('mock experiment')\n", encoding="utf-8")
    metric_name = (spec.get("metrics") or ["score"])[0]
    metric_value = 0.75
    if spec["end_stage"] >= 2:
        _write_json(log_dir / "baseline_summary.json", {"primary_metric": {metric_name: metric_value - 0.05}})
    if spec["end_stage"] >= 3:
        _write_json(log_dir / "research_summary.json", {"primary_metric": {metric_name: metric_value}})
    if spec["end_stage"] >= 4:
        _write_json(log_dir / "ablation_summary.json", {"primary_metric": {metric_name: metric_value - 0.01}})
    _write_json(log_dir / "experiment_results.json", {"status": "success"})
    return 0, exp_dir, None


def _run_real(run_root: Path, spec: Dict[str, Any], timeout_seconds: int) -> tuple[int, Optional[Path], Optional[str]]:
    repo_root = Path(__file__).resolve().parents[3]
    tool_root = repo_root / "consortium" / "external_tools" / "run_experiment_tool"
    script_path = tool_root / "launch_scientist_bfts.py"
    idea_payload = [_build_ai_scientist_idea(spec)]
    idea_path = run_root / "idea.json"
    _write_json(idea_path, idea_payload)

    cmd = [
        _get_python_executable(),
        str(script_path),
        "--load_ideas",
        str(idea_path),
        "--idea_idx",
        "0",
        "--debug",
        "--skip_writeup",
        "--skip_review",
        "--end_stage",
        str(spec["end_stage"]),
    ]
    env = os.environ.copy()
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(run_root),
            env=env,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, _find_experiment_dir(run_root), f"timed out after {timeout_seconds} seconds"

    exp_dir = _find_experiment_dir(run_root)
    if result.returncode != 0:
        return result.returncode, exp_dir, f"launcher failed with rc={result.returncode}"
    _ = time.time() - t0
    return 0, exp_dir, None


def _normalize_stage_results(stage_results: Any) -> Dict[str, Any]:
    if isinstance(stage_results, dict):
        return stage_results
    return {"raw_output": stage_results}


def _write_cli_stage_summaries(exp_dir: Path, all_results: Dict[str, Any], requested_stage: int) -> None:
    log_dir = exp_dir / "logs" / "0-run"
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_to_key = {
        1: "baseline",
        2: "tuning",
        3: "creative",
        4: "ablation",
    }
    summary_targets = {
        2: "baseline_summary.json",
        3: "research_summary.json",
        4: "ablation_summary.json",
    }
    for stage, filename in summary_targets.items():
        if requested_stage < stage:
            continue
        source_key = stage_to_key[stage]
        payload = _normalize_stage_results(all_results.get(source_key, {}))
        _write_json(log_dir / filename, payload)
    _write_json(log_dir / "experiment_results.json", {"stages": all_results})


def _run_cli_stages(
    run_root: Path,
    spec: Dict[str, Any],
    timeout_seconds: int,
    backend: Optional[str],
    model: Optional[str],
) -> tuple[int, Optional[Path], Optional[str]]:
    from ...experiment_runner import run_experiment_stages

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = run_root / "experiments" / f"{timestamp}_{_slugify(spec['experiment_id'])}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    idea_payload = _build_ai_scientist_idea(spec)
    try:
        summary = run_experiment_stages(
            idea_spec=json.dumps(idea_payload, indent=2),
            workspace_dir=str(run_root),
            backend=backend or os.environ.get("CONSORTIUM_ACTIVE_CLI_BACKEND", "claude"),
            model=model or os.environ.get("CONSORTIUM_ACTIVE_CLI_MODEL"),
            end_stage=spec["end_stage"],
            timeout_per_stage=timeout_seconds,
            stage_root=str(exp_dir),
        )
    except Exception as exc:
        return 1, exp_dir, str(exc)

    try:
        all_results = json.loads(summary)
    except json.JSONDecodeError:
        all_results = {"raw_output": summary}
    _write_json(exp_dir / "all_results.json", all_results)
    _write_cli_stage_summaries(exp_dir, all_results, spec["end_stage"])
    (exp_dir / "research_idea.md").write_text(spec["hypothesis"] + "\n", encoding="utf-8")
    return 0, exp_dir, None


def _build_result(
    workspace_dir: Path,
    run_root: Path,
    spec: Dict[str, Any],
    exp_dir: Optional[Path],
    requested_stage: int,
    status_hint: str,
    failure_reason: Optional[str],
    wall_time_seconds: int,
) -> Dict[str, Any]:
    executed_stage = _infer_end_stage_executed(exp_dir)
    if status_hint == "success" and executed_stage and executed_stage < requested_stage:
        status_hint = "partial"

    primary_metric_name = (spec.get("metrics") or ["metric"])[0]
    primary_metric = _extract_primary_metric(exp_dir, primary_metric_name)
    run_dir_rel = os.path.relpath(run_root, workspace_dir)
    entry = {
        "experiment_id": spec["experiment_id"],
        "goal_id": spec.get("goal_id"),
        "status": status_hint,
        "end_stage_executed": executed_stage,
        "end_stage_requested": requested_stage,
        "run_dir": run_dir_rel,
        "primary_metric": primary_metric,
        "failure_reason": failure_reason,
        "wall_time_seconds": wall_time_seconds,
    }
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a canonical experiment wrapper and update execution_log.json"
    )
    parser.add_argument("--workspace", default=".", help="Workspace root (default: current directory)")
    parser.add_argument("--spec-file", required=True, help="Path to a single experiment spec or experiment_design.json")
    parser.add_argument("--experiment-id", help="Experiment id to select from experiment_design.json")
    parser.add_argument("--mock", action="store_true", help="Create canonical mock artifacts instead of launching AI-Scientist")
    parser.add_argument(
        "--engine",
        choices=["cli_stages", "ai_scientist"],
        default=os.environ.get("CONSORTIUM_EXPERIMENT_ENGINE", "ai_scientist"),
        help="Execution engine (default: CONSORTIUM_EXPERIMENT_ENGINE or ai_scientist)",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("CONSORTIUM_ACTIVE_CLI_BACKEND", "claude"),
        help="CLI backend for cli_stages engine (default: active agent backend or claude)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CONSORTIUM_ACTIVE_CLI_MODEL"),
        help="CLI model override for cli_stages engine (default: active agent model)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("CONSORTIUM_EXPERIMENT_TIMEOUT", "3600")),
        help="Execution timeout for the selected engine (default: CONSORTIUM_EXPERIMENT_TIMEOUT or 3600)",
    )
    parser.add_argument("--no-update-log", action="store_true", help="Do not append to experiment_workspace/execution_log.json")
    args = parser.parse_args()

    workspace_dir = Path(args.workspace).resolve()
    spec_payload = _load_json(Path(args.spec_file).resolve())
    spec = _validate_spec(_extract_experiment_spec(spec_payload, args.experiment_id))

    run_root = workspace_dir / "experiment_runs" / str(uuid.uuid4())
    run_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.mock:
        rc, exp_dir, failure_reason = _run_mock(run_root, spec)
    elif args.engine == "cli_stages":
        rc, exp_dir, failure_reason = _run_cli_stages(
            run_root, spec, args.timeout_seconds, args.backend, args.model
        )
    else:
        rc, exp_dir, failure_reason = _run_real(run_root, spec, args.timeout_seconds)
    elapsed = int(max(1, round(time.time() - t0)))

    if rc == 0:
        status = "success"
    elif rc == 124:
        status = "timeout"
    else:
        status = "failed"

    result = _build_result(
        workspace_dir=workspace_dir,
        run_root=run_root,
        spec=spec,
        exp_dir=exp_dir,
        requested_stage=spec["end_stage"],
        status_hint=status,
        failure_reason=failure_reason,
        wall_time_seconds=elapsed,
    )

    _write_json(run_root / "execution_result.json", result)
    if not args.no_update_log:
        _append_execution_log(workspace_dir, result)
        if result["status"] in {"partial", "timeout"}:
            _append_partial_notes(workspace_dir, result)

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if result["status"] in {"success", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
