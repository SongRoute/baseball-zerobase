from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from baseball_zerobase import __version__
from baseball_zerobase.baseline.behavior import EmpiricalBehaviorModel
from baseball_zerobase.baseline.transition import EmpiricalTransitionModel
from baseball_zerobase.data.contracts import OutcomeLabel, TransitionAtom
from baseball_zerobase.data.manifest import manifest_path_for, sha256_file
from baseball_zerobase.data.splits import DatasetRole, guard_dev_path, require_dev_role
from baseball_zerobase.evaluation.metrics import (
    ActionRankingMetrics,
    action_ranking_metrics,
    inning_distribution_metrics,
    outcome_brier_score,
)
from baseball_zerobase.simulation.inning import InningSimulator
from baseball_zerobase.simulation.state import GameState


@dataclass(frozen=True, slots=True)
class Fold:
    train_years: tuple[int, ...]
    validation_year: int


@dataclass(frozen=True, slots=True)
class FoldEvaluationReport:
    train_years: tuple[int, ...]
    validation_year: int
    training_row_count: int
    validation_row_count: int
    dataset_manifest_hash: str
    code_version: str
    behavior_top1_accuracy: float
    behavior_top3_accuracy: float
    behavior_negative_log_likelihood: float
    transition_negative_log_likelihood: float
    transition_outcome_brier_score: float
    behavior_backoff_distribution: dict[str, int]
    transition_backoff_distribution: dict[str, int]
    simulated_mean_runs: float
    actual_mean_runs: float
    simulated_vs_actual_mean_run_error: float
    inning_run_wasserstein_distance: float
    zero_run_probability_error: float
    multi_run_probability_error: float
    simulation_truncation_rate: float
    simulated_inning_count: int


@dataclass(frozen=True, slots=True)
class RollingEvaluationSummary:
    dataset_path: Path
    output_dir: Path
    dataset_manifest_hash: str
    code_version: str
    fold_reports: tuple[FoldEvaluationReport, ...]
    markdown_path: Path


ROLLING_FOLDS = (
    Fold(train_years=(2022,), validation_year=2023),
    Fold(train_years=(2022, 2023), validation_year=2024),
    Fold(train_years=(2022, 2023, 2024), validation_year=2025),
)


def rolling_folds() -> tuple[Fold, ...]:
    return ROLLING_FOLDS


def evaluate_rolling(
    dataset: Path,
    output_dir: Path,
    *,
    project_root: Path | None = None,
    trials: int = 100,
    seed: int = 20240604,
) -> RollingEvaluationSummary:
    require_dev_role(DatasetRole.DEV_REGULAR)
    dataset_path = dataset.resolve()
    report_dir = output_dir.resolve()
    root = project_root.resolve() if project_root is not None else _infer_project_root(dataset_path)
    locked_dir = root / "data" / "locked"
    guard_dev_path(dataset_path, locked_dir)
    guard_dev_path(report_dir, locked_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest_hash = _dataset_manifest_hash(dataset_path)
    frame = pl.read_parquet(dataset_path)
    reports = tuple(
        evaluate_fold(
            frame,
            train_years=fold.train_years,
            validation_year=fold.validation_year,
            trials=trials,
            seed=seed + fold.validation_year,
            dataset_manifest_hash=dataset_manifest_hash,
            code_version=__version__,
        )
        for fold in rolling_folds()
    )
    for report in reports:
        _write_fold_report(report, report_dir / _fold_report_name(report))

    markdown_path = report_dir / "rolling_summary.md"
    markdown_path.write_text(_markdown_summary(reports, dataset_path, dataset_manifest_hash), encoding="utf-8")
    return RollingEvaluationSummary(
        dataset_path=dataset_path,
        output_dir=report_dir,
        dataset_manifest_hash=dataset_manifest_hash,
        code_version=__version__,
        fold_reports=reports,
        markdown_path=markdown_path,
    )


def evaluate_fold(
    frame: object,
    *,
    train_years: Sequence[int],
    validation_year: int,
    trials: int = 100,
    seed: int = 20240604,
    dataset_manifest_hash: str = "unknown",
    code_version: str = __version__,
) -> FoldEvaluationReport:
    if not train_years:
        raise ValueError("train_years must not be empty")
    if validation_year in train_years:
        raise ValueError("validation_year cannot appear in train_years")

    rows = _iter_rows(frame)
    train_year_set = {int(year) for year in train_years}
    train_rows = [row for row in rows if _row_year(row) in train_year_set]
    validation_rows = [row for row in rows if _row_year(row) == validation_year]
    if not train_rows:
        raise ValueError(f"no training rows for years {tuple(sorted(train_year_set))}")
    if not validation_rows:
        raise ValueError(f"no validation rows for year {validation_year}")

    behavior_metrics = ActionRankingMetrics(0.0, 0.0, 0.0, 0)
    transition_nll = 0.0
    transition_brier = 0.0
    behavior_backoffs: Counter[str] = Counter()
    transition_backoffs: Counter[str] = Counter()
    predicted_runs: list[int] = []
    actual_runs: list[int] = []
    truncated_trials = 0
    simulated_trials = 0

    train_frame = pl.DataFrame(train_rows)
    behavior_model = EmpiricalBehaviorModel().fit(
        train_frame,
        training_manifest_hash=dataset_manifest_hash,
    )
    transition_model = EmpiricalTransitionModel().fit(
        train_frame,
        training_manifest_hash=dataset_manifest_hash,
    )
    behavior_metrics, behavior_backoffs = _evaluate_behavior(behavior_model, validation_rows)
    transition_nll, transition_brier, transition_backoffs = _evaluate_transitions(
        transition_model,
        validation_rows,
    )
    (
        predicted_runs,
        actual_runs,
        truncated_trials,
        simulated_trials,
    ) = _simulate_validation_innings(
        behavior_model,
        transition_model,
        validation_rows,
        trials=trials,
        seed=seed,
    )

    inning_metrics = inning_distribution_metrics(predicted=predicted_runs, actual=actual_runs)
    return FoldEvaluationReport(
        train_years=tuple(int(year) for year in train_years),
        validation_year=int(validation_year),
        training_row_count=len(train_rows),
        validation_row_count=len(validation_rows),
        dataset_manifest_hash=dataset_manifest_hash,
        code_version=code_version,
        behavior_top1_accuracy=behavior_metrics.top1_accuracy,
        behavior_top3_accuracy=behavior_metrics.top3_accuracy,
        behavior_negative_log_likelihood=behavior_metrics.negative_log_likelihood,
        transition_negative_log_likelihood=transition_nll,
        transition_outcome_brier_score=transition_brier,
        behavior_backoff_distribution=dict(sorted(behavior_backoffs.items())),
        transition_backoff_distribution=dict(sorted(transition_backoffs.items())),
        simulated_mean_runs=inning_metrics.predicted_mean_runs,
        actual_mean_runs=inning_metrics.actual_mean_runs,
        simulated_vs_actual_mean_run_error=inning_metrics.mean_run_error,
        inning_run_wasserstein_distance=inning_metrics.wasserstein_distance,
        zero_run_probability_error=inning_metrics.zero_run_probability_error,
        multi_run_probability_error=inning_metrics.multi_run_probability_error,
        simulation_truncation_rate=truncated_trials / simulated_trials if simulated_trials else 0.0,
        simulated_inning_count=len(actual_runs),
    )


def _evaluate_behavior(
    model: EmpiricalBehaviorModel,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[ActionRankingMetrics, Counter[str]]:
    predicted: list[Mapping[Any, float]] = []
    actual: list[object] = []
    backoffs: Counter[str] = Counter()
    for row in rows:
        action = _action_label(row)
        context = _behavior_context(row)
        if action is None or context is None:
            continue
        probabilities = model.predict_proba(**context)
        if model.last_backoff_level is not None:
            backoffs[model.last_backoff_level] += 1
        predicted.append(probabilities)
        actual.append(action)
    return action_ranking_metrics(predicted=predicted, actual=actual), backoffs


def _evaluate_transitions(
    model: EmpiricalTransitionModel,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, float, Counter[str]]:
    losses: list[float] = []
    outcome_probabilities: list[dict[str, float]] = []
    actual_outcomes: list[str] = []
    backoffs: Counter[str] = Counter()
    for row in rows:
        atom = _transition_atom(row)
        context = _transition_context(row)
        if atom is None or context is None:
            continue
        distribution = model.predict_distribution(**context)
        if model.last_backoff_level is not None:
            backoffs[model.last_backoff_level] += 1
        losses.append(-model.log_probability(atom, context))
        outcome_probabilities.append(_outcome_distribution(distribution))
        actual_outcomes.append(atom.outcome.value)
    return (
        sum(losses) / len(losses) if losses else 0.0,
        outcome_brier_score(predicted=outcome_probabilities, actual=actual_outcomes)
        if outcome_probabilities
        else 0.0,
        backoffs,
    )


def _simulate_validation_innings(
    behavior_model: EmpiricalBehaviorModel,
    transition_model: EmpiricalTransitionModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    trials: int,
    seed: int,
) -> tuple[list[int], list[int], int, int]:
    simulator = InningSimulator(cast(Any, behavior_model), cast(Any, transition_model))
    predicted_runs: list[int] = []
    actual_runs: list[int] = []
    truncated_trials = 0
    simulated_trials = 0
    for index, inning_rows in enumerate(_half_inning_groups(rows).values()):
        if not inning_rows:
            continue
        initial_state = _game_state(inning_rows[0])
        if initial_state is None:
            continue
        actual_runs.append(sum(_int(row.get("runs_scored"), default=0) for row in inning_rows))
        result = simulator.simulate_many(initial_state, trials=trials, seed=seed + index)
        predicted_runs.extend(result.runs)
        truncated_trials += result.truncated_trials
        simulated_trials += trials
    return predicted_runs, actual_runs, truncated_trials, simulated_trials


def _half_inning_groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[object, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("game_pk"),
            row.get("inning"),
            row.get("inning_topbot", "half"),
        )
        groups[key].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=_sort_key)
    return dict(groups)


def _game_state(row: Mapping[str, Any]) -> GameState | None:
    lineup_ids = _lineup(row.get("lineup_ids"))
    lineup_stands = tuple(str(value) for value in _lineup(row.get("lineup_stands")))
    slot = _int(row.get("batting_order_slot"), default=1) - 1
    if not lineup_ids or len(lineup_ids) != len(lineup_stands) or not 0 <= slot < len(lineup_ids):
        return None
    try:
        return GameState(
            balls=_int(row.get("balls"), default=0),
            strikes=_int(row.get("strikes"), default=0),
            outs=_int(row.get("outs"), default=0),
            runners=_int(row.get("runners"), default=0),
            inning=max(_int(row.get("inning"), default=1), 1),
            score_diff=_int(row.get("score_diff"), default=0),
            batting_order_index=slot,
            lineup_ids=tuple(_int(value, default=0) for value in lineup_ids),
            lineup_stands=lineup_stands,
            stand=str(row.get("stand") or lineup_stands[slot]),
            p_throws=str(row.get("p_throws") or row.get("pitcher_throws") or "R"),
        )
    except (TypeError, ValueError):
        return None


def _write_fold_report(report: FoldEvaluationReport, path: Path) -> None:
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_summary(
    reports: Sequence[FoldEvaluationReport],
    dataset_path: Path,
    dataset_manifest_hash: str,
) -> str:
    lines = [
        "# Rolling Baseline Evaluation",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Dataset manifest hash: `{dataset_manifest_hash}`",
        f"- Code version: `{__version__}`",
        "",
        "## Fold Summary",
        "",
        "| Train years | Validation year | Rows | Action Top-1 | Transition NLL | Mean run error | Truncation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        lines.append(
            "| "
            f"{', '.join(str(year) for year in report.train_years)} | "
            f"{report.validation_year} | "
            f"{report.validation_row_count} | "
            f"{report.behavior_top1_accuracy:.3f} | "
            f"{report.transition_negative_log_likelihood:.3f} | "
            f"{report.simulated_vs_actual_mean_run_error:.3f} | "
            f"{report.simulation_truncation_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Korean Summary / 한국어 요약",
            "",
            "이 보고서는 2022-2025 정규시즌 개발 데이터만 사용해 고정 롤링 검증을 수행합니다.",
            "각 폴드는 검증 연도보다 과거인 학습 연도만 사용하므로 미래 정보 누수를 피합니다.",
            "표의 Action Top-1, 전이 NLL, 득점 분포 오차, 시뮬레이션 절단률을 검토해 "
            "경험적 베이스라인의 안정성을 확인하세요.",
            "",
        ]
    )
    return "\n".join(lines)


def _fold_report_name(report: FoldEvaluationReport) -> str:
    train = "_".join(str(year) for year in report.train_years)
    return f"fold_{train}_to_{report.validation_year}.json"


def _dataset_manifest_hash(dataset_path: Path) -> str:
    actual_hash = sha256_file(dataset_path)
    manifest_path = manifest_path_for(dataset_path)
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("sha256"), str):
            manifest_hash = payload["sha256"]
            if manifest_hash != actual_hash:
                raise ValueError(
                    "dataset manifest sha256 does not match dataset bytes: "
                    f"{manifest_hash} != {actual_hash}"
                )
            return manifest_hash
    return actual_hash


def _infer_project_root(dataset_path: Path) -> Path:
    parts = dataset_path.resolve().parts
    if "data" in parts:
        data_index = len(parts) - 1 - tuple(reversed(parts)).index("data")
        if data_index > 0:
            return Path(*parts[:data_index])
    return Path.cwd()


def _iter_rows(frame: object) -> list[Mapping[str, Any]]:
    to_dicts = getattr(frame, "to_dicts", None)
    if callable(to_dicts):
        rows = to_dicts()
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    if isinstance(frame, list):
        return [row for row in frame if isinstance(row, Mapping)]
    return [row for row in frame if isinstance(row, Mapping)]  # type: ignore[operator]


def _row_year(row: Mapping[str, Any]) -> int | None:
    value = row.get("game_date")
    if isinstance(value, datetime):
        return value.year
    if isinstance(value, date):
        return value.year
    if value is None:
        return None
    return date.fromisoformat(str(value)[:10]).year


def _behavior_context(row: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        return {
            "balls": _int(row.get("balls"), default=0),
            "strikes": _int(row.get("strikes"), default=0),
            "stand": str(row.get("stand") or "R"),
            "p_throws": str(row.get("p_throws") or row.get("pitcher_throws") or "R"),
        }
    except ValueError:
        return None


def _transition_context(row: Mapping[str, Any]) -> dict[str, Any] | None:
    pitch_type = row.get("pitch_type")
    relative_zone = row.get("relative_zone")
    if pitch_type is None or relative_zone is None:
        return None
    behavior_context = _behavior_context(row)
    if behavior_context is None:
        return None
    try:
        return {
            **behavior_context,
            "pitch_type": str(pitch_type),
            "relative_zone": str(relative_zone),
            "outs": _int(row.get("outs"), default=0),
            "runners": _int(row.get("runners"), default=0),
        }
    except ValueError:
        return None


def _transition_atom(row: Mapping[str, Any]) -> TransitionAtom | None:
    try:
        return TransitionAtom(
            outcome=OutcomeLabel(str(row.get("outcome"))),
            balls_after=_int(row.get("balls_after"), default=0),
            strikes_after=_int(row.get("strikes_after"), default=0),
            outs_after=_int(row.get("outs_after"), default=0),
            runners_after=_runners_tuple(row.get("runners_after")),
            runs_scored=_int(row.get("runs_scored"), default=0),
            plate_appearance_ended=_bool(row.get("plate_appearance_ended")),
            half_inning_ended=_bool(row.get("half_inning_ended")),
            terminal_reason=None
            if row.get("terminal_reason") is None
            else str(row.get("terminal_reason")),
        )
    except (TypeError, ValueError):
        return None


def _outcome_distribution(distribution: Mapping[TransitionAtom, float]) -> dict[str, float]:
    probabilities: dict[str, float] = defaultdict(float)
    for atom, probability in distribution.items():
        probabilities[atom.outcome.value] += probability
    return dict(probabilities)


def _action_label(row: Mapping[str, Any]) -> str | None:
    action = row.get("action")
    if action is not None:
        return str(action)
    pitch_type = row.get("pitch_type")
    relative_zone = row.get("relative_zone")
    if pitch_type is None or relative_zone is None:
        return None
    return f"{pitch_type}:{relative_zone}"


def _runners_tuple(value: object) -> tuple[bool, bool, bool]:
    if isinstance(value, tuple | list):
        if len(value) != 3:
            raise ValueError("runners_after must have three entries")
        return tuple(bool(entry) for entry in value)  # type: ignore[return-value]
    runner_bits = _int(value, default=0)
    if not 0 <= runner_bits <= 7:
        raise ValueError("runners_after must be a 3-bit mask")
    return (bool(runner_bits & 1), bool(runner_bits & 2), bool(runner_bits & 4))


def _lineup(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple | list):
        return tuple(value)
    return ()


def _sort_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        row.get("pitch_timestamp") or datetime.min,
        _int(row.get("at_bat_number"), default=0),
        _int(row.get("pitch_number"), default=0),
    )


def _int(value: object, *, default: int) -> int:
    if value is None:
        return default
    return int(value)  # type: ignore[arg-type]


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).lower()
    if normalized in {"true", "t", "1"}:
        return True
    if normalized in {"false", "f", "0"}:
        return False
    raise ValueError("value cannot be coerced to bool")
