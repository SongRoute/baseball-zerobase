from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from baseball_zerobase.data.contracts import OutcomeLabel, TransitionAtom
from baseball_zerobase.inference.candidates import (
    generate_candidate_grid,
    normalize_pitch_types,
    resolve_candidate_pitch_types,
)
from baseball_zerobase.inference.schemas import (
    PitchCandidate,
    PitchRecommendation,
    RecommendationReport,
)
from baseball_zerobase.models.transition import SharedTransitionModelV0
from baseball_zerobase.models.transition_context import (
    TransitionContext,
    transition_context_from_row,
)


_VALUE_TYPE = "transition_proxy"
_ZONE_FILTERING = "disabled"
_FORBIDDEN_SERVING_COLUMNS = {
    "zone",
    "plate_x",
    "plate_z",
    "release_speed",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
}
_CANDIDATE_DEPENDENT_PROFILE_COLUMNS = {
    "pitcher_profile_pitch_type_prior_count",
    "pitcher_profile_pitch_type_usage_rate",
}
_CANDIDATE_DEPENDENT_PROFILE_PREFIXES = (
    "pitcher_profile_release_",
    "pitcher_profile_pfx_",
)
_ON_BASE_DAMAGE_OUTCOMES = {
    OutcomeLabel.SINGLE,
    OutcomeLabel.DOUBLE,
    OutcomeLabel.TRIPLE,
    OutcomeLabel.HOME_RUN,
    OutcomeLabel.WALK,
    OutcomeLabel.HBP,
    OutcomeLabel.REACH_OTHER,
}
_EXTRA_BASE_OR_HOME_RUN_OUTCOMES = {
    OutcomeLabel.DOUBLE,
    OutcomeLabel.TRIPLE,
    OutcomeLabel.HOME_RUN,
}
_STRIKE_OUTCOMES = {
    OutcomeLabel.CALLED_STRIKE,
    OutcomeLabel.SWINGING_STRIKE,
    OutcomeLabel.FOUL,
    OutcomeLabel.STRIKEOUT,
}
_BALL_OUTCOMES = {
    OutcomeLabel.BALL,
    OutcomeLabel.WALK,
    OutcomeLabel.HBP,
}


def recommend_pitches(
    model: SharedTransitionModelV0,
    row: Mapping[str, Any],
    *,
    pitch_types: Iterable[object] | object | None = None,
    top_k: int | None = 10,
    top_outcomes: int = 5,
) -> RecommendationReport:
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when provided")
    if top_outcomes < 1:
        raise ValueError("top_outcomes must be positive")

    _reject_base_timestamp_leakage(row)
    _reject_forbidden_serving_columns(row)
    resolved_pitch_types = resolve_candidate_pitch_types(row, pitch_types)
    candidates = generate_candidate_grid(resolved_pitch_types)
    scored = [
        _score_candidate(
            model,
            row,
            candidate,
            top_outcomes=top_outcomes,
        )
        for candidate in candidates
    ]
    scored.sort(
        key=lambda item: (
            item.ranking_score,
            item.pitch_type,
            item.relative_zone,
        )
    )
    selected = scored if top_k is None else scored[:top_k]
    ranked = tuple(
        PitchRecommendation(
            rank=index,
            pitch_type=item.pitch_type,
            relative_zone=item.relative_zone,
            action=item.action,
            ranking_score=item.ranking_score,
            value_type=item.value_type,
            explanation=item.explanation,
            transition_distribution=item.transition_distribution,
        )
        for index, item in enumerate(selected, start=1)
    )
    return RecommendationReport(
        value_type=_VALUE_TYPE,
        zone_filtering=_ZONE_FILTERING,
        candidate_count=len(candidates),
        pitch_types=resolved_pitch_types,
        model_training_manifest_hash=model.training_manifest_hash,
        recommendations=ranked,
        input_summary={
            "candidate_pitch_type_source": _candidate_pitch_type_source(row, pitch_types),
            "zones_per_pitch_type": 13,
        },
    )


def _score_candidate(
    model: SharedTransitionModelV0,
    row: Mapping[str, Any],
    candidate: PitchCandidate,
    *,
    top_outcomes: int,
) -> PitchRecommendation:
    candidate_row = _candidate_row(row, candidate)
    context = transition_context_from_row(candidate_row)
    distribution = model.predict_distribution(context)
    score_parts = _score_distribution(context, distribution)
    support = model.support(context)
    distribution_records = tuple(
        _atom_record(atom, probability)
        for atom, probability in sorted(
            distribution.items(),
            key=lambda item: (-item[1], _atom_sort_key(item[0])),
        )
    )
    explanation = {
        **score_parts,
        "support": support,
        "confidence": "medium" if support >= 10 else "low",
        "pitcher_pitch_type_owned": bool(candidate_row.get("pitcher_pitch_type_owned", False)),
        "batter_weakness_archetype": candidate_row.get("batter_weakness_archetype"),
        "batter_threat_score": candidate_row.get("batter_threat_score"),
        "top_transition_atoms": list(distribution_records[:top_outcomes]),
        "scoring_formula": "expected_runs_scored + p_unfavorable - p_favorable",
    }
    return PitchRecommendation(
        rank=0,
        pitch_type=candidate.pitch_type,
        relative_zone=candidate.relative_zone,
        action=candidate.action,
        ranking_score=float(score_parts["ranking_score"]),
        value_type=_VALUE_TYPE,
        explanation=explanation,
        transition_distribution=distribution_records,
    )


def _candidate_row(row: Mapping[str, Any], candidate: PitchCandidate) -> dict[str, Any]:
    candidate_row = dict(row)
    candidate_row["pitch_type"] = candidate.pitch_type
    candidate_row["relative_zone"] = candidate.relative_zone
    candidate_row["action"] = candidate.action
    _normalize_state_aliases(candidate_row)
    _drop_candidate_dependent_profile_features(candidate_row)
    owned_pitch_types = _owned_pitch_type_set(row.get("pitcher_owned_pitch_types"))
    if owned_pitch_types is not None:
        candidate_row["pitcher_pitch_type_owned"] = candidate.pitch_type in owned_pitch_types
    elif "pitcher_pitch_type_owned" in candidate_row:
        candidate_row["pitcher_pitch_type_owned"] = False
    return candidate_row


def _normalize_state_aliases(candidate_row: dict[str, Any]) -> None:
    if "outs" not in candidate_row and "outs_when_up" in candidate_row:
        candidate_row["outs"] = candidate_row["outs_when_up"]
    if "runners" in candidate_row:
        return
    runners = 0
    for bit, column in ((1, "on_1b"), (2, "on_2b"), (4, "on_3b")):
        if candidate_row.get(column) is not None:
            runners |= bit
    candidate_row["runners"] = runners


def _drop_candidate_dependent_profile_features(candidate_row: dict[str, Any]) -> None:
    for column in tuple(candidate_row):
        if column in _CANDIDATE_DEPENDENT_PROFILE_COLUMNS or column.startswith(
            _CANDIDATE_DEPENDENT_PROFILE_PREFIXES
        ):
            candidate_row.pop(column)


def _owned_pitch_type_set(value: object) -> set[str] | None:
    if value is None:
        return None
    try:
        return set(normalize_pitch_types(value))
    except ValueError as exc:
        if "at least one pitch type" in str(exc):
            return set()
        raise


def _reject_forbidden_serving_columns(row: Mapping[str, Any]) -> None:
    present = sorted(column for column in _FORBIDDEN_SERVING_COLUMNS if row.get(column) is not None)
    if present:
        raise ValueError(
            f"serving input cannot include current-pitch measurement columns: {present}"
        )


def _reject_base_timestamp_leakage(row: Mapping[str, Any]) -> None:
    if "as_of_timestamp" not in row:
        return
    pitch_timestamp = _datetime_or_none(row.get("pitch_timestamp"))
    if pitch_timestamp is None:
        return
    as_of_timestamp = _datetime_or_none(row.get("as_of_timestamp"))
    if as_of_timestamp is None or as_of_timestamp >= pitch_timestamp:
        raise ValueError("as_of_timestamp must be before pitch_timestamp")


def _datetime_or_none(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _score_distribution(
    context: TransitionContext,
    distribution: Mapping[TransitionAtom, float],
) -> dict[str, float]:
    expected_runs_scored = sum(
        probability * atom.runs_scored for atom, probability in distribution.items()
    )
    p_favorable = sum(
        probability for atom, probability in distribution.items() if _is_favorable(context, atom)
    )
    p_unfavorable = sum(
        probability for atom, probability in distribution.items() if _is_unfavorable(context, atom)
    )
    p_out_added = sum(
        probability for atom, probability in distribution.items() if atom.outs_after > context.outs
    )
    p_strike_added = sum(
        probability
        for atom, probability in distribution.items()
        if not atom.plate_appearance_ended and atom.strikes_after > context.strikes
    )
    p_ball_added = sum(
        probability
        for atom, probability in distribution.items()
        if not atom.plate_appearance_ended and atom.balls_after > context.balls
    )
    p_half_inning_ended = sum(
        probability for atom, probability in distribution.items() if atom.half_inning_ended
    )
    p_on_base_damage = sum(
        probability
        for atom, probability in distribution.items()
        if atom.outcome in _ON_BASE_DAMAGE_OUTCOMES
    )
    p_extra_base_or_home_run = sum(
        probability
        for atom, probability in distribution.items()
        if atom.outcome in _EXTRA_BASE_OR_HOME_RUN_OUTCOMES
    )
    p_home_run = sum(
        probability
        for atom, probability in distribution.items()
        if atom.outcome == OutcomeLabel.HOME_RUN
    )
    strike_probability = sum(
        probability
        for atom, probability in distribution.items()
        if atom.outcome in _STRIKE_OUTCOMES
    )
    ball_probability = sum(
        probability for atom, probability in distribution.items() if atom.outcome in _BALL_OUTCOMES
    )
    reach_probability = p_on_base_damage
    return {
        "ranking_score": expected_runs_scored + p_unfavorable - p_favorable,
        "expected_runs_scored": expected_runs_scored,
        "p_favorable": p_favorable,
        "p_unfavorable": p_unfavorable,
        "p_out_added": p_out_added,
        "p_strike_added": p_strike_added,
        "p_ball_added": p_ball_added,
        "p_half_inning_ended": p_half_inning_ended,
        "p_on_base_damage": p_on_base_damage,
        "p_extra_base_or_home_run": p_extra_base_or_home_run,
        "home_run_probability": p_home_run,
        "strike_probability": strike_probability,
        "ball_probability": ball_probability,
        "reach_probability": reach_probability,
    }


def _is_favorable(context: TransitionContext, atom: TransitionAtom) -> bool:
    return (
        atom.half_inning_ended
        or atom.outs_after > context.outs
        or (not atom.plate_appearance_ended and atom.strikes_after > context.strikes)
        or atom.outcome == OutcomeLabel.STRIKEOUT
    )


def _is_unfavorable(context: TransitionContext, atom: TransitionAtom) -> bool:
    return (
        atom.runs_scored > 0
        or atom.outcome in _ON_BASE_DAMAGE_OUTCOMES
        or (not atom.plate_appearance_ended and atom.balls_after > context.balls)
    )


def _atom_record(atom: TransitionAtom, probability: float) -> dict[str, object]:
    return {
        "probability": probability,
        "outcome": atom.outcome.value,
        "balls_after": atom.balls_after,
        "strikes_after": atom.strikes_after,
        "outs_after": atom.outs_after,
        "runners_after": list(atom.runners_after),
        "runs_scored": atom.runs_scored,
        "plate_appearance_ended": atom.plate_appearance_ended,
        "half_inning_ended": atom.half_inning_ended,
        "terminal_reason": atom.terminal_reason,
    }


def _atom_sort_key(atom: TransitionAtom) -> tuple[object, ...]:
    return (
        atom.outcome.value,
        atom.balls_after,
        atom.strikes_after,
        atom.outs_after,
        atom.runners_after,
        atom.runs_scored,
        atom.plate_appearance_ended,
        atom.half_inning_ended,
        atom.terminal_reason or "",
    )


def _candidate_pitch_type_source(
    row: Mapping[str, Any],
    explicit_pitch_types: Iterable[object] | object | None,
) -> str:
    if explicit_pitch_types is not None:
        return "explicit"
    if row.get("pitcher_owned_pitch_types") is not None:
        return "pitcher_owned_pitch_types"
    if row.get("eligible_pitch_types") is not None:
        return "eligible_pitch_types"
    return "row_pitch_type_fallback"
