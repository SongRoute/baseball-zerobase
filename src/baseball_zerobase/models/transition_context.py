from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping


LABEL_COLUMNS = {
    "outcome",
    "balls_after",
    "strikes_after",
    "outs_after",
    "runners_after",
    "runs_scored",
    "plate_appearance_ended",
    "half_inning_ended",
    "terminal_reason",
    "transition_atom",
    "zone",
}
_CONTEXT_COLUMNS = {
    "pitch_type",
    "relative_zone",
    "balls",
    "strikes",
    "outs",
    "runners",
    "stand",
    "p_throws",
    "pitch_timestamp",
    "as_of_timestamp",
    "game_pk",
    "game_date",
    "game_type",
}


@dataclass(frozen=True, slots=True)
class TransitionContext:
    pitch_type: str
    relative_zone: str
    balls: int
    strikes: int
    outs: int
    runners: int
    stand: str | None = None
    p_throws: str | None = None
    features: Mapping[str, object] = MappingProxyType({})


def transition_context_from_row(row: Mapping[str, Any]) -> TransitionContext:
    _reject_feature_timestamp_leakage(row)
    features = {
        str(key): value
        for key, value in row.items()
        if key not in LABEL_COLUMNS and key not in _CONTEXT_COLUMNS
    }
    return TransitionContext(
        pitch_type=str(row["pitch_type"]),
        relative_zone=str(row["relative_zone"]),
        balls=int(row["balls"]),
        strikes=int(row["strikes"]),
        outs=int(row["outs"]),
        runners=_runner_mask(row.get("runners", 0)),
        stand=_string_or_none(row.get("stand")),
        p_throws=_string_or_none(row.get("p_throws")),
        features=MappingProxyType(features),
    )


def _runner_mask(value: object) -> int:
    if isinstance(value, tuple):
        mask = 0
        for index, occupied in enumerate(value):
            if bool(occupied):
                mask |= 1 << index
        return mask
    if value is None:
        return 0
    return int(str(value))


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _reject_feature_timestamp_leakage(row: Mapping[str, Any]) -> None:
    if "pitch_timestamp" not in row:
        return
    pitch_timestamp = _datetime_or_none(row["pitch_timestamp"])
    if pitch_timestamp is None:
        return
    for key, value in row.items():
        if not str(key).endswith("_as_of_timestamp") or key == "as_of_timestamp":
            continue
        feature_timestamp = _datetime_or_none(value)
        if feature_timestamp is None or feature_timestamp >= pitch_timestamp:
            raise ValueError(f"feature timestamp {key} must be before pitch_timestamp")


def _datetime_or_none(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
