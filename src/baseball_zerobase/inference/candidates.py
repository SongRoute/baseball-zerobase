from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from baseball_zerobase.data.contracts import RelativeZone
from baseball_zerobase.inference.schemas import PitchCandidate


def generate_candidate_grid(pitch_types: Iterable[object] | object) -> tuple[PitchCandidate, ...]:
    normalized_pitch_types = normalize_pitch_types(pitch_types)
    return tuple(
        PitchCandidate(pitch_type=pitch_type, relative_zone=zone.value)
        for pitch_type in normalized_pitch_types
        for zone in RelativeZone
    )


def resolve_candidate_pitch_types(
    row: Mapping[str, Any],
    explicit_pitch_types: Iterable[object] | object | None,
) -> tuple[str, ...]:
    if explicit_pitch_types is not None:
        return normalize_pitch_types(explicit_pitch_types)
    for column in ("pitcher_owned_pitch_types", "eligible_pitch_types"):
        if column in row and row[column] is not None:
            return normalize_pitch_types(row[column])
    if row.get("pitch_type") is not None:
        return normalize_pitch_types([row["pitch_type"]])
    raise ValueError("pitch types are required for recommendation candidates")


def normalize_pitch_types(value: Iterable[object] | object) -> tuple[str, ...]:
    raw_values = _flatten_pitch_type_values(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        pitch_type = _normalize_pitch_type(raw_value)
        if pitch_type in seen:
            continue
        normalized.append(pitch_type)
        seen.add(pitch_type)
    if not normalized:
        raise ValueError("at least one pitch type is required")
    return tuple(normalized)


def _flatten_pitch_type_values(value: Iterable[object] | object) -> list[object]:
    if isinstance(value, str):
        return [part for part in value.split(",")]
    if isinstance(value, Mapping):
        raise ValueError("pitch types must be a string or sequence, not a mapping")
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalize_pitch_type(value: object) -> str:
    text = str(value).strip().upper()
    if not text:
        raise ValueError("pitch type values must be non-empty")
    if ":" in text or "," in text:
        raise ValueError("pitch type values cannot contain ':' or ','")
    return text
