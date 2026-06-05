from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import polars as pl

from baseball_zerobase.data.contracts import OutcomeLabel, RelativeZone
from baseball_zerobase.data.splits import DatasetRole, classify_row


class LeakageError(RuntimeError):
    pass


_JOIN_KEYS = ("game_pk", "at_bat_number", "pitch_number")
_LOCKED_ROLES = {
    DatasetRole.LOCKED_POSTSEASON_2025.value,
    DatasetRole.LOCKED_REGULAR_2026.value,
}
_REQUIRED_COLUMNS = {
    "action",
    "as_of_timestamp",
    "at_bat_number",
    "balls",
    "balls_after",
    "game_date",
    "game_pk",
    "game_type",
    "half_inning_ended",
    "is_official_starter_pitch",
    "lineup_stable",
    "outcome",
    "outs",
    "outs_after",
    "pitch_number",
    "pitch_timestamp",
    "pitch_type",
    "relative_zone",
    "runners",
    "runners_after",
    "runs_scored",
    "starter_eligible",
    "strikes",
    "strikes_after",
    "terminal_reason",
    "timestamp_joined",
}
_RELATIVE_ZONE_VALUES = {zone.value for zone in RelativeZone}
_SUPPORTED_OUTCOME_VALUES = {label.value for label in OutcomeLabel} - {OutcomeLabel.OTHER.value}
_TERMINAL_REASON_VALUES = {"game_end", "three_outs"}


@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    game_count: int
    action_row_count: int
    included_counts: dict[str, int]
    excluded_counts: dict[str, int]
    pitch_type_counts: dict[str, int]
    relative_zone_counts: dict[str, int]
    starter_eligible_counts: dict[str, int]
    transition_outcome_counts: dict[str, int]
    timestamp_join_rate: float
    unknown_action_rate: float
    unknown_outcome_rate: float
    action_counts: dict[str, int]
    outcome_counts: dict[str, int]
    terminal_reason_counts: dict[str, int]
    half_inning_ended_counts: dict[str, int]
    locked_row_count: int
    timestamp_joined_counts: dict[str, int]
    dataset_role_counts: dict[str, int]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_snapshots(frame: pl.DataFrame) -> ValidationReport:
    warnings: list[str] = []
    _reject_missing_required_columns(frame)
    _reject_duplicate_pitch_keys(frame)
    _reject_timestamp_leakage(frame)
    _reject_feature_timestamp_leakage(frame)

    dataset_role_counts = _dataset_role_counts(frame)
    locked_row_count = sum(
        count for role, count in dataset_role_counts.items() if role in _LOCKED_ROLES
    )
    if locked_row_count:
        locked_roles = sorted(role for role in _LOCKED_ROLES if dataset_role_counts.get(role, 0))
        raise LeakageError(
            "development dataset cannot contain locked rows; "
            f"locked roles: {locked_roles}, locked row count: {locked_row_count}"
        )

    _reject_count_domains(frame)
    _reject_relative_zones(frame)
    _reject_starter_context(frame)
    _reject_transition_fields(frame)
    _reject_terminal_fields(frame)
    _reject_outcome_values(frame)
    _reject_action_values(frame)

    action_frame = _action_frame(frame, warnings)
    unknown_action_count = _unknown_action_count(frame)
    unknown_outcome_count = _unknown_outcome_count(frame)
    included_counts = _included_counts(
        frame,
        dataset_role_counts=dataset_role_counts,
        unknown_action_count=unknown_action_count,
        unknown_outcome_count=unknown_outcome_count,
    )

    return ValidationReport(
        row_count=frame.height,
        game_count=_unique_count(frame, "game_pk"),
        action_row_count=action_frame.height,
        included_counts=included_counts,
        excluded_counts=_excluded_counts(frame, included_counts, locked_row_count),
        pitch_type_counts=_value_counts(frame, "pitch_type", warnings=warnings),
        relative_zone_counts=_value_counts(action_frame, "relative_zone"),
        starter_eligible_counts=_value_counts(frame, "starter_eligible", warnings=warnings),
        transition_outcome_counts=_value_counts(frame, "outcome", warnings=warnings),
        timestamp_join_rate=_rate(_true_count(frame, "timestamp_joined"), frame.height),
        unknown_action_rate=_rate(unknown_action_count, frame.height),
        unknown_outcome_rate=_rate(unknown_outcome_count, frame.height),
        action_counts=_value_counts(action_frame, "action"),
        outcome_counts=_value_counts(frame, "outcome", warnings=warnings),
        terminal_reason_counts=_value_counts(frame, "terminal_reason", warnings=warnings),
        half_inning_ended_counts=_value_counts(frame, "half_inning_ended", warnings=warnings),
        locked_row_count=locked_row_count,
        timestamp_joined_counts=_value_counts(frame, "timestamp_joined", warnings=warnings),
        dataset_role_counts=dataset_role_counts,
        warnings=tuple(warnings),
    )


def _reject_missing_required_columns(frame: pl.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise LeakageError(f"validation frame is missing required columns: {missing}")


def _reject_duplicate_pitch_keys(frame: pl.DataFrame) -> None:
    seen: set[tuple[object, ...]] = set()
    duplicates: list[str] = []
    malformed: list[str] = []
    for index, row in enumerate(frame.select(_JOIN_KEYS).iter_rows(named=True)):
        key = tuple(row[column] for column in _JOIN_KEYS)
        if any(value is None for value in key):
            malformed.append(str(index))
            continue
        if key in seen:
            duplicates.append(str(key))
        seen.add(key)

    if malformed:
        raise LeakageError(
            f"snapshot join keys cannot be null; malformed row indexes: {_sample(malformed)}"
        )
    if duplicates:
        raise LeakageError(
            "snapshot rows must be unique by (game_pk, at_bat_number, pitch_number); "
            f"duplicate keys: {_sample(duplicates)}"
        )


def _reject_timestamp_leakage(frame: pl.DataFrame) -> None:
    missing = sorted({"as_of_timestamp", "pitch_timestamp"}.difference(frame.columns))
    if missing:
        raise LeakageError(f"missing leakage timestamp columns: {missing}")

    malformed_rows: list[str] = []
    leaky_keys: list[str] = []
    for index, row in enumerate(
        frame.select(["as_of_timestamp", "pitch_timestamp"]).iter_rows(named=True)
    ):
        as_of_timestamp = _datetime_or_none(row["as_of_timestamp"])
        pitch_timestamp = _datetime_or_none(row["pitch_timestamp"])
        if as_of_timestamp is None or pitch_timestamp is None:
            malformed_rows.append(str(index))
            continue
        if as_of_timestamp >= pitch_timestamp:
            leaky_keys.append(str(index))

    if malformed_rows:
        raise LeakageError(
            "snapshot rows must have non-null parseable as_of_timestamp and pitch_timestamp; "
            f"malformed row indexes: {_sample(malformed_rows)}"
        )
    if leaky_keys:
        raise LeakageError(
            "snapshot rows must have as_of_timestamp strictly before pitch_timestamp "
            f"when both exist; leaky row indexes: {_sample(leaky_keys)}"
        )


def _reject_feature_timestamp_leakage(frame: pl.DataFrame) -> None:
    feature_columns = [
        column
        for column in frame.columns
        if column.endswith("_as_of_timestamp") and column != "as_of_timestamp"
    ]
    if not feature_columns:
        return

    malformed_rows: list[str] = []
    leaky_rows: list[str] = []
    selected_columns = ["pitch_timestamp", *feature_columns]
    for index, row in enumerate(frame.select(selected_columns).iter_rows(named=True)):
        pitch_timestamp = _datetime_or_none(row["pitch_timestamp"])
        if pitch_timestamp is None:
            malformed_rows.append(str(index))
            continue
        for column in feature_columns:
            feature_timestamp = _datetime_or_none(row[column])
            if feature_timestamp is None:
                malformed_rows.append(f"{index}:{column}")
                continue
            if feature_timestamp >= pitch_timestamp:
                leaky_rows.append(f"{index}:{column}")

    if malformed_rows:
        raise LeakageError(
            "profile feature timestamp columns must be non-null and parseable; "
            f"malformed row indexes: {_sample(malformed_rows)}"
        )
    if leaky_rows:
        raise LeakageError(
            "profile feature timestamp columns must be strictly before pitch_timestamp; "
            f"leaky feature timestamp rows: {_sample(leaky_rows)}"
        )


def _dataset_role_counts(frame: pl.DataFrame) -> dict[str, int]:
    missing = sorted({"game_date", "game_type"}.difference(frame.columns))
    if missing:
        raise LeakageError(f"missing dataset role columns: {missing}")

    counts: dict[str, int] = {}
    for index, row in enumerate(frame.select(["game_date", "game_type"]).iter_rows(named=True)):
        try:
            game_date = _date_value(row["game_date"])
            game_type = "" if row["game_type"] is None else str(row["game_type"])
        except (TypeError, ValueError) as exc:
            raise LeakageError(f"invalid dataset role fields at row index {index}") from exc
        role = classify_row(game_date, game_type).value
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def _reject_count_domains(frame: pl.DataFrame) -> None:
    domains = {
        "balls": (0, 3),
        "balls_after": (0, 3),
        "outs": (0, 2),
        "outs_after": (0, 3),
        "runners": (0, 7),
        "runners_after": (0, 7),
        "strikes": (0, 2),
        "strikes_after": (0, 2),
    }
    for column, (minimum, maximum) in domains.items():
        bad_rows = [
            str(index)
            for index, row in enumerate(frame.select(column).iter_rows(named=True))
            if not _integer_in_range(row[column], minimum, maximum)
        ]
        if bad_rows:
            raise LeakageError(
                f"{column} values must be integers between {minimum} and {maximum}; "
                f"invalid row indexes: {_sample(bad_rows)}"
            )


def _reject_relative_zones(frame: pl.DataFrame) -> None:
    bad_rows = [
        str(index)
        for index, row in enumerate(frame.select("relative_zone").iter_rows(named=True))
        if row["relative_zone"] not in _RELATIVE_ZONE_VALUES
    ]
    if bad_rows:
        raise LeakageError(
            "relative_zone values must match RelativeZone values; "
            f"invalid row indexes: {_sample(bad_rows)}"
        )


def _reject_starter_context(frame: pl.DataFrame) -> None:
    bad_official_rows = [
        str(index)
        for index, row in enumerate(frame.select("is_official_starter_pitch").iter_rows(named=True))
        if _bool_value(row["is_official_starter_pitch"]) is not True
    ]
    if bad_official_rows:
        raise LeakageError(
            "main development dataset rows must be official starter pitches; "
            f"invalid row indexes: {_sample(bad_official_rows)}"
        )

    if {"pitcher_id", "expected_starter_id"}.issubset(frame.columns):
        mismatched_rows: list[str] = []
        for index, row in enumerate(
            frame.select(["pitcher_id", "expected_starter_id"]).iter_rows(named=True)
        ):
            pitcher_id = row["pitcher_id"]
            expected_starter_id = row["expected_starter_id"]
            if pitcher_id is None or expected_starter_id is None:
                continue
            if _integer_value(pitcher_id) != _integer_value(expected_starter_id):
                mismatched_rows.append(str(index))
        if mismatched_rows:
            raise LeakageError(
                "main development dataset official starter mismatch; "
                f"invalid row indexes: {_sample(mismatched_rows)}"
            )

    bad_lineup_rows = [
        str(index)
        for index, row in enumerate(frame.select("lineup_stable").iter_rows(named=True))
        if _bool_value(row["lineup_stable"]) is not True
    ]
    if bad_lineup_rows:
        raise LeakageError(
            "main development dataset rows must have lineup_stable true; "
            f"invalid row indexes: {_sample(bad_lineup_rows)}"
        )


def _reject_transition_fields(frame: pl.DataFrame) -> None:
    negative_runs_rows = [
        str(index)
        for index, row in enumerate(frame.select("runs_scored").iter_rows(named=True))
        if (runs_scored := _integer_value(row["runs_scored"])) is None or runs_scored < 0
    ]
    if negative_runs_rows:
        raise LeakageError(
            f"runs_scored cannot be negative; invalid row indexes: {_sample(negative_runs_rows)}"
        )

    decreasing_out_rows = []
    for index, row in enumerate(frame.select(["outs", "outs_after"]).iter_rows(named=True)):
        outs = _integer_value(row["outs"])
        outs_after = _integer_value(row["outs_after"])
        if outs is None or outs_after is None or outs_after < outs:
            decreasing_out_rows.append(str(index))
    if decreasing_out_rows:
        raise LeakageError(
            "observed transition cannot decrease outs_after below outs; "
            f"invalid row indexes: {_sample(decreasing_out_rows)}"
        )


def _reject_terminal_fields(frame: pl.DataFrame) -> None:
    missing_reason_rows: list[str] = []
    invalid_reason_rows: list[str] = []
    for index, row in enumerate(
        frame.select(["half_inning_ended", "terminal_reason"]).iter_rows(named=True)
    ):
        half_inning_ended = _bool_value(row["half_inning_ended"])
        terminal_reason = _string_or_none(row["terminal_reason"])
        if half_inning_ended is None:
            invalid_reason_rows.append(str(index))
            continue
        if half_inning_ended and terminal_reason is None:
            missing_reason_rows.append(str(index))
            continue
        if not half_inning_ended and terminal_reason is not None:
            invalid_reason_rows.append(str(index))
            continue
        if terminal_reason is not None and terminal_reason not in _TERMINAL_REASON_VALUES:
            invalid_reason_rows.append(str(index))

    if missing_reason_rows:
        raise LeakageError(
            "half_inning_ended true rows must have terminal_reason; "
            f"invalid row indexes: {_sample(missing_reason_rows)}"
        )
    if invalid_reason_rows:
        raise LeakageError(
            "terminal_reason values must be one of "
            f"{sorted(_TERMINAL_REASON_VALUES)} when present; "
            f"invalid row indexes: {_sample(invalid_reason_rows)}"
        )


def _reject_outcome_values(frame: pl.DataFrame) -> None:
    bad_rows = [
        str(index)
        for index, row in enumerate(frame.select("outcome").iter_rows(named=True))
        if row["outcome"] not in _SUPPORTED_OUTCOME_VALUES
    ]
    if bad_rows:
        raise LeakageError(
            "outcome values must be supported OutcomeLabel values for strategic validation; "
            f"invalid row indexes: {_sample(bad_rows)}"
        )


def _reject_action_values(frame: pl.DataFrame) -> None:
    bad_rows: list[str] = []
    for index, row in enumerate(
        frame.select(["action", "pitch_type", "relative_zone"]).iter_rows(named=True)
    ):
        pitch_type = _string_or_none(row["pitch_type"])
        relative_zone = _string_or_none(row["relative_zone"])
        action = _string_or_none(row["action"])
        if pitch_type is None or relative_zone is None or action != f"{pitch_type}:{relative_zone}":
            bad_rows.append(str(index))
    if bad_rows:
        raise LeakageError(
            "action values must match '<pitch_type>:<relative_zone>' for strategic validation; "
            f"invalid row indexes: {_sample(bad_rows)}"
        )


def _action_frame(frame: pl.DataFrame, warnings: list[str]) -> pl.DataFrame:
    missing = sorted({"action", "relative_zone"}.difference(frame.columns))
    if missing:
        raise LeakageError(f"missing action distribution columns: {missing}")
    return frame.filter(pl.col("action").is_not_null() & pl.col("relative_zone").is_not_null())


def _value_counts(
    frame: pl.DataFrame,
    column: str,
    *,
    warnings: list[str] | None = None,
) -> dict[str, int]:
    if column not in frame.columns:
        if warnings is not None:
            warnings.append(f"missing count column: {column}")
        return {}

    counts: dict[str, int] = {}
    for row in frame.select(column).iter_rows(named=True):
        value = row[column]
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _date_value(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _datetime_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return datetime.fromisoformat(str(value))


def _integer_in_range(value: Any, minimum: int, maximum: int) -> bool:
    integer = _integer_value(value)
    return integer is not None and minimum <= integer <= maximum


def _integer_value(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(text)
    except ValueError:
        return None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _true_count(frame: pl.DataFrame, column: str) -> int:
    return sum(
        1 for row in frame.select(column).iter_rows(named=True) if _bool_value(row[column]) is True
    )


def _unique_count(frame: pl.DataFrame, column: str) -> int:
    values = {
        row[column] for row in frame.select(column).iter_rows(named=True) if row[column] is not None
    }
    return len(values)


def _unknown_action_count(frame: pl.DataFrame) -> int:
    count = 0
    for row in frame.select(["action", "pitch_type", "relative_zone"]).iter_rows(named=True):
        pitch_type = _string_or_none(row["pitch_type"])
        relative_zone = _string_or_none(row["relative_zone"])
        action = _string_or_none(row["action"])
        if pitch_type is None or relative_zone is None or action != f"{pitch_type}:{relative_zone}":
            count += 1
    return count


def _unknown_outcome_count(frame: pl.DataFrame) -> int:
    return sum(
        1
        for row in frame.select("outcome").iter_rows(named=True)
        if row["outcome"] not in _SUPPORTED_OUTCOME_VALUES
    )


def _included_counts(
    frame: pl.DataFrame,
    *,
    dataset_role_counts: dict[str, int],
    unknown_action_count: int,
    unknown_outcome_count: int,
) -> dict[str, int]:
    return {
        "dev_regular": dataset_role_counts.get(DatasetRole.DEV_REGULAR.value, 0),
        "known_action": frame.height - unknown_action_count,
        "known_outcome": frame.height - unknown_outcome_count,
        "lineup_stable": _true_count(frame, "lineup_stable"),
        "non_null_action": frame.height - _null_count(frame, "action"),
        "official_starter_pitch": _true_count(frame, "is_official_starter_pitch"),
        "starter_eligible": _true_count(frame, "starter_eligible"),
        "timestamp_joined": _true_count(frame, "timestamp_joined"),
    }


def _excluded_counts(
    frame: pl.DataFrame,
    included_counts: dict[str, int],
    locked_row_count: int,
) -> dict[str, int]:
    return {
        "locked": locked_row_count,
        "unknown_action": frame.height - included_counts["known_action"],
        "unknown_outcome": frame.height - included_counts["known_outcome"],
        "unstable_lineup": frame.height - included_counts["lineup_stable"],
        "missing_action": frame.height - included_counts["non_null_action"],
        "non_dev_regular": frame.height - included_counts["dev_regular"],
        "non_official_starter_pitch": frame.height - included_counts["official_starter_pitch"],
        "starter_ineligible": frame.height - included_counts["starter_eligible"],
        "timestamp_not_joined": frame.height - included_counts["timestamp_joined"],
    }


def _null_count(frame: pl.DataFrame, column: str) -> int:
    return sum(1 for row in frame.select(column).iter_rows(named=True) if row[column] is None)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _sample(values: list[str]) -> str:
    return ", ".join(values[:5])
