from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, Mapping, Never, Protocol

from baseball_zerobase.data.contracts import RelativeZone

ActionKey = tuple[str, str]
BackoffLevel = Literal["balls_strikes_stand_p_throws", "balls_strikes", "global"]
_FullKey = tuple[int, int, str, str]
_CountKey = tuple[int, int]


class _ChoiceRng(Protocol):
    def choice(self, a: int, *, p: list[float]) -> int: ...


@dataclass
class EmpiricalBehaviorModel:
    """Estimate observed starter pitch actions with simple empirical backoff."""

    min_support: int = 1
    alpha: float = 0.5
    last_backoff_level: BackoffLevel | None = field(default=None, init=False)
    training_manifest_hash: str | None = field(default=None, init=False)

    _actions: list[ActionKey] = field(default_factory=list, init=False)
    _global_counts: Counter[ActionKey] = field(default_factory=Counter, init=False)
    _full_counts: dict[_FullKey, Counter[ActionKey]] = field(
        default_factory=lambda: defaultdict(Counter),
        init=False,
    )
    _count_counts: dict[_CountKey, Counter[ActionKey]] = field(
        default_factory=lambda: defaultdict(Counter),
        init=False,
    )
    _fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.min_support < 1:
            raise ValueError("min_support must be at least 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

    def fit(self, frame: object, *, training_manifest_hash: str) -> EmpiricalBehaviorModel:
        if not _string_or_none(training_manifest_hash):
            raise ValueError("training_manifest_hash is required")

        self._actions = []
        self._global_counts = Counter()
        self._full_counts = defaultdict(Counter)
        self._count_counts = defaultdict(Counter)
        self.last_backoff_level = None
        self.training_manifest_hash = training_manifest_hash

        observed_actions: set[ActionKey] = set()
        for row in _iter_rows(frame):
            action = _action_from_row(row)
            if action is None:
                continue

            weight = _row_weight(row)
            if weight <= 0:
                continue

            observed_actions.add(action)
            self._global_counts[action] += weight

            count_key = _count_key_from_row(row)
            if count_key is not None:
                self._count_counts[count_key][action] += weight

            full_key = _full_key_from_row(row)
            if full_key is not None:
                self._full_counts[full_key][action] += weight

        if not observed_actions:
            raise ValueError("cannot fit EmpiricalBehaviorModel without observed actions")

        self._actions = sorted(observed_actions, key=lambda action: (action[0], action[1] or ""))
        self._fitted = True
        return self

    def predict_proba(
        self,
        *,
        balls: int,
        strikes: int,
        stand: str,
        p_throws: str,
    ) -> dict[ActionKey, float]:
        counts, level = self._select_counts(balls, strikes, stand, p_throws)
        self.last_backoff_level = level

        support = float(counts.total())
        denominator = support + self.alpha * len(self._actions)
        if denominator == 0:
            return {}

        return {
            action: (float(counts[action]) + self.alpha) / denominator for action in self._actions
        }

    def sample(
        self,
        rng: _ChoiceRng,
        *,
        balls: int,
        strikes: int,
        stand: str,
        p_throws: str,
    ) -> ActionKey:
        probabilities = self.predict_proba(
            balls=balls,
            strikes=strikes,
            stand=stand,
            p_throws=p_throws,
        )
        actions = list(probabilities)
        weights = list(probabilities.values())
        index = int(rng.choice(len(actions), p=weights))
        return actions[index]

    def support(
        self,
        *,
        balls: int,
        strikes: int,
        stand: str,
        p_throws: str,
    ) -> int:
        counts, level = self._select_counts(balls, strikes, stand, p_throws)
        self.last_backoff_level = level
        return int(counts.total())

    def _select_counts(
        self,
        balls: int,
        strikes: int,
        stand: str,
        p_throws: str,
    ) -> tuple[Counter[ActionKey], BackoffLevel]:
        if not self._fitted:
            raise ValueError("EmpiricalBehaviorModel must be fit before prediction")

        count = _coerce_count_value(balls, "balls")
        strike_count = _coerce_count_value(strikes, "strikes")
        handedness = _normalize_handedness(stand)
        pitcher_handedness = _normalize_handedness(p_throws)

        full_counts = self._full_counts.get(
            (count, strike_count, handedness, pitcher_handedness),
            Counter(),
        )
        if full_counts.total() >= self.min_support:
            return full_counts, "balls_strikes_stand_p_throws"

        count_counts = self._count_counts.get((count, strike_count), Counter())
        if count_counts.total() >= self.min_support:
            return count_counts, "balls_strikes"

        return self._global_counts, "global"

    def to_json(self) -> str:
        if not self._fitted:
            raise ValueError("EmpiricalBehaviorModel must be fit before serialization")
        if self.training_manifest_hash is None:
            raise ValueError("training_manifest_hash is required")

        payload = {
            "model_type": "EmpiricalBehaviorModel",
            "version": 1,
            "training_manifest_hash": self.training_manifest_hash,
            "settings": {
                "min_support": self.min_support,
                "alpha": self.alpha,
                "backoff_levels": [
                    "balls_strikes_stand_p_throws",
                    "balls_strikes",
                    "global",
                ],
            },
            "actions": [_action_to_json(action) for action in self._actions],
            "counts": {
                "global": _counter_to_json(self._global_counts),
                "balls_strikes_stand_p_throws": _keyed_counts_to_json(self._full_counts),
                "balls_strikes": _keyed_counts_to_json(self._count_counts),
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> EmpiricalBehaviorModel:
        payload = json.loads(value)
        settings = payload.get("settings")
        if not isinstance(settings, Mapping):
            raise ValueError("serialized behavior model is missing settings")

        model = cls(
            min_support=int(settings["min_support"]),
            alpha=float(settings["alpha"]),
        )
        training_manifest_hash = _string_or_none(payload.get("training_manifest_hash"))
        if training_manifest_hash is None:
            raise ValueError("serialized behavior model is missing training_manifest_hash")
        model.training_manifest_hash = training_manifest_hash

        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise ValueError("serialized behavior model is missing actions")
        model._actions = [_action_from_json(action) for action in actions]

        counts = payload.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("serialized behavior model is missing counts")

        model._global_counts = _counter_from_json(counts.get("global"))
        model._full_counts = defaultdict(
            Counter,
            _keyed_counts_from_json(counts.get("balls_strikes_stand_p_throws"), key_size=4),
        )
        model._count_counts = defaultdict(
            Counter,
            _keyed_counts_from_json(counts.get("balls_strikes"), key_size=2),
        )
        model._fitted = True
        return model


def _iter_rows(frame: object) -> list[Mapping[str, Any]]:
    to_dicts = getattr(frame, "to_dicts", None)
    if callable(to_dicts):
        rows = to_dicts()
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
        return []
    to_dict = getattr(frame, "to_dict", None)
    if callable(to_dict):
        try:
            rows = to_dict(orient="records")
        except TypeError:
            rows = to_dict()
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    if isinstance(frame, list):
        return [row for row in frame if isinstance(row, Mapping)]
    return [row for row in frame if isinstance(row, Mapping)]  # type: ignore[operator]


def _action_from_row(row: Mapping[str, Any]) -> ActionKey | None:
    if _string_or_none(row.get("zone")) is not None:
        _raise_raw_zone_error()

    pitch_type = _string_or_none(row.get("pitch_type"))
    relative_zone = _relative_zone_or_none(row.get("relative_zone"))

    if pitch_type is not None and relative_zone is not None:
        return (pitch_type, relative_zone)

    action = row.get("action")
    if action is None:
        return None
    return _parse_action(action)


def _parse_action(action: object) -> ActionKey | None:
    if isinstance(action, Mapping):
        if _string_or_none(action.get("zone")) is not None:
            _raise_raw_zone_error()
        pitch_type = _string_or_none(action.get("pitch_type"))
        relative_zone = _relative_zone_or_none(action.get("relative_zone"))
        if pitch_type is None or relative_zone is None:
            return None
        return (pitch_type, relative_zone)

    if isinstance(action, tuple | list) and len(action) == 2:
        pitch_type = _string_or_none(action[0])
        relative_zone = _relative_zone_or_none(action[1])
        if pitch_type is None or relative_zone is None:
            return None
        return (pitch_type, relative_zone)

    pitch_type_attr = _string_or_none(getattr(action, "pitch_type", None))
    if pitch_type_attr is not None:
        if _string_or_none(getattr(action, "zone", None)) is not None:
            _raise_raw_zone_error()
        relative_zone_attr = _relative_zone_or_none(getattr(action, "relative_zone", None))
        if relative_zone_attr is None:
            return None
        return (pitch_type_attr, relative_zone_attr)

    action_text = _string_or_none(action)
    if action_text is None:
        return None

    for delimiter in ("|", ":", "@", ","):
        if delimiter in action_text:
            pitch_type, relative_zone = action_text.split(delimiter, 1)
            normalized_pitch_type = _string_or_none(pitch_type)
            normalized_zone = _relative_zone_or_none(relative_zone)
            if normalized_pitch_type is None or normalized_zone is None:
                return None
            return (normalized_pitch_type, normalized_zone)

    return None


def _row_weight(row: Mapping[str, Any]) -> int:
    if "count" not in row:
        return 1
    value = row["count"]
    try:
        weight = int(value)
    except (TypeError, ValueError):
        return 0
    if weight != value and not isinstance(value, bool):
        try:
            if float(value) != weight:
                return 0
        except (TypeError, ValueError):
            return 0
    return weight


def _full_key_from_row(row: Mapping[str, Any]) -> _FullKey | None:
    count_key = _count_key_from_row(row)
    if count_key is None:
        return None

    stand = _optional_handedness(row.get("stand"))
    p_throws = _optional_handedness(row.get("p_throws"))
    if p_throws is None:
        p_throws = _optional_handedness(row.get("pitcher_throws"))
    if stand is None or p_throws is None:
        return None

    return (count_key[0], count_key[1], stand, p_throws)


def _count_key_from_row(row: Mapping[str, Any]) -> _CountKey | None:
    if "balls" not in row or "strikes" not in row:
        return None
    try:
        return (
            _coerce_count_value(row["balls"], "balls"),
            _coerce_count_value(row["strikes"], "strikes"),
        )
    except ValueError:
        return None


def _coerce_count_value(value: object, field_name: str) -> int:
    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer") from None

    if field_name == "balls" and not 0 <= normalized <= 3:
        raise ValueError("balls must be between 0 and 3")
    if field_name == "strikes" and not 0 <= normalized <= 2:
        raise ValueError("strikes must be between 0 and 2")
    return normalized


def _normalize_handedness(value: object) -> str:
    normalized = _optional_handedness(value)
    if normalized is None:
        raise ValueError("handedness must be 'R' or 'L'")
    return normalized


def _optional_handedness(value: object) -> str | None:
    normalized = _string_or_none(value)
    if normalized is None:
        return None
    normalized = normalized.upper()
    if normalized not in {"R", "L"}:
        return None
    return normalized


def _relative_zone_or_none(value: object) -> str | None:
    normalized = _string_or_none(value)
    if normalized is None:
        return None
    try:
        return RelativeZone(normalized).value
    except ValueError:
        return None


def _raise_raw_zone_error() -> Never:
    raise ValueError("raw Statcast zone is not allowed; use batter-relative relative_zone")


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not isfinite(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _action_to_json(action: ActionKey) -> list[str]:
    return [action[0], action[1]]


def _action_from_json(value: object) -> ActionKey:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("serialized action must be a two-item list")
    pitch_type = _string_or_none(value[0])
    relative_zone = _relative_zone_or_none(value[1])
    if pitch_type is None or relative_zone is None:
        raise ValueError("serialized action is invalid")
    return (pitch_type, relative_zone)


def _counter_to_json(counter: Counter[ActionKey]) -> list[dict[str, object]]:
    return [
        {"action": _action_to_json(action), "count": count}
        for action, count in sorted(counter.items(), key=lambda item: item[0])
    ]


def _counter_from_json(value: object) -> Counter[ActionKey]:
    if not isinstance(value, list):
        raise ValueError("serialized counter must be a list")
    counter: Counter[ActionKey] = Counter()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("serialized counter item must be an object")
        action = _action_from_json(item.get("action"))
        counter[action] = int(item["count"])
    return counter


def _keyed_counts_to_json(
    counts: Mapping[tuple[Any, ...], Counter[ActionKey]],
) -> list[dict[str, object]]:
    return [
        {"key": list(key), "actions": _counter_to_json(counter)}
        for key, counter in sorted(counts.items(), key=lambda item: item[0])
    ]


def _keyed_counts_from_json(
    value: object, *, key_size: int
) -> dict[tuple[Any, ...], Counter[ActionKey]]:
    if not isinstance(value, list):
        raise ValueError("serialized keyed counts must be a list")
    counts: dict[tuple[Any, ...], Counter[ActionKey]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("serialized keyed count item must be an object")
        key = item.get("key")
        if not isinstance(key, list) or len(key) != key_size:
            raise ValueError("serialized keyed count key has invalid shape")
        counts[tuple(key)] = _counter_from_json(item.get("actions"))
    return counts
