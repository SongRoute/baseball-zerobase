from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite, log
from typing import Any, Mapping, Protocol

import polars as pl

from baseball_zerobase.data.contracts import OutcomeLabel, TransitionAtom
from baseball_zerobase.models.transition_context import (
    TransitionContext,
    transition_context_from_row,
)
from baseball_zerobase.models.transition_heads import is_legal_transition


class _ChoiceRng(Protocol):
    def choice(self, a: int, *, p: list[float]) -> object: ...


BackoffKey = tuple[str, tuple[object, ...]]


@dataclass(slots=True)
class SharedTransitionModelV0:
    min_support: int = 1
    prior_weight: float = 1.0
    epsilon: float = 1e-12
    training_manifest_hash: str | None = None
    feature_columns: tuple[str, ...] = field(default_factory=tuple)

    _counts: dict[BackoffKey, Counter[TransitionAtom]] = field(
        default_factory=lambda: defaultdict(Counter),
        init=False,
    )
    _fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.min_support < 1:
            raise ValueError("min_support must be at least 1")
        if self.prior_weight < 0 or not isfinite(self.prior_weight):
            raise ValueError("prior_weight must be a non-negative finite value")
        if self.epsilon <= 0 or not isfinite(self.epsilon):
            raise ValueError("epsilon must be a positive finite value")

    def fit(self, frame: pl.DataFrame, *, training_manifest_hash: str) -> SharedTransitionModelV0:
        if not training_manifest_hash:
            raise ValueError("training_manifest_hash is required")
        self.training_manifest_hash = training_manifest_hash
        self._counts = defaultdict(Counter)
        self.feature_columns = _feature_columns(frame)

        for row in frame.iter_rows(named=True):
            context = transition_context_from_row(row)
            atom = _atom_from_row(row)
            if not is_legal_transition(context, atom):
                continue
            for key in _backoff_keys(context):
                self._counts[key][atom] += 1

        if not self._counts.get(("global", ())):
            raise ValueError("cannot fit SharedTransitionModelV0 without legal transitions")
        self._fitted = True
        return self

    def predict_distribution(self, context: TransitionContext) -> dict[TransitionAtom, float]:
        self._require_fitted()
        selected_counts = self._select_counts(context)
        legal_selected = Counter(
            {
                atom: count
                for atom, count in selected_counts.items()
                if is_legal_transition(context, atom)
            }
        )
        if not legal_selected:
            legal_selected = Counter(
                {
                    atom: count
                    for atom, count in self._counts[("global", ())].items()
                    if is_legal_transition(context, atom)
                }
            )
        if not legal_selected:
            raise ValueError("no legal transitions available for context")

        global_counts = Counter(
            {
                atom: count
                for atom, count in self._counts[("global", ())].items()
                if is_legal_transition(context, atom)
            }
        )
        global_total = float(global_counts.total())
        weighted: dict[TransitionAtom, float] = {}
        for atom in sorted(set(legal_selected) | set(global_counts), key=_atom_sort_key):
            prior_probability = (
                float(global_counts.get(atom, 0)) / global_total if global_total else 0.0
            )
            weighted[atom] = (
                float(legal_selected.get(atom, 0)) + self.prior_weight * prior_probability
            )
        total = sum(weighted.values())
        return {atom: value / total for atom, value in weighted.items() if value > 0}

    def sample(
        self, rng: _ChoiceRng, context: TransitionContext | None = None, **kwargs: Any
    ) -> TransitionAtom:
        if context is None:
            context = transition_context_from_row(kwargs)
        distribution = self.predict_distribution(context)
        atoms = list(distribution)
        weights = list(distribution.values())
        index = int(str(rng.choice(len(atoms), p=weights)))
        return atoms[index]

    def log_probability(
        self,
        actual_atom: TransitionAtom | Mapping[str, Any],
        context: TransitionContext,
    ) -> float:
        atom = (
            actual_atom if isinstance(actual_atom, TransitionAtom) else _atom_from_row(actual_atom)
        )
        distribution = self.predict_distribution(context)
        return log(max(distribution.get(atom, 0.0), self.epsilon))

    def support(self, context: TransitionContext) -> int:
        self._require_fitted()
        return int(self._select_counts(context).total())

    def to_dict(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "model_type": "SharedTransitionModelV0",
            "version": 1,
            "training_manifest_hash": self.training_manifest_hash,
            "feature_columns": list(self.feature_columns),
            "backoff_order": [key for key, _ in _backoff_keys(_dummy_context())],
            "smoothing_settings": {
                "min_support": self.min_support,
                "prior_weight": self.prior_weight,
                "epsilon": self.epsilon,
            },
            "counts": _counts_to_json(self._counts),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SharedTransitionModelV0:
        settings = dict(payload["smoothing_settings"])
        model = cls(
            min_support=int(settings["min_support"]),
            prior_weight=float(settings["prior_weight"]),
            epsilon=float(settings["epsilon"]),
        )
        model.training_manifest_hash = str(payload["training_manifest_hash"])
        model.feature_columns = tuple(str(value) for value in payload.get("feature_columns", []))
        model._counts = _counts_from_json(payload["counts"])
        model._fitted = True
        return model

    def _select_counts(self, context: TransitionContext) -> Counter[TransitionAtom]:
        for key in _backoff_keys(context):
            counts = self._counts.get(key)
            if counts is not None and counts.total() >= self.min_support:
                return counts
        return self._counts[("global", ())]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("SharedTransitionModelV0 must be fit before prediction")


def _backoff_keys(context: TransitionContext) -> tuple[BackoffKey, ...]:
    action = (context.pitch_type, context.relative_zone)
    return (
        (
            "personalized_action",
            (
                *action,
                bool(context.features.get("pitcher_pitch_type_owned", False)),
                str(context.features.get("batter_weakness_archetype", "unknown")),
                _threat_bucket(context.features.get("batter_threat_score")),
            ),
        ),
        ("state_action", (*action, context.balls, context.strikes, context.outs, context.runners)),
        ("count_action", (*action, context.balls, context.strikes)),
        ("action", action),
        ("global", ()),
    )


def _feature_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    from baseball_zerobase.models.transition_context import LABEL_COLUMNS

    context_columns = {"pitch_type", "relative_zone", "balls", "strikes", "outs", "runners"}
    return tuple(
        sorted(column for column in frame.columns if column not in LABEL_COLUMNS | context_columns)
    )


def _atom_from_row(row: Mapping[str, Any]) -> TransitionAtom:
    return TransitionAtom(
        outcome=OutcomeLabel(str(row["outcome"])),
        balls_after=int(row["balls_after"]),
        strikes_after=int(row["strikes_after"]),
        outs_after=int(row["outs_after"]),
        runners_after=_runners_tuple(row["runners_after"]),
        runs_scored=int(row["runs_scored"]),
        plate_appearance_ended=bool(row["plate_appearance_ended"]),
        half_inning_ended=bool(row["half_inning_ended"]),
        terminal_reason=None if row.get("terminal_reason") is None else str(row["terminal_reason"]),
    )


def _runners_tuple(value: object) -> tuple[bool, bool, bool]:
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError("runners_after tuple must have three entries")
        return (bool(value[0]), bool(value[1]), bool(value[2]))
    if value is None:
        mask = 0
    else:
        mask = int(str(value))
    return (bool(mask & 1), bool(mask & 2), bool(mask & 4))


def _threat_bucket(value: object) -> str:
    if value is None:
        return "unknown"
    numeric = float(str(value))
    if numeric >= 0.66:
        return "high"
    if numeric >= 0.33:
        return "medium"
    return "low"


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


def _counts_to_json(
    counts: Mapping[BackoffKey, Counter[TransitionAtom]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for (level, key), counter in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1])):
        atoms = [
            {"atom": _atom_to_json(atom), "count": int(count)}
            for atom, count in sorted(counter.items(), key=lambda item: _atom_sort_key(item[0]))
        ]
        records.append({"level": level, "key": list(key), "atoms": atoms})
    return records


def _counts_from_json(value: object) -> dict[BackoffKey, Counter[TransitionAtom]]:
    counts: dict[BackoffKey, Counter[TransitionAtom]] = defaultdict(Counter)
    if not isinstance(value, list):
        raise ValueError("counts payload must be a list")
    for record in value:
        if not isinstance(record, dict):
            raise ValueError("count record must be a mapping")
        key = (str(record["level"]), tuple(record["key"]))
        for atom_record in record["atoms"]:
            counts[key][_atom_from_json(atom_record["atom"])] = int(atom_record["count"])
    return counts


def _atom_to_json(atom: TransitionAtom) -> dict[str, object]:
    return {
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


def _atom_from_json(payload: Mapping[str, object]) -> TransitionAtom:
    runners_after = payload["runners_after"]
    if not isinstance(runners_after, Sequence) or isinstance(runners_after, str):
        raise ValueError("runners_after must be a sequence")
    if len(runners_after) != 3:
        raise ValueError("runners_after must have three entries")
    runners_after_tuple = (bool(runners_after[0]), bool(runners_after[1]), bool(runners_after[2]))
    return TransitionAtom(
        outcome=OutcomeLabel(str(payload["outcome"])),
        balls_after=int(str(payload["balls_after"])),
        strikes_after=int(str(payload["strikes_after"])),
        outs_after=int(str(payload["outs_after"])),
        runners_after=runners_after_tuple,
        runs_scored=int(str(payload["runs_scored"])),
        plate_appearance_ended=bool(payload["plate_appearance_ended"]),
        half_inning_ended=bool(payload["half_inning_ended"]),
        terminal_reason=None
        if payload.get("terminal_reason") is None
        else str(payload["terminal_reason"]),
    )


def _dummy_context() -> TransitionContext:
    return TransitionContext("FF", "middle_middle", 0, 0, 0, 0)
