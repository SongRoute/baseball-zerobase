from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import isfinite, log
from typing import Any, Literal, Mapping, Protocol

from pydantic import ValidationError

from baseball_zerobase.data.contracts import OutcomeLabel, RelativeZone, TransitionAtom

ActionKey = tuple[str, str]
BackoffLevel = Literal[
    "action_balls_strikes_outs_runners_stand_p_throws",
    "action_balls_strikes_stand_p_throws",
    "action_balls_strikes",
    "action",
    "global",
]
_FullKey = tuple[str, str, int, int, int, int, str, str]
_HandedCountKey = tuple[str, str, int, int, str, str]
_ActionCountKey = tuple[str, str, int, int]


class _ChoiceRng(Protocol):
    def choice(self, a: int, *, p: list[float]) -> int: ...


@dataclass
class EmpiricalTransitionModel:
    """Estimate observed transition atoms with hierarchical context backoff."""

    min_support: int = 1
    epsilon: float = 1e-12
    last_backoff_level: BackoffLevel | None = field(default=None, init=False)
    training_manifest_hash: str | None = field(default=None, init=False)

    _actions: list[ActionKey] = field(default_factory=list, init=False)
    _global_counts: Counter[TransitionAtom] = field(default_factory=Counter, init=False)
    _action_counts: dict[ActionKey, Counter[TransitionAtom]] = field(
        default_factory=lambda: defaultdict(Counter),
        init=False,
    )
    _action_count_counts: dict[_ActionCountKey, Counter[TransitionAtom]] = field(
        default_factory=lambda: defaultdict(Counter),
        init=False,
    )
    _handed_count_counts: dict[_HandedCountKey, Counter[TransitionAtom]] = field(
        default_factory=lambda: defaultdict(Counter),
        init=False,
    )
    _full_counts: dict[_FullKey, Counter[TransitionAtom]] = field(
        default_factory=lambda: defaultdict(Counter),
        init=False,
    )
    _fitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.min_support < 1:
            raise ValueError("min_support must be at least 1")
        if self.epsilon <= 0 or not isfinite(self.epsilon):
            raise ValueError("epsilon must be a positive finite value")

    def fit(
        self,
        frame: object,
        *,
        training_manifest_hash: str,
    ) -> EmpiricalTransitionModel:
        if not _string_or_none(training_manifest_hash):
            raise ValueError("training_manifest_hash is required")

        self._actions = []
        self._global_counts = Counter()
        self._action_counts = defaultdict(Counter)
        self._action_count_counts = defaultdict(Counter)
        self._handed_count_counts = defaultdict(Counter)
        self._full_counts = defaultdict(Counter)
        self.last_backoff_level = None
        self.training_manifest_hash = training_manifest_hash

        observed_actions: set[ActionKey] = set()
        for row in _iter_rows(frame):
            action = _action_from_row(row)
            if action is None:
                continue

            atom = _transition_atom_from_row(row)
            if atom is None or not _atom_preserves_state_invariants(atom, row):
                continue

            weight = _row_weight(row)
            if weight <= 0:
                continue

            observed_actions.add(action)
            self._global_counts[atom] += weight
            self._action_counts[action][atom] += weight

            action_count_key = _action_count_key_from_row(row, action)
            if action_count_key is not None:
                self._action_count_counts[action_count_key][atom] += weight

            handed_count_key = _handed_count_key_from_row(row, action)
            if handed_count_key is not None:
                self._handed_count_counts[handed_count_key][atom] += weight

            full_key = _full_key_from_row(row, action)
            if full_key is not None:
                self._full_counts[full_key][atom] += weight

        if not observed_actions:
            raise ValueError("cannot fit EmpiricalTransitionModel without observed transitions")

        self._actions = sorted(observed_actions, key=lambda action: (action[0], action[1]))
        self._fitted = True
        return self

    def predict_distribution(
        self,
        *,
        pitch_type: str,
        relative_zone: str | RelativeZone,
        balls: int,
        strikes: int,
        outs: int,
        runners: int | tuple[bool, bool, bool],
        stand: str,
        p_throws: str,
    ) -> dict[TransitionAtom, float]:
        counts, level = self._select_counts(
            pitch_type=pitch_type,
            relative_zone=relative_zone,
            balls=balls,
            strikes=strikes,
            outs=outs,
            runners=runners,
            stand=stand,
            p_throws=p_throws,
        )
        self.last_backoff_level = level

        support = float(counts.total())
        if support == 0:
            return {}
        return {atom: float(count) / support for atom, count in counts.items()}

    def sample(
        self,
        rng: _ChoiceRng,
        *,
        pitch_type: str,
        relative_zone: str | RelativeZone,
        balls: int,
        strikes: int,
        outs: int,
        runners: int | tuple[bool, bool, bool],
        stand: str,
        p_throws: str,
    ) -> TransitionAtom:
        distribution = self.predict_distribution(
            pitch_type=pitch_type,
            relative_zone=relative_zone,
            balls=balls,
            strikes=strikes,
            outs=outs,
            runners=runners,
            stand=stand,
            p_throws=p_throws,
        )
        atoms = list(distribution)
        if not atoms:
            raise ValueError("no valid transition atoms for context")
        weights = list(distribution.values())
        index = int(rng.choice(len(atoms), p=weights))
        return atoms[index]

    def support(
        self,
        *,
        pitch_type: str,
        relative_zone: str | RelativeZone,
        balls: int,
        strikes: int,
        outs: int,
        runners: int | tuple[bool, bool, bool],
        stand: str,
        p_throws: str,
    ) -> int:
        counts, level = self._select_counts(
            pitch_type=pitch_type,
            relative_zone=relative_zone,
            balls=balls,
            strikes=strikes,
            outs=outs,
            runners=runners,
            stand=stand,
            p_throws=p_throws,
        )
        self.last_backoff_level = level
        return int(counts.total())

    def log_probability(
        self,
        actual_atom: TransitionAtom | Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        **context_kwargs: Any,
    ) -> float:
        atom = _atom_from_value(actual_atom)
        if atom is None:
            raise ValueError("actual_atom must be a valid TransitionAtom")
        resolved_context = _merge_context(context, context_kwargs)
        distribution = self.predict_distribution(
            pitch_type=resolved_context["pitch_type"],
            relative_zone=resolved_context["relative_zone"],
            balls=resolved_context["balls"],
            strikes=resolved_context["strikes"],
            outs=resolved_context["outs"],
            runners=resolved_context["runners"],
            stand=resolved_context["stand"],
            p_throws=resolved_context["p_throws"],
        )
        return log(max(distribution.get(atom, 0.0), self.epsilon))

    def to_json(self) -> str:
        if not self._fitted:
            raise ValueError("EmpiricalTransitionModel must be fit before serialization")
        if self.training_manifest_hash is None:
            raise ValueError("training_manifest_hash is required")

        payload = {
            "model_type": "EmpiricalTransitionModel",
            "version": 1,
            "training_manifest_hash": self.training_manifest_hash,
            "settings": {
                "min_support": self.min_support,
                "epsilon": self.epsilon,
                "backoff_levels": [
                    "action_balls_strikes_outs_runners_stand_p_throws",
                    "action_balls_strikes_stand_p_throws",
                    "action_balls_strikes",
                    "action",
                    "global",
                ],
            },
            "actions": [_action_to_json(action) for action in self._actions],
            "counts": {
                "global": _counter_to_json(self._global_counts),
                "action": _keyed_counts_to_json(self._action_counts),
                "action_balls_strikes": _keyed_counts_to_json(self._action_count_counts),
                "action_balls_strikes_stand_p_throws": _keyed_counts_to_json(
                    self._handed_count_counts
                ),
                "action_balls_strikes_outs_runners_stand_p_throws": _keyed_counts_to_json(
                    self._full_counts
                ),
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> EmpiricalTransitionModel:
        payload = json.loads(value)
        settings = payload.get("settings")
        if not isinstance(settings, Mapping):
            raise ValueError("serialized transition model is missing settings")

        model = cls(
            min_support=int(settings["min_support"]),
            epsilon=float(settings["epsilon"]),
        )
        model.training_manifest_hash = _require_string(
            payload.get("training_manifest_hash"),
            "serialized transition model is missing training_manifest_hash",
        )

        actions = payload.get("actions")
        if not isinstance(actions, list):
            raise ValueError("serialized transition model is missing actions")
        model._actions = [_action_from_json(action) for action in actions]

        counts = payload.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("serialized transition model is missing counts")

        model._global_counts = _counter_from_json(counts.get("global"))
        model._action_counts = defaultdict(
            Counter,
            _keyed_counts_from_json(counts.get("action"), key_size=2),
        )
        model._action_count_counts = defaultdict(
            Counter,
            _keyed_counts_from_json(counts.get("action_balls_strikes"), key_size=4),
        )
        model._handed_count_counts = defaultdict(
            Counter,
            _keyed_counts_from_json(
                counts.get("action_balls_strikes_stand_p_throws"),
                key_size=6,
            ),
        )
        model._full_counts = defaultdict(
            Counter,
            _keyed_counts_from_json(
                counts.get("action_balls_strikes_outs_runners_stand_p_throws"),
                key_size=8,
            ),
        )
        model._fitted = True
        return model

    def _select_counts(
        self,
        *,
        pitch_type: str,
        relative_zone: str | RelativeZone,
        balls: int,
        strikes: int,
        outs: int,
        runners: int | tuple[bool, bool, bool],
        stand: str,
        p_throws: str,
    ) -> tuple[Counter[TransitionAtom], BackoffLevel]:
        if not self._fitted:
            raise ValueError("EmpiricalTransitionModel must be fit before prediction")

        action = _action_from_values(pitch_type, relative_zone)
        ball_count = _coerce_count_value(balls, "balls")
        strike_count = _coerce_count_value(strikes, "strikes")
        out_count = _coerce_outs_value(outs)
        runner_bits = _coerce_runner_bits(runners)
        handedness = _normalize_handedness(stand)
        pitcher_handedness = _normalize_handedness(p_throws)

        full_counts = _valid_counts_for_context(
            self._full_counts.get(
                (
                    action[0],
                    action[1],
                    ball_count,
                    strike_count,
                    out_count,
                    runner_bits,
                    handedness,
                    pitcher_handedness,
                ),
                Counter(),
            ),
            balls=ball_count,
            strikes=strike_count,
            outs=out_count,
            runners=runner_bits,
        )
        if full_counts.total() >= self.min_support:
            return full_counts, "action_balls_strikes_outs_runners_stand_p_throws"

        handed_count_counts = _valid_counts_for_context(
            self._handed_count_counts.get(
                (action[0], action[1], ball_count, strike_count, handedness, pitcher_handedness),
                Counter(),
            ),
            balls=ball_count,
            strikes=strike_count,
            outs=out_count,
            runners=runner_bits,
        )
        if handed_count_counts.total() >= self.min_support:
            return handed_count_counts, "action_balls_strikes_stand_p_throws"

        action_count_counts = _valid_counts_for_context(
            self._action_count_counts.get(
                (action[0], action[1], ball_count, strike_count),
                Counter(),
            ),
            balls=ball_count,
            strikes=strike_count,
            outs=out_count,
            runners=runner_bits,
        )
        if action_count_counts.total() >= self.min_support:
            return action_count_counts, "action_balls_strikes"

        action_counts = _valid_counts_for_context(
            self._action_counts.get(action, Counter()),
            balls=ball_count,
            strikes=strike_count,
            outs=out_count,
            runners=runner_bits,
        )
        if action_counts.total() >= self.min_support:
            return action_counts, "action"

        return (
            _valid_counts_for_context(
                self._global_counts,
                balls=ball_count,
                strikes=strike_count,
                outs=out_count,
                runners=runner_bits,
            ),
            "global",
        )


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


def _action_from_values(pitch_type: str, relative_zone: str | RelativeZone) -> ActionKey:
    normalized_pitch_type = _string_or_none(pitch_type)
    normalized_zone = _relative_zone_or_none(relative_zone)
    if normalized_pitch_type is None or normalized_zone is None:
        raise ValueError("pitch_type and relative_zone must define a valid action")
    return (normalized_pitch_type, normalized_zone)


def _transition_atom_from_row(row: Mapping[str, Any]) -> TransitionAtom | None:
    transition_atom = row.get("transition_atom")
    if transition_atom is not None:
        parsed = _atom_from_value(transition_atom)
        if parsed is not None:
            return parsed

    return _atom_from_value(row)


def _atom_from_value(value: TransitionAtom | Mapping[str, Any] | object) -> TransitionAtom | None:
    if isinstance(value, TransitionAtom):
        return value
    if not isinstance(value, Mapping):
        return None

    outcome = _outcome_or_none(value.get("outcome"))
    balls_after = _int_or_none(value.get("balls_after"))
    strikes_after = _int_or_none(value.get("strikes_after"))
    outs_after = _int_or_none(value.get("outs_after"))
    runners_after = _runners_after_or_none(value.get("runners_after"))
    runs_scored = _int_or_none(value.get("runs_scored"))
    plate_appearance_ended = _bool_or_none(value.get("plate_appearance_ended"))
    half_inning_ended = _bool_or_none(value.get("half_inning_ended"))

    if (
        outcome is None
        or balls_after is None
        or strikes_after is None
        or outs_after is None
        or runners_after is None
        or runs_scored is None
        or plate_appearance_ended is None
        or half_inning_ended is None
    ):
        return None

    try:
        return TransitionAtom(
            outcome=outcome,
            balls_after=balls_after,
            strikes_after=strikes_after,
            outs_after=outs_after,
            runners_after=runners_after,
            runs_scored=runs_scored,
            plate_appearance_ended=plate_appearance_ended,
            half_inning_ended=half_inning_ended,
            terminal_reason=_string_or_none(value.get("terminal_reason")),
        )
    except (TypeError, ValueError, ValidationError):
        return None


def _atom_preserves_state_invariants(atom: TransitionAtom, row: Mapping[str, Any]) -> bool:
    outs_before = _int_or_none(row.get("outs"))
    if outs_before is not None and atom.outs_after < outs_before:
        return False
    if atom.half_inning_ended and atom.terminal_reason is None:
        return False
    if not atom.half_inning_ended and atom.terminal_reason is not None:
        return False
    return True


def _valid_counts_for_context(
    counts: Counter[TransitionAtom],
    *,
    balls: int,
    strikes: int,
    outs: int,
    runners: int,
) -> Counter[TransitionAtom]:
    return Counter(
        {
            atom: count
            for atom, count in counts.items()
            if _atom_is_valid_for_context(
                atom,
                balls=balls,
                strikes=strikes,
                outs=outs,
                runners=runners,
            )
        }
    )


def _atom_is_valid_for_context(
    atom: TransitionAtom,
    *,
    balls: int,
    strikes: int,
    outs: int,
    runners: int,
) -> bool:
    if atom.outs_after < outs:
        return False
    outs_delta = atom.outs_after - outs

    if atom.half_inning_ended:
        if not atom.plate_appearance_ended or atom.outs_after != 3 or atom.terminal_reason is None:
            return False
    elif atom.terminal_reason is not None:
        return False

    if atom.plate_appearance_ended:
        if atom.balls_after != 0 or atom.strikes_after != 0:
            return False
        if not _terminal_outcome_is_valid(atom, balls=balls, strikes=strikes, outs_delta=outs_delta):
            return False
    elif not _non_terminal_count_is_valid(atom, balls=balls, strikes=strikes, outs=outs):
        return False

    return _runner_state_is_reachable(
        atom,
        before_runners=runners,
        outs_delta=outs_delta,
    )


def _terminal_outcome_is_valid(
    atom: TransitionAtom,
    *,
    balls: int,
    strikes: int,
    outs_delta: int,
) -> bool:
    if atom.outcome == OutcomeLabel.STRIKEOUT:
        return strikes == 2 and outs_delta >= 1
    if atom.outcome == OutcomeLabel.WALK:
        return balls == 3 and outs_delta == 0
    if atom.outcome == OutcomeLabel.HBP:
        return outs_delta == 0
    if atom.outcome == OutcomeLabel.IN_PLAY_OUT:
        return outs_delta > 0
    if atom.outcome in {
        OutcomeLabel.SINGLE,
        OutcomeLabel.DOUBLE,
        OutcomeLabel.TRIPLE,
        OutcomeLabel.HOME_RUN,
    }:
        return outs_delta == 0
    if atom.outcome == OutcomeLabel.REACH_OTHER:
        return True
    return True


def _non_terminal_count_is_valid(
    atom: TransitionAtom,
    *,
    balls: int,
    strikes: int,
    outs: int,
) -> bool:
    if atom.outs_after != outs or atom.half_inning_ended:
        return False

    if atom.outcome == OutcomeLabel.BALL:
        return balls < 3 and atom.balls_after == balls + 1 and atom.strikes_after == strikes

    if atom.outcome in {OutcomeLabel.CALLED_STRIKE, OutcomeLabel.SWINGING_STRIKE}:
        return strikes < 2 and atom.balls_after == balls and atom.strikes_after == strikes + 1

    if atom.outcome == OutcomeLabel.FOUL:
        return atom.balls_after == balls and atom.strikes_after == min(strikes + 1, 2)

    return atom.balls_after >= balls and atom.strikes_after >= strikes


def _runner_count(runners: int) -> int:
    return int(bool(runners & 1)) + int(bool(runners & 2)) + int(bool(runners & 4))


def _runner_state_is_reachable(
    atom: TransitionAtom,
    *,
    before_runners: int,
    outs_delta: int,
) -> bool:
    before_count = _runner_count(before_runners)
    after_bits = _runner_bits_from_tuple(atom.runners_after)
    after_count = _runner_count(after_bits)
    batter_added = 1 if atom.plate_appearance_ended else 0
    if after_count + atom.runs_scored + outs_delta != before_count + batter_added:
        return False

    if atom.outcome == OutcomeLabel.HOME_RUN:
        return after_bits == 0 and atom.runs_scored == before_count + 1

    if not atom.plate_appearance_ended:
        return after_bits == before_runners and atom.runs_scored == 0 and outs_delta == 0

    if atom.outcome == OutcomeLabel.WALK or atom.outcome == OutcomeLabel.HBP:
        return _forced_walk_state(before_runners) == after_bits

    if atom.outcome in {OutcomeLabel.SINGLE, OutcomeLabel.DOUBLE, OutcomeLabel.TRIPLE}:
        return _hit_runner_state_is_reachable(
            before_runners=before_runners,
            after_bits=after_bits,
            hit_bases=_hit_bases(atom.outcome),
        )

    if atom.outcome == OutcomeLabel.REACH_OTHER and outs_delta == 0:
        return _reach_other_safe_state_is_reachable(
            before_runners=before_runners,
            after_bits=after_bits,
        )

    if atom.outcome in {OutcomeLabel.IN_PLAY_OUT, OutcomeLabel.STRIKEOUT}:
        return _remaining_runners_do_not_move_backward(before_runners, after_bits)

    return _remaining_runners_do_not_move_backward(before_runners, after_bits)


def _runner_bits_from_tuple(runners_after: tuple[bool, bool, bool]) -> int:
    return (
        int(runners_after[0])
        | (int(runners_after[1]) << 1)
        | (int(runners_after[2]) << 2)
    )


def _forced_walk_state(before_runners: int) -> int:
    after = before_runners | 1
    if before_runners & 1:
        after |= 2
    if before_runners & 1 and before_runners & 2:
        after |= 4
    return after & 7


def _hit_bases(outcome: OutcomeLabel) -> int:
    if outcome == OutcomeLabel.SINGLE:
        return 1
    if outcome == OutcomeLabel.DOUBLE:
        return 2
    if outcome == OutcomeLabel.TRIPLE:
        return 3
    raise ValueError(f"{outcome} is not a hit outcome")


def _hit_runner_state_is_reachable(
    *,
    before_runners: int,
    after_bits: int,
    hit_bases: int,
) -> bool:
    possible: set[int] = set()

    def advance_existing(base_index: int, current_bits: int) -> None:
        if base_index > 3:
            place_batter(current_bits)
            return
        base_bit = 1 << (base_index - 1)
        if not before_runners & base_bit:
            advance_existing(base_index + 1, current_bits)
            return

        for destination in range(base_index, 5):
            if destination >= 4:
                advance_existing(base_index + 1, current_bits)
                continue
            destination_bit = 1 << (destination - 1)
            if current_bits & destination_bit:
                continue
            advance_existing(base_index + 1, current_bits | destination_bit)

    def place_batter(current_bits: int) -> None:
        batter_bit = 1 << (hit_bases - 1)
        if current_bits & batter_bit:
            return
        possible.add(current_bits | batter_bit)

    advance_existing(1, 0)
    return after_bits in possible


def _reach_other_safe_state_is_reachable(*, before_runners: int, after_bits: int) -> bool:
    if not after_bits & 1:
        return False
    return _remaining_runners_do_not_move_backward(before_runners, after_bits & ~1)


def _remaining_runners_do_not_move_backward(before_runners: int, after_bits: int) -> bool:
    before_bases = [base for base in range(1, 4) if before_runners & (1 << (base - 1))]
    after_bases = [base for base in range(1, 4) if after_bits & (1 << (base - 1))]
    if len(after_bases) > len(before_bases):
        return False
    return all(after_base >= before_base for before_base, after_base in zip(before_bases, after_bases))


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


def _full_key_from_row(row: Mapping[str, Any], action: ActionKey) -> _FullKey | None:
    ball_count = _row_count_value(row, "balls")
    strike_count = _row_count_value(row, "strikes")
    out_count = _row_outs_value(row)
    runner_bits = _row_runners_value(row)
    stand = _optional_handedness(row.get("stand"))
    p_throws = _optional_handedness(row.get("p_throws"))
    if p_throws is None:
        p_throws = _optional_handedness(row.get("pitcher_throws"))
    if (
        ball_count is None
        or strike_count is None
        or out_count is None
        or runner_bits is None
        or stand is None
        or p_throws is None
    ):
        return None
    return (
        action[0],
        action[1],
        ball_count,
        strike_count,
        out_count,
        runner_bits,
        stand,
        p_throws,
    )


def _handed_count_key_from_row(row: Mapping[str, Any], action: ActionKey) -> _HandedCountKey | None:
    ball_count = _row_count_value(row, "balls")
    strike_count = _row_count_value(row, "strikes")
    stand = _optional_handedness(row.get("stand"))
    p_throws = _optional_handedness(row.get("p_throws"))
    if p_throws is None:
        p_throws = _optional_handedness(row.get("pitcher_throws"))
    if ball_count is None or strike_count is None or stand is None or p_throws is None:
        return None
    return (action[0], action[1], ball_count, strike_count, stand, p_throws)


def _action_count_key_from_row(row: Mapping[str, Any], action: ActionKey) -> _ActionCountKey | None:
    ball_count = _row_count_value(row, "balls")
    strike_count = _row_count_value(row, "strikes")
    if ball_count is None or strike_count is None:
        return None
    return (action[0], action[1], ball_count, strike_count)


def _row_count_value(row: Mapping[str, Any], field_name: Literal["balls", "strikes"]) -> int | None:
    try:
        return _coerce_count_value(row[field_name], field_name)
    except (KeyError, ValueError):
        return None


def _row_outs_value(row: Mapping[str, Any]) -> int | None:
    try:
        return _coerce_outs_value(row["outs"])
    except (KeyError, ValueError):
        return None


def _row_runners_value(row: Mapping[str, Any]) -> int | None:
    try:
        return _coerce_runner_bits(row["runners"])
    except (KeyError, ValueError):
        return None


def _coerce_count_value(value: object, field_name: Literal["balls", "strikes"]) -> int:
    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer") from None

    if field_name == "balls" and not 0 <= normalized <= 3:
        raise ValueError("balls must be between 0 and 3")
    if field_name == "strikes" and not 0 <= normalized <= 2:
        raise ValueError("strikes must be between 0 and 2")
    return normalized


def _coerce_outs_value(value: object) -> int:
    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("outs must be an integer") from None
    if not 0 <= normalized <= 2:
        raise ValueError("outs must be between 0 and 2")
    return normalized


def _coerce_runner_bits(value: object) -> int:
    if isinstance(value, tuple | list):
        if len(value) != 3:
            raise ValueError("runners tuple must have three bases")
        runners = 0
        for bit, occupied in ((1, value[0]), (2, value[1]), (4, value[2])):
            if bool(occupied):
                runners |= bit
        return runners

    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("runners must be an integer bitmask") from None
    if not 0 <= normalized <= 7:
        raise ValueError("runners must be between 0 and 7")
    return normalized


def _runners_after_or_none(value: object) -> tuple[bool, bool, bool] | None:
    if isinstance(value, tuple | list):
        if len(value) != 3:
            return None
        return tuple(bool(occupied) for occupied in value)  # type: ignore[return-value]
    try:
        runner_bits = _coerce_runner_bits(value)
    except ValueError:
        return None
    return (
        bool(runner_bits & 1),
        bool(runner_bits & 2),
        bool(runner_bits & 4),
    )


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = _string_or_none(value)
    if normalized is None:
        return None
    if normalized.lower() in {"true", "t", "1"}:
        return True
    if normalized.lower() in {"false", "f", "0"}:
        return False
    return None


def _outcome_or_none(value: object) -> OutcomeLabel | None:
    normalized = _string_or_none(value)
    if normalized is None:
        return None
    try:
        return OutcomeLabel(normalized)
    except ValueError:
        return None


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


def _raise_raw_zone_error() -> None:
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


def _merge_context(
    context: Mapping[str, Any] | None,
    context_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = dict(context or {})
    resolved.update(context_kwargs)
    missing = sorted(
        {
            "pitch_type",
            "relative_zone",
            "balls",
            "strikes",
            "outs",
            "runners",
            "stand",
            "p_throws",
        }.difference(resolved)
    )
    if missing:
        raise ValueError(f"transition context is missing required fields: {missing}")
    return resolved


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


def _atom_to_json(atom: TransitionAtom) -> dict[str, object]:
    return atom.model_dump(mode="json")


def _counter_to_json(counter: Counter[TransitionAtom]) -> list[dict[str, object]]:
    return [
        {"atom": _atom_to_json(atom), "count": count}
        for atom, count in sorted(counter.items(), key=lambda item: _atom_sort_key(item[0]))
    ]


def _counter_from_json(value: object) -> Counter[TransitionAtom]:
    if not isinstance(value, list):
        raise ValueError("serialized counter must be a list")
    counter: Counter[TransitionAtom] = Counter()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("serialized counter item must be an object")
        atom = _atom_from_value(item.get("atom"))
        if atom is None:
            raise ValueError("serialized counter atom is invalid")
        counter[atom] = int(item["count"])
    return counter


def _keyed_counts_to_json(
    counts: Mapping[tuple[Any, ...], Counter[TransitionAtom]],
) -> list[dict[str, object]]:
    return [
        {"key": list(key), "atoms": _counter_to_json(counter)}
        for key, counter in sorted(counts.items(), key=lambda item: item[0])
    ]


def _keyed_counts_from_json(
    value: object,
    *,
    key_size: int,
) -> dict[tuple[Any, ...], Counter[TransitionAtom]]:
    if not isinstance(value, list):
        raise ValueError("serialized keyed counts must be a list")
    counts: dict[tuple[Any, ...], Counter[TransitionAtom]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("serialized keyed count item must be an object")
        key = item.get("key")
        if not isinstance(key, list) or len(key) != key_size:
            raise ValueError("serialized keyed count key has invalid shape")
        counts[tuple(key)] = _counter_from_json(item.get("atoms"))
    return counts


def _atom_sort_key(atom: TransitionAtom) -> str:
    return json.dumps(_atom_to_json(atom), sort_keys=True, separators=(",", ":"))


def _require_string(value: object, message: str) -> str:
    normalized = _string_or_none(value)
    if normalized is None:
        raise ValueError(message)
    return normalized
