from baseball_zerobase.data.contracts import OutcomeLabel


_TERMINAL_OUTCOMES: dict[str, OutcomeLabel] = {
    "single": OutcomeLabel.SINGLE,
    "double": OutcomeLabel.DOUBLE,
    "triple": OutcomeLabel.TRIPLE,
    "home_run": OutcomeLabel.HOME_RUN,
    "walk": OutcomeLabel.WALK,
    "intent_walk": OutcomeLabel.WALK,
    "hit_by_pitch": OutcomeLabel.HBP,
    "strikeout": OutcomeLabel.STRIKEOUT,
    "strikeout_double_play": OutcomeLabel.STRIKEOUT,
}

_IN_PLAY_OUT_EVENTS = {
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "double_play",
    "triple_play",
    "sac_fly",
    "sac_fly_double_play",
    "sac_bunt",
    "fielders_choice_out",
}

_REACH_OTHER_EVENTS = {
    "field_error",
    "fielders_choice",
    "catcher_interf",
}

_DESCRIPTION_OUTCOMES: dict[str, OutcomeLabel] = {
    "ball": OutcomeLabel.BALL,
    "blocked_ball": OutcomeLabel.BALL,
    "pitchout": OutcomeLabel.BALL,
    "called_strike": OutcomeLabel.CALLED_STRIKE,
    "swinging_strike": OutcomeLabel.SWINGING_STRIKE,
    "swinging_strike_blocked": OutcomeLabel.SWINGING_STRIKE,
    "foul": OutcomeLabel.FOUL,
    "foul_tip": OutcomeLabel.FOUL,
    "foul_bunt": OutcomeLabel.FOUL,
}


def map_outcome(description: str | None, event: str | None) -> OutcomeLabel:
    normalized_event = _normalize_label(event)
    if normalized_event is not None:
        if normalized_event in _TERMINAL_OUTCOMES:
            return _TERMINAL_OUTCOMES[normalized_event]
        if normalized_event in _IN_PLAY_OUT_EVENTS:
            return OutcomeLabel.IN_PLAY_OUT
        if normalized_event in _REACH_OTHER_EVENTS:
            return OutcomeLabel.REACH_OTHER

    normalized_description = _normalize_label(description)
    if normalized_description is not None and normalized_description in _DESCRIPTION_OUTCOMES:
        return _DESCRIPTION_OUTCOMES[normalized_description]
    return OutcomeLabel.OTHER


def _normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        return None
    return normalized
