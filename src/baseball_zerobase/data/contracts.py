from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RelativeZone(StrEnum):
    HIGH_INSIDE = "high_inside"
    HIGH_MIDDLE = "high_middle"
    HIGH_AWAY = "high_away"
    MIDDLE_INSIDE = "middle_inside"
    MIDDLE_MIDDLE = "middle_middle"
    MIDDLE_AWAY = "middle_away"
    LOW_INSIDE = "low_inside"
    LOW_MIDDLE = "low_middle"
    LOW_AWAY = "low_away"
    CHASE_HIGH = "chase_high"
    CHASE_LOW = "chase_low"
    CHASE_INSIDE = "chase_inside"
    CHASE_AWAY = "chase_away"


class Action(BaseModel):
    model_config = ConfigDict(frozen=True)

    pitch_type: str
    zone: RelativeZone


class OutcomeLabel(StrEnum):
    BALL = "ball"
    CALLED_STRIKE = "called_strike"
    SWINGING_STRIKE = "swinging_strike"
    FOUL = "foul"
    IN_PLAY_OUT = "in_play_out"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    HOME_RUN = "home_run"
    WALK = "walk"
    STRIKEOUT = "strikeout"
    HBP = "hit_by_pitch"
    REACH_OTHER = "reach_other"
    OTHER = "other"


class TransitionAtom(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    outcome: OutcomeLabel
    balls_after: int = Field(ge=0, le=3)
    strikes_after: int = Field(ge=0, le=2)
    outs_after: int = Field(ge=0, le=3)
    runners_after: tuple[bool, bool, bool]
    runs_scored: int = Field(ge=0)
    plate_appearance_ended: bool
    half_inning_ended: bool
    terminal_reason: str | None
