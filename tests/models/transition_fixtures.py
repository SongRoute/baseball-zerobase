from datetime import date, datetime, timedelta

import polars as pl


def transition_training_frame() -> pl.DataFrame:
    start = datetime(2024, 4, 1, 18, 0)
    rows = []
    outcomes = [
        ("ball", 1, 0, 0, 0, 0, False, False, None),
        ("called_strike", 0, 1, 0, 0, 0, False, False, None),
        ("swinging_strike", 0, 1, 0, 0, 0, False, False, None),
        ("home_run", 0, 0, 0, 0, 1, True, False, None),
    ]
    for index, outcome in enumerate(outcomes):
        (
            outcome_name,
            balls_after,
            strikes_after,
            outs_after,
            runners_after,
            runs_scored,
            plate_appearance_ended,
            half_inning_ended,
            terminal_reason,
        ) = outcome
        rows.append(
            {
                "game_pk": 1000 + index,
                "game_date": date(2024, 4, 1) + timedelta(days=index),
                "game_type": "R",
                "pitch_timestamp": start + timedelta(days=index),
                "as_of_timestamp": start + timedelta(days=index, seconds=-1),
                "pitch_type": "FF",
                "relative_zone": "middle_middle",
                "balls": 0,
                "strikes": 0,
                "outs": 0,
                "runners": 0,
                "stand": "R",
                "p_throws": "R",
                "outcome": outcome_name,
                "balls_after": balls_after,
                "strikes_after": strikes_after,
                "outs_after": outs_after,
                "runners_after": runners_after,
                "runs_scored": runs_scored,
                "plate_appearance_ended": plate_appearance_ended,
                "half_inning_ended": half_inning_ended,
                "terminal_reason": terminal_reason,
                "pitcher_pitch_type_owned": True,
                "batter_weakness_archetype": "chase_vulnerable",
                "batter_threat_score": 0.75,
            }
        )
    return pl.DataFrame(rows)
