from baseball_zerobase.data.contracts import RelativeZone
from baseball_zerobase.inference.candidates import (
    generate_candidate_grid,
    resolve_candidate_pitch_types,
)


def test_candidate_grid_uses_every_relative_zone_for_each_pitch_type() -> None:
    candidates = generate_candidate_grid(["FF", "SL"])

    assert len(candidates) == 2 * len(RelativeZone)
    assert {candidate.relative_zone for candidate in candidates} == {
        zone.value for zone in RelativeZone
    }
    assert {(candidate.pitch_type, candidate.relative_zone) for candidate in candidates} == {
        (pitch_type, zone.value) for pitch_type in ("FF", "SL") for zone in RelativeZone
    }


def test_candidate_pitch_types_prefer_explicit_values_over_row_action() -> None:
    row = {
        "pitch_type": "CU",
        "pitcher_owned_pitch_types": ["FF", "SL"],
    }

    pitch_types = resolve_candidate_pitch_types(row, explicit_pitch_types=["CH", "FF", "FF"])

    assert pitch_types == ("CH", "FF")


def test_candidate_pitch_types_can_come_from_prior_owned_pitch_type_list() -> None:
    row = {
        "pitch_type": "CU",
        "pitcher_owned_pitch_types": ["FF", "SL"],
    }

    pitch_types = resolve_candidate_pitch_types(row, explicit_pitch_types=None)

    assert pitch_types == ("FF", "SL")
