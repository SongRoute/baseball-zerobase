from typing import cast

from baseball_zerobase.data.contracts import RelativeZone
from baseball_zerobase.data.zone_mapper import map_relative_zone


def test_zone_mirrors_inside_and_away_by_batter_hand() -> None:
    right = map_relative_zone(plate_x=0.60, plate_z=2.50, sz_bot=1.50, sz_top=3.50, stand="R")
    left = map_relative_zone(plate_x=-0.60, plate_z=2.50, sz_bot=1.50, sz_top=3.50, stand="L")
    assert right is RelativeZone.MIDDLE_INSIDE
    assert left is RelativeZone.MIDDLE_INSIDE


def test_vertical_chase_takes_precedence_at_corner() -> None:
    zone = map_relative_zone(plate_x=1.20, plate_z=4.00, sz_bot=1.50, sz_top=3.50, stand="R")
    assert zone is RelativeZone.CHASE_HIGH


def test_invalid_inputs_return_none() -> None:
    zone = map_relative_zone(
        plate_x=cast(float, "bad"),
        plate_z=2.50,
        sz_bot=1.50,
        sz_top=3.50,
        stand="R",
    )
    assert zone is None
