from math import isfinite

from baseball_zerobase.data.contracts import RelativeZone


_PLATE_LEFT = -0.83
_PLATE_RIGHT = 0.83


def map_relative_zone(
    plate_x: float | None,
    plate_z: float | None,
    sz_bot: float | None,
    sz_top: float | None,
    stand: str | None,
) -> RelativeZone | None:
    values = _coerce_inputs(plate_x, plate_z, sz_bot, sz_top)
    normalized_stand = _normalize_stand(stand)
    if values is None or normalized_stand is None:
        return None

    x, z, bottom, top = values
    if top <= bottom:
        return None

    relative_x = -x if normalized_stand == "R" else x

    if z > top:
        return RelativeZone.CHASE_HIGH
    if z < bottom:
        return RelativeZone.CHASE_LOW
    if relative_x > _PLATE_RIGHT:
        return RelativeZone.CHASE_INSIDE
    if relative_x < _PLATE_LEFT:
        return RelativeZone.CHASE_AWAY

    vertical_band = _vertical_band(z, bottom, top)
    horizontal_band = _horizontal_band(relative_x)
    return RelativeZone(f"{vertical_band}_{horizontal_band}")


def _coerce_inputs(
    plate_x: float | None,
    plate_z: float | None,
    sz_bot: float | None,
    sz_top: float | None,
) -> tuple[float, float, float, float] | None:
    if plate_x is None or plate_z is None or sz_bot is None or sz_top is None:
        return None

    try:
        values = (float(plate_x), float(plate_z), float(sz_bot), float(sz_top))
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in values):
        return None
    return values


def _normalize_stand(stand: str | None) -> str | None:
    if stand is None:
        return None
    normalized = stand.strip().upper()
    if normalized not in {"R", "L"}:
        return None
    return normalized


def _vertical_band(z: float, bottom: float, top: float) -> str:
    zone_height = top - bottom
    low_cutoff = bottom + zone_height / 3
    high_cutoff = top - zone_height / 3
    if z < low_cutoff:
        return "low"
    if z > high_cutoff:
        return "high"
    return "middle"


def _horizontal_band(relative_x: float) -> str:
    zone_width = _PLATE_RIGHT - _PLATE_LEFT
    away_cutoff = _PLATE_LEFT + zone_width / 3
    inside_cutoff = _PLATE_RIGHT - zone_width / 3
    if relative_x < away_cutoff:
        return "away"
    if relative_x > inside_cutoff:
        return "inside"
    return "middle"
