from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class GameState:
    balls: int
    strikes: int
    outs: int
    runners: int
    inning: int
    score_diff: int
    batting_order_index: int
    lineup_ids: tuple[int, ...]
    lineup_stands: tuple[str, ...]
    stand: str
    p_throws: str

    def __post_init__(self) -> None:
        if not 0 <= self.balls <= 3:
            raise ValueError("balls must be between 0 and 3")
        if not 0 <= self.strikes <= 2:
            raise ValueError("strikes must be between 0 and 2")
        if not 0 <= self.outs <= 3:
            raise ValueError("outs must be between 0 and 3")
        if not 0 <= self.runners <= 7:
            raise ValueError("runners must be a 3-bit occupancy mask")
        if self.inning < 1:
            raise ValueError("inning must be positive")
        if not self.lineup_ids:
            raise ValueError("lineup_ids must not be empty")
        if len(self.lineup_ids) != len(self.lineup_stands):
            raise ValueError("lineup_ids and lineup_stands must have the same length")
        if not 0 <= self.batting_order_index < len(self.lineup_ids):
            raise ValueError("batting_order_index must reference lineup_ids")
        if self.stand != self.lineup_stands[self.batting_order_index]:
            raise ValueError("stand must match lineup_stands at batting_order_index")

    def advance_batting_order(self) -> GameState:
        next_index = (self.batting_order_index + 1) % len(self.lineup_ids)
        return replace(
            self,
            batting_order_index=next_index,
            stand=self.lineup_stands[next_index],
        )
