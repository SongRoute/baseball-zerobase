from baseball_zerobase.simulation.state import GameState


def test_advance_batting_order_rotates_and_updates_stand() -> None:
    state = GameState(
        balls=0,
        strikes=0,
        outs=0,
        runners=0,
        inning=1,
        score_diff=0,
        batting_order_index=0,
        lineup_ids=(10, 20),
        lineup_stands=("R", "L"),
        stand="R",
        p_throws="R",
    )

    advanced = state.advance_batting_order()
    rotated = advanced.advance_batting_order()

    assert advanced.batting_order_index == 1
    assert advanced.stand == "L"
    assert rotated.batting_order_index == 0
    assert rotated.stand == "R"
