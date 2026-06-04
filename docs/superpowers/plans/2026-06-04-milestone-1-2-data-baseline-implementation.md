# Milestone 1-2 Data Foundation and Baseline System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean-room, reproducible pipeline that downloads MLB source data, reconstructs leakage-safe pre-pitch snapshots for qualified starting pitchers, and evaluates empirical behavior and transition baselines through an inning simulator.

**Architecture:** Raw Statcast and MLB StatsAPI responses are immutable inputs with manifests. Normalization produces official-starter metadata, stable initial lineups, batter-relative 13-zone actions, and pre-pitch snapshots with direct observed transition atoms. League-level empirical behavior and transition models provide a fully testable baseline before pitcher profiles, batter archetypes, neural transition models, recommendations, or web UI are introduced.

**Tech Stack:** Python 3.12, uv, Polars, PyArrow, pybaseball, httpx, Pydantic, Typer, NumPy, SciPy, scikit-learn metrics, pytest, Ruff, Pyright.

---

## Scope Boundary

This plan implements only Milestone 1 and Milestone 2 from the approved design:

- immutable source-data acquisition and manifests
- locked-test partition protection
- official starter and stable-lineup reconstruction
- batter-relative 13-zone mapping
- leakage-safe pre-pitch snapshots and observed transition atoms
- as-of starter eligibility
- empirical behavior and transition baselines
- deterministic inning simulation
- rolling validation and baseline reports

This plan deliberately does not implement:

- pitcher profile features beyond as-of eligibility counts
- batter archetypes or threat scores
- day-state features beyond fields already present in snapshots
- PyTorch or LightGBM models
- action ranking, recommendation confidence, replay product reports, API, or web UI
- execution of the 2025 postseason or 2026 locked tests

## Locked Decisions

- Python is exactly `>=3.12,<3.13`.
- Existing deleted scaffold files are not restored from Git; new files are created from this plan.
- Raw data is never modified in place.
- Development commands cannot read paths classified as locked data.
- Locked partitions are:
  - 2025 postseason: `game_type` in `F`, `D`, `L`, `W`
  - 2026 regular season from `2026-03-25` through `2026-05-31`
- Main development data is 2022-2025 regular season.
- An action is `pitch_type × batter-relative 13-zone`.
- Zone frequency and reachability never remove a zone from future recommendation candidates.
- Main-evaluation snapshots are marked unstable after the first lineup substitution.
- All as-of features must have timestamps strictly before the target pitch.
- User-facing English Markdown documents must have Korean review counterparts. At minimum,
  maintain `README.md` and `README.ko.md` together; generated reports must include a Korean
  summary.

## Planned File Structure

```text
pyproject.toml
README.md
README.ko.md
configs/
  base.yaml
data/
  .gitignore
artifacts/
  .gitignore
reports/generated/
  .gitignore
scripts/
  check.sh
src/baseball_zerobase/
  __init__.py
  cli.py
  config.py
  paths.py
  data/
    __init__.py
    contracts.py
    outcomes.py
    splits.py
    manifest.py
    zone_mapper.py
    statcast.py
    game_feed.py
    starter_lineup.py
    snapshots.py
    eligibility.py
    validation.py
  baseline/
    __init__.py
    behavior.py
    transition.py
  simulation/
    __init__.py
    state.py
    inning.py
  evaluation/
    __init__.py
    metrics.py
    rolling.py
tests/
  conftest.py
  fixtures/
    game_feed_minimal.json
  test_config.py
  data/
    test_splits.py
    test_manifest.py
    test_outcomes.py
    test_zone_mapper.py
    test_statcast.py
    test_game_feed.py
    test_starter_lineup.py
    test_snapshots.py
    test_eligibility.py
    test_validation.py
  baseline/
    test_behavior.py
    test_transition.py
  simulation/
    test_state.py
    test_inning.py
  evaluation/
    test_metrics.py
    test_rolling.py
  test_cli_smoke.py
```

## Task 1: Create the Clean-Room Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `README.ko.md`
- Create: `configs/base.yaml`
- Create: `scripts/check.sh`
- Create: `src/baseball_zerobase/__init__.py`
- Create: `src/baseball_zerobase/cli.py`
- Create: `tests/test_cli_smoke.py`
- Create: `tests/test_docs_language.py`
- Create: `data/.gitignore`
- Create: `artifacts/.gitignore`
- Create: `reports/generated/.gitignore`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
# tests/test_cli_smoke.py
from typer.testing import CliRunner

from baseball_zerobase.cli import app


def test_cli_help_lists_pipeline_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout
```

```python
# tests/test_docs_language.py
from pathlib import Path


def test_korean_readme_exists_for_user_review() -> None:
    readme_ko = Path("README.ko.md")
    assert readme_ko.exists()
    assert "클린룸" in readme_ko.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the smoke test and verify it fails**

Run: `uv run pytest tests/test_cli_smoke.py tests/test_docs_language.py -q`

Expected: FAIL because the new package and CLI do not exist.

- [ ] **Step 3: Create the package and quality configuration**

Use this dependency set in `pyproject.toml`:

```toml
[project]
name = "baseball-zerobase"
version = "0.1.0"
description = "Leakage-safe MLB starting-pitcher strategy research pipeline"
requires-python = ">=3.12,<3.13"
dependencies = [
  "httpx>=0.28",
  "numpy>=2.2",
  "pandas>=2.3",
  "polars>=1.30",
  "pyarrow>=19",
  "pybaseball>=2.2.7",
  "pydantic>=2.11",
  "pydantic-settings>=2.9",
  "pyyaml>=6.0",
  "scikit-learn>=1.7",
  "scipy>=1.15",
  "tenacity>=9.1",
  "typer>=0.16",
]

[project.scripts]
baseball-zerobase = "baseball_zerobase.cli:app"

[dependency-groups]
dev = [
  "pyright>=1.1",
  "pytest>=8.4",
  "pytest-cov>=6.2",
  "ruff>=0.12",
]

[tool.hatch.build.targets.wheel]
packages = ["src/baseball_zerobase"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

Create a Typer app with a working version command. Pipeline commands are added only by the tasks that fully implement them:

```python
# src/baseball_zerobase/cli.py
import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo("0.1.0")
```

Create `scripts/check.sh` to run Ruff, Pyright, and pytest. Do not modify the already-dirty root `.gitignore`; create local ignore files in `data/`, `artifacts/`, and `reports/generated/` instead:

```gitignore
*
!.gitignore
```

Create `README.md` in English and `README.ko.md` as its Korean review translation. Both files
must state that this is a clean-room rewrite, the project is restricted to MLB starting pitchers,
and Milestone 1-2 builds only leakage-safe data foundations and empirical baselines.

- [ ] **Step 4: Install and verify the scaffold**

Run:

```bash
uv sync
uv run pytest tests/test_cli_smoke.py tests/test_docs_language.py -q
uv run ruff check .
uv run pyright src tests
```

Expected: all commands PASS.

- [ ] **Step 5: Commit the scaffold**

```bash
git add pyproject.toml README.md README.ko.md configs/base.yaml scripts/check.sh \
  src/baseball_zerobase/__init__.py src/baseball_zerobase/cli.py \
  tests/test_cli_smoke.py tests/test_docs_language.py \
  data/.gitignore artifacts/.gitignore reports/generated/.gitignore uv.lock
git commit -m "chore: scaffold clean-room data baseline project"
```

## Task 2: Implement Configuration, Paths, and Locked-Data Guard

**Files:**
- Create: `src/baseball_zerobase/config.py`
- Create: `src/baseball_zerobase/paths.py`
- Create: `src/baseball_zerobase/data/__init__.py`
- Create: `src/baseball_zerobase/data/splits.py`
- Create: `tests/test_config.py`
- Create: `tests/data/test_splits.py`
- Modify: `configs/base.yaml`

- [ ] **Step 1: Write failing configuration and split-guard tests**

```python
# tests/data/test_splits.py
from datetime import date

import pytest

from pathlib import Path

from baseball_zerobase.data.splits import (
    DatasetRole,
    LockedDataError,
    classify_row,
    guard_dev_path,
    require_dev_role,
)


def test_classifies_locked_partitions() -> None:
    assert classify_row(date(2025, 10, 1), "F") is DatasetRole.LOCKED_POSTSEASON_2025
    assert classify_row(date(2026, 4, 15), "R") is DatasetRole.LOCKED_REGULAR_2026
    assert classify_row(date(2024, 7, 1), "R") is DatasetRole.DEV_REGULAR


def test_dev_guard_rejects_locked_role() -> None:
    with pytest.raises(LockedDataError):
        require_dev_role(DatasetRole.LOCKED_REGULAR_2026)


def test_dev_guard_rejects_locked_path(tmp_path: Path) -> None:
    locked = tmp_path / "data" / "locked"
    with pytest.raises(LockedDataError):
        guard_dev_path(locked / "snapshots.parquet", locked)
```

```python
# tests/test_config.py
from baseball_zerobase.config import Settings


def test_settings_resolve_paths_under_project_root(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    assert settings.raw_dir == tmp_path / "data" / "raw"
    assert settings.locked_dir == tmp_path / "data" / "locked"
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `uv run pytest tests/test_config.py tests/data/test_splits.py -q`

Expected: FAIL because settings and split guards do not exist.

- [ ] **Step 3: Implement settings and split contracts**

```python
# src/baseball_zerobase/data/splits.py
from datetime import date
from enum import StrEnum


class DatasetRole(StrEnum):
    DEV_REGULAR = "dev_regular"
    LOCKED_POSTSEASON_2025 = "locked_postseason_2025"
    LOCKED_REGULAR_2026 = "locked_regular_2026"
    EXCLUDED = "excluded"


class LockedDataError(RuntimeError):
    pass


def classify_row(game_date: date, game_type: str) -> DatasetRole:
    if game_date.year == 2025 and game_type in {"F", "D", "L", "W"}:
        return DatasetRole.LOCKED_POSTSEASON_2025
    if date(2026, 3, 25) <= game_date <= date(2026, 5, 31) and game_type == "R":
        return DatasetRole.LOCKED_REGULAR_2026
    if 2022 <= game_date.year <= 2025 and game_type == "R":
        return DatasetRole.DEV_REGULAR
    return DatasetRole.EXCLUDED


def require_dev_role(role: DatasetRole) -> None:
    if role is not DatasetRole.DEV_REGULAR:
        raise LockedDataError(f"development pipeline cannot read dataset role {role}")


def guard_dev_path(path: Path, locked_dir: Path) -> None:
    if path.resolve().is_relative_to(locked_dir.resolve()):
        raise LockedDataError(f"development pipeline cannot read locked path {path}")
```

Implement `Settings` with Pydantic Settings, `project_root`, derived directories, thresholds, and `random_seed=42`. Keep all thresholds in `configs/base.yaml`, including:

```yaml
starter:
  prior_two_season_pitches: 1000
baseline:
  transition_min_support: 50
  behavior_min_support: 25
simulation:
  trials: 2000
  max_pitches_per_inning: 150
```

Implement path construction in `paths.py` so callers do not concatenate data paths themselves:

```python
def statcast_partition(root: Path, role: DatasetRole, start: date, end: date) -> Path:
    base = root / "data" / ("locked/raw/statcast" if role is not DatasetRole.DEV_REGULAR else "raw/statcast")
    return base / f"start={start.isoformat()}_end={end.isoformat()}.parquet"


def require_dev_input(path: Path, settings: Settings) -> Path:
    guard_dev_path(path, settings.locked_dir)
    return path
```

- [ ] **Step 4: Verify configuration and locked-data tests**

Run: `uv run pytest tests/test_config.py tests/data/test_splits.py -q`

Expected: PASS.

- [ ] **Step 5: Commit configuration contracts**

```bash
git add configs/base.yaml src/baseball_zerobase/config.py src/baseball_zerobase/paths.py \
  src/baseball_zerobase/data/__init__.py src/baseball_zerobase/data/splits.py \
  tests/test_config.py tests/data/test_splits.py
git commit -m "feat: add configuration and locked data guard"
```

## Task 3: Define Core Data Contracts and Batter-Relative 13 Zones

**Files:**
- Create: `src/baseball_zerobase/data/contracts.py`
- Create: `src/baseball_zerobase/data/outcomes.py`
- Create: `src/baseball_zerobase/data/zone_mapper.py`
- Create: `tests/data/test_outcomes.py`
- Create: `tests/data/test_zone_mapper.py`

- [ ] **Step 1: Write failing zone and contract tests**

```python
# tests/data/test_zone_mapper.py
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
```

```python
# tests/data/test_outcomes.py
from baseball_zerobase.data.contracts import OutcomeLabel
from baseball_zerobase.data.outcomes import map_outcome


def test_terminal_event_overrides_pitch_description() -> None:
    assert map_outcome(description="hit_into_play", event="home_run") is OutcomeLabel.HOME_RUN


def test_nonterminal_pitch_description_is_preserved() -> None:
    assert map_outcome(description="foul", event=None) is OutcomeLabel.FOUL
```

- [ ] **Step 2: Run the zone tests and verify they fail**

Run: `uv run pytest tests/data/test_zone_mapper.py tests/data/test_outcomes.py -q`

Expected: FAIL because the contracts and mapper do not exist.

- [ ] **Step 3: Implement immutable contracts and the zone mapper**

Define these enums and Pydantic models in `contracts.py`:

```python
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
    model_config = ConfigDict(frozen=True)
    outcome: OutcomeLabel
    balls_after: int
    strikes_after: int
    outs_after: int
    runners_after: int
    runs_scored: int
    plate_appearance_ended: bool
    half_inning_ended: bool
    terminal_reason: str
```

Implement the mapper with:

- horizontal strike-zone bounds fixed at `-0.83` and `0.83` feet
- vertical bounds from `sz_bot` and `sz_top`
- vertical outside zones taking precedence over horizontal outside zones
- `relative_x = plate_x` for right-handed batters and `-plate_x` for left-handed batters
- positive relative X meaning inside
- missing or invalid inputs returning `None`

Implement `map_outcome` with terminal `events` taking precedence over `description`:

- `single`, `double`, `triple`, `home_run`, `walk`, `hit_by_pitch`, `strikeout` map directly
- `field_out`, `force_out`, `grounded_into_double_play`, `double_play`, `triple_play`, `sac_fly`, `sac_bunt`, and `fielders_choice_out` map to `IN_PLAY_OUT`
- `field_error`, `fielders_choice`, and `catcher_interf` map to `REACH_OTHER`
- nonterminal ball, called-strike, swinging-strike, and foul descriptions map explicitly
- unrecognized values map to `OTHER` and are counted by validation

- [ ] **Step 4: Run contract and zone tests**

Run:

```bash
uv run pytest tests/data/test_zone_mapper.py tests/data/test_outcomes.py -q
uv run pyright src/baseball_zerobase/data
```

Expected: PASS.

- [ ] **Step 5: Commit contracts and zone mapping**

```bash
git add src/baseball_zerobase/data/contracts.py src/baseball_zerobase/data/outcomes.py \
  src/baseball_zerobase/data/zone_mapper.py tests/data/test_outcomes.py \
  tests/data/test_zone_mapper.py
git commit -m "feat: define pitch contracts and relative zones"
```

## Task 4: Add Immutable Manifests and Statcast Acquisition

**Files:**
- Create: `src/baseball_zerobase/data/manifest.py`
- Create: `src/baseball_zerobase/data/statcast.py`
- Create: `tests/data/test_manifest.py`
- Create: `tests/data/test_statcast.py`
- Modify: `src/baseball_zerobase/cli.py`

- [ ] **Step 1: Write failing manifest and downloader tests**

```python
# tests/data/test_manifest.py
from baseball_zerobase.data.manifest import sha256_file, write_manifest


def test_manifest_records_file_checksum(tmp_path) -> None:
    raw = tmp_path / "sample.bin"
    raw.write_bytes(b"immutable")
    manifest = write_manifest(raw, source="test", request={"start": "2024-04-01"})
    assert manifest.sha256 == sha256_file(raw)
    assert manifest.row_count is None
```

```python
# tests/data/test_statcast.py
from datetime import date

import pandas as pd

from baseball_zerobase.data.statcast import download_statcast_range


def test_download_statcast_writes_partition_and_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "baseball_zerobase.data.statcast.pybaseball_statcast",
        lambda **_: pd.DataFrame({"game_pk": [1], "game_date": ["2024-04-01"], "game_type": ["R"]}),
    )
    result = download_statcast_range(date(2024, 4, 1), date(2024, 4, 1), tmp_path)
    assert result.data_path.exists()
    assert result.manifest_path.exists()
```

- [ ] **Step 2: Run acquisition tests and verify they fail**

Run: `uv run pytest tests/data/test_manifest.py tests/data/test_statcast.py -q`

Expected: FAIL because manifest and downloader modules do not exist.

- [ ] **Step 3: Implement immutable acquisition**

`manifest.py` must:

- calculate SHA-256 in chunks
- record source, request parameters, retrieval timestamp, byte size, row count, schema names, and checksum
- write JSON beside the immutable source file
- reject overwriting a path whose checksum differs

`statcast.py` must:

- wrap `pybaseball.statcast(start_dt=..., end_dt=...)`
- immediately convert the returned pandas DataFrame to Polars
- add no derived columns
- fail before writing if any required source column is missing:

```text
game_pk, game_date, game_type, pitcher, batter, at_bat_number, pitch_number,
inning, inning_topbot, home_team, away_team, stand, p_throws,
balls, strikes, outs_when_up, on_1b, on_2b, on_3b,
pitch_type, zone, plate_x, plate_z, sz_top, sz_bot,
description, events, type, bat_score, fld_score, post_bat_score, post_fld_score
```

- write raw Parquet partitioned by requested date range
- classify every returned row with `classify_row`
- reject a response containing mixed dataset roles
- route locked roles only to `data/locked/raw/statcast`
- never read a locked partition

Add the complete acquisition command:

```python
@app.command("download-statcast")
def download_statcast_command(
    start: date = typer.Option(...),
    end: date = typer.Option(...),
    config: Path = typer.Option(Path("configs/base.yaml")),
) -> None:
    settings = load_settings(config)
    result = download_statcast_range(start, end, settings.project_root)
    typer.echo(result.data_path)
```

- [ ] **Step 4: Verify acquisition and CLI**

Run:

```bash
uv run pytest tests/data/test_manifest.py tests/data/test_statcast.py tests/test_cli_smoke.py -q
uv run ruff check src/baseball_zerobase/data tests/data
```

Expected: PASS without making network calls.

- [ ] **Step 5: Commit immutable acquisition**

```bash
git add src/baseball_zerobase/data/manifest.py src/baseball_zerobase/data/statcast.py \
  src/baseball_zerobase/cli.py tests/data/test_manifest.py tests/data/test_statcast.py
git commit -m "feat: add immutable Statcast acquisition"
```

## Task 5: Download and Normalize MLB Game Feeds

**Files:**
- Create: `src/baseball_zerobase/data/game_feed.py`
- Create: `tests/fixtures/game_feed_minimal.json`
- Create: `tests/data/test_game_feed.py`
- Modify: `src/baseball_zerobase/cli.py`

- [ ] **Step 1: Add a minimal game-feed fixture and failing normalization tests**

The fixture must contain:

- one home and one away official starter
- nine starting batters per team with batting-order codes ending in `00`
- two half-innings
- at least one nonstarter batter appearing later

```python
# tests/data/test_game_feed.py
import json

from baseball_zerobase.data.game_feed import normalize_game_feed


def test_normalize_game_feed_extracts_starters_and_initial_lineups(fixture_dir) -> None:
    feed = json.loads((fixture_dir / "game_feed_minimal.json").read_text())
    game = normalize_game_feed(feed)
    assert game.home_starter_id == 501
    assert game.away_starter_id == 601
    assert len(game.home_initial_lineup) == 9
    assert len(game.away_initial_lineup) == 9
```

- [ ] **Step 2: Run the game-feed tests and verify they fail**

Run: `uv run pytest tests/data/test_game_feed.py -q`

Expected: FAIL because `game_feed.py` does not exist.

- [ ] **Step 3: Implement feed download and normalization**

Use `httpx.Client`, `tenacity`, and this endpoint:

```text
https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
```

Implement:

```python
class NormalizedGame(BaseModel):
    game_pk: int
    game_date: date
    game_type: str
    home_team_id: int
    away_team_id: int
    home_starter_id: int
    away_starter_id: int
    home_starter_throws: str
    away_starter_throws: str
    home_initial_lineup: tuple[int, ...]
    away_initial_lineup: tuple[int, ...]
    home_initial_lineup_stands: tuple[str, ...]
    away_initial_lineup_stands: tuple[str, ...]
    game_start_timestamp: datetime
    first_substitution_at_bat: int | None


class NormalizedPitchEvent(BaseModel):
    game_pk: int
    at_bat_number: int
    pitch_number: int
    pitch_timestamp: datetime
    completed_event_timestamp: datetime
```

Normalization rules:

- official starter is the first pitcher ID listed for each team in the boxscore
- initial lineup consists of players whose numeric `battingOrder` ends in `00`, sorted by slot
- initial lineup stands are derived before the game from `gameData.players[*].batSide.code`; switch hitters use the opposite side of the official starter's known throwing hand
- each `isPitch=True` play event is normalized to `(game_pk, at_bat_number, pitch_number)` and its event timestamps
- the first substitution boundary is the earliest at-bat containing a batter not in that team’s initial lineup or any explicit substitution event
- save raw JSON immutably with a manifest
- save normalized game metadata and normalized pitch events separately under `data/normalized/games`

Add a `download-games --game-pks-parquet PATH` CLI command.

- [ ] **Step 4: Verify game-feed normalization**

Run: `uv run pytest tests/data/test_game_feed.py -q`

Expected: PASS.

- [ ] **Step 5: Commit game-feed acquisition**

```bash
git add src/baseball_zerobase/data/game_feed.py src/baseball_zerobase/cli.py \
  tests/fixtures/game_feed_minimal.json tests/data/test_game_feed.py
git commit -m "feat: normalize starters and lineups from game feeds"
```

## Task 6: Filter Official Starter Pitches and Build Stable Lineup Timelines

**Files:**
- Create: `src/baseball_zerobase/data/starter_lineup.py`
- Create: `tests/conftest.py`
- Create: `tests/data/test_starter_lineup.py`

- [ ] **Step 1: Write failing starter and lineup tests**

```python
# tests/data/test_starter_lineup.py
from baseball_zerobase.data.starter_lineup import attach_starter_and_lineup_context


def test_keeps_only_official_starter_pitches(statcast_frame, normalized_game_frame) -> None:
    result = attach_starter_and_lineup_context(statcast_frame, normalized_game_frame)
    assert result["is_official_starter_pitch"].to_list() == [True, True, False]


def test_marks_snapshots_unstable_at_first_substitution(statcast_frame, normalized_game_frame) -> None:
    result = attach_starter_and_lineup_context(statcast_frame, normalized_game_frame)
    assert result["lineup_stable"].to_list() == [True, True, False]
```

- [ ] **Step 2: Run starter-lineup tests and verify they fail**

Run: `uv run pytest tests/data/test_starter_lineup.py -q`

Expected: FAIL because starter-lineup normalization does not exist.

- [ ] **Step 3: Implement starter and lineup context**

`attach_starter_and_lineup_context` must:

- join Statcast rows to normalized games by `game_pk`
- choose the home starter in the top half and away starter in the bottom half
- flag `is_official_starter_pitch = pitcher == expected_starter_id`
- attach offense initial lineup as a nine-ID list
- attach matching initial-lineup stand codes as a nine-value list
- derive the current lineup slot from the batter ID
- set `lineup_stable=False` at and after `first_substitution_at_bat`
- preserve all rows and add flags; filtering happens only in dataset-building code

Create reusable Polars fixtures in `tests/conftest.py`:

- `statcast_frame`: three pitches in one game, with two by the official starter and one by a reliever
- `normalized_game_frame`: matching starters, nine-player lineups, stands, game start, and substitution boundary
- `prepared_pitch_frame`: two same-PA pitches followed by a PA-ending pitch
- `starter_snapshot_frame`: two games for one starter so shifted eligibility can be tested
- `baseline_snapshot_frame`: at least two actions and three transition atoms
- `valid_snapshot_frame` and `leaky_snapshot_frame`
- `initial_game_state` and `fitted_baselines`

Every fixture uses small literal dictionaries converted with `pl.DataFrame`; no fixture reads data from `/Users/song/Projects/baseball`.

- [ ] **Step 4: Verify starter and lineup behavior**

Run: `uv run pytest tests/data/test_starter_lineup.py -q`

Expected: PASS.

- [ ] **Step 5: Commit starter and lineup context**

```bash
git add src/baseball_zerobase/data/starter_lineup.py tests/conftest.py \
  tests/data/test_starter_lineup.py
git commit -m "feat: attach official starter and stable lineup context"
```

## Task 7: Build Leakage-Safe Pre-Pitch Snapshots and Transition Atoms

**Files:**
- Create: `src/baseball_zerobase/data/snapshots.py`
- Create: `tests/data/test_snapshots.py`
- Modify: `src/baseball_zerobase/cli.py`

- [ ] **Step 1: Write failing snapshot and transition tests**

```python
# tests/data/test_snapshots.py
from baseball_zerobase.data.snapshots import build_snapshots


def test_snapshot_uses_only_pre_pitch_state(prepared_pitch_frame) -> None:
    snapshots = build_snapshots(prepared_pitch_frame)
    first = snapshots.row(0, named=True)
    assert first["balls"] == 0
    assert first["strikes"] == 0
    assert first["runs_scored"] == 0
    assert first["as_of_timestamp"] < first["pitch_timestamp"]


def test_transition_atom_uses_next_observed_state(prepared_pitch_frame) -> None:
    snapshots = build_snapshots(prepared_pitch_frame)
    first = snapshots.row(0, named=True)
    assert first["balls_after"] == 0
    assert first["strikes_after"] == 1
    assert first["plate_appearance_ended"] is False
```

- [ ] **Step 2: Run snapshot tests and verify they fail**

Run: `uv run pytest tests/data/test_snapshots.py -q`

Expected: FAIL because snapshot building does not exist.

- [ ] **Step 3: Implement snapshots and observed transitions**

Snapshot columns must include:

```text
game_pk, game_date, game_type, pitch_timestamp
at_bat_number, pitch_number, inning, inning_topbot
pitcher_id, batter_id, stand, p_throws
balls, strikes, outs, runners, batting_order_slot
bat_score, fld_score, score_diff
lineup_ids, lineup_stable, is_official_starter_pitch
lineup_stands
timestamp_joined
pitch_type, relative_zone
as_of_timestamp
outcome, balls_after, strikes_after, outs_after, runners_after
runs_scored, plate_appearance_ended, half_inning_ended, terminal_reason
```

Implementation rules:

- sort by `game_pk`, half-inning, `at_bat_number`, `pitch_number`
- encode runners as bits `1`, `2`, `4`
- map the actual arrival action from `pitch_type` and relative zone
- derive `runs_scored = post_bat_score - bat_score`
- use the next row only when it belongs to the same game and half-inning
- reset count to `0-0` when a plate appearance ends
- set `half_inning_ended=True` when no next row exists in the same half-inning
- set `terminal_reason="three_outs"` for normal half-inning changes and `"game_end"` for game-ending rows with fewer than three outs
- classify unmapped or missing actions as nullable and preserve the row
- join normalized pitch events by `(game_pk, at_bat_number, pitch_number)`
- use the previous completed event timestamp as `as_of_timestamp`; use game-start timestamp for the first pitch
- set `timestamp_joined=False` and preserve rows whose event timestamp cannot be joined
- require joined rows to have `as_of_timestamp` strictly before the target pitch timestamp
- define `score_diff = fld_score - bat_score` from the starter's defensive perspective

Add a complete `build-snapshots` command that:

- accepts only development regular-season normalized paths
- calls `require_dev_role`
- writes partitioned snapshots and a dataset manifest

- [ ] **Step 4: Verify snapshots and transition atoms**

Run:

```bash
uv run pytest tests/data/test_snapshots.py tests/data/test_zone_mapper.py -q
uv run pyright src/baseball_zerobase/data/snapshots.py
```

Expected: PASS.

- [ ] **Step 5: Commit snapshot construction**

```bash
git add src/baseball_zerobase/data/snapshots.py src/baseball_zerobase/cli.py \
  tests/data/test_snapshots.py
git commit -m "feat: build leakage-safe pre-pitch snapshots"
```

## Task 8: Add As-Of Starter Eligibility and Development Dataset Builder

**Files:**
- Create: `src/baseball_zerobase/data/eligibility.py`
- Create: `tests/data/test_eligibility.py`
- Modify: `src/baseball_zerobase/data/snapshots.py`
- Modify: `src/baseball_zerobase/cli.py`

- [ ] **Step 1: Write failing as-of eligibility tests**

```python
# tests/data/test_eligibility.py
from baseball_zerobase.data.eligibility import add_starter_eligibility


def test_eligibility_counts_only_prior_games(starter_snapshot_frame) -> None:
    result = add_starter_eligibility(starter_snapshot_frame, min_prior_pitches=3)
    first_game = result.filter(result["game_pk"] == 1)
    second_game = result.filter(result["game_pk"] == 2)
    assert first_game["starter_eligible"].unique().to_list() == [False]
    assert second_game["prior_two_season_starter_pitches"].min() == 3
    assert second_game["starter_eligible"].unique().to_list() == [True]
```

- [ ] **Step 2: Run eligibility tests and verify they fail**

Run: `uv run pytest tests/data/test_eligibility.py -q`

Expected: FAIL because eligibility code does not exist.

- [ ] **Step 3: Implement prior-game-only eligibility**

Implementation rules:

- aggregate official starter pitches by pitcher and game before calculating cumulative values
- shift cumulative values by one game so the current game never contributes
- use only dates in `[game_date - 2 years, game_date)` for `prior_two_season_starter_pitches`
- calculate `current_season_prior_pitches`
- set `starter_eligible` from the configured prior-two-season threshold
- retain ineligible rows with flags for audit
- development dataset builder filters to:
  - `DatasetRole.DEV_REGULAR`
  - official starter pitch
  - stable lineup
  - starter eligible
  - non-null action
  - `timestamp_joined=True`
  - supported strategic event, excluding automatic calls and intentional walks

Add `build-dev-dataset` CLI command and write a dataset manifest containing input checksums and filter counts.

- [ ] **Step 4: Verify eligibility and development dataset rules**

Run: `uv run pytest tests/data/test_eligibility.py tests/data/test_snapshots.py -q`

Expected: PASS.

- [ ] **Step 5: Commit eligibility and dataset building**

```bash
git add src/baseball_zerobase/data/eligibility.py src/baseball_zerobase/data/snapshots.py \
  src/baseball_zerobase/cli.py tests/data/test_eligibility.py
git commit -m "feat: add as-of starter eligibility"
```

## Task 9: Implement Dataset Validation and Leakage Audits

**Files:**
- Create: `src/baseball_zerobase/data/validation.py`
- Create: `tests/data/test_validation.py`
- Modify: `src/baseball_zerobase/cli.py`

- [ ] **Step 1: Write failing validation tests**

```python
# tests/data/test_validation.py
import pytest

from baseball_zerobase.data.validation import LeakageError, audit_snapshots


def test_audit_rejects_future_as_of_timestamp(leaky_snapshot_frame) -> None:
    with pytest.raises(LeakageError):
        audit_snapshots(leaky_snapshot_frame)


def test_audit_reports_action_and_terminal_distributions(valid_snapshot_frame) -> None:
    report = audit_snapshots(valid_snapshot_frame)
    assert report.row_count > 0
    assert sum(report.relative_zone_counts.values()) == report.action_row_count
    assert report.locked_row_count == 0
```

- [ ] **Step 2: Run validation tests and verify they fail**

Run: `uv run pytest tests/data/test_validation.py -q`

Expected: FAIL because validation code does not exist.

- [ ] **Step 3: Implement fail-closed audits**

The audit must raise on:

- any locked row in a development dataset
- `as_of_timestamp >= pitch_timestamp`
- duplicated `(game_pk, at_bat_number, pitch_number)`
- invalid count, outs, runners, or relative-zone values
- official starter mismatch
- `lineup_stable=False` in the main development dataset
- negative runs or decreasing outs within a sampled observed transition
- malformed probability or terminal fields

The audit report must include:

- row and game counts
- included and excluded counts by reason
- pitch type and relative-zone distributions
- starter eligibility distribution
- transition outcome distribution
- pitch-event timestamp join rate
- unknown action and unknown outcome rates

Add `validate-dataset --input PATH --report PATH` CLI command. Generated reports go under `reports/generated/` and are not committed.

- [ ] **Step 4: Verify audits**

Run:

```bash
uv run pytest tests/data/test_validation.py -q
uv run ruff check src/baseball_zerobase/data/validation.py
```

Expected: PASS.

- [ ] **Step 5: Commit validation**

```bash
git add src/baseball_zerobase/data/validation.py src/baseball_zerobase/cli.py \
  tests/data/test_validation.py
git commit -m "feat: add dataset validation and leakage audits"
```

## Task 10: Implement the Empirical Behavior Baseline

**Files:**
- Create: `src/baseball_zerobase/baseline/__init__.py`
- Create: `src/baseball_zerobase/baseline/behavior.py`
- Create: `tests/baseline/test_behavior.py`

- [ ] **Step 1: Write failing behavior-model tests**

```python
# tests/baseline/test_behavior.py
import numpy as np

from baseball_zerobase.baseline.behavior import EmpiricalBehaviorModel


def test_behavior_probabilities_sum_to_one(baseline_snapshot_frame) -> None:
    model = EmpiricalBehaviorModel(min_support=2).fit(baseline_snapshot_frame)
    probs = model.predict_proba(balls=0, strikes=0, stand="R", p_throws="R")
    assert np.isclose(sum(probs.values()), 1.0)


def test_behavior_model_backs_off_when_context_is_sparse(baseline_snapshot_frame) -> None:
    model = EmpiricalBehaviorModel(min_support=100).fit(baseline_snapshot_frame)
    probs = model.predict_proba(balls=3, strikes=2, stand="L", p_throws="L")
    assert probs
    assert model.last_backoff_level == "global"
```

- [ ] **Step 2: Run behavior tests and verify they fail**

Run: `uv run pytest tests/baseline/test_behavior.py -q`

Expected: FAIL because the behavior baseline does not exist.

- [ ] **Step 3: Implement hierarchical empirical behavior**

The model estimates actual MLB action frequencies and uses these backoff levels:

```text
1. balls, strikes, stand, p_throws
2. balls, strikes
3. global
```

Requirements:

- action key is `(pitch_type, relative_zone)`
- use additive smoothing `alpha=0.5` over actions observed in the training set
- choose the first level meeting `min_support`
- expose `predict_proba`, `sample(rng, ...)`, `support`, and `last_backoff_level`
- never use behavior probabilities to remove future recommendation zones
- serialize to JSON containing counts, actions, settings, and training manifest hash

- [ ] **Step 4: Verify behavior baseline**

Run: `uv run pytest tests/baseline/test_behavior.py -q`

Expected: PASS.

- [ ] **Step 5: Commit behavior baseline**

```bash
git add src/baseball_zerobase/baseline/__init__.py \
  src/baseball_zerobase/baseline/behavior.py tests/baseline/test_behavior.py
git commit -m "feat: add empirical behavior baseline"
```

## Task 11: Implement the Empirical Transition Baseline

**Files:**
- Create: `src/baseball_zerobase/baseline/transition.py`
- Create: `tests/baseline/test_transition.py`

- [ ] **Step 1: Write failing transition-model tests**

```python
# tests/baseline/test_transition.py
import numpy as np

from baseball_zerobase.baseline.transition import EmpiricalTransitionModel


def test_transition_distribution_sums_to_one(baseline_snapshot_frame) -> None:
    model = EmpiricalTransitionModel(min_support=2).fit(baseline_snapshot_frame)
    distribution = model.predict_distribution(
        pitch_type="FF",
        relative_zone="middle_middle",
        balls=0,
        strikes=0,
        outs=0,
        runners=0,
        stand="R",
        p_throws="R",
    )
    assert np.isclose(sum(distribution.values()), 1.0)


def test_sampled_transition_preserves_state_invariants(baseline_snapshot_frame) -> None:
    model = EmpiricalTransitionModel(min_support=2).fit(baseline_snapshot_frame)
    atom = model.sample(np.random.default_rng(42), pitch_type="FF", relative_zone="middle_middle",
                        balls=0, strikes=0, outs=0, runners=0, stand="R", p_throws="R")
    assert atom.outs_after >= 0
    assert 0 <= atom.runners_after <= 7
```

- [ ] **Step 2: Run transition tests and verify they fail**

Run: `uv run pytest tests/baseline/test_transition.py -q`

Expected: FAIL because the transition baseline does not exist.

- [ ] **Step 3: Implement hierarchical empirical transitions**

The model predicts a distribution over immutable `TransitionAtom` values using:

```text
1. action, balls, strikes, outs, runners, stand, p_throws
2. action, balls, strikes, stand, p_throws
3. action, balls, strikes
4. action
5. global
```

Requirements:

- choose the first level meeting `min_support`
- use observed atom frequencies at the chosen level
- expose `predict_distribution`, `sample`, `support`, and `last_backoff_level`
- expose an epsilon-floored `log_probability(actual_atom, context)` for evaluation
- serialize counts and training manifest hash
- validate every fitted atom against state invariants

- [ ] **Step 4: Verify transition baseline**

Run: `uv run pytest tests/baseline/test_transition.py -q`

Expected: PASS.

- [ ] **Step 5: Commit transition baseline**

```bash
git add src/baseball_zerobase/baseline/transition.py tests/baseline/test_transition.py
git commit -m "feat: add empirical transition baseline"
```

## Task 12: Build the Deterministic Inning Simulator

**Files:**
- Create: `src/baseball_zerobase/simulation/__init__.py`
- Create: `src/baseball_zerobase/simulation/state.py`
- Create: `src/baseball_zerobase/simulation/inning.py`
- Create: `tests/simulation/test_state.py`
- Create: `tests/simulation/test_inning.py`

- [ ] **Step 1: Write failing simulator tests**

```python
# tests/simulation/test_inning.py
from baseball_zerobase.simulation.inning import InningSimulator


def test_same_seed_produces_same_run_distribution(fitted_baselines, initial_game_state) -> None:
    simulator = InningSimulator(*fitted_baselines, max_pitches=50)
    first = simulator.simulate_many(initial_game_state, trials=100, seed=42)
    second = simulator.simulate_many(initial_game_state, trials=100, seed=42)
    assert first.runs == second.runs


def test_simulation_always_terminates(fitted_baselines, initial_game_state) -> None:
    simulator = InningSimulator(*fitted_baselines, max_pitches=50)
    result = simulator.simulate_many(initial_game_state, trials=20, seed=7)
    assert result.truncated_trials == 0
```

```python
# tests/simulation/test_state.py
from baseball_zerobase.simulation.state import GameState


def test_plate_appearance_advance_rotates_batter_and_stand() -> None:
    state = GameState(
        balls=0, strikes=0, outs=0, runners=0, inning=1, score_diff=0,
        batting_order_index=0, lineup_ids=(10, 20), lineup_stands=("R", "L"),
        stand="R", p_throws="R",
    )
    advanced = state.advance_batting_order()
    assert advanced.batting_order_index == 1
    assert advanced.stand == "L"
```

- [ ] **Step 2: Run simulator tests and verify they fail**

Run: `uv run pytest tests/simulation -q`

Expected: FAIL because simulation modules do not exist.

- [ ] **Step 3: Implement game state and simulation**

```python
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
```

The simulator must:

- sample subsequent actions from `EmpiricalBehaviorModel`
- sample observed transition atoms from `EmpiricalTransitionModel`
- update count, outs, runners, score difference, and batting-order index
- advance batting order only when `plate_appearance_ended=True`
- update `stand` from `lineup_stands` whenever batting order advances
- terminate on `half_inning_ended=True`
- reject transitions that decrease outs or violate count/runners bounds
- enforce `max_pitches` and report truncations instead of silently looping
- return runs, pitch counts, zero-run probability, and 2+-run probability
- accept an optional fixed first action for later single-decision value work, without ranking actions in this milestone

- [ ] **Step 4: Verify simulator invariants and determinism**

Run:

```bash
uv run pytest tests/simulation -q
uv run pyright src/baseball_zerobase/simulation
```

Expected: PASS.

- [ ] **Step 5: Commit simulator**

```bash
git add src/baseball_zerobase/simulation tests/simulation
git commit -m "feat: add deterministic inning simulator"
```

## Task 13: Add Baseline Metrics and Rolling Validation

**Files:**
- Create: `src/baseball_zerobase/evaluation/__init__.py`
- Create: `src/baseball_zerobase/evaluation/metrics.py`
- Create: `src/baseball_zerobase/evaluation/rolling.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/evaluation/test_rolling.py`
- Modify: `src/baseball_zerobase/cli.py`

- [ ] **Step 1: Write failing metric and fold tests**

```python
# tests/evaluation/test_rolling.py
from baseball_zerobase.evaluation.rolling import rolling_folds


def test_rolling_folds_never_train_on_validation_year() -> None:
    folds = rolling_folds()
    assert folds[0].train_years == (2022,)
    assert folds[0].validation_year == 2023
    assert folds[-1].train_years == (2022, 2023, 2024)
    assert folds[-1].validation_year == 2025
```

```python
# tests/evaluation/test_metrics.py
from baseball_zerobase.evaluation.metrics import inning_distribution_metrics


def test_inning_metrics_compare_zero_and_multi_run_rates() -> None:
    metrics = inning_distribution_metrics(predicted=[0, 0, 1, 2], actual=[0, 1, 1, 3])
    assert metrics.zero_run_probability_error == 0.25
    assert metrics.multi_run_probability_error == 0.0
```

- [ ] **Step 2: Run evaluation tests and verify they fail**

Run: `uv run pytest tests/evaluation -q`

Expected: FAIL because evaluation modules do not exist.

- [ ] **Step 3: Implement rolling baseline evaluation**

Implement fixed folds:

```python
ROLLING_FOLDS = (
    Fold(train_years=(2022,), validation_year=2023),
    Fold(train_years=(2022, 2023), validation_year=2024),
    Fold(train_years=(2022, 2023, 2024), validation_year=2025),
)
```

Metrics must include:

- behavior action Top-1, Top-3, and negative log likelihood
- transition atom negative log likelihood
- transition outcome-label Brier score
- selected backoff-level distribution
- simulated versus actual inning-run mean error
- Wasserstein distance
- zero-run probability error
- 2+-run probability error
- simulation truncation rate

`evaluate_rolling` must:

- call `require_dev_role` before reading data
- fit only on fold training years
- evaluate only on the validation year
- start simulations from observed half-inning starting states
- write one JSON report per fold and a Markdown summary
- include a Korean summary section in the Markdown report so the user can review conclusions
- include dataset manifest hash and code version

Add the complete `evaluate-rolling --dataset PATH --output-dir PATH` command.

- [ ] **Step 4: Verify rolling evaluation**

Run:

```bash
uv run pytest tests/evaluation tests/test_cli_smoke.py -q
uv run ruff check src/baseball_zerobase/evaluation
```

Expected: PASS.

- [ ] **Step 5: Commit evaluation**

```bash
git add src/baseball_zerobase/evaluation src/baseball_zerobase/cli.py tests/evaluation
git commit -m "feat: add rolling baseline evaluation"
```

## Task 14: Connect the End-to-End Development Pipeline

**Files:**
- Create: `tests/test_end_to_end_pipeline.py`
- Modify: `src/baseball_zerobase/cli.py`
- Modify: `README.md`
- Modify: `README.ko.md`
- Modify: `scripts/check.sh`

- [ ] **Step 1: Write the failing end-to-end fixture test**

```python
# tests/test_end_to_end_pipeline.py
from baseball_zerobase.data.eligibility import add_starter_eligibility
from baseball_zerobase.data.snapshots import build_snapshots
from baseball_zerobase.data.starter_lineup import attach_starter_and_lineup_context
from baseball_zerobase.evaluation.rolling import evaluate_fold


def test_fixture_pipeline_builds_and_evaluates_baseline(raw_fixture_frame, normalized_game_frame) -> None:
    prepared = attach_starter_and_lineup_context(raw_fixture_frame, normalized_game_frame)
    snapshots = add_starter_eligibility(build_snapshots(prepared), min_prior_pitches=1)
    report = evaluate_fold(snapshots, train_years=(2022,), validation_year=2023, trials=20)
    assert report.transition_nll >= 0
    assert report.simulation_truncation_rate == 0
```

- [ ] **Step 2: Run the end-to-end test and verify it fails**

Run: `uv run pytest tests/test_end_to_end_pipeline.py -q`

Expected: FAIL until fixture wiring and public interfaces are consistent.

- [ ] **Step 3: Complete public interfaces and documentation**

Make public function signatures consistent with the end-to-end test. Update README with:

- the clean-room constraint
- exact development and locked partitions
- install and quality commands
- a small-range acquisition example
- the sequence of pipeline CLI commands
- warning that full-season downloads are long-running and are never part of automated tests
- warning that locked tests must not be run during development

Update `README.ko.md` with the same user-facing content in Korean, including the clean-room
constraint, split warnings, command sequence, and locked-test warning.

Add a top-level `pipeline-smoke` CLI command that uses test-size local inputs only and never downloads network data.

Update `scripts/check.sh` to run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright src tests
uv run pytest -q
```

- [ ] **Step 4: Run full verification**

Run:

```bash
scripts/check.sh
uv run baseball-zerobase pipeline-smoke
git diff --check
```

Expected:

- `scripts/check.sh` exits 0
- `pipeline-smoke` exits 0 and prints a baseline metric summary
- `git diff --check` has no output

- [ ] **Step 5: Commit the connected pipeline**

```bash
git add README.md README.ko.md scripts/check.sh src/baseball_zerobase/cli.py \
  tests/test_end_to_end_pipeline.py tests/conftest.py
git commit -m "feat: connect data and baseline pipeline"
```

## Post-Implementation Acceptance

After all tasks are complete:

1. Run `scripts/check.sh`.
2. Run `uv run baseball-zerobase pipeline-smoke`.
3. Run a network smoke download for one completed regular-season day into a temporary project root.
4. Validate the downloaded raw manifest and normalized game feed.
5. Build snapshots for that day and run `validate-dataset`.
6. Confirm that no path under `data/locked/` was read by any development command.
7. Confirm that no code, model weight, or processed data from `/Users/song/Projects/baseball` was imported or copied.
8. Confirm that user-facing English Markdown changes have Korean counterparts, especially `README.ko.md`.
9. Record the exact commands and generated report paths in the implementation completion summary.

The next implementation plan begins only after the rolling baseline reports demonstrate that the data contracts, state transitions, and simulator are trustworthy enough to compare against personalized models.
