from __future__ import annotations

import json
from pathlib import Path

from baseball_zerobase.models.transition import SharedTransitionModelV0


def write_transition_artifact(model: SharedTransitionModelV0, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.to_json() + "\n", encoding="utf-8")


def read_transition_artifact(path: Path) -> SharedTransitionModelV0:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("transition artifact must be a JSON object")
    return SharedTransitionModelV0.from_dict(payload)
