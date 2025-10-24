import json
from pathlib import Path
from typing import Any

DEFAULT_STATE = {"open_position": None, "pending_orders": []}


class StateStore:
    def __init__(self, path: str = "state.json"):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        try:
            if self.path.exists():
                return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return DEFAULT_STATE.copy()

    def save(self, state: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.path)
