"""Persisted auto-draw session state.

This keeps `auto_draw_info` across restarts so users don't lose their auto-draw
settings after the bot restarts.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class AutoDrawSession(BaseModel):
    enabled: bool = True
    presets: list[str] = Field(default_factory=list)
    opener_user_id: str = ""


class AutoDrawStore(BaseModel):
    sessions: dict[str, AutoDrawSession] = Field(default_factory=dict)


class AutoDrawStoreManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_file = data_dir / "auto_draw_info.json"
        self._store: AutoDrawStore | None = None

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> AutoDrawStore:
        if self._store is not None:
            return self._store
        self._ensure_dir()
        if self.data_file.exists():
            try:
                raw = json.loads(self.data_file.read_text("utf-8"))
                self._store = AutoDrawStore.model_validate(raw)
            except Exception:
                self._store = AutoDrawStore()
        else:
            self._store = AutoDrawStore()
        return self._store

    def save_from_runtime(self, auto_draw_info: dict[str, dict | None]) -> None:
        store = self.load()
        sessions: dict[str, AutoDrawSession] = {}
        for umo, state in auto_draw_info.items():
            if not state:
                continue
            try:
                sessions[umo] = AutoDrawSession.model_validate(state)
            except Exception:
                continue
        store.sessions = sessions
        self._ensure_dir()
        self.data_file.write_text(store.model_dump_json(indent=2), "utf-8")

    def to_runtime(self) -> dict[str, dict | None]:
        store = self.load()
        return {umo: sess.model_dump() for umo, sess in store.sessions.items()}
