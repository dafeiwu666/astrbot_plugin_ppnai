"""Persisted auto-draw session state.

This keeps `auto_draw_info` across restarts so users don't lose their auto-draw
settings after the bot restarts.
"""

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from astrbot.api import logger


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
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[nai] Failed to load auto-draw store; reset to default. file={self.data_file}",
                    exc_info=e,
                )
                self._store = AutoDrawStore()
        else:
            self._store = AutoDrawStore()
        return self._store

    async def aload(self) -> AutoDrawStore:
        return await asyncio.to_thread(self.load)

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

    async def asave_from_runtime(self, auto_draw_info: dict[str, dict | None]) -> None:
        await asyncio.to_thread(self.save_from_runtime, auto_draw_info)

    def to_runtime(self) -> dict[str, dict | None]:
        store = self.load()
        return {umo: sess.model_dump() for umo, sess in store.sessions.items()}

    async def ato_runtime(self) -> dict[str, dict | None]:
        return await asyncio.to_thread(self.to_runtime)
