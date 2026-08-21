"""Persistent first-result previews for named generation resources."""

from __future__ import annotations

import json
import re
from pathlib import Path

from astrbot.api import logger


class PreviewManager:
    """Store one preview image per logical resource without affecting quotas."""

    def __init__(self, data_dir: Path) -> None:
        self.base_dir = data_dir / "previews"
        self.image_dir = self.base_dir / "images"
        self.index_file = self.base_dir / "index.json"
        self._index: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._index is not None:
            return self._index
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self._index = {}
            return self._index
        try:
            raw = json.loads(self.index_file.read_text("utf-8"))
            self._index = {
                key: value
                for key, value in raw.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        except (OSError, ValueError) as exc:
            logger.warning("[nai] preview index load failed: %s", exc)
            self._index = {}
        return self._index

    def _save(self) -> None:
        if self._index is None:
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), "utf-8"
        )

    @staticmethod
    def _filename(resource_key: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", resource_key).strip("._")
        return f"{safe or 'resource'}.png"

    def save_first_if_missing(self, resource_keys: list[str], image: bytes) -> list[str]:
        """Save ``image`` for keys without an existing preview; never overwrite."""
        index = self._load()
        saved: list[str] = []
        for key in dict.fromkeys(resource_keys):
            if key in index:
                continue
            self.image_dir.mkdir(parents=True, exist_ok=True)
            filename = self._filename(key)
            path = self.image_dir / filename
            if path.exists():
                logger.warning("[nai] preview path already exists, refusing overwrite: %s", path)
                continue
            path.write_bytes(image)
            index[key] = filename
            saved.append(key)
        if saved:
            self._save()
        return saved

    def save_or_replace(self, resource_keys: list[str], image: bytes) -> None:
        """Save a manually supplied preview, replacing an existing one."""
        index = self._load()
        self.image_dir.mkdir(parents=True, exist_ok=True)
        for key in dict.fromkeys(resource_keys):
            old_filename = index.get(key)
            filename = self._filename(key)
            path = self.image_dir / filename
            path.write_bytes(image)
            index[key] = filename
            if old_filename and old_filename != filename:
                old_path = self.image_dir / old_filename
                if old_path.is_file():
                    old_path.unlink()
        self._save()

    def read(self, resource_key: str) -> bytes | None:
        filename = self._load().get(resource_key)
        if not filename:
            return None
        path = self.image_dir / filename
        if not path.is_file():
            logger.warning("[nai] preview file missing for %s: %s", resource_key, path)
            return None
        return path.read_bytes()
