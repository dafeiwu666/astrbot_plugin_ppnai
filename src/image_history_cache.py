"""Bounded, expiring cache for uploaded vibe inputs and generated images."""

from __future__ import annotations

import base64
import hashlib
import json
import time
import threading
from pathlib import Path


class ImageHistoryCache:
    def __init__(
        self,
        data_dir: Path,
        *,
        enabled: bool = False,
        vibe_enabled: bool = True,
        ttl_days: int,
        max_size_mb: int,
    ) -> None:
        # ``enabled`` controls generated-image caching. Vibe input caching is
        # intentionally independent because it is also useful for diagnostics.
        self.enabled = enabled
        self.vibe_enabled = vibe_enabled
        self.ttl_seconds = max(0, ttl_days) * 86400
        self.max_bytes = max(0, max_size_mb) * 1024 * 1024
        self.base_dir = data_dir / "image_cache"
        self.index_file = self.base_dir / "index.json"
        self._index: dict[str, dict[str, object]] | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _extension(data: bytes) -> str:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return ".webp"
        return ".bin"

    def _load(self) -> dict[str, dict[str, object]]:
        if self._index is not None:
            return self._index
        if not self.index_file.exists():
            self._index = {}
            return self._index
        raw = json.loads(self.index_file.read_text("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("图片缓存索引必须是对象")
        self._index = {str(key): value for key, value in raw.items() if isinstance(value, dict)}
        return self._index

    def _save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.index_file.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._load(), ensure_ascii=False, indent=2), "utf-8"
        )
        temporary.replace(self.index_file)

    def put_bytes(self, kind: str, data: bytes, owner_id: str) -> str | None:
        if kind == "vibe_inputs":
            if not self.vibe_enabled:
                return None
        elif not self.enabled:
            return None
        with self._lock:
            digest = hashlib.sha256(data).hexdigest()
            key = f"{kind}:{digest}"
            index = self._load()
            now = time.time()
            if key in index:
                index[key]["ts"] = now
                owners = index[key].setdefault("owner_ids", [])
                if isinstance(owners, list) and owner_id not in owners:
                    owners.append(owner_id)
                self._save()
                return key
            directory = self.base_dir / kind
            directory.mkdir(parents=True, exist_ok=True)
            filename = f"{digest}{self._extension(data)}"
            (directory / filename).write_bytes(data)
            index[key] = {
                "kind": kind,
                "file": f"{kind}/{filename}",
                "owner_id": owner_id,
                "owner_ids": [owner_id],
                "ts": now,
                "size": len(data),
            }
            self._save()
            self.cleanup()
            return key

    def put_data_uri(self, kind: str, data_uri: str, owner_id: str) -> str | None:
        encoded = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
        return self.put_bytes(kind, base64.b64decode(encoded), owner_id)

    def cleanup(self) -> int:
        if not self.enabled:
            return 0
        with self._lock:
            return self._cleanup_locked()

    def _cleanup_locked(self) -> int:
        index = self._load()
        now = time.time()
        removed = 0

        def remove(key: str) -> None:
            nonlocal removed
            entry = index.pop(key)
            path = self.base_dir / str(entry.get("file", ""))
            if path.is_file():
                path.unlink()
            removed += 1

        if self.ttl_seconds > 0:
            for key, entry in list(index.items()):
                if now - float(entry.get("ts", 0)) > self.ttl_seconds:
                    remove(key)
        total = sum(int(entry.get("size", 0)) for entry in index.values())
        if self.max_bytes > 0 and total > self.max_bytes:
            for key, entry in sorted(index.items(), key=lambda item: float(item[1].get("ts", 0))):
                if total <= self.max_bytes:
                    break
                total -= int(entry.get("size", 0))
                remove(key)
        if removed:
            self._save()
        return removed
