"""Named image library used by image-generation parameters."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


class ImageLibraryManager:
    def __init__(self, data_dir: Path) -> None:
        self.base_dir = data_dir / "image_library"
        self.image_dir = self.base_dir / "images"
        self.index_file = self.base_dir / "index.json"
        self._index: dict[str, dict[str, str]] | None = None

    @staticmethod
    def validate_name(name: str) -> str:
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("图库名称不能为空且不能超过 80 个字符")
        if any(ch in name for ch in "\\/:*?\"<>|"):
            raise ValueError("图库名称包含非法字符")
        return name

    def _load(self) -> dict[str, dict[str, str]]:
        if self._index is not None:
            return self._index
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self._index = {}
            return self._index
        try:
            raw = json.loads(self.index_file.read_text("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("图库索引必须是对象")
            self._index = {
                str(name): value
                for name, value in raw.items()
                if isinstance(value, dict) and isinstance(value.get("file"), str)
            }
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"图库索引加载失败: {exc}") from exc
        return self._index

    def _save(self) -> None:
        if self._index is None:
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), "utf-8"
        )

    def exists(self, name: str) -> bool:
        return self.validate_name(name) in self._load()

    def add(self, name: str, data_uri: str, *, overwrite: bool = False) -> None:
        name = self.validate_name(name)
        index = self._load()
        if name in index and not overwrite:
            raise FileExistsError(f"图库图片已存在: {name}")
        header, encoded = data_uri.split(",", 1)
        mime = header.split(";", 1)[0].removeprefix("data:")
        raw = base64.b64decode(encoded)
        digest = hashlib.sha256(raw).hexdigest()
        filename = f"{digest}.bin"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        (self.image_dir / filename).write_bytes(raw)
        old_entry = index.get(name)
        if old_entry and old_entry.get("file") != filename:
            old_path = self.image_dir / old_entry["file"]
            if old_path.exists():
                old_path.unlink()
        index[name] = {"file": filename, "mime": mime or "image/jpeg"}
        self._save()

    def delete(self, name: str) -> bool:
        name = self.validate_name(name)
        entry = self._load().pop(name, None)
        if entry is None:
            return False
        path = self.image_dir / entry["file"]
        if path.exists():
            path.unlink()
        self._save()
        return True

    def list_names(self) -> list[str]:
        return sorted(self._load())

    def read_data_uri(self, name: str) -> str:
        entry = self._load().get(self.validate_name(name))
        if entry is None:
            raise FileNotFoundError(f"图库图片不存在: {name}")
        path = self.image_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"图库图片文件缺失: {name}")
        mime = entry.get("mime", "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

    def read_bytes(self, name: str) -> tuple[bytes, str]:
        data_uri = self.read_data_uri(name)
        header, encoded = data_uri.split(",", 1)
        return base64.b64decode(encoded), header.split(";", 1)[0].removeprefix("data:")


class LibraryImage:
    """Minimal image component adapter for the existing request assembler."""

    def __init__(self, data_uri: str) -> None:
        self._data_uri = data_uri

    async def convert_to_base64(self) -> str:
        return self._data_uri
