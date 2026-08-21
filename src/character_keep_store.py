"""Character keep (cs) storage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


TAG_BLOCK_PATTERN = re.compile(r"<tag>\s*(.+?)\s*</tag>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class CharacterKeepEntry:
    name: str
    content: str


class CharacterKeepStore:
    def __init__(self, base_dir: Path, cssaying_path: Path):
        self.base_dir = base_dir
        self.cssaying_path = cssaying_path

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _sanitize_segment(self, segment: str) -> str:
        s = segment.strip()
        if not s:
            raise ValueError("名称不能为空")
        # Avoid path traversal and invalid filename characters.
        s = s.replace("/", "_").replace("\\", "_")
        s = re.sub(r"[<>:\\|?*]", "_", s)
        return s

    def _get_user_dir(self, user_id: str) -> Path:
        safe_id = self._sanitize_segment(user_id)
        return self.base_dir / safe_id

    def _get_entry_path(self, user_id: str, name: str) -> Path:
        safe_name = self._sanitize_segment(name)
        return self._get_user_dir(user_id) / f"{safe_name}.txt"

    def list_names(self, user_id: str) -> list[str]:
        user_dir = self._get_user_dir(user_id)
        if not user_dir.exists():
            return []
        return sorted([p.stem for p in user_dir.glob("*.txt") if p.is_file()])

    def list_grouped(self, user_id: str | None = None) -> dict[str, list[str]]:
        if user_id is not None:
            return {user_id: self.list_names(user_id)} if self.list_names(user_id) else {}
        if not self.base_dir.exists():
            return {}
        grouped: dict[str, list[str]] = {}
        for directory in self.base_dir.iterdir():
            if directory.is_dir():
                names = sorted(p.stem for p in directory.glob("*.txt") if p.is_file())
                if names:
                    grouped[directory.name] = names
        return dict(sorted(grouped.items()))

    def exists(self, user_id: str, name: str) -> bool:
        return self._get_entry_path(user_id, name).exists()

    def read(self, user_id: str, name: str) -> str:
        path = self._get_entry_path(user_id, name)
        return path.read_text("utf-8")

    def write(self, user_id: str, name: str, content: str, *, overwrite: bool) -> None:
        path = self._get_entry_path(user_id, name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"角色保持 {name} 已存在")
        self._ensure_dir(path.parent)
        path.write_text(content, "utf-8")

    def delete(self, user_id: str, name: str) -> bool:
        path = self._get_entry_path(user_id, name)
        if not path.exists():
            return False
        path.unlink()
        return True

    def load_cssaying(self) -> str:
        if not self.cssaying_path.exists():
            raise FileNotFoundError(f"缺少提示词文件：{self.cssaying_path}")
        return self.cssaying_path.read_text("utf-8")


def extract_nai_tag(content: str) -> str | None:
    if not content:
        return None
    match = TAG_BLOCK_PATTERN.search(content)
    if match:
        return match.group(1).strip()
    return None


def replace_nai_tag(content: str, new_tag: str) -> tuple[str, bool]:
    if not content:
        return new_tag.strip(), False
    if TAG_BLOCK_PATTERN.search(content):
        replaced = TAG_BLOCK_PATTERN.sub(
            lambda _m: f"<tag>\n{new_tag.strip()}\n</tag>",
            content,
            count=1,
        )
        return replaced, True
    return new_tag.strip(), False
