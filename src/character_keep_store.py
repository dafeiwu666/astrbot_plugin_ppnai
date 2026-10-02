"""Character Tag library storage helpers (legacy module name retained)."""

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
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

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

    def read_tag(self, user_id: str, name: str) -> str:
        """Read a character's NovelAI tags, upgrading legacy wrapped entries."""
        content = self.read(user_id, name)
        return extract_nai_tag(content) or ""

    def find_matching(self, user_id: str, query: str) -> list[tuple[str, str]]:
        """Find saved character names mentioned in a query (case-insensitive)."""
        normalized_query = query.casefold()
        matches: list[tuple[int, str, str]] = []
        for name in self.list_names(user_id):
            position = normalized_query.find(name.casefold())
            if position < 0:
                continue
            tags = self.read_tag(user_id, name)
            if tags:
                matches.append((position, name, tags))
        matches.sort(key=lambda match: (match[0], match[1]))
        return [(name, tags) for _, name, tags in matches]

    def write(self, user_id: str, name: str, content: str, *, overwrite: bool) -> None:
        path = self._get_entry_path(user_id, name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"角色Tag {name} 已存在")
        self._ensure_dir(path.parent)
        path.write_text(content, "utf-8")

    def delete(self, user_id: str, name: str) -> bool:
        path = self._get_entry_path(user_id, name)
        if not path.exists():
            return False
        path.unlink()
        return True

def extract_nai_tag(content: str) -> str | None:
    if not content:
        return None
    match = TAG_BLOCK_PATTERN.search(content)
    if match:
        return match.group(1).strip()
    # New role-library entries are stored as plain NovelAI tags. This fallback
    # also keeps old ``nn=`` entries usable without a migration step.
    return content.strip() or None


def replace_nai_tag(content: str, new_tag: str) -> tuple[str, bool]:
    return new_tag.strip(), bool(content.strip())
