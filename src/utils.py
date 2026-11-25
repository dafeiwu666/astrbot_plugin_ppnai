import re
from collections.abc import Iterable
from typing import overload


def with_compact_parameter_aliases(*keys: str) -> tuple[str, ...]:
    """Keep existing parameter names and add their underscore-free spellings."""
    return tuple(dict.fromkeys((*keys, *(key.replace("_", "") for key in keys))))


def match_resource_names(names: Iterable[str], query: str) -> list[str]:
    """Rank exact names, names mentioned in the query, then partial keyword hits."""
    query = query.strip().casefold()
    if not query:
        return []
    keywords = {query, *re.findall(r"\w+(?:[.-]\w+)*", query)}
    matches = []
    for name in set(names):
        normalized = name.casefold()
        if normalized == query:
            score = (0, 0, 0)
        elif (position := query.find(normalized)) >= 0:
            score = (1, position, -len(normalized))
        else:
            hits = [
                (-len(keyword), position)
                for keyword in keywords
                if (position := normalized.find(keyword)) >= 0
            ]
            if not hits:
                continue
            length, position = min(hits)
            score = (2, length, position)
        matches.append((score, name))
    return [name for _, name in sorted(matches)]


BASE64_FILE_MAGIC = [
    ("image/png", "iVBORw0KGgo"),
    ("image/jpeg", "/9j/4AAQ"),
    ("image/gif", "R0lGOD"),
    ("image/webp", "UklGR"),
]


@overload
def get_base64_mime(b64str: str, default: None = None) -> str | None: ...
@overload
def get_base64_mime(b64str: str, default: str) -> str: ...
def get_base64_mime(b64str: str, default: str | None = None) -> str | None:
    for mime, magic in BASE64_FILE_MAGIC:
        if b64str.startswith(magic):
            return mime
    return default
