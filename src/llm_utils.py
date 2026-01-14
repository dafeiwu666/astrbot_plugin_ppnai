"""LLM-side helpers for formatting and cleanup."""

import re

from astrbot.api import logger

BASE64_BLOB_RE = re.compile(r"(?:data:[^;]+;base64,)?[A-Za-z0-9+/]{512,}={0,2}")


def _shorten_base64_segments(text: str) -> str:
    """Replace long base64 blobs with placeholders for readability."""

    def _replace(match: re.Match[str]) -> str:
        chunk = match.group(0)
        if chunk.startswith("data:"):
            prefix, _, payload = chunk.partition(",")
            mime = prefix[5:].split(";")[0] if len(prefix) > 5 else "unknown"
            return f"<base64:{mime},len={len(payload)}>"
        return f"<base64:len={len(chunk)}>"

    return BASE64_BLOB_RE.sub(_replace, text)


def apply_regex_replacements(content: str, regex_replacements: list[str]) -> str:
    """Apply cleanup regex rules to content."""
    if not regex_replacements:
        return content

    result = content
    for rule in regex_replacements:
        if not rule.strip():
            continue
        parts = rule.split("|||", 1)
        pattern = parts[0]
        replacement = parts[1] if len(parts) > 1 else ""
        try:
            result = re.sub(pattern, replacement, result, flags=re.DOTALL)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            continue

    if result != content:
        logger.debug(f"Regex cleanup applied: {len(content)} -> {len(result)} chars")

    return result


def format_readable_error(exc: Exception) -> str:
    e_li: list[str] = []
    e = exc
    while e:
        msg = _shorten_base64_segments(str(e))
        e_li.append(f"{'Caused by: ' if e_li else ''}{type(e).__name__}: {msg}")
        e = e.__cause__
    return "\n".join(e_li)
