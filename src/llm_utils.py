"""LLM-side helpers for formatting and cleanup."""

import re

from .text_sanitize import shorten_base64_segments

from astrbot.api import logger


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
        msg = shorten_base64_segments(str(e))
        e_li.append(f"{'Caused by: ' if e_li else ''}{type(e).__name__}: {msg}")
        e = e.__cause__
    return "\n".join(e_li)
