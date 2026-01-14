"""Small text-sanitization helpers shared across modules.

Keep this module dependency-light so it can be reused by logging/exception paths.
"""

from __future__ import annotations

import re

BASE64_BLOB_RE = re.compile(r"(?:data:[^;]+;base64,)?[A-Za-z0-9+/]{512,}={0,2}")


def shorten_base64_segments(text: str) -> str:
    """Replace long base64 blobs with placeholders for readability."""

    def _replace(match: re.Match[str]) -> str:
        chunk = match.group(0)
        if chunk.startswith("data:"):
            prefix, _, payload = chunk.partition(",")
            mime = prefix[5:].split(";")[0] if len(prefix) > 5 else "unknown"
            return f"<base64:{mime},len={len(payload)}>"
        return f"<base64:len={len(chunk)}>"

    return BASE64_BLOB_RE.sub(_replace, text)
