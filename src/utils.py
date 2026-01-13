from typing import overload

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
