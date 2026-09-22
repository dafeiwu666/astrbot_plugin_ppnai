"""Anonymous ShareBin hosting for standalone QR output."""
from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import quote

import httpx
from astrbot.api import logger

_SHAREBIN_UPLOAD_URL = "https://sharebin.eu/"
_FILE_PATH_RE = re.compile(
    r'''(?:href|value)=["'](?:https?://sharebin\.eu)?/files/([^"'?#\s<>]+)''',
    re.IGNORECASE,
)


async def upload_images_to_sharebin(plugin, images: Sequence[bytes]) -> list[str]:
    """Upload images anonymously and return public embed URLs."""
    if not images:
        return []
    proxy = str(getattr(plugin.config.request, "proxy", "") or "").strip() or None
    urls: list[str] = []
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30),
            proxy=proxy,
            follow_redirects=True,
            trust_env=False,
            headers={"User-Agent": "AstrBot-ppnai/1.0"},
        ) as client:
            for index, image in enumerate(images, 1):
                try:
                    response = await client.post(
                        _SHAREBIN_UPLOAD_URL,
                        data={"expiry_days": "1"},
                        files={"file": (f"ppnai-{index}.png", image, "image/png")},
                    )
                    response.raise_for_status()
                    match = _FILE_PATH_RE.search(response.text)
                    if not match:
                        raise ValueError("ShareBin response did not contain a file URL")
                    urls.append(
                        "https://sharebin.eu/embed/" + quote(match.group(1), safe="%")
                    )
                except Exception as exc:
                    logger.warning(
                        "[ppnai] ShareBin upload failed for image %s: %s", index, exc
                    )
    except Exception as exc:
        logger.warning("[ppnai] ShareBin upload unavailable: %s", exc)
    return urls
