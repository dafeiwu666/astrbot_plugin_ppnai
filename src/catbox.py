"""Anonymous Catbox hosting for standalone QR output."""
from __future__ import annotations

from collections.abc import Sequence

import httpx
from astrbot.api import logger

_CATBOX_UPLOAD_URL = "https://catbox.moe/user/api.php"


async def upload_images_to_catbox(plugin, images: Sequence[bytes]) -> list[str]:
    """Upload images anonymously and return public URLs on a best-effort basis."""
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
                        _CATBOX_UPLOAD_URL,
                        data={"reqtype": "fileupload"},
                        files={"fileToUpload": (f"ppnai-{index}.png", image, "image/png")},
                    )
                    response.raise_for_status()
                    url = response.text.strip()
                    if not url.startswith("https://files.catbox.moe/"):
                        raise ValueError(f"unexpected Catbox response: {url[:160]!r}")
                    urls.append(url)
                except Exception as exc:
                    logger.warning("[ppnai] Catbox upload failed for image %s: %s", index, exc)
    except Exception as exc:
        logger.warning("[ppnai] Catbox upload unavailable: %s", exc)
    return urls
