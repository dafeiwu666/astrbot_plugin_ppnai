"""Session-scoped curtain image storage and mirage compositing."""
import asyncio
import hashlib
import io
import json
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image as PILImage
from astrbot.api.message_components import Image as AstrImage


class CurtainStore:
    def __init__(self, data_dir: Path, max_bytes: int = 10 * 1024 * 1024):
        self.root = data_dir / "curtain"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def _paths(self, session_id: str) -> tuple[Path, Path]:
        stem = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / f"{stem}.png", self.root / f"{stem}.json"

    def get(self, session_id: str) -> Path | None:
        image, _ = self._paths(session_id)
        return image if image.is_file() else None

    def delete(self, session_id: str) -> None:
        image, meta = self._paths(session_id)
        image.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)

    async def save_component(self, session_id: str, component: AstrImage) -> Path:
        url = str(getattr(component, "url", "") or "")
        if url.startswith(("/", "file://")):
            path = Path(url[7:] if url.startswith("file://") else url)
            if not path.is_file():
                raise ValueError("本地图片文件不存在或已被清理")
            payload = await asyncio.to_thread(path.read_bytes)
        elif url.startswith(("http://", "https://")):
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as client:
                async with client.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"下载图片失败（HTTP {response.status}）")
                    payload = await response.read()
        else:
            raise ValueError("不支持的图片地址格式")
        if len(payload) > self.max_bytes:
            raise ValueError("图片超过帷幕允许的最大体积")
        image, meta = self._paths(session_id)
        def save() -> None:
            with PILImage.open(io.BytesIO(payload)) as source:
                source.convert("RGBA").save(image, "PNG")
            meta.write_text(json.dumps({"persistent": True}), encoding="utf-8")
        await asyncio.to_thread(save)
        return image


def _compose(front_path: Path, back_bytes: bytes, output: Path, mode: str, a: float, b: float, weight: float) -> None:
    with PILImage.open(front_path) as front_raw, PILImage.open(io.BytesIO(back_bytes)) as back_raw:
        if mode == "gray":
            front = front_raw.convert("L")
            back = back_raw.convert("L").resize(front.size, PILImage.Resampling.LANCZOS)
            af, ab = np.array(front, dtype=float), np.array(back, dtype=float)
            alpha = 1.0 - (af * 0.5 / 10 + 128) / 255.0 + (ab * 0.5) / 255.0
            pixel = np.where(np.abs(alpha) > 1e-6, (ab * 0.5) / alpha, 255.0)
            rgba = np.empty((front.height, front.width, 4), dtype=np.uint8)
            rgba[:, :, :3] = np.clip(pixel, 0, 255).astype(np.uint8)[:, :, None]
            rgba[:, :, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        else:
            front = front_raw.convert("RGB")
            back = back_raw.convert("RGB").resize(front.size, PILImage.Resampling.LANCZOS)
            fa, ba = np.array(front, dtype=float), np.array(back, dtype=float)
            fg = .299 * fa[:, :, 0] + .587 * fa[:, :, 1] + .114 * fa[:, :, 2]
            bg = a * (.299 * ba[:, :, 0] + .587 * ba[:, :, 1] + .114 * ba[:, :, 2]) + b
            alpha = np.clip(255.0 - fg + bg, 1, 255).astype(np.uint8)
            alpha3 = alpha[:, :, None].astype(float)
            base = (1 - weight) * fa + weight * ba
            rgb = np.clip((base - (255 - alpha3)) / (alpha3 / 255), 0, 255).astype(np.uint8)
            rgba = np.concatenate((rgb, alpha[:, :, None]), axis=2)
        PILImage.fromarray(rgba, "RGBA").save(output, "PNG")


async def compose_curtain(front: Path, image: bytes, output: Path, mode: str, a: float, b: float, weight: float) -> bytes:
    await asyncio.to_thread(_compose, front, image, output, mode, a, b, weight)
    return await asyncio.to_thread(output.read_bytes)
