"""Helpers for resolving uploaded images used by AI draw commands."""

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from astrbot.api.message_components import Image

from .image_io import resolve_image

FALSE_VALUES = {"false", "0", "off", "关", "否", "no"}
I2I_KEYS = {"i2i", "图生图"}
VIBE_TRANSFER_KEYS = {"vibe_transfer", "v_t", "氛围转移"}
CHARACTER_KEEP_KEYS = {"character_keep", "c_k", "ck", "角色保持"}


class ResolvedImageParams(BaseModel):
    """Resolved image parameters plus images left for vision reference."""

    i2i_image: str | None = None
    vibe_transfer_images: list[str] = Field(default_factory=list)
    character_keep_image: str | None = None
    vision_images: list[Any] = Field(default_factory=list)

    def summary(self) -> list[str]:
        parts: list[str] = []
        if self.i2i_image:
            parts.append("图生图")
        if self.vibe_transfer_images:
            parts.append(f"氛围转移×{len(self.vibe_transfer_images)}")
        if self.character_keep_image:
            parts.append("角色保持")
        return parts


def _is_enabled(value: str) -> bool:
    return value.strip().lower() not in FALSE_VALUES


def iter_key_values(texts: Iterable[str]) -> Iterable[tuple[str, str]]:
    """Yield key-value pairs from command/preset texts, ignoring non key-value lines."""
    for text in texts:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            yield key.strip(), value.strip()


def has_enabled_image_params(params: Mapping[str, str]) -> bool:
    """Return whether any image-consuming parameter is enabled."""
    for key, value in params.items():
        if key in I2I_KEYS | VIBE_TRANSFER_KEYS | CHARACTER_KEEP_KEYS and _is_enabled(
            value
        ):
            return True
    return False


async def resolve_image_params(
    params: Iterable[tuple[str, str]],
    images: Iterable[Image],
) -> ResolvedImageParams:
    """Resolve i2i/vibe-transfer/character-keep images in /nai-compatible order."""
    image_queue = list(images)
    result = ResolvedImageParams()

    def pop_image(param_name: str) -> Image:
        if not image_queue:
            raise ValueError(f"参数 {param_name} 需要上传图片")
        return image_queue.pop(0)

    for key, value in params:
        if not _is_enabled(value):
            continue
        if key in I2I_KEYS:
            if result.i2i_image:
                raise ValueError("Param `i2i` already set")
            result.i2i_image = await resolve_image(pop_image(key))
        elif key in VIBE_TRANSFER_KEYS:
            result.vibe_transfer_images.append(await resolve_image(pop_image(key)))
        elif key in CHARACTER_KEEP_KEYS:
            if result.character_keep_image:
                raise ValueError("Param `character_keep` already set")
            # 使用相同的 resolve_image() 处理，与氛围转移保持一致
            result.character_keep_image = await resolve_image(pop_image(key))

    result.vision_images = image_queue
    return result
