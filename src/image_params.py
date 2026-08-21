"""Helpers for resolving uploaded images used by AI draw commands."""

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from astrbot.api.message_components import Image

from .image_io import aconvert_to_jpeg_for_character_keep, resolve_image, resolve_image_as_jpeg

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
    resource_keys: list[str] = Field(default_factory=list)

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
    image_library: Any | None = None,
) -> ResolvedImageParams:
    """Resolve i2i/vibe-transfer/character-keep images in /nai-compatible order."""
    image_queue = list(images)
    result = ResolvedImageParams()

    def pop_image(param_name: str) -> Image:
        if not image_queue:
            raise ValueError(f"参数 {param_name} 需要上传图片")
        return image_queue.pop(0)

    async def resolve_reference(value: str, param_name: str) -> str:
        normalized = value.strip().lower()
        if normalized in FALSE_VALUES:
            return ""
        if normalized in {"true", "1", "on", "yes", "是"}:
            return await resolve_image(pop_image(param_name))
        if image_library is None:
            raise ValueError(f"参数 {param_name} 引用了图库，但图库未启用")
        result.resource_keys.append(f"image:{value.strip()}")
        return await asyncio.to_thread(image_library.read_data_uri, value.strip())

    for key, value in params:
        if key in I2I_KEYS:
            if result.i2i_image:
                raise ValueError("Param `i2i` already set")
            result.i2i_image = await resolve_reference(value, key) or None
        elif key in VIBE_TRANSFER_KEYS:
            resolved = await resolve_reference(value, key)
            if resolved:
                result.vibe_transfer_images.append(resolved)
        elif key in CHARACTER_KEEP_KEYS:
            if result.character_keep_image:
                raise ValueError("Param `character_keep` already set")
            normalized = value.strip().lower()
            if normalized in FALSE_VALUES:
                continue
            if normalized in {"true", "1", "on", "yes", "是"}:
                # Character Reference requires the API's allowed size and JPEG format.
                result.character_keep_image = await resolve_image_as_jpeg(
                    pop_image(key)
                )
            else:
                # Apply the same resize/padding/JPEG conversion to named library images.
                library_data_uri = await resolve_reference(value, key)
                result.character_keep_image = (
                    await aconvert_to_jpeg_for_character_keep(library_data_uri)
                    if library_data_uri
                    else None
                )

    result.vision_images = image_queue
    return result
