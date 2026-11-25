"""Helpers for resolving uploaded images used by AI draw commands."""

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from astrbot.api.message_components import Image

from .image_io import resolve_image

FALSE_VALUES = {"false", "0", "off", "关", "否", "no"}
I2I_KEYS = {"i2i", "图生图"}
VIBE_TRANSFER_KEYS = {"vibe_transfer", "v_t", "氛围转移"}
CHARACTER_KEEP_KEYS = {"character_keep", "c_k", "ck", "角色保持"}
CHARACTER_KEEP_MODE_ALIASES = ["角色参考模式", "ck_mode", "c_k_m"]
CHARACTER_KEEP_MODE_KEYS = {"character_keep_mode", *CHARACTER_KEEP_MODE_ALIASES}
IMAGE_PARAM_KEYS = I2I_KEYS | VIBE_TRANSFER_KEYS | CHARACTER_KEEP_KEYS


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


def with_implicit_character_keep(params: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """An explicit mode can consume an attachment at its position in the command."""
    pairs = list(params)
    if any(key in CHARACTER_KEEP_KEYS for key, _ in pairs):
        return pairs

    for index, (key, _) in enumerate(pairs):
        if key in CHARACTER_KEEP_MODE_KEYS:
            pairs.insert(index + 1, ("character_keep", "true"))
            break
    return pairs


def has_enabled_image_params(params: Mapping[str, str]) -> bool:
    """Return whether any image-consuming parameter is enabled."""
    return any(
        key in IMAGE_PARAM_KEYS and _is_enabled(value)
        for key, value in with_implicit_character_keep(params.items())
    )


async def resolve_image_params(
    params: Iterable[tuple[str, str]],
    images: Iterable[Image],
    image_library: Any | None = None,
    *,
    owner_id: str | None = None,
    allow_all: bool = True,
) -> ResolvedImageParams:
    """Resolve i2i/vibe-transfer/character-keep images in /nai-compatible order."""
    image_queue = list(images)
    result = ResolvedImageParams()

    def pop_image(param_name: str) -> Image:
        if not image_queue:
            raise ValueError(f"参数 {param_name} 需要上传图片")
        return image_queue.pop(0)

    async def is_library_name(value: str) -> bool:
        if image_library is None:
            return False
        try:
            if owner_id is None:
                return await asyncio.to_thread(image_library.exists, value.strip())
            return await asyncio.to_thread(
                image_library.is_accessible,
                value.strip(),
                owner_id,
                allow_all,
            )
        except (OSError, RuntimeError, ValueError):
            return False

    async def resolve_reference(value: str, param_name: str) -> str:
        normalized = value.strip().lower()
        if normalized in FALSE_VALUES:
            return ""
        if image_library is None:
            if normalized in {"true", "1", "on", "yes", "是"}:
                return await resolve_image(pop_image(param_name))
            raise ValueError(f"参数 {param_name} 引用了图库，但图库未启用")

        # 图库名称可能是数字（例如 ``1``）。优先检查图库，避免把它
        # 误判成布尔值 true，从而错误地要求用户上传图片。
        library_name = value.strip()
        if await is_library_name(library_name):
            result.resource_keys.append(f"image:{library_name}")
            return await asyncio.to_thread(
                image_library.read_data_uri,
                library_name,
                owner_id=owner_id,
                allow_all=allow_all,
            )

        if normalized in {"true", "1", "on", "yes", "是"}:
            return await resolve_image(pop_image(param_name))
        raise ValueError(f"图库图片不存在：{library_name}")

    for key, value in with_implicit_character_keep(params):
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
            result.character_keep_image = await resolve_reference(value, key) or None

    result.vision_images = image_queue
    return result
