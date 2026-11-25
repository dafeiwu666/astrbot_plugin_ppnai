"""Helpers for resolving uploaded images used by AI draw commands."""

import asyncio
import re
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
CHARACTER_KEEP_STRENGTH_ALIASES = ["角色保持强度", "ck_strength", "c_k_s"]
_CHARACTER_KEEP_PARAMS = {
    **dict.fromkeys(CHARACTER_KEEP_KEYS, "character_keep"),
    **dict.fromkeys(CHARACTER_KEEP_MODE_KEYS, "character_keep_mode"),
    **dict.fromkeys(["character_keep_strength", *CHARACTER_KEEP_STRENGTH_ALIASES], "character_keep_strength"),
}
IMAGE_PARAM_KEYS = I2I_KEYS | VIBE_TRANSFER_KEYS | CHARACTER_KEEP_KEYS


class ResolvedImageParams(BaseModel):
    """Resolved image parameters plus images left for vision reference."""

    i2i_image: str | None = None
    vibe_transfer_images: list[str] = Field(default_factory=list)
    character_keep_images: dict[int, str] = Field(default_factory=dict)
    vision_images: list[Any] = Field(default_factory=list)
    resource_keys: list[str] = Field(default_factory=list)

    def summary(self) -> list[str]:
        parts: list[str] = []
        if self.i2i_image:
            parts.append("图生图")
        if self.vibe_transfer_images:
            parts.append(f"氛围转移×{len(self.vibe_transfer_images)}")
        if self.character_keep_images:
            parts.append(f"精准参考×{len(self.character_keep_images)}")
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


def character_keep_param(key: str) -> tuple[str, int] | None:
    if key in _CHARACTER_KEEP_PARAMS:
        return _CHARACTER_KEEP_PARAMS[key], 1
    match = re.fullmatch(r"(.+?)([1-9][0-9]*)", key)
    if match and match[1] in _CHARACTER_KEEP_PARAMS:
        return _CHARACTER_KEEP_PARAMS[match[1]], int(match[2])
    return None


def character_keep_key(param: str, index: int) -> str:
    return param if index == 1 else f"{param}{index}"


def with_implicit_character_keep(params: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Fill image selectors and consume precise references in numerical order."""
    pairs = [(key, value, character_keep_param(key)) for key, value in params]
    sources = {ref[1] for _, _, ref in pairs if ref and ref[0] == "character_keep"}
    expanded: list[tuple[str, str, int | None]] = []
    for key, value, ref in pairs:
        if ref is None:
            expanded.append((key, value, None))
            continue
        param, index = ref
        source_index = index if param == "character_keep" else None
        expanded.append((character_keep_key(param, index), value, source_index))
        if index not in sources:
            sources.add(index)
            expanded.append((character_keep_key("character_keep", index), "true", index))

    ordered_sources = iter(sorted(
        ((index, key, value) for key, value, index in expanded if index is not None),
        key=lambda source: source[0],
    ))
    return [
        (key, value) if index is None else next(ordered_sources)[1:]
        for key, value, index in expanded
    ]


def is_character_keep_source(key: str) -> bool:
    ref = character_keep_param(key)
    return ref is not None and ref[0] == "character_keep"


def has_enabled_image_params(params: Mapping[str, str]) -> bool:
    """Return whether any image-consuming parameter is enabled."""
    return any(
        (key in IMAGE_PARAM_KEYS or is_character_keep_source(key)) and _is_enabled(value)
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

    # The handlers list the command first, then presets in descending priority.
    parameters = []
    seen_references: set[tuple[str, int]] = set()
    for key, value in params:
        ref = character_keep_param(key)
        if ref and ref in seen_references:
            continue
        if ref:
            seen_references.add(ref)
        parameters.append((key, value))

    for key, value in with_implicit_character_keep(parameters):
        if key in I2I_KEYS:
            if result.i2i_image:
                raise ValueError("Param `i2i` already set")
            result.i2i_image = await resolve_reference(value, key) or None
        elif key in VIBE_TRANSFER_KEYS:
            resolved = await resolve_reference(value, key)
            if resolved:
                result.vibe_transfer_images.append(resolved)
        elif (ref := character_keep_param(key)) and ref[0] == "character_keep":
            if resolved := await resolve_reference(value, key):
                result.character_keep_images[ref[1]] = resolved

    result.vision_images = image_queue
    return result
