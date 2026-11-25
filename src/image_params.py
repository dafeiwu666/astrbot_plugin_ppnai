"""Helpers for resolving uploaded images used by AI draw commands."""

import asyncio
import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from astrbot.api.message_components import Image

from .image_io import resolve_image

FALSE_VALUES = {"false", "0", "off", "关", "否", "no"}
TRUE_VALUES = {"true", "1", "on", "yes", "是"}
I2I_KEYS = {"i2i", "图生图"}
VIBE_TRANSFER_KEYS = {"vibe_transfer", "v_t", "氛围转移"}
VIBE_SETTING_PARAMS = {
    **dict.fromkeys(("vibe_transfer_info_extract", "v_t_i_e", "氛围转移信息提取度"), "vibe_transfer_info_extract"),
    **dict.fromkeys(("vibe_transfer_ref_strength", "v_t_r_s", "氛围转移参考强度"), "vibe_transfer_ref_strength"),
}
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
    vibe_transfer_settings: list[dict[str, str]] = Field(default_factory=list)
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
    match = re.fullmatch(r"(.+?)([0-9]+)", key)
    if match and match[1] in _CHARACTER_KEEP_PARAMS:
        index = int(match[2])
        if index > 0:
            return _CHARACTER_KEEP_PARAMS[match[1]], index
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
    saved_images: Mapping[str, str] | None = None,
    recipe_sources: Mapping[str, str] | None = None,
) -> ResolvedImageParams:
    """Resolve i2i/vibe-transfer/character-keep images in /nai-compatible order."""
    result = ResolvedImageParams()

    # The handlers list the command first, then presets in descending priority.
    parameters = []
    seen_references: set[tuple[str, int]] = set()
    for key, value in params:
        ref = ("i2i", 1) if key in I2I_KEYS else character_keep_param(key)
        if ref and ref in seen_references:
            continue
        if ref:
            seen_references.add(ref)
        parameters.append((key, value))

    pairs, bound, result.vision_images, result.resource_keys = await bind_image_sources(
        parameters, images, image_library, owner_id=owner_id, allow_all=allow_all,
        saved_images=saved_images, recipe_sources=recipe_sources,
    )
    first_vibe_overrides: dict[str, str] = {}
    for index, (key, value) in enumerate(pairs):
        if key in I2I_KEYS:
            result.i2i_image = bound.get(index)
        elif key in VIBE_TRANSFER_KEYS:
            resolved = bound.get(index)
            if resolved:
                result.vibe_transfer_images.append(resolved)
                settings = first_vibe_overrides.copy() if not result.vibe_transfer_settings else {}
                result.vibe_transfer_settings.append(settings)
        elif key in VIBE_SETTING_PARAMS:
            setting = VIBE_SETTING_PARAMS[key]
            if not result.vibe_transfer_settings:
                first_vibe_overrides[setting] = value
            elif len(result.vibe_transfer_settings) > 1 or setting not in first_vibe_overrides:
                result.vibe_transfer_settings[-1][setting] = value
        elif (ref := character_keep_param(key)) and ref[0] == "character_keep":
            if resolved := bound.get(index):
                result.character_keep_images[ref[1]] = resolved

    return result


async def bind_image_sources(
    params: Iterable[tuple[str, str]],
    images: Iterable[Image],
    image_library: Any | None,
    *,
    owner_id: str | None,
    allow_all: bool,
    saved_images: Mapping[str, str] | None = None,
    recipe_sources: Mapping[str, str] | None = None,
) -> tuple[list[tuple[str, str]], dict[int, str], list[Image], list[str]]:
    """Allocate this message's uploads before gallery fallbacks, preserving recipe images."""
    pairs = with_implicit_character_keep(params)
    uploads = list(images)
    saved_images = saved_images or {}
    recipe_sources = recipe_sources or {}
    bound: dict[int, str] = {}
    direct_uploads: list[int] = []
    recipe_uploads: list[int] = []
    gallery_fallbacks: list[int] = []
    gallery_sources: dict[int, str] = {}
    resource_keys: list[str] = []

    for index, (key, selector) in enumerate(pairs):
        if key not in IMAGE_PARAM_KEYS and not is_character_keep_source(key):
            continue
        if selector.lower() in FALSE_VALUES:
            continue
        if selector in saved_images:
            bound[index] = saved_images[selector]
            continue
        in_recipe = selector in recipe_sources
        value = recipe_sources.get(selector, selector)
        wants_upload = value.lower() in TRUE_VALUES
        if wants_upload and value.lower() != "true" and image_library is not None:
            if owner_id is None:
                gallery_exists = await asyncio.to_thread(image_library.exists, value)
            else:
                gallery_exists = await asyncio.to_thread(image_library.is_accessible, value, owner_id, allow_all)
            wants_upload = not gallery_exists
        if wants_upload:
            (recipe_uploads if in_recipe else direct_uploads).append(index)
        else:
            gallery_sources[index] = value
            if not in_recipe:
                gallery_fallbacks.append(index)

    required = [*direct_uploads, *recipe_uploads]
    if len(uploads) < len(required):
        key, _ = pairs[required[len(uploads)]]
        raise ValueError(f"参数 {key} 需要上传图片")

    spare = len(uploads) - len(required)
    direct_positions = sorted([*direct_uploads, *gallery_fallbacks[:spare]])
    upload_positions = [*direct_positions, *recipe_uploads]
    for index, image in zip(upload_positions, uploads):
        bound[index] = await resolve_image(image)

    for index, name in gallery_sources.items():
        if index in bound:
            continue
        if image_library is None:
            raise ValueError(f"参数 {pairs[index][0]} 引用了图库，但图库未启用")
        bound[index] = await asyncio.to_thread(
            image_library.read_data_uri, name, owner_id=owner_id, allow_all=allow_all,
        )
        resource_keys.append(f"image:{name}")
    return pairs, bound, uploads[len(upload_positions):], resource_keys
