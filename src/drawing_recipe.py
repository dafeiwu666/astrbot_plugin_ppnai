"""Drawing parameters with private, reusable attachments and an optional preview."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from pydantic import BaseModel, Field

from .image_io import resolve_image
from .image_params import (
    FALSE_VALUES,
    I2I_KEYS,
    TRUE_VALUES,
    VIBE_SETTING_PARAMS,
    VIBE_TRANSFER_KEYS,
    character_keep_key,
    character_keep_param,
    with_implicit_character_keep,
)
from .utils import with_compact_parameter_aliases

_I2I_SETTINGS = {
    **dict.fromkeys(with_compact_parameter_aliases("i2i_force", "i_f", "重绘力度"), "i2i_force"),
    **dict.fromkeys(with_compact_parameter_aliases("i2i_cl", "图片处理"), "i2i_cl"),
}


class DrawingRecipe(BaseModel):
    content: str
    images: dict[str, str] = Field(default_factory=dict)
    preview_image: str | None = None

    def visible_content(self) -> str:
        lines = []
        for line in self.content.splitlines():
            key, separator, value = line.partition("=")
            lines.append(f"{key}=true" if separator and value.strip() in self.images else line)
        return "\n".join(lines)

    def bound_image_sources(self) -> list[tuple[str, str]]:
        """Return the parameter and image for each saved attachment in recipe order."""
        return [
            (key, self.images[value])
            for key, value in recipe_pairs(self.content)
            if value in self.images
        ]

    def runtime_content(self, recipe_sources: dict[str, str]) -> str:
        lines = []
        for key, value in with_implicit_character_keep(recipe_pairs(self.content)):
            if _source_key(key) and value not in self.images and value.lower() not in FALSE_VALUES:
                token = f"ppnai-recipe:{uuid4().hex}"
                recipe_sources[token] = value
                value = token
            lines.append(f"{key}={value}")
        return "\n".join(lines)

    def parameters_without_tag(self) -> tuple[list[str], str]:
        tags: list[str] = []
        parameters: list[str] = []
        for key, value in recipe_pairs(self.content):
            if key in {"tag", "正向提示词"}:
                tags.append(value)
            else:
                parameters.append(f"{key}={value}")
        return tags, "\n".join(parameters)

    def preview_bytes(self, preview_manager, resource_key: str) -> bytes | None:
        if "preview_image" not in self.model_fields_set:
            return preview_manager.read(resource_key)
        if self.preview_image:
            return base64.b64decode(self.preview_image.split(",", 1)[1])
        return None


def recipe_pairs(content: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_line in content.splitlines():
        if line := raw_line.strip():
            key, sep, value = line.partition("=")
            pairs.append((key.strip(), value.strip()) if sep else ("tag", line))
    return pairs


def numbered_selectors(params: Iterable[tuple[str, str]]) -> tuple[dict[str, dict[int, str]], list[str]]:
    selectors: dict[str, dict[int, str]] = {"s": {}, "cs": {}}
    source_lines: dict[tuple[str, int], str] = {}
    conflicts = []
    for key, value in params:
        if match := re.fullmatch(r"(cs|s)([0-9]+)?", key):
            family = match[1]
            index = int(match[2]) if match[2] else 1
            line = f"{key}={value}"
            if index < 1:
                conflicts.append(f"编号须从 1 开始：{line}")
            elif index in selectors[family]:
                conflicts.append(f"编号 {family}{index} 冲突：{source_lines[family, index]} 与 {line}")
            else:
                selectors[family][index] = value
                source_lines[family, index] = line
    return selectors, conflicts


def inspect_drawing_slots(groups: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """A precise-reference number or the i2i image belongs to one parameter source."""
    claims: dict[str, list[tuple[str, str]]] = defaultdict(list)
    conflicts = []
    vibe_sources: list[str] = []
    precise_sources: list[str] = []
    for label, content in groups:
        group_claims: dict[str, dict[str, str]] = defaultdict(dict)
        current_vibe: str | None = None
        vibe_settings: dict[str, str] = {}
        precise_slots: set[str] = set()
        disabled_precise_slots: set[str] = set()
        for key, value in recipe_pairs(content):
            line = f"{key}={value}"
            if ref := character_keep_param(key):
                field, index = ref
                slot = f"精准参考 #{index}"
                precise_slots.add(slot)
                if field == "character_keep" and value.lower() in FALSE_VALUES:
                    disabled_precise_slots.add(slot)
            elif key.endswith("0") and character_keep_param(key + "1"):
                conflicts.append(f"精准参考编号须从 1 开始：{label}（{line}）")
                continue
            elif key in I2I_KEYS or key in _I2I_SETTINGS:
                slot = "图生图 i2i"
                field = _I2I_SETTINGS.get(key, "i2i")
            elif key in VIBE_TRANSFER_KEYS:
                current_vibe = None if value.lower() in FALSE_VALUES else line
                vibe_settings.clear()
                if current_vibe:
                    vibe_sources.append(f"{label}（{line}）")
                continue
            elif key in VIBE_SETTING_PARAMS:
                setting = VIBE_SETTING_PARAMS[key]
                if current_vibe is None:
                    conflicts.append(f"{label} 的 {line} 缺少同一条目中对应的 v_t 参考图")
                elif previous := vibe_settings.get(setting):
                    conflicts.append(f"{label} 的 {current_vibe} 同时填写了 {previous} 与 {line}")
                vibe_settings[setting] = line
                continue
            else:
                continue
            if old_line := group_claims[slot].get(field):
                conflicts.append(f"{slot} 冲突：{label} 同时填写 {old_line} 与 {line}")
            group_claims[slot][field] = line
        for slot, fields in group_claims.items():
            summary = "、".join(fields.values())
            claims[slot].append((label, summary))
            if slot in precise_slots and slot not in disabled_precise_slots:
                precise_sources.append(f"{label}（{summary}）")
    if vibe_sources and precise_sources:
        conflicts.append(f"氛围转移和精准参考无法同用：{'；'.join(vibe_sources + precise_sources)}")
    occupied = []
    for slot, entries in claims.items():
        detail = "；".join(f"{label}（{params}）" for label, params in entries)
        occupied.append(f"{slot} · {detail}")
        if len(entries) > 1:
            conflicts.append(f"{slot} 冲突：{detail}")
    return occupied, conflicts


def check_drawing_slots(groups: Iterable[tuple[str, str]]) -> None:
    _, conflicts = inspect_drawing_slots(groups)
    if conflicts:
        raise ValueError("\n".join(conflicts) + "\n请给不同来源使用互不重复的编号。")


def recipe_reference_signature(recipes: Iterable[DrawingRecipe]) -> str:
    image_parameters = [
        [(key, value) for key, value in recipe_pairs(recipe.content)
         if character_keep_param(key) or _source_key(key) or key in VIBE_SETTING_PARAMS or key in _I2I_SETTINGS]
        for recipe in recipes
    ]
    return hashlib.sha256(json.dumps(image_parameters, ensure_ascii=False).encode()).hexdigest()


def _source_key(key: str) -> str | None:
    if key in I2I_KEYS:
        return "i2i"
    if key in VIBE_TRANSFER_KEYS:
        return "vibe_transfer"
    if (ref := character_keep_param(key)) and ref[0] == "character_keep":
        return character_keep_key(*ref)
    return None


async def prepare_drawing_recipe(plugin, event, content: str, previous: DrawingRecipe | None = None) -> DrawingRecipe:
    from .params import _collect_images_with_replies

    check_drawing_slots([("保存内容", content)])
    uploads = deque(_collect_images_with_replies(event.message_obj.message))
    previous_images: dict[str, deque[str]] = defaultdict(deque)
    if previous:
        for key, value in recipe_pairs(previous.content):
            if (source := _source_key(key)) and value in previous.images:
                previous_images[source].append(previous.images[value])

    bindings: dict[str, str] = {}
    lines = []
    for key, value in with_implicit_character_keep(recipe_pairs(content)):
        source = _source_key(key)
        if source and value.lower() in TRUE_VALUES:
            # The gallery may have a numeric name such as "1".
            is_gallery_name = await asyncio.to_thread(
                plugin.image_library.is_accessible,
                value,
                plugin._get_resource_owner(event),
                plugin._check_resource_admin(event) or plugin.config.general.list_all_resources,
            ) if value.lower() != "true" else False
            if not is_gallery_name:
                existing = previous_images[source].popleft() if previous_images[source] else None
                image = await resolve_image(uploads.popleft()) if uploads else existing
                if image is None:
                    value = "true"
                else:
                    value = f"ppnai-image:{uuid4().hex}"
                    bindings[value] = image
        lines.append(f"{key}={value}")

    preview = await resolve_image(uploads[-1]) if uploads else None
    return DrawingRecipe(content="\n".join(lines), images=bindings, preview_image=preview)


def write_recipe_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary_path = Path(file.name)
    try:
        with file:
            file.write(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
