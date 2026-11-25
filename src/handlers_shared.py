"""Shared helper logic for handlers.

Keep handler modules thin by extracting repeated parameter merging and
"explicit param override" behaviors.
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING

from astrbot.api.message_components import Image, Node, Nodes, Plain
from pydantic import ValidationError

from .drawing_recipe import check_drawing_slots
from .image_library import LibraryImage
from .image_params import character_keep_key, character_keep_param, is_character_keep_source
from .llm_utils import format_readable_error
from .models import Req
from .params import parse_req, req_model_assembler

if TYPE_CHECKING:
    from .config import Config
    from .drawing_recipe import DrawingRecipe


async def drawing_recipe_view(event, recipe: DrawingRecipe, title: str, preview_manager, resource_key: str):
    attachments = recipe.bound_image_sources()
    preview = await asyncio.to_thread(recipe.preview_bytes, preview_manager, resource_key)
    content = recipe.visible_content()
    if recipe.locked:
        content = f"lock=true\n{content}"
    sections = [[Plain(f"{title}\n\n{content}\n\n已保存参考图：{len(attachments)} 张")]]
    for index, (key, data_uri) in enumerate(attachments, 1):
        label = f"参考图 {index}（{key}）"
        try:
            data = base64.b64decode(data_uri.split(",", 1)[-1], validate=True)
        except ValueError:
            sections.append([Plain(f"{label} 无法读取")])
        else:
            sections.append([Plain(f"{label}\n"), Image.fromBytes(data)])
    if preview is not None:
        sections.append([Plain("预览图\n"), Image.fromBytes(preview)])
    return event.chain_result([
        Nodes([
            Node(uin=event.get_sender_id(), name=event.get_sender_name(), content=section)
            for section in sections
        ])
    ])


async def load_character_recipes(store, user_id: str, names: list[str], *, include_all: bool) -> list[tuple[str, DrawingRecipe]]:
    recipes = []
    for name in dict.fromkeys(names):
        try:
            resolved = await asyncio.to_thread(store.resolve_recipe, user_id, name, include_all=include_all)
        except FileNotFoundError:
            resolved = None
        except ValidationError:
            raise ValueError(f"角色预设 {name} 内容无法读取") from None
        if resolved is None:
            raise ValueError(f"角色预设 {name} 不存在，请先使用 /cs添加 创建")
        recipes.append(resolved)
    return recipes


def combine_drawing_recipes(
    presets: list[DrawingRecipe], characters: list[tuple[str, DrawingRecipe]],
) -> tuple[list[str], list[str], dict[str, str], list[str], dict[str, str]]:
    recipes = list(presets)
    character_tags: list[str] = []
    for _, recipe in characters:
        tags, parameters = recipe.parameters_without_tag()
        character_tags.extend(tags)
        if parameters:
            recipes.append(recipe.model_copy(update={"content": parameters}))
    recipe_sources: dict[str, str] = {}
    return (
        [recipe.runtime_content(recipe_sources) for recipe in recipes],
        [recipe.visible_content() for recipe in recipes if not recipe.locked],
        {key: image for recipe in recipes for key, image in recipe.images.items()},
        character_tags,
        recipe_sources,
    )


def drawing_recipe_entries(presets, characters, character_names: list[str]) -> list[tuple[str, DrawingRecipe]]:
    return [
        *((f"预设 {preset.title}", preset) for preset in presets),
        *((f"CS {name}", recipe) for name, (_, recipe) in zip(dict.fromkeys(character_names), characters, strict=True)),
    ]


def drawing_slot_groups(direct: str, entries: list[tuple[str, DrawingRecipe]]) -> list[tuple[str, str]]:
    return [("本次指令", direct), *((label, recipe.visible_content()) for label, recipe in entries)]


def locked_recipe_sources(entries: list[tuple[str, DrawingRecipe]]) -> frozenset[str]:
    return frozenset(label for label, recipe in entries if recipe.locked)


def format_recipe_error(exc: Exception, *, uses_locked_recipe: bool) -> str:
    if uses_locked_recipe:
        return "本次使用了锁定的预设或 CS，详细错误已隐藏。"
    return format_readable_error(exc)


def _falsy(value: str) -> bool:
    return value.strip().lower() in {"false", "0", "no", "off", "关", "否"}


def _expand_count_to_lines(key: str, value: str) -> list[str]:
    s = value.strip()
    if s.isdigit():
        n = int(s)
        return [f"{key}=true" for _ in range(n)]
    if _falsy(s):
        return []
    return [f"{key}={value}"]


def _extract_params_from_text(text: str, treat_non_kv_as_tag: bool) -> list[tuple[str, str]]:
    res: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            res.append((k.strip(), v.strip()))
        elif treat_non_kv_as_tag:
            res.append(("tag", line))
    return res


def _canon_factory(appliers_map):
    def canon(k: str) -> str | None:
        if reference_param := character_keep_param(k):
            return character_keep_key(*reference_param)
        infos = appliers_map.get(k)
        return infos[0].id if infos else None

    return canon


def _extract_direct_pairs(direct_text: str) -> list[tuple[str, str]]:
    direct_pairs: list[tuple[str, str]] = []
    if not direct_text:
        return direct_pairs

    for raw_line in direct_text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "ds":
            continue
        if k.startswith("s") and k[1:].isdigit():
            continue
        direct_pairs.append((k, v))
    return direct_pairs


def _collect_param_groups(
    preset_contents: list[str],
    direct_pairs: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    groups: list[list[tuple[str, str]]] = []
    for p in reversed(preset_contents):
        pairs = _extract_params_from_text(p, treat_non_kv_as_tag=True)
        if pairs:
            groups.append(pairs)
    if direct_pairs:
        groups.append(direct_pairs)
    return groups


def _merge_groups(groups: list[list[tuple[str, str]]], canon) -> tuple[
    dict[str, str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    set[str],
]:
    merged_scalar: dict[str, str] = {}
    tag_parts: list[str] = []
    negative_parts: list[str] = []
    prepend_tag_parts: list[str] = []
    append_tag_parts: list[str] = []
    prepend_negative_parts: list[str] = []
    append_negative_parts: list[str] = []
    multi_lines: list[str] = []
    explicit_ids: set[str] = set()

    for params in groups:
        for raw_k, raw_v in params:
            cid = canon(raw_k)
            if not cid:
                continue
            explicit_ids.add(cid)
            if cid == "tag":
                tag_parts.append(raw_v)
            elif cid == "negative":
                negative_parts.append(raw_v)
            elif cid == "prepend_tag":
                prepend_tag_parts.insert(0, raw_v)
            elif cid == "append_tag":
                append_tag_parts.append(raw_v)
            elif cid == "prepend_negative":
                prepend_negative_parts.insert(0, raw_v)
            elif cid == "append_negative":
                append_negative_parts.append(raw_v)
            elif cid in {"vibe_transfer", "role"}:
                if cid == "vibe_transfer":
                    multi_lines.extend(_expand_count_to_lines("vibe_transfer", raw_v))
                else:
                    multi_lines.append(f"role={raw_v}")
            else:
                merged_scalar[cid] = raw_v

    return (
        merged_scalar,
        tag_parts,
        negative_parts,
        prepend_tag_parts,
        append_tag_parts,
        prepend_negative_parts,
        append_negative_parts,
        multi_lines,
        explicit_ids,
    )


def _build_merged_lines(
    *,
    merged_scalar: dict[str, str],
    tag_parts: list[str],
    negative_parts: list[str],
    prepend_tag_parts: list[str],
    append_tag_parts: list[str],
    prepend_negative_parts: list[str],
    append_negative_parts: list[str],
    multi_lines: list[str],
) -> list[str]:
    merged_lines: list[str] = []
    if tag_parts:
        merged_lines.append(f"tag={', '.join(tag_parts)}")
    if prepend_tag_parts:
        merged_lines.append(f"prepend_tag={', '.join(prepend_tag_parts)}")
    if append_tag_parts:
        merged_lines.append(f"append_tag={', '.join(append_tag_parts)}")
    if prepend_negative_parts:
        merged_lines.append(f"prepend_negative={', '.join(prepend_negative_parts)}")
    if append_negative_parts:
        merged_lines.append(f"append_negative={', '.join(append_negative_parts)}")
    for k, v in merged_scalar.items():
        merged_lines.append(f"{k}={v}")
    if negative_parts:
        merged_lines.append(f"negative={', '.join(negative_parts)}")
    merged_lines.extend(multi_lines)
    return merged_lines


def merge_nai_params(
    preset_contents: list[str],
    direct_text: str = "",
) -> tuple[str, dict[str, str], set[str]]:
    """Merge preset contents and direct kv params into a unified raw param text.

    Returns:
    - merged_raw: final k=v lines
    - wrappers: prompt wrappers (prepend/append tag/negative)
    - explicit_ids: canonical param ids explicitly set by user/presets

    Notes:
    - `direct_text` supports filtering out non-params (e.g. `ds=`) and preset selectors (`s1=...`).
    """

    appliers_map = req_model_assembler.appliers_map
    canon = _canon_factory(appliers_map)
    direct_pairs = _extract_direct_pairs(direct_text)
    groups = _collect_param_groups(preset_contents, direct_pairs)

    (
        merged_scalar,
        tag_parts,
        negative_parts,
        prepend_tag_parts,
        append_tag_parts,
        prepend_negative_parts,
        append_negative_parts,
        multi_lines,
        explicit_ids,
    ) = _merge_groups(groups, canon)

    merged_lines = _build_merged_lines(
        merged_scalar=merged_scalar,
        tag_parts=tag_parts,
        negative_parts=negative_parts,
        prepend_tag_parts=prepend_tag_parts,
        append_tag_parts=append_tag_parts,
        prepend_negative_parts=prepend_negative_parts,
        append_negative_parts=append_negative_parts,
        multi_lines=multi_lines,
    )

    wrappers = {
        "prepend_tag": ", ".join(prepend_tag_parts),
        "append_tag": ", ".join(append_tag_parts),
        "prepend_negative": ", ".join(prepend_negative_parts),
        "append_negative": ", ".join(append_negative_parts),
    }
    return "\n".join(merged_lines), wrappers, explicit_ids


async def parse_llm_overrides(
    merged_raw: str,
    config: Config,
    is_whitelisted: bool,
    character_keep_images: dict[int, str],
    vibe_transfer_images: list[str] | None = None,
    vibe_transfer_settings: list[dict[str, str]] | None = None,
) -> Req | None:
    blocked_keys = {
        "i2i", "vibe_transfer", "vibe_transfer_info_extract", "vibe_transfer_ref_strength",
        "QR", "QK", "qr", "qk",
    }
    lines = [
        line for line in merged_raw.splitlines()
        if (key := line.partition("=")[0]) not in blocked_keys
        and not is_character_keep_source(key)
    ]
    images = []
    for index, image in sorted(character_keep_images.items()):
        lines.append(f"{character_keep_key('character_keep', index)}=true")
        images.append(LibraryImage(image))
    if vibe_transfer_images and vibe_transfer_settings and any(vibe_transfer_settings):
        for image, settings in zip(vibe_transfer_images, vibe_transfer_settings, strict=True):
            lines.append("vibe_transfer=true")
            lines.extend(f"{key}={value}" for key, value in settings.items())
            images.append(LibraryImage(image))
    if not lines:
        return None
    return await parse_req("\n".join(lines), images, config, is_whitelisted)


def extract_batch_count(
    preset_contents: list[str],
    direct_text: str = "",
    *,
    max_n: int = 0,
) -> int:
    direct_pairs = _extract_direct_pairs(direct_text)
    groups = _collect_param_groups(preset_contents, direct_pairs)
    batch_count = 1

    for params in groups:
        for key, value in params:
            if key != "n":
                continue
            raw_count = value.strip()
            if not raw_count:
                continue
            if not raw_count.isdigit() or int(raw_count) < 1:
                raise ValueError("参数 n 必须是大于等于 1 的整数")
            batch_count = int(raw_count)

    if max_n > 0 and batch_count > max_n:
        raise ValueError(f"参数 n 不能超过 {max_n}")

    return batch_count


def _apply_prompt_wrappers(base: str, prepend: str, append: str) -> str:
    s = base.strip()
    if prepend.strip():
        s = f"{prepend.strip()}, {s}" if s else prepend.strip()
    if append.strip():
        s = f"{s}, {append.strip()}" if s else append.strip()
    return s


def apply_explicit_overrides(
    req: Req,
    user_req: Req,
    explicit_ids: set[str],
    wrappers: dict[str, str],
) -> None:
    """Apply user-specified params (and wrapper-only changes) onto an LLM-generated req."""

    if "model" in explicit_ids:
        req.model = user_req.model
    if "size" in explicit_ids:
        req.size = user_req.size
    if "seed" in explicit_ids:
        req.seed = user_req.seed
    if "steps" in explicit_ids:
        req.steps = user_req.steps
    if "scale" in explicit_ids:
        req.scale = user_req.scale
    if "cfg" in explicit_ids:
        req.cfg = user_req.cfg
    if "sampler" in explicit_ids:
        req.sampler = user_req.sampler
    if "noise_schedule" in explicit_ids:
        req.noise_schedule = user_req.noise_schedule
    if "other" in explicit_ids:
        req.other = user_req.other
    if "i2i_force" in explicit_ids:
        req.i2i_force = user_req.i2i_force
    if "i2i_cl" in explicit_ids:
        req.i2i_cl = user_req.i2i_cl
    if "artist" in explicit_ids:
        req.artist = user_req.artist
    if "data" in explicit_ids:
        req.data = user_req.data

    if user_req.addition.character_keep:
        req.addition.character_keep = user_req.addition.character_keep
    if user_req.addition.character_keep_extra:
        req.addition.character_keep_extra = user_req.addition.character_keep_extra

    if (
        "role" in explicit_ids
        and user_req.addition
        and req.addition
        and user_req.addition.multi_role_list
    ):
        req.addition.multi_role_list = user_req.addition.multi_role_list

    if user_req.addition.vibe_transfer_list:
        req.addition.vibe_transfer_list = user_req.addition.vibe_transfer_list

    if "tag" in explicit_ids:
        req.tag = user_req.tag
    elif ("prepend_tag" in explicit_ids) or ("append_tag" in explicit_ids):
        req.tag = _apply_prompt_wrappers(
            req.tag,
            wrappers.get("prepend_tag", ""),
            wrappers.get("append_tag", ""),
        )

    if "negative" in explicit_ids:
        req.negative = user_req.negative
    elif ("prepend_negative" in explicit_ids) or ("append_negative" in explicit_ids):
        req.negative = _apply_prompt_wrappers(
            req.negative,
            wrappers.get("prepend_negative", ""),
            wrappers.get("append_negative", ""),
        )
