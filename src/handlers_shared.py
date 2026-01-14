"""Shared helper logic for handlers.

Keep handler modules thin by extracting repeated parameter merging and
"explicit param override" behaviors.
"""

from __future__ import annotations

from .models import Req
from .params import req_model_assembler


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

    def canon(k: str) -> str | None:
        infos = appliers_map.get(k)
        return infos[0].id if infos else None

    direct_pairs: list[tuple[str, str]] = []
    if direct_text:
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

    groups: list[list[tuple[str, str]]] = []
    for p in reversed(preset_contents):
        pairs = _extract_params_from_text(p, treat_non_kv_as_tag=True)
        if pairs:
            groups.append(pairs)
    if direct_pairs:
        groups.append(direct_pairs)

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

    wrappers = {
        "prepend_tag": ", ".join(prepend_tag_parts),
        "append_tag": ", ".join(append_tag_parts),
        "prepend_negative": ", ".join(prepend_negative_parts),
        "append_negative": ", ".join(append_negative_parts),
    }
    return "\n".join(merged_lines), wrappers, explicit_ids


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

    if (
        "character_keep" in explicit_ids
        and user_req.addition
        and req.addition
        and user_req.addition.character_keep
    ):
        req.addition.character_keep = user_req.addition.character_keep

    if (
        "role" in explicit_ids
        and user_req.addition
        and req.addition
        and user_req.addition.multi_role_list
    ):
        req.addition.multi_role_list = user_req.addition.multi_role_list

    if (
        user_req.addition
        and req.addition
        and user_req.addition.vibe_transfer_list
        and (
            ("vibe_transfer_info_extract" in explicit_ids)
            or ("vibe_transfer_ref_strength" in explicit_ids)
        )
    ):
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
