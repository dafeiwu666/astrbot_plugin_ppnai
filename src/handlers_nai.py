"""Command handlers for nai draw and nai."""

import asyncio
import random
from collections.abc import AsyncIterator

from astrbot.api import logger
from astrbot.api.message_components import Image, Node, Nodes

from .data_source import GenerateError, wrapped_generate
from .llm import ReturnToLLMError, llm_generate_advanced_req
from .llm_utils import format_readable_error
from .params import _is_image_component, parse_req_with_remaining_images, req_model_assembler
from .queue_flow import QueueRejected, acquire_generation_semaphore, reserve_queue


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on", "开", "是"}


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


def _merge_nai_params(preset_contents: list[str], direct_text: str) -> tuple[str, dict[str, str], set[str]]:
    appliers_map = req_model_assembler.appliers_map

    def canon(k: str) -> str | None:
        infos = appliers_map.get(k)
        return infos[0].id if infos else None

    direct_pairs: list[tuple[str, str]] = []
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


async def handle_nai_draw(plugin, event, waiting_replies: list[str]) -> AsyncIterator:
    """Handle nai画图 command; yields AstrBot results."""
    if not plugin.config.request.tokens:
        logger.warning("配置项中 Token 列表为空，忽略本次指令响应")
        yield event.plain_result("❌ 配置项中 Token 列表为空，请管理员先配置 Token")
        return

    user_id = plugin._get_user_id(event)

    if plugin.user_manager.is_blacklisted(user_id):
        yield event.plain_result("你已被加入黑名单，无法使用画图功能")
        return

    is_whitelisted = plugin.user_manager.is_whitelisted(user_id)
    quota_enabled = plugin.config.quota.enable_quota

    if quota_enabled and not is_whitelisted:
        can_use, reason = plugin.user_manager.can_use(user_id)
        if not can_use:
            yield event.plain_result(reason)
            return

    raw_input = event.message_str.removeprefix("nai画图").strip()
    preset_names, other_params = plugin._parse_presets_from_params(raw_input)

    description = other_params.get("ds", "")

    reply_text = plugin._get_reply_text(event)
    if reply_text:
        if description:
            description = f"参考：{reply_text}\n\n{description}"
        else:
            description = f"参考：{reply_text}"

    preset_contents: list[str] = []
    for preset_name in preset_names:
        preset = plugin.preset_manager.get_preset(preset_name)
        if preset is None:
            yield event.plain_result(f"预设 {preset_name} 不存在，使用 nai预设列表 查看可用预设")
            return
        preset_contents.append(preset.content)

    merged_raw, wrappers, explicit_ids = _merge_nai_params(preset_contents, raw_input)
    try:
        if merged_raw.strip():
            user_req, remaining_images = await parse_req_with_remaining_images(
                merged_raw,
                event.message_obj.message,
                plugin.config,
                is_whitelisted=is_whitelisted,
            )
        else:
            user_req = None
            remaining_images = [x for x in event.message_obj.message if _is_image_component(x)]
    except Exception as e:  # noqa: BLE001
        yield event.plain_result(f"参数/图片解析失败：{format_readable_error(e)}")
        return

    i2i_image = (
        user_req.addition.image_to_image_base64 if user_req and user_req.addition else None
    )
    vibe_transfer_images = None
    if user_req and user_req.addition and user_req.addition.vibe_transfer_list:
        vibe_transfer_images = [
            x.base64 for x in user_req.addition.vibe_transfer_list if x.base64
        ]
    vision_images = remaining_images

    if not preset_contents and not description and not vision_images and not i2i_image and not vibe_transfer_images:
        yield event.plain_result(
            "请输入画图描述，格式：\n"
            "nai画图\n"
            "s1=猫娘\n"
            "ds=画一个可爱的女孩"
        )
        return

    full_description_parts = list(reversed(preset_contents))
    if description:
        full_description_parts.append(description)
    full_description = "\n\n".join(full_description_parts)

    logger.debug(
        f"[nai画图] presets={preset_names}, description={description[:50] if description else 'None'}"
    )

    consume_quota = (
        (lambda: plugin.user_manager.consume_quota(user_id))
        if quota_enabled and not is_whitelisted
        else None
    )

    try:
        async with reserve_queue(
            plugin,
            user_id,
            is_whitelisted=is_whitelisted,
            consume_quota=consume_quota,
        ) as reservation:
            queue_total = reservation.queue_total

            token = plugin._get_next_token()
            queue_status = f"（当前队列：{queue_total}）" if queue_total > 1 else ""
            yield event.plain_result(f"{random.choice(waiting_replies)}{queue_status}")

            try:
                async with acquire_generation_semaphore(plugin):
                    req = await llm_generate_advanced_req(
                        instructions=f"画一张图\n{full_description}",
                        config=plugin.config,
                        ctx=plugin.context,
                        event=event,
                        i2i_image=i2i_image,
                        vibe_transfer_images=vibe_transfer_images,
                        vision_images=vision_images,
                        skip_default_prompts=bool(preset_contents),
                    )

                    if user_req is not None:
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
                            and user_req.addition.character_keep
                        ):
                            req.addition.character_keep = user_req.addition.character_keep
                        if "role" in explicit_ids and user_req.addition.multi_role_list:
                            req.addition.multi_role_list = user_req.addition.multi_role_list
                        if (
                            ("vibe_transfer_info_extract" in explicit_ids)
                            or ("vibe_transfer_ref_strength" in explicit_ids)
                        ) and user_req.addition.vibe_transfer_list:
                            req.addition.vibe_transfer_list = user_req.addition.vibe_transfer_list

                        if "tag" in explicit_ids:
                            req.tag = user_req.tag
                        elif ("prepend_tag" in explicit_ids) or (
                            "append_tag" in explicit_ids
                        ):
                            req.tag = _apply_prompt_wrappers(
                                req.tag,
                                wrappers.get("prepend_tag", ""),
                                wrappers.get("append_tag", ""),
                            )
                        if "negative" in explicit_ids:
                            req.negative = user_req.negative
                        elif ("prepend_negative" in explicit_ids) or (
                            "append_negative" in explicit_ids
                        ):
                            req.negative = _apply_prompt_wrappers(
                                req.negative,
                                wrappers.get("prepend_negative", ""),
                                wrappers.get("append_negative", ""),
                            )

                    async def _do_generate():
                        nonlocal token
                        token = plugin._get_next_token()
                        return await wrapped_generate(req, plugin.config, token=token)

                    image = await plugin._run_with_retry(_do_generate)

                sender_id = event.get_sender_id()
                sender_name = event.get_sender_name()
                nodes = Nodes([
                    Node(
                        uin=sender_id,
                        name=sender_name,
                        content=[Image.fromBytes(image)],
                    )
                ])
                yield event.chain_result([nodes])
            except ReturnToLLMError as e:
                yield event.plain_result(f"画图失败：{e}")
            except asyncio.CancelledError:
                await plugin._queue.mark_wait_finished(
                    max_concurrent=plugin.config.request.max_concurrent
                )
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("nai画图 failed")
                yield event.plain_result(f"画图失败：{format_readable_error(e)}")
    except QueueRejected as e:
        if e.reason == "inflight":
            yield event.plain_result("你的上一张还没画完呢~")
        elif e.reason == "queue_full":
            yield event.plain_result(
                f"⚠️ 队列已满（{plugin.config.request.max_queue_size}），请稍后再试"
            )
        elif e.reason == "quota":
            yield event.plain_result("你的画图次数已用完，请/nai签到获取额度")
        return


async def handle_cmd_nai(plugin, event, waiting_replies: list[str]) -> AsyncIterator:
    """Handle nai command; yields AstrBot results."""
    if not plugin.config.request.tokens:
        logger.warning("配置项中 Token 列为空，忽略本次指令响应")
        yield event.plain_result("❌ 配置项中 Token 列表为空，请管理员先配置 Token")
        return

    user_id = plugin._get_user_id(event)

    if plugin.user_manager.is_blacklisted(user_id):
        yield event.plain_result("你已被加入黑名单，无法使用画图功能")
        return

    is_whitelisted = plugin.user_manager.is_whitelisted(user_id)
    quota_enabled = plugin.config.quota.enable_quota

    try:
        req = await plugin._parse_args(event, is_whitelisted)
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to parse args", exc_info=e)
        yield event.plain_result(
            f"你提供的参数貌似有些问题呢 xwx\n{format_readable_error(e)}"
        )
        return

    if req is None:
        help_msg = plugin.generate_help(event.unified_msg_origin)
        if plugin.config.general.help_t2i:
            try:
                image_paths = await plugin._render_markdown_to_images(help_msg)
                if image_paths:
                    yield event.chain_result([Image.fromFileSystem(p) for p in image_paths])
                else:
                    yield event.image_result(await plugin.text_to_image(help_msg))
            except Exception:
                logger.exception("帮助图片渲染失败")
                yield event.plain_result(help_msg)
        else:
            yield event.plain_result(help_msg)
        return

    if quota_enabled and not is_whitelisted:
        can_use, reason = plugin.user_manager.can_use(user_id)
        if not can_use:
            yield event.plain_result(reason)
            return

    consume_quota = (
        (lambda: plugin.user_manager.consume_quota(user_id))
        if quota_enabled and not is_whitelisted
        else None
    )

    try:
        async with reserve_queue(
            plugin,
            user_id,
            is_whitelisted=is_whitelisted,
            consume_quota=consume_quota,
        ) as reservation:
            queue_total = reservation.queue_total

            token = plugin._get_next_token()
            queue_status = f"（当前队列：{queue_total}）" if queue_total > 1 else ""
            yield event.plain_result(f"{random.choice(waiting_replies)}{queue_status}")

            try:
                async with acquire_generation_semaphore(plugin):
                    req.token = token

                    async def _do_generate():
                        nonlocal token
                        token = plugin._get_next_token()
                        req.token = token
                        return await wrapped_generate(req, plugin.config, token=token)

                    image = await plugin._run_with_retry(_do_generate)

                sender_id = event.get_sender_id()
                sender_name = event.get_sender_name()
                nodes = Nodes([
                    Node(
                        uin=sender_id,
                        name=sender_name,
                        content=[Image.fromBytes(image)],
                    )
                ])
                yield event.chain_result([nodes])
            except GenerateError as e:
                logger.error(f"Generation failed: {e}")
                readable = format_readable_error(e)
                extra = f" ({readable})" if readable else ""
                yield event.plain_result(
                    f"呱！画图的时候好像出现了点问题 xwx{extra}"
                )
            except asyncio.CancelledError:
                await plugin._queue.mark_wait_finished(
                    max_concurrent=plugin.config.request.max_concurrent
                )
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Failed to fetch")
                yield event.plain_result("呱！画图的时候好像出现了点奇怪问题 xwx")
    except QueueRejected as e:
        if e.reason == "inflight":
            yield event.plain_result("你的上一张还没画完呢~")
        elif e.reason == "queue_full":
            yield event.plain_result(
                f"⚠️ 队列已满（{plugin.config.request.max_queue_size}），请稍后再试"
            )
        elif e.reason == "quota":
            yield event.plain_result("你的画图次数已用完，请/nai签到获取额度")
        return
