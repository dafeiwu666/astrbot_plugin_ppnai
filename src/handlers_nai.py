"""Command handlers for nai draw and nai."""

import asyncio
import os
import random
import time
import tempfile
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

from astrbot.api import logger
from astrbot.api.message_components import Image, Node, Nodes, Plain
from astrbot.api.star import StarTools

from .data_source import GenerateError, wrapped_generate
from .image_params import iter_key_values, resolve_image_params
from .llm import ReturnToLLMError, llm_generate_advanced_req
from .llm_utils import format_readable_error
from .generation_report import format_generation_report, get_input_image_bytes
from .params import _collect_images_with_replies, parse_req
from .handlers_shared import (
    apply_explicit_overrides,
    extract_batch_count,
    merge_nai_params,
)
from .queue_flow import QueueRejected, acquire_generation_semaphore, reserve_queue
from .curtain import compose_curtain
from .qr_code import build_qr_image
from .sharebin import upload_images_to_sharebin
from .quark import build_quark_result, upload_images_to_quark
from .anlas_audit import read_balance, record_generation


_LAST_REACTION_AT = 0.0


def _flag(event: Any, name: str) -> bool:
    for line in event.message_str.splitlines():
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key.lower() == name.lower():
            return value.lower() in {"true", "1", "on", "yes", "是", "开"}
    return False


def build_draw_reaction_result(plugin: Any, event: Any):
    global _LAST_REACTION_AT
    reaction = getattr(plugin.config, "reaction", None)
    if reaction is None or not reaction.enabled:
        return None
    now = time.monotonic()
    if now - _LAST_REACTION_AT < reaction.cooldown_seconds:
        return None
    comments = [item.strip() for item in str(reaction.comments).splitlines() if item.strip()]
    components = [Plain(random.choice(comments))] if comments else []
    if reaction.sticker_enabled and random.random() < reaction.sticker_probability:
        sticker_dir = StarTools.get_data_dir("astrbot_plugin_ppnai") / "reaction_stickers"
        stickers = [p for p in sticker_dir.iterdir() if p.is_file()] if sticker_dir.is_dir() else []
        if stickers:
            components.append(Image.fromFileSystem(random.choice(stickers)))
    if not components:
        return None
    _LAST_REACTION_AT = now
    return event.chain_result(components)


async def _apply_curtain(plugin: Any, event: Any, images: list[bytes]) -> list[bytes]:
    front = plugin.curtain_store.get(event.get_session_id())
    if front is None:
        return images
    mode = "gray" if "帷幕=灰度" in event.message_str or "curtain=gray" in event.message_str.lower() else plugin.config.curtain.default_mode
    output_dir = plugin.curtain_store.root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for image in images:
        _handle, filename = tempfile.mkstemp(suffix=".png", dir=output_dir)
        os.close(_handle)
        await asyncio.to_thread(Path(filename).unlink, missing_ok=True)
        try:
            result.append(await compose_curtain(front, image, Path(filename), mode, plugin.config.curtain.color_a, plugin.config.curtain.color_b, plugin.config.curtain.color_weight))
        finally:
            await asyncio.to_thread(Path(filename).unlink, missing_ok=True)
    return result


async def _send_optional_outputs(plugin: Any, event: Any, images: list[bytes]):
    qr_requested = plugin.config.general.send_qr or _flag(event, "QR")
    quark_requested = plugin.config.general.send_quark_link or _flag(event, "QK")
    if not qr_requested and not quark_requested:
        return

    if qr_requested:
        # QR is independent from Quark and uses a public image-hosting URL.
        qr_urls = await upload_images_to_sharebin(plugin, images)
        qr_images = await asyncio.gather(
            *(asyncio.to_thread(build_qr_image, url) for url in qr_urls)
        )
        if qr_images:
            yield event.chain_result([Image.fromBytes(qr) for qr in qr_images])
    if quark_requested:
        urls = await upload_images_to_quark(
            plugin,
            images,
            force=_flag(event, "QK"),
        )
        if not urls:
            yield event.plain_result(
                "夸克网盘尚未完成配置或登录，请管理员查看 AstrBot 日志中的二维码，"
                "使用夸克 APP 扫码登录后再试。"
            )
        result = build_quark_result(event, urls)
        if result is not None:
            yield result


def _strip_image_param_lines(raw_params: str) -> str:
    blocked_keys = {
        "i2i",
        "vibe_transfer",
        "character_keep",
        "vibe_transfer_info_extract",
        "vibe_transfer_ref_strength",
        "character_keep_vibe",
        "character_keep_strength",
        "QR",
        "QK",
        "qr",
        "qk",
    }
    kept_lines: list[str] = []
    for raw_line in raw_params.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, _value = line.split("=", 1)
        if key.strip() in blocked_keys:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


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
    preset_names, other_params, cs_names, image_params = (
        plugin._parse_presets_from_params(raw_input)
    )

    # 默认预设兜底：用户未指定任何 sN= 时，自动应用配置中的默认预设
    if not preset_names:
        default_name = (plugin.config.defaults.default_preset or "").strip()
        if default_name:
            default_preset = await asyncio.to_thread(
                plugin.preset_manager.get_preset, default_name
            )
            if default_preset is None:
                logger.warning(
                    f"[nai] defaults.default_preset 配置为 {default_name!r}，"
                    f"但该预设不存在，已跳过"
                )
            else:
                preset_names = [default_name]

    description = other_params.get("ds", "")

    cs_content_parts: list[str] = []
    if cs_names:
        for cs_name in cs_names:
            exists = await asyncio.to_thread(plugin.cs_store.exists, user_id, cs_name)
            if not exists:
                yield event.plain_result(f"角色保持 {cs_name} 不存在，请先使用 /cs 创建")
                return
            cs_content_parts.append(
                await asyncio.to_thread(plugin.cs_store.read, user_id, cs_name)
            )
    cs_content = "\n\n".join(cs_content_parts)

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

    uploaded_images = _collect_images_with_replies(event.message_obj.message)
    try:
        resolved_images = await resolve_image_params(
            [*image_params, *iter_key_values(preset_contents)],
            uploaded_images,
            plugin.image_library,
        )
    except Exception as e:  # noqa: BLE001
        yield event.plain_result(f"图片参数解析失败：{format_readable_error(e)}")
        return

    try:
        batch_count = extract_batch_count(
            preset_contents,
            raw_input,
            max_n=plugin.config.request.max_n,
        )
    except Exception as e:  # noqa: BLE001
        yield event.plain_result(f"参数解析失败：{format_readable_error(e)}")
        return

    merged_raw, wrappers, explicit_ids = merge_nai_params(preset_contents, raw_input)
    filtered_raw = _strip_image_param_lines(merged_raw)
    try:
        if filtered_raw.strip():
            user_req = await parse_req(
                filtered_raw,
                [],
                plugin.config,
                is_whitelisted=is_whitelisted,
            )
        else:
            user_req = None
    except Exception as e:  # noqa: BLE001
        yield event.plain_result(f"参数解析失败：{format_readable_error(e)}")
        return

    i2i_image = resolved_images.i2i_image
    vibe_transfer_images = resolved_images.vibe_transfer_images
    character_keep_image = resolved_images.character_keep_image
    vision_images = resolved_images.vision_images

    if (
        not preset_contents
        and not description
        and not vision_images
        and not i2i_image
        and not vibe_transfer_images
        and not character_keep_image
    ):
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
    resource_keys = [
        *(f"preset:{name}" for name in preset_names),
        *(f"ck:{user_id}:{name}" for name in cs_names),
        *resolved_images.resource_keys,
    ]

    logger.debug(
        f"[nai画图] presets={preset_names}, description={description[:50] if description else 'None'}"
    )

    if quota_enabled and not is_whitelisted:
        quota = plugin.user_manager.get_quota(user_id)
        if quota < batch_count:
            yield event.plain_result(
                f"你的画图次数不足，当前剩余 {quota} 次，本次需要 {batch_count} 次"
            )
            return

    consume_quota = (
        (lambda: plugin.user_manager.consume_quota_n(user_id, batch_count))
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
                    images: list[bytes] = []
                    last_req = None
                    for _ in range(batch_count):
                        req = await llm_generate_advanced_req(
                            instructions=f"画一张图\n{full_description}",
                            config=plugin.config,
                            ctx=plugin.context,
                            event=event,
                            i2i_image=i2i_image,
                            vibe_transfer_images=vibe_transfer_images,
                            character_keep_image=character_keep_image,
                            vision_images=vision_images,
                            skip_default_prompts=bool(preset_contents),
                            extra_system_prompt=cs_content,
                        )

                        if user_req is not None:
                            apply_explicit_overrides(req, user_req, explicit_ids, wrappers)

                        async def _do_generate():
                            nonlocal token
                            token = plugin._get_next_token()
                            token_index = ((plugin._token_index - 1) % len(plugin.config.request.tokens)) + 1
                            balance_before = await read_balance(plugin, token)
                            image = await wrapped_generate(
                                req,
                                plugin.config,
                                token=token,
                                client_getter=plugin.get_http_client,
                                vibe_cache=plugin.vibe_cache_manager,
                                image_cache=plugin.image_history_cache,
                                owner_id=event.get_sender_id(),
                            )
                            plugin._create_background_task(record_generation(plugin, token_index=token_index, token=token, req=req, user_id=event.get_sender_id(), sender_name=event.get_sender_name(), session=event.unified_msg_origin, outcome="success", balance_before=balance_before, batch_count=batch_count), name="nai:anlas_audit")
                            return image

                        images.append(await plugin._run_with_retry(_do_generate))
                        last_req = req

                if images and resource_keys:
                    try:
                        await asyncio.to_thread(
                            plugin.preview_manager.save_first_if_missing,
                            resource_keys,
                            images[0],
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("Failed to save generation preview")

                images = await _apply_curtain(plugin, event, images)
                await plugin.user_manager.arecord_successful_draw(
                    user_id, len(images), event.get_sender_name()
                )
                sender_id = event.get_sender_id()
                sender_name = event.get_sender_name()
                if plugin.config.general.merge_draw_to_chat_record:
                    nodes = Nodes(
                        [
                            Node(
                                uin=sender_id,
                                name=sender_name,
                                content=[Image.fromBytes(img)],
                            )
                            for img in images
                        ]
                    )
                    yield event.chain_result([nodes])
                else:
                    yield event.chain_result([Image.fromBytes(img) for img in images])
                async for result in _send_optional_outputs(plugin, event, images):
                    yield result
                reaction_result = build_draw_reaction_result(plugin, event)
                if reaction_result is not None:
                    yield reaction_result
                if last_req is not None and (
                    plugin.config.general.send_generation_details
                    or last_req.data is True
                ):
                    report = format_generation_report(
                        raw_input, last_req
                    )
                    report_content = [Plain(report)]
                    report_content.extend(
                        Image.fromBytes(image)
                        for image in get_input_image_bytes(last_req)
                    )
                    yield event.chain_result([
                        Nodes([
                            Node(
                                uin=sender_id,
                                name=sender_name,
                                content=report_content,
                            )
                        ])
                    ])
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
        parsed = await plugin._parse_args(event, is_whitelisted)
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to parse args", exc_info=e)
        yield event.plain_result(
            f"你提供的参数貌似有些问题呢 xwx\n{format_readable_error(e)}"
        )
        return

    if parsed is None:
        help_msg = await plugin.generate_help(event.unified_msg_origin)
        if plugin.config.general.help_t2i:
            try:
                pages = await plugin._render_markdown_to_images(help_msg)
                if pages:
                    yield event.chain_result([Image.fromBytes(b) for b in pages])
                else:
                    yield event.plain_result(help_msg)
            except Exception:
                logger.exception("帮助图片渲染失败")
                yield event.plain_result(help_msg)
        else:
            yield event.plain_result(help_msg)
        return

    req, batch_count, preset_names, cs_names = parsed
    user_id = plugin._get_user_id(event)
    resource_keys = [
        *(f"preset:{name}" for name in preset_names),
        *(f"ck:{user_id}:{name}" for name in cs_names),
    ]
    for line in event.message_str.splitlines():
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in {"i2i", "c_k", "ck", "character_keep", "vibe_transfer", "v_t"}:
            if value.lower() not in {"true", "false", "1", "0", "on", "off", "是", "否", "关"}:
                resource_keys.append(f"image:{value}")

    if quota_enabled and not is_whitelisted:
        can_use, reason = plugin.user_manager.can_use(user_id)
        if not can_use:
            yield event.plain_result(reason)
            return
        quota = plugin.user_manager.get_quota(user_id)
        if quota < batch_count:
            yield event.plain_result(
                f"你的画图次数不足，当前剩余 {quota} 次，本次需要 {batch_count} 次"
            )
            return

    consume_quota = (
        (lambda: plugin.user_manager.consume_quota_n(user_id, batch_count))
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
                        token_index = ((plugin._token_index - 1) % len(plugin.config.request.tokens)) + 1
                        balance_before = await read_balance(plugin, token)
                        image = await wrapped_generate(
                            req,
                            plugin.config,
                            token=token,
                            client_getter=plugin.get_http_client,
                            vibe_cache=plugin.vibe_cache_manager,
                            image_cache=plugin.image_history_cache,
                            owner_id=event.get_sender_id(),
                        )
                        plugin._create_background_task(record_generation(plugin, token_index=token_index, token=token, req=req, user_id=event.get_sender_id(), sender_name=event.get_sender_name(), session=event.unified_msg_origin, outcome="success", balance_before=balance_before, batch_count=batch_count), name="nai:anlas_audit")
                        return image

                    images: list[bytes] = []
                    for _ in range(batch_count):
                        images.append(await plugin._run_with_retry(_do_generate))

                if images and resource_keys:
                    try:
                        await asyncio.to_thread(
                            plugin.preview_manager.save_first_if_missing,
                            resource_keys,
                            images[0],
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("Failed to save generation preview")

                images = await _apply_curtain(plugin, event, images)
                await plugin.user_manager.arecord_successful_draw(
                    user_id, len(images), event.get_sender_name()
                )
                sender_id = event.get_sender_id()
                sender_name = event.get_sender_name()
                if plugin.config.general.merge_draw_to_chat_record:
                    nodes = Nodes([
                        Node(
                            uin=sender_id,
                            name=sender_name,
                            content=[Image.fromBytes(img)],
                        )
                        for img in images
                    ])
                    yield event.chain_result([nodes])
                else:
                    yield event.chain_result([Image.fromBytes(img) for img in images])
                async for result in _send_optional_outputs(plugin, event, images):
                    yield result
                reaction_result = build_draw_reaction_result(plugin, event)
                if reaction_result is not None:
                    yield reaction_result
                if plugin.config.general.send_generation_details or req.data is True:
                    report = format_generation_report(
                        event.message_str.removeprefix("nai").strip(),
                        req,
                    )
                    report_content = [Plain(report)]
                    report_content.extend(
                        Image.fromBytes(image)
                        for image in get_input_image_bytes(req)
                    )
                    yield event.chain_result([
                        Nodes([
                            Node(
                                uin=sender_id,
                                name=sender_name,
                                content=report_content,
                            )
                        ])
                    ])
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
