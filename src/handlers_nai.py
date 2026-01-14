"""Command handlers for nai draw and nai."""

import asyncio
import random
from collections.abc import AsyncIterator

from astrbot.api import logger
from astrbot.api.message_components import Image, Node, Nodes

from .data_source import GenerateError, wrapped_generate
from .llm import ReturnToLLMError, llm_generate_advanced_req
from .llm_utils import format_readable_error
from .params import _is_image_component, parse_req_with_remaining_images
from .handlers_shared import apply_explicit_overrides, merge_nai_params
from .queue_flow import QueueRejected, acquire_generation_semaphore, reserve_queue


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

    merged_raw, wrappers, explicit_ids = merge_nai_params(preset_contents, raw_input)
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
            remaining_images = [
                x for x in event.message_obj.message if _is_image_component(x)
            ]
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

    if (
        not preset_contents
        and not description
        and not vision_images
        and not i2i_image
        and not vibe_transfer_images
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
                        apply_explicit_overrides(req, user_req, explicit_ids, wrappers)

                    async def _do_generate():
                        nonlocal token
                        token = plugin._get_next_token()
                        return await wrapped_generate(
                            req,
                            plugin.config,
                            token=token,
                            client_getter=plugin.get_http_client,
                        )

                    image = await plugin._run_with_retry(_do_generate)

                sender_id = event.get_sender_id()
                sender_name = event.get_sender_name()
                nodes = Nodes(
                    [
                        Node(
                            uin=sender_id,
                            name=sender_name,
                            content=[Image.fromBytes(image)],
                        )
                    ]
                )
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
                        return await wrapped_generate(
                            req,
                            plugin.config,
                            token=token,
                            client_getter=plugin.get_http_client,
                        )

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
