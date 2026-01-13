"""Command handlers for nai draw and nai."""

import asyncio
import random
from collections.abc import AsyncIterator

from astrbot import logger
from astrbot.api.message_components import Image, Node, Nodes

from .data_source import GenerateError, wrapped_generate
from .llm import ReturnToLLMError, llm_generate_advanced_req
from .llm_utils import format_readable_error


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

    vision_images = [x for x in event.message_obj.message if isinstance(x, Image)]

    preset_contents: list[str] = []
    for preset_name in preset_names:
        preset = plugin.preset_manager.get_preset(preset_name)
        if preset is None:
            yield event.plain_result(f"预设 {preset_name} 不存在，使用 nai预设列表 查看可用预设")
            return
        preset_contents.append(preset.content)

    if not preset_contents and not description and not vision_images:
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

    res = await plugin._queue.reserve(
        user_id,
        is_whitelisted=is_whitelisted,
        max_queue_size=plugin.config.request.max_queue_size,
        max_concurrent=plugin.config.request.max_concurrent,
        consume_quota=(lambda: plugin.user_manager.consume_quota(user_id))
        if quota_enabled and not is_whitelisted
        else None,
    )

    if not res.ok:
        if res.reason == "inflight":
            yield event.plain_result("你的上一张还没画完呢~")
        elif res.reason == "queue_full":
            yield event.plain_result(
                f"⚠️ 队列已满（{plugin.config.request.max_queue_size}），请稍后再试"
            )
        elif res.reason == "quota":
            yield event.plain_result("你的画图次数已用完，请/nai签到获取额度")
        return

    reserved_user = res.reserved_user
    queue_total = res.queue_total

    token = plugin._get_next_token()
    queue_status = f"（当前队列：{queue_total}）" if queue_total > 1 else ""
    yield event.plain_result(f"{random.choice(waiting_replies)}{queue_status}")

    try:
        sem = plugin._ensure_semaphore()
        async with sem:
            await plugin._queue.mark_wait_finished(
                max_concurrent=plugin.config.request.max_concurrent
            )

            req = await llm_generate_advanced_req(
                instructions=f"画一张图\n{full_description}",
                config=plugin.config,
                ctx=plugin.context,
                event=event,
                vision_images=vision_images,
                skip_default_prompts=bool(preset_contents),
            )

            async def _do_generate():
                nonlocal token
                token = plugin._get_next_token()
                return await wrapped_generate(req, plugin.config, token=token)

            image = await plugin._run_with_retry(_do_generate)

        user_id = event.get_sender_id()
        user_name = event.get_sender_name()
        nodes = Nodes([
            Node(
                uin=user_id,
                name=user_name,
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
    finally:
        await plugin._queue.release(
            user_id=user_id,
            reserved_user=reserved_user,
            max_concurrent=plugin.config.request.max_concurrent,
        )


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

    res = await plugin._queue.reserve(
        user_id,
        is_whitelisted=is_whitelisted,
        max_queue_size=plugin.config.request.max_queue_size,
        max_concurrent=plugin.config.request.max_concurrent,
        consume_quota=(lambda: plugin.user_manager.consume_quota(user_id))
        if quota_enabled and not is_whitelisted
        else None,
    )

    if not res.ok:
        if res.reason == "inflight":
            yield event.plain_result("你的上一张还没画完呢~")
        elif res.reason == "queue_full":
            yield event.plain_result(
                f"⚠️ 队列已满（{plugin.config.request.max_queue_size}），请稍后再试"
            )
        elif res.reason == "quota":
            yield event.plain_result("你的画图次数已用完，请/nai签到获取额度")
        return

    reserved_user = res.reserved_user
    queue_total = res.queue_total

    token = plugin._get_next_token()
    queue_status = f"（当前队列：{queue_total}）" if queue_total > 1 else ""
    yield event.plain_result(f"{random.choice(waiting_replies)}{queue_status}")

    try:
        sem = plugin._ensure_semaphore()
        async with sem:
            await plugin._queue.mark_wait_finished(
                max_concurrent=plugin.config.request.max_concurrent
            )

            req.token = token

            async def _do_generate():
                nonlocal token
                token = plugin._get_next_token()
                req.token = token
                return await wrapped_generate(req, plugin.config, token=token)

            image = await plugin._run_with_retry(_do_generate)

        user_id = event.get_sender_id()
        user_name = event.get_sender_name()
        nodes = Nodes([
            Node(
                uin=user_id,
                name=user_name,
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
    finally:
        await plugin._queue.release(
            user_id=user_id,
            reserved_user=reserved_user,
            max_concurrent=plugin.config.request.max_concurrent,
        )
