"""Auto-draw command handlers and hook logic."""

import asyncio
import io
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

from PIL import Image as PILImage

from astrbot.api import logger
from astrbot.api.message_components import Image, Node, Nodes
from astrbot.api.provider import LLMResponse

from .data_source import wrapped_generate
from .image_io import resolve_image
from .llm import llm_generate_advanced_req
from .llm_utils import format_readable_error
from .models import ReqAdditionCharacterKeep
from .params import _is_image_component, parse_req_with_remaining_images
from .image_params import iter_key_values, resolve_image_params
from .handlers_shared import (
    apply_explicit_overrides,
    extract_batch_count,
    merge_nai_params,
)
from .queue_flow import QueueRejected, acquire_generation_semaphore, reserve_queue


try:
    from astrbot.api.star import StarTools
    _ORIGINALS_DIR = StarTools.get_data_dir("astrbot_plugin_ppnai") / "auto_draw_originals"
except Exception:
    _ORIGINALS_DIR = Path(__file__).parent.parent / "data" / "auto_draw_originals"

ORIGINALS_DIR = _ORIGINALS_DIR
JPEG_QUALITY = 85


def _save_original_and_compress(img_bytes: bytes, max_bytes: int, save_original: bool = True) -> bytes:
    """Save original image to disk and return a platform-safe compressed version.

    Args:
        img_bytes: raw bytes from NAI.
        max_bytes: threshold in bytes above which compression kicks in.
        save_original: if True, save raw image to plugin_data/auto_draw_originals/.

    Returns:
        JPEG bytes under max_bytes, or the original bytes if already small enough.
    """
    original_len = len(img_bytes)

    # Save original (if enabled)
    if save_original:
        ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        is_png = img_bytes[:4] == b'\x89PNG'
        ext = ".png" if is_png else ".jpg"
        filename = f"{ts}_{original_len}{ext}"
        filepath = ORIGINALS_DIR / filename
        filepath.write_bytes(img_bytes)
        logger.info(f"[auto_draw] 原图已保存: {filepath} ({original_len:,} bytes)")

    # If already under limit, return as-is
    if original_len <= max_bytes:
        logger.info(f"[auto_draw] 图片 {original_len:,} bytes ≤ {max_bytes:,}, 无需压缩")
        return img_bytes

    # Compress with PIL
    pil_img = PILImage.open(io.BytesIO(img_bytes))
    orig_w, orig_h = pil_img.size
    rgb_img = pil_img.convert("RGB")

    # Try progressive quality reduction
    for quality in [95, 90, 85, 80, 75, 70, 65, 60]:
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=quality, optimize=True)
        compressed = buf.getvalue()
        if len(compressed) <= max_bytes:
            logger.info(
                f"[auto_draw] 压缩完成: {original_len:,} → {len(compressed):,} bytes "
                f"(JPEG q={quality}, {orig_w}x{orig_h})"
            )
            return compressed

    # Last resort: resize to half dimensions
    half_w, half_h = max(1, orig_w // 2), max(1, orig_h // 2)
    rgb_img = rgb_img.resize((half_w, half_h), PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb_img.save(buf, format="JPEG", quality=80, optimize=True)
    compressed = buf.getvalue()
    logger.warning(
        f"[auto_draw] 极端压缩: {original_len:,} → {len(compressed):,} bytes "
        f"(resize {orig_w}x{orig_h}→{half_w}x{half_h}, JPEG q=80)"
    )
    return compressed


async def handle_auto_draw_off(plugin, event) -> AsyncIterator:
    plugin.auto_draw_info.pop(event.unified_msg_origin, None)
    if hasattr(plugin, "persist_auto_draw_info"):
        await plugin.persist_auto_draw_info()
    yield event.plain_result("❌ 自动画图已关闭")


def _strip_image_param_lines(raw_params: str) -> str:
    blocked_keys = {
        "i2i",
        "vibe_transfer",
        "character_keep",
        "vibe_transfer_info_extract",
        "vibe_transfer_ref_strength",
        "character_keep_vibe",
        "character_keep_strength",
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


async def _enable_auto_draw(
    plugin, event, raw_input: str, require_preset: bool
) -> tuple[bool, str]:
    umo = event.unified_msg_origin
    user_id = plugin._get_user_id(event)
    preset_names, _other_params, cs_names, image_params = (
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

    if require_preset and not preset_names:
        return False, "请使用键值对格式设置预设，例如：\nnai自动画图\ns1=猫娘"

    preset_contents: list[str] = []
    for preset_name in preset_names:
        preset = await asyncio.to_thread(plugin.preset_manager.get_preset, preset_name)
        if preset is None:
            return False, f"预设 {preset_name} 不存在，使用 nai预设列表 查看可用预设"
        preset_contents.append(preset.content)

    if cs_names:
        for cs_name in cs_names:
            exists = await asyncio.to_thread(plugin.cs_store.exists, user_id, cs_name)
            if not exists:
                return False, f"角色保持 {cs_name} 不存在，请先使用 /cs 创建"

    uploaded_images = [
        x for x in event.message_obj.message if _is_image_component(x)
    ]
    try:
        resolved_images = await resolve_image_params(
            [*image_params, *iter_key_values(preset_contents)],
            uploaded_images,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"图片参数解析失败：{format_readable_error(e)}"

    vision_images = [
        await resolve_image(img) if isinstance(img, Image) else img
        for img in resolved_images.vision_images
    ]

    plugin.auto_draw_info[umo] = {
        "enabled": True,
        "presets": preset_names,
        "opener_user_id": user_id,
        "cs_names": cs_names,
        "i2i_image": resolved_images.i2i_image,
        "vibe_transfer_images": resolved_images.vibe_transfer_images,
        "character_keep_image": resolved_images.character_keep_image,
        "vision_images": vision_images,
    }
    if hasattr(plugin, "persist_auto_draw_info"):
        await plugin.persist_auto_draw_info()

    image_summary = resolved_images.summary()
    extra = f"\n图片参数：{', '.join(image_summary)}" if image_summary else ""
    if preset_names:
        preset_str = ", ".join(f"#{name}" for name in preset_names)
        return True, (
            f"✅ 自动画图已开启\n"
            f"使用预设：{preset_str}{extra}\n"
            f"主 AI 的回复将与预设内容结合后生成图片\n"
            f"⚠️ 后续触发的画图将消耗你的额度"
        )
    return True, (
        f"✅ 自动画图已开启{extra}\n"
        f"主 AI 的回复将被自动分析生成图片\n"
        f"⚠️ 后续触发的画图将消耗你的额度"
    )


async def handle_auto_draw_on(plugin, event) -> AsyncIterator:
    user_id = plugin._get_user_id(event)

    if await asyncio.to_thread(plugin.user_manager.is_blacklisted, user_id):
        yield event.plain_result("你已被加入黑名单，无法开启自动画图")
        return

    raw_input = event.message_str.removeprefix("nai自动画图开").strip()
    _ok, message = await _enable_auto_draw(
        plugin, event, raw_input, require_preset=False
    )
    yield event.plain_result(message)


async def handle_auto_draw(plugin, event) -> AsyncIterator:
    umo = event.unified_msg_origin
    user_id = plugin._get_user_id(event)
    raw_input = event.message_str.removeprefix("nai自动画图").strip()

    if raw_input:
        if await asyncio.to_thread(plugin.user_manager.is_blacklisted, user_id):
            yield event.plain_result("你已被加入黑名单，无法开启自动画图")
            return

        _ok, message = await _enable_auto_draw(
            plugin, event, raw_input, require_preset=True
        )
        yield event.plain_result(message)
        return

    current = plugin.auto_draw_info.get(umo)
    if current is None:
        yield event.plain_result(
            "当前会话自动画图状态：❌ 关闭\n\n"
            "使用 nai自动画图开 来开启自动画图"
        )
        return

    presets = current.get("presets", [])
    cs_names = current.get("cs_names") or []
    if not cs_names:
        legacy_name = current.get("cs_name", "")
        if legacy_name:
            cs_names = [legacy_name]
    opener_id = current.get("opener_user_id", "")
    opener_quota = await asyncio.to_thread(plugin.user_manager.get_quota, opener_id)
    is_whitelisted = await asyncio.to_thread(plugin.user_manager.is_whitelisted, opener_id)

    status_parts = ["当前会话自动画图状态：✅ 开启"]
    if presets:
        preset_str = ", ".join(f"#{name}" for name in presets)
        status_parts.append(f"使用预设：{preset_str}")
    else:
        status_parts.append("未使用预设")
    if cs_names:
        status_parts.append(f"角色保持：{', '.join(cs_names)}")
    image_parts: list[str] = []
    if current.get("i2i_image"):
        image_parts.append("图生图")
    vibe_count = len(current.get("vibe_transfer_images") or [])
    if vibe_count:
        image_parts.append(f"氛围转移×{vibe_count}")
    if current.get("character_keep_image"):
        image_parts.append("角色保持图片")
    if image_parts:
        status_parts.append(f"图片参数：{', '.join(image_parts)}")
    status_parts.append(f"开启者：{opener_id}")
    if is_whitelisted:
        status_parts.append("额度：无限（白名单）")
    else:
        status_parts.append(f"剩余额度：{opener_quota} 次")
    status_parts.append("\n使用 nai自动画图关 来关闭")

    yield event.plain_result("\n".join(status_parts))


async def handle_llm_response_auto_draw(plugin, event, resp: LLMResponse):
    umo = event.unified_msg_origin
    auto_info = plugin.auto_draw_info.get(umo)
    if auto_info is None:
        return

    presets = auto_info.get("presets", [])
    opener_user_id = auto_info.get("opener_user_id", "")
    cs_names = auto_info.get("cs_names") or []
    if not cs_names:
        legacy_name = auto_info.get("cs_name", "")
        if legacy_name:
            cs_names = [legacy_name]

    if not plugin.config.request.tokens:
        return

    ai_response = resp.completion_text if hasattr(resp, "completion_text") else str(resp)
    if not ai_response or len(ai_response.strip()) < 10:
        return

    if await asyncio.to_thread(plugin.user_manager.is_blacklisted, opener_user_id):
        logger.debug(f"[nai] Auto draw: opener {opener_user_id} is blacklisted, skipping")
        return

    is_whitelisted = await asyncio.to_thread(plugin.user_manager.is_whitelisted, opener_user_id)
    quota_enabled = plugin.config.quota.enable_quota

    if quota_enabled and not is_whitelisted:
        can_use, reason = await asyncio.to_thread(plugin.user_manager.can_use, opener_user_id)
        if not can_use:
            await event.send(
                event.plain_result(
                    "⚠️ 自动画图已暂停：开启者额度不足\n"
                    f"开启者 {opener_user_id} 的额度已用完，请签到获取额度后重新开启"
                )
            )
            plugin.auto_draw_info[umo] = None
            if hasattr(plugin, "persist_auto_draw_info"):
                await plugin.persist_auto_draw_info()
            return

    preset_contents: list[str] = []
    for preset_name in presets:
        preset = await asyncio.to_thread(plugin.preset_manager.get_preset, preset_name)
        if preset:
            preset_contents.append(preset.content)

    logger.debug(
        f"[nai] Auto draw: generating from response ({len(ai_response)} chars), "
        f"presets={presets}, opener={opener_user_id}"
    )

    coro = _auto_draw_generate(
        plugin,
        event,
        ai_response,
        preset_contents,
        opener_user_id,
        is_whitelisted,
        cs_names,
        auto_info,
    )
    if hasattr(plugin, "_create_background_task"):
        plugin._create_background_task(coro, name="nai:auto_draw")
    else:
        task = asyncio.create_task(coro)
        def _log_task_exception(t: asyncio.Task):
            exc = t.exception()
            if exc is not None:
                logger.error("[nai] Auto draw task failed", exc_info=exc)

        task.add_done_callback(_log_task_exception)


async def _auto_draw_generate(
    plugin,
    event,
    ai_response: str,
    preset_contents: list[str],
    opener_user_id: str,
    is_whitelisted: bool,
    cs_names: list[str],
    auto_info: dict,
):
    quota_enabled = plugin.config.quota.enable_quota
    umo = event.unified_msg_origin

    try:
        batch_count = extract_batch_count(
            preset_contents,
            max_n=plugin.config.request.max_n,
        )
    except Exception as e:  # noqa: BLE001
        await event.send(
            event.plain_result(f"🎨 自动画图失败：{format_readable_error(e)}")
        )
        return

    if quota_enabled and not is_whitelisted:
        quota = await asyncio.to_thread(plugin.user_manager.get_quota, opener_user_id)
        if quota < batch_count:
            await event.send(
                event.plain_result(
                    "⚠️ 自动画图已暂停：开启者额度不足\n"
                    f"开启者 {opener_user_id} 的额度不足，本次需要 {batch_count} 次"
                )
            )
            plugin.auto_draw_info.pop(umo, None)
            if hasattr(plugin, "persist_auto_draw_info"):
                await plugin.persist_auto_draw_info()
            return

    consume_quota = (
        (lambda: plugin.user_manager.consume_quota_n(opener_user_id, batch_count))
        if quota_enabled and not is_whitelisted
        else None
    )

    try:
        async with reserve_queue(
            plugin,
            opener_user_id,
            is_whitelisted=is_whitelisted,
            consume_quota=consume_quota,
        ) as reservation:
            queue_total = reservation.queue_total

            token = plugin._get_next_token()
            queue_status = f"（当前队列：{queue_total}）" if queue_total > 1 else ""

            try:
                ai_response_with_prefix = f"参考：{ai_response}"
                merged_raw, wrappers, explicit_ids = merge_nai_params(preset_contents)
                filtered_raw = _strip_image_param_lines(merged_raw)
                if filtered_raw.strip():
                    user_req, _remaining_images = await parse_req_with_remaining_images(
                        filtered_raw,
                        [],
                        plugin.config,
                        is_whitelisted=is_whitelisted,
                    )
                else:
                    user_req = None

                i2i_image = auto_info.get("i2i_image")
                vibe_transfer_images = auto_info.get("vibe_transfer_images")
                character_keep_image = auto_info.get("character_keep_image")
                vision_images = auto_info.get("vision_images")

                cs_content_parts: list[str] = []
                if cs_names:
                    for cs_name in cs_names:
                        exists = await asyncio.to_thread(
                            plugin.cs_store.exists, opener_user_id, cs_name
                        )
                        if not exists:
                            await event.send(
                                event.plain_result(
                                    f"🎨 自动画图失败：角色保持 {cs_name} 不存在"
                                )
                            )
                            return
                        cs_content_parts.append(
                            await asyncio.to_thread(
                                plugin.cs_store.read, opener_user_id, cs_name
                            )
                        )
                cs_content = "\n\n".join(cs_content_parts)

                full_parts = list(reversed(preset_contents)) + [ai_response_with_prefix]
                full_instructions = "\n\n".join(full_parts)

                await event.send(event.plain_result(f"🎨 自动画图中...{queue_status}"))

                async with acquire_generation_semaphore(plugin):
                    images: list[bytes] = []
                    for _ in range(batch_count):
                        req = await llm_generate_advanced_req(
                            instructions=f"画一张图\n{full_instructions}",
                            config=plugin.config,
                            ctx=plugin.context,
                            event=event,
                            i2i_image=i2i_image,
                            vibe_transfer_images=vibe_transfer_images,
                            vision_images=vision_images,
                            skip_default_prompts=bool(preset_contents),
                            extra_system_prompt=cs_content,
                        )

                        if character_keep_image and req.addition is not None:
                            req.addition.character_keep = ReqAdditionCharacterKeep(
                                base64=character_keep_image,
                                keep_vibe=plugin.config.defaults.character_keep_vibe,
                                strength=plugin.config.defaults.character_keep_strength,
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

                        images.append(await plugin._run_with_retry(_do_generate))

                # ── 原图留底 + 压缩 ──
                images = [
                    _save_original_and_compress(
                        img,
                        max_bytes=plugin.config.general.auto_draw_compress_threshold_mb * 1024 * 1024,
                        save_original=plugin.config.general.auto_draw_save_original,
                    )
                    for img in images
                ]

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
                    await event.send(event.chain_result([nodes]))
                else:
                    await event.send(event.chain_result([Image.fromBytes(img) for img in images]))

            except asyncio.CancelledError:
                await plugin._queue.mark_wait_finished(
                    max_concurrent=plugin.config.request.max_concurrent
                )
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Auto draw generation failed: {e}")
                await event.send(
                    event.plain_result(f"🎨 自动画图失败：{format_readable_error(e)}")
                )
    except QueueRejected as e:
        close_auto = False
        if e.reason == "inflight":
            await event.send(event.plain_result("🎨 自动画图跳过：你的上一张还没画完呢~"))
        elif e.reason == "queue_full":
            await event.send(
                event.plain_result(
                    f"⚠️ 自动画图跳过：队列已满（{plugin.config.request.max_queue_size}）"
                )
            )
        elif e.reason == "quota":
            close_auto = True
            await event.send(
                event.plain_result(
                    "⚠️ 自动画图已暂停：开启者额度不足\n"
                    f"开启者 {opener_user_id} 的额度已用完，请签到获取额度后重新开启"
                )
            )
        if close_auto:
            plugin.auto_draw_info.pop(umo, None)
            if hasattr(plugin, "persist_auto_draw_info"):
                await plugin.persist_auto_draw_info()
        return
