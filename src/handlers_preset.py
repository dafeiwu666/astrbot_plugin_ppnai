"""Preset command handlers.

Extracted from main.py to keep Plugin wiring lightweight.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator

from astrbot.api.message_components import Image, Node, Nodes, Plain

from .image_io import resolve_image
from .params import _collect_images_with_replies


async def handle_preset_list(plugin, event) -> AsyncIterator:
    owner_id = plugin._get_resource_owner(event)
    show_all = plugin._check_resource_admin(event) or plugin.config.general.list_all_resources
    grouped = await asyncio.to_thread(
        plugin.preset_manager.list_grouped, None if show_all else owner_id
    )
    if not grouped:
        yield event.plain_result("暂无预设，管理员可使用 nai预设添加 命令添加预设")
        return
    result = "预设列表：\n" + "\n".join(
        f"- {owner}:\n" + "\n".join(f"  • {title}" for title in titles)
        for owner, titles in grouped.items()
    )
    yield event.plain_result(result)


async def handle_preset_view(plugin, event) -> AsyncIterator:
    args = event.message_str.removeprefix("nai预设查看").strip()
    if not args:
        async for result in handle_preset_list(plugin, event):
            yield result
        return

    title = args.split()[0]
    preset = await asyncio.to_thread(plugin.preset_manager.get_preset, title)

    if preset is None:
        yield event.plain_result(f"预设 #{title} 不存在")
        return

    preview = await asyncio.to_thread(
        plugin.preview_manager.read, f"preset:{title}"
    )
    content = [Plain(f"📝 预设 #{title}\n\n{preset.content}")]
    if preview is not None:
        content.append(Image.fromBytes(preview))
    yield event.chain_result([
        Nodes([
            Node(
                uin=event.get_sender_id(),
                name=event.get_sender_name(),
                content=content,
            )
        ])
    ])


async def handle_preset_add(plugin, event) -> AsyncIterator:
    full_text = event.message_str
    lines = full_text.split("\n", 1)

    first_line = lines[0].removeprefix("nai预设添加").strip()
    if not first_line:
        yield event.plain_result(
            "请指定预设标题和内容，格式：\n"
            "nai预设添加 标题名\n"
            "这里是预设内容..."
        )
        return

    title = first_line

    if len(lines) < 2 or not lines[1].strip():
        yield event.plain_result(
            f"请在标题后换行添加预设内容，格式：\n"
            f"nai预设添加 {title}\n"
            f"这里是预设内容..."
        )
        return

    content = lines[1]

    if await asyncio.to_thread(plugin.preset_manager.get_preset, title) is not None:
        yield event.plain_result(f"预设 #{title} 已存在，如需修改请先删除再添加")
        return

    await asyncio.to_thread(
        plugin.preset_manager.add_preset,
        title,
        content,
        plugin._get_resource_owner(event),
    )
    images = _collect_images_with_replies(event.message_obj.message)
    if images:
        preview_data = await resolve_image(images[0])
        encoded = preview_data.split(",", 1)[1]
        await asyncio.to_thread(
            plugin.preview_manager.save_or_replace,
            [f"preset:{title}"],
            base64.b64decode(encoded),
        )
    preview = content[:200] + ("..." if len(content) > 200 else "")
    yield event.plain_result(f"✅ 预设 #{title} 添加成功！\n\n预览：\n{preview}")


async def handle_preset_modify(plugin, event) -> AsyncIterator:
    full_text = event.message_str
    lines = full_text.split("\n", 1)
    title = lines[0].removeprefix("nai预设修改").strip()
    if not title:
        yield event.plain_result("请指定预设标题和内容，格式：\nnai预设修改 标题名\n这里是新内容...")
        return
    if len(lines) < 2 or not lines[1].strip():
        yield event.plain_result(f"请在标题后换行添加新内容：\nnai预设修改 {title}\n这里是新内容...")
        return

    owner_id = None if plugin._check_resource_admin(event) else plugin._get_resource_owner(event)
    existing = await asyncio.to_thread(plugin.preset_manager.get_preset, title)
    if existing is None or (owner_id is not None and existing.owner_id != owner_id):
        yield event.plain_result(f"预设 #{title} 不存在或无权修改")
        return

    content = lines[1]
    await asyncio.to_thread(plugin.preset_manager.update_preset, title, content)
    images = _collect_images_with_replies(event.message_obj.message)
    if images:
        preview_data = await resolve_image(images[0])
        await asyncio.to_thread(
            plugin.preview_manager.save_or_replace,
            [f"preset:{title}"],
            base64.b64decode(preview_data.split(",", 1)[1]),
        )
    yield event.plain_result(f"✅ 预设 #{title} 修改成功")


async def handle_preset_delete(plugin, event) -> AsyncIterator:
    args = event.message_str.removeprefix("nai预设删除").strip()
    if not args:
        yield event.plain_result("请指定预设名称，例如：nai预设删除 猫娘")
        return

    title = args.split()[0]

    owner_id = None if plugin._check_resource_admin(event) else plugin._get_resource_owner(event)
    deleted = await asyncio.to_thread(plugin.preset_manager.delete_preset, title, owner_id)
    if deleted:
        yield event.plain_result(f"✅ 预设 #{title} 已删除")
    else:
        yield event.plain_result(f"预设 #{title} 不存在")
