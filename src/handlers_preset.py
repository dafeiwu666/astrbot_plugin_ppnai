"""Preset command handlers.

Extracted from main.py to keep Plugin wiring lightweight.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


async def handle_preset_list(plugin, event) -> AsyncIterator:
    presets = await asyncio.to_thread(plugin.preset_manager.list_presets)
    if not presets:
        yield event.plain_result("暂无预设，管理员可使用 nai预设添加 命令添加预设")
        return

    result = "📝 预设列表：\n" + "\n".join(f"• {title}" for title in presets)
    result += "\n\n使用方式：\nnai\ns1=预设名"
    yield event.plain_result(result)


async def handle_preset_view(plugin, event) -> AsyncIterator:
    args = event.message_str.removeprefix("nai预设查看").strip()
    if not args:
        yield event.plain_result("请指定预设名称，例如：nai预设查看 猫娘")
        return

    title = args.split()[0]
    preset = await asyncio.to_thread(plugin.preset_manager.get_preset, title)

    if preset is None:
        yield event.plain_result(f"预设 #{title} 不存在")
        return

    yield event.plain_result(f"📝 预设 #{title}\n\n```\n{preset.content}\n```")


async def handle_preset_add(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

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

    await asyncio.to_thread(plugin.preset_manager.add_preset, title, content)
    preview = content[:200] + ("..." if len(content) > 200 else "")
    yield event.plain_result(f"✅ 预设 #{title} 添加成功！\n\n预览：\n{preview}")


async def handle_preset_delete(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    args = event.message_str.removeprefix("nai预设删除").strip()
    if not args:
        yield event.plain_result("请指定预设名称，例如：nai预设删除 猫娘")
        return

    title = args.split()[0]

    deleted = await asyncio.to_thread(plugin.preset_manager.delete_preset, title)
    if deleted:
        yield event.plain_result(f"✅ 预设 #{title} 已删除")
    else:
        yield event.plain_result(f"预设 #{title} 不存在")
