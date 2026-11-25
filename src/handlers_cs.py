"""Character keep (cs) command handlers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .drawing_recipe import prepare_drawing_recipe
from .handlers_shared import drawing_recipe_view


async def handle_cs(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("cs").strip()
    user_id = plugin._get_user_id(event)

    if raw:
        yield event.plain_result("查看角色库请使用 /cs。添加角色用 /cs添加 名称，换行填写绘图参数。")
        return

    show_all = plugin._check_resource_admin(event) or plugin.config.general.list_all_resources
    grouped = await asyncio.to_thread(
        plugin.cs_store.list_grouped, None if show_all else user_id
    )
    if not grouped:
        yield event.plain_result("角色库为空，可使用 /cs添加 名称\ntag=角色标签 添加角色。")
        return
    result = "角色Tag库：\n" + "\n".join(
        f"- {owner}:\n" + "\n".join(f"  • {name}" for name in names)
        for owner, names in grouped.items()
    )
    yield event.plain_result(result)


async def handle_cs_add(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("cs添加").strip()
    user_id = plugin._get_user_id(event)
    if not raw:
        yield event.plain_result("请按格式添加：/cs添加 夕立\ntag=yuudachi kai ni (Kancolle)")
        return

    lines = raw.splitlines()
    name = lines[0].strip()
    if "=" in name:
        key, value = name.split("=", 1)
        if key.strip().lower() in {"name", "名称", "na"}:
            name = value.strip()
    content = "\n".join(lines[1:]).strip()
    if not name or not content:
        yield event.plain_result("名称和绘图内容均不能为空。格式：/cs添加 夕立\ntag=yuudachi kai ni (Kancolle)")
        return

    try:
        recipe = await prepare_drawing_recipe(plugin, event, content)
        await asyncio.to_thread(plugin.cs_store.write, user_id, name, recipe, overwrite=False)
    except FileExistsError:
        yield event.plain_result(f"角色 {name} 已存在，修改请使用 /ccs {name} 后换行填写绘图参数。")
        return
    except Exception as exc:  # noqa: BLE001
        yield event.plain_result(f"保存角色Tag失败：{exc}")
        return

    yield event.plain_result(f"✅ 角色预设 {name} 已保存")


async def handle_dcs(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("dcs").strip()
    parts = raw.split(maxsplit=1)
    target_user = plugin._get_user_id(event)
    name = raw
    if plugin._check_resource_admin(event) and len(parts) == 2:
        target_user, name = parts
    if not name:
        yield event.plain_result("请提供要删除的名称，例如：/dcs 角色名")
        return

    deleted = await asyncio.to_thread(plugin.cs_store.delete, target_user, name)
    if deleted:
        yield event.plain_result(f"✅ 角色Tag {name} 已从角色库删除")
    else:
        yield event.plain_result(f"角色Tag {name} 不存在")


async def handle_scs(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("scs").strip()
    parts = raw.split(maxsplit=1)
    target_user = plugin._get_user_id(event)
    name = raw
    if plugin._check_resource_admin(event) and len(parts) == 2:
        target_user, name = parts
    if not name:
        user_id = plugin._get_user_id(event)
        show_all = plugin._check_resource_admin(event) or plugin.config.general.list_all_resources
        grouped = await asyncio.to_thread(plugin.cs_store.list_grouped, None if show_all else user_id)
        if not grouped:
            yield event.plain_result("角色Tag库为空，可使用 /cs添加 名称 添加角色")
            return
        result = "角色Tag库：\n" + "\n".join(
            f"- {owner}:\n" + "\n".join(f"  • {item}" for item in names)
            for owner, names in grouped.items()
        )
        yield event.plain_result(result)
        return

    resolved = await asyncio.to_thread(
        plugin.cs_store.resolve_recipe,
        target_user,
        name,
        include_all=plugin.config.general.list_all_resources
        or plugin._check_resource_admin(event),
    )
    if resolved is None:
        yield event.plain_result(f"角色Tag {name} 不存在")
        return

    owner_id, recipe = resolved
    yield await drawing_recipe_view(
        event, recipe, f"📝 角色预设：{name}", plugin.preview_manager, f"ck:{owner_id}:{name}",
    )


async def handle_ccs(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("ccs").strip()
    if not raw:
        yield event.plain_result("请提供名称和新的绘图参数，例如：/ccs 夕立\ntag=yuudachi kai ni (Kancolle)")
        return

    lines = raw.splitlines()
    first_line = lines[0].strip()
    remainder = "\n".join(lines[1:]).strip()
    parts = first_line.split(maxsplit=2)
    if len(parts) < 2 and not remainder:
        yield event.plain_result("请提供新的 NovelAI Tag，例如：/ccs 夕立\nyuudachi kai ni (Kancolle)")
        return

    target_user = plugin._get_user_id(event)
    if remainder and await asyncio.to_thread(plugin.cs_store.exists, target_user, first_line):
        name, new_content = first_line, remainder
    elif plugin._check_resource_admin(event) and len(parts) >= 2 and await asyncio.to_thread(
        plugin.cs_store.exists, parts[0], parts[1]
    ):
        target_user, name = parts[:2]
        new_content = "\n".join(part for part in (" ".join(parts[2:]), remainder) if part)
    else:
        name = parts[0]
        new_content = "\n".join(part for part in (first_line[len(name):].strip(), remainder) if part)
    if not new_content:
        yield event.plain_result("新的绘图内容不能为空")
        return
    try:
        previous = await asyncio.to_thread(plugin.cs_store.read_recipe, target_user, name)
        recipe = await prepare_drawing_recipe(plugin, event, new_content, previous)
        await asyncio.to_thread(plugin.cs_store.write, target_user, name, recipe, overwrite=True)
    except FileNotFoundError:
        yield event.plain_result(f"角色Tag {name} 不存在")
        return
    except (ValueError, OSError, RuntimeError) as exc:
        yield event.plain_result(f"保存角色预设失败：{exc}")
        return
    yield event.plain_result(f"✅ 角色预设 {name} 已更新")
