"""Character keep (cs) command handlers."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator

from astrbot.api.message_components import Image, Node, Nodes, Plain

from .character_keep_store import replace_nai_tag
from .image_io import resolve_image
from .params import _collect_images_with_replies


async def _save_preview(plugin, event, resource_key: str) -> None:
    images = _collect_images_with_replies(event.message_obj.message)
    if not images:
        return
    data_uri = await resolve_image(images[0])
    await asyncio.to_thread(
        plugin.preview_manager.save_or_replace,
        [resource_key],
        base64.b64decode(data_uri.split(",", 1)[1]),
    )


async def handle_cs(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("cs").strip()
    user_id = plugin._get_user_id(event)

    if raw:
        yield event.plain_result("查看角色Tag库请使用 /cs；添加角色请使用 /cs添加 名称 后换行填写 NovelAI Tag。")
        return

    show_all = plugin._check_resource_admin(event) or plugin.config.general.list_all_resources
    grouped = await asyncio.to_thread(
        plugin.cs_store.list_grouped, None if show_all else user_id
    )
    if not grouped:
        yield event.plain_result("角色Tag库为空，可使用 /cs添加 名称\n角色 NovelAI Tag 添加角色。")
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
        yield event.plain_result("请按格式添加：/cs添加 夕立\nyuudachi kai ni (Kancolle)")
        return

    lines = raw.splitlines()
    name = lines[0].strip()
    if "=" in name:
        key, value = name.split("=", 1)
        if key.strip().lower() in {"name", "名称", "na"}:
            name = value.strip()
    tags = "\n".join(lines[1:]).strip()
    if not name or not tags:
        yield event.plain_result("名称和 NovelAI Tag 均不能为空。格式：/cs添加 夕立\nyuudachi kai ni (Kancolle)")
        return

    try:
        await asyncio.to_thread(
            plugin.cs_store.write, user_id, name, tags, overwrite=False
        )
    except FileExistsError:
        yield event.plain_result(f"角色 {name} 已存在；修改请使用 /ccs {name} 新Tag。")
        return
    except Exception as exc:  # noqa: BLE001
        yield event.plain_result(f"保存角色Tag失败：{exc}")
        return

    yield event.plain_result(f"✅ 角色 {name} 已加入角色Tag库")


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

    if not await asyncio.to_thread(plugin.cs_store.exists, target_user, name):
        yield event.plain_result(f"角色Tag {name} 不存在")
        return

    content = await asyncio.to_thread(plugin.cs_store.read_tag, target_user, name)
    preview = await asyncio.to_thread(
        plugin.preview_manager.read, f"ck:{target_user}:{name}"
    )
    node_content = [Plain(f"📝 角色Tag：{name}\n\n{content}")]
    if preview is not None:
        node_content.append(Image.fromBytes(preview))
    yield event.chain_result([
        Nodes([
            Node(
                uin=event.get_sender_id(),
                name=event.get_sender_name(),
                content=node_content,
            )
        ])
    ])


async def handle_ccs(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("ccs").strip()
    if not raw:
        yield event.plain_result("请提供名称和新的 NovelAI Tag，例如：/ccs 夕立\nyuudachi kai ni (Kancolle)")
        return

    lines = raw.splitlines()
    first_line = lines[0].strip()
    remainder = "\n".join(lines[1:]).strip()
    parts = first_line.split(maxsplit=2)
    if len(parts) < 2 and not remainder:
        yield event.plain_result("请提供新的 NovelAI Tag，例如：/ccs 夕立\nyuudachi kai ni (Kancolle)")
        return

    target_user = plugin._get_user_id(event)
    if plugin._check_resource_admin(event) and len(parts) >= 3:
        target_user, name = parts[0], parts[1]
        new_tag = " ".join(parts[2:]).strip()
    else:
        name = parts[0]
        new_tag = first_line[len(name) :].strip()
    if remainder:
        new_tag = f"{new_tag}\n{remainder}".strip()
    if not new_tag:
        yield event.plain_result("新的角色Tag不能为空")
        return

    if not await asyncio.to_thread(plugin.cs_store.exists, target_user, name):
        yield event.plain_result(f"角色Tag {name} 不存在")
        return

    content = await asyncio.to_thread(plugin.cs_store.read, target_user, name)
    updated, replaced = replace_nai_tag(content, new_tag)
    await asyncio.to_thread(plugin.cs_store.write, target_user, name, updated, overwrite=True)
    await _save_preview(plugin, event, f"ck:{target_user}:{name}")

    if replaced:
        yield event.plain_result(f"✅ 角色Tag {name} 已更新")
    else:
        yield event.plain_result(f"✅ 角色Tag {name} 已覆盖")
