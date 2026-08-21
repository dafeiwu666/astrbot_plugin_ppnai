"""Character keep (cs) command handlers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from astrbot.api import logger
from astrbot.core.agent.message import Message

from .character_keep_store import extract_nai_tag, replace_nai_tag


def _split_lines_preserve(value: str) -> list[str]:
    return [line for line in value.splitlines()]


def _parse_kv_with_continuation(raw: str) -> dict[str, str]:
    data: dict[str, str] = {}
    last_key: str | None = None
    for line in _split_lines_preserve(raw):
        striped = line.strip()
        if not striped:
            continue
        if "=" in striped:
            key, value = striped.split("=", 1)
            key = key.strip()
            value = value.strip()
            data[key] = value
            last_key = key
            continue
        if last_key:
            data[last_key] = f"{data[last_key]}\n{striped}"
    return data


async def handle_cs(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("cs").strip()
    user_id = plugin._get_user_id(event)

    if not raw:
        show_all = plugin._check_resource_admin(event) or plugin.config.general.list_all_resources
        grouped = await asyncio.to_thread(
            plugin.cs_store.list_grouped, None if show_all else user_id
        )
        if not grouped:
            yield event.plain_result("暂无角色保持记录，可使用 /cs 创建")
            return
        result = "角色保持列表：\n" + "\n".join(
            f"- {owner}:\n" + "\n".join(f"  • {name}" for name in names)
            for owner, names in grouped.items()
        )
        yield event.plain_result(result)
        return

    kv = _parse_kv_with_continuation(raw)
    name = kv.get("na", "").strip()
    aa_prompt = kv.get("aa", "").strip()
    nn_prompt = kv.get("nn", "").strip()

    if not name:
        yield event.plain_result("请提供角色保持名称：na=名称")
        return

    if bool(aa_prompt) == bool(nn_prompt):
        yield event.plain_result("请提供 aa= 或 nn= 其中一个")
        return

    if await asyncio.to_thread(plugin.cs_store.exists, user_id, name):
        yield event.plain_result(f"角色保持 {name} 已存在，如需修改请先删除或使用 /ccs")
        return

    if nn_prompt:
        content = nn_prompt
    else:
        quota_enabled = plugin.config.quota.enable_quota
        is_whitelisted = plugin.user_manager.is_whitelisted(user_id)
        if quota_enabled and not is_whitelisted:
            can_use, reason = plugin.user_manager.can_use(user_id)
            if not can_use:
                yield event.plain_result(reason)
                return
            if not plugin.user_manager.consume_quota_n(user_id, 1):
                yield event.plain_result("你的画图次数已用完，请/nai签到获取额度")
                return

        try:
            cssaying = await asyncio.to_thread(plugin.cs_store.load_cssaying)
        except Exception as e:  # noqa: BLE001
            yield event.plain_result(f"读取提示词失败：{e}")
            return

        prompt = f"{aa_prompt}\n\n{cssaying}"
        try:
            provider_id = await plugin.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
            contexts = [Message(role="user", content=prompt)]
            llm_resp = await plugin.context.llm_generate(
                chat_provider_id=provider_id,
                contexts=contexts,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("cs llm_generate failed")
            yield event.plain_result(f"生成角色保持失败：{e}")
            return

        content = (llm_resp.completion_text or "").strip()
        if not content:
            yield event.plain_result("生成角色保持失败：返回内容为空")
            return

    try:
        await asyncio.to_thread(plugin.cs_store.write, user_id, name, content, overwrite=False)
    except FileExistsError:
        yield event.plain_result(f"角色保持 {name} 已存在，如需修改请先删除或使用 /ccs")
        return
    except Exception as e:  # noqa: BLE001
        yield event.plain_result(f"保存失败：{e}")
        return

    yield event.plain_result(f"✅ 角色保持 {name} 已保存")


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
        yield event.plain_result(f"✅ 角色保持 {name} 已删除")
    else:
        yield event.plain_result(f"角色保持 {name} 不存在")


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
            yield event.plain_result("暂无角色保持记录，可使用 /cs 创建")
            return
        result = "角色保持列表：\n" + "\n".join(
            f"- {owner}:\n" + "\n".join(f"  • {item}" for item in names)
            for owner, names in grouped.items()
        )
        yield event.plain_result(result)
        return

    if not await asyncio.to_thread(plugin.cs_store.exists, target_user, name):
        yield event.plain_result(f"角色保持 {name} 不存在")
        return

    content = await asyncio.to_thread(plugin.cs_store.read, target_user, name)
    tag = extract_nai_tag(content)
    if not tag:
        yield event.plain_result("未找到 NovelAI tag style 外貌提示词内容")
        return

    yield event.plain_result(tag)


async def handle_ccs(plugin, event) -> AsyncIterator:
    raw = event.message_str.removeprefix("ccs").strip()
    if not raw:
        yield event.plain_result("请提供名称和修改内容，例如：/ccs 角色名 新内容")
        return

    parts = raw.split()
    if len(parts) < 2:
        yield event.plain_result("请提供修改内容，例如：/ccs 角色名 新内容")
        return

    target_user = plugin._get_user_id(event)
    if plugin._check_resource_admin(event) and len(parts) >= 3:
        target_user, name = parts[0], parts[1]
        new_tag = " ".join(parts[2:]).strip()
    else:
        name, new_tag = parts[0], " ".join(parts[1:]).strip()
    if not new_tag:
        yield event.plain_result("修改内容不能为空")
        return

    if not await asyncio.to_thread(plugin.cs_store.exists, target_user, name):
        yield event.plain_result(f"角色保持 {name} 不存在")
        return

    content = await asyncio.to_thread(plugin.cs_store.read, target_user, name)
    updated, replaced = replace_nai_tag(content, new_tag)
    await asyncio.to_thread(plugin.cs_store.write, target_user, name, updated, overwrite=True)

    if replaced:
        yield event.plain_result(f"✅ 角色保持 {name} 已更新")
    else:
        yield event.plain_result(
            f"✅ 角色保持 {name} 已覆盖（原内容中未找到目标字段）"
        )
