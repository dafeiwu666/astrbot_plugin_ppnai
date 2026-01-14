"""Admin/user command handlers (checkin/quota/blacklist/whitelist).

These are extracted from main.py to keep Plugin wiring lightweight.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


async def handle_checkin(plugin, event) -> AsyncIterator:
    user_id = plugin._get_user_id(event)
    _success, _gained, message = await asyncio.to_thread(
        plugin.user_manager.checkin, user_id, plugin.config
    )
    yield event.plain_result(message)


async def handle_queue_status(plugin, event) -> AsyncIterator:
    max_concurrent = plugin.config.request.max_concurrent
    max_queue = plugin.config.request.max_queue_size

    processing = max(plugin._queue.queue_count - plugin._queue.waiting_count, 0)
    waiting = plugin._queue.waiting_count

    status_lines = [
        "📊 当前队列状态",
        f"• 正在处理：{processing}/{max_concurrent}",
        f"• 排队等待：{waiting}/{max_queue if max_queue > 0 else '∞'}",
    ]

    if plugin._queue.queue_count == 0:
        status_lines.append("\n✅ 队列空闲，可以立即开始画图")
    elif max_queue > 0 and waiting >= max_queue:
        status_lines.append("\n⚠️ 队列已满，新请求将被拒绝")
    else:
        if max_queue > 0:
            status_lines.append(f"\n📝 还可加入 {max_queue - waiting} 个请求")

    yield event.plain_result("\n".join(status_lines))


async def handle_query_quota(plugin, event) -> AsyncIterator:
    user_id = plugin._get_user_id(event)

    if await asyncio.to_thread(plugin.user_manager.is_blacklisted, user_id):
        yield event.plain_result("你已被加入黑名单")
        return

    if await asyncio.to_thread(plugin.user_manager.is_whitelisted, user_id):
        yield event.plain_result("你在白名单中，可无限使用画图功能")
        return

    if not plugin.config.quota.enable_quota:
        yield event.plain_result("当前未启用额度系统，可无限使用画图功能")
        return

    quota = await asyncio.to_thread(plugin.user_manager.get_quota, user_id)
    yield event.plain_result(f"你当前剩余 {quota} 次画图额度")


async def handle_add_blacklist(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    args = event.message_str.removeprefix("nai黑名单添加").strip()
    if not args:
        yield event.plain_result("请指定用户ID，例如：nai黑名单添加 123456")
        return

    user_id = args.split()[0]
    if await asyncio.to_thread(plugin.user_manager.add_to_blacklist, user_id):
        yield event.plain_result(f"已将用户 {user_id} 添加到黑名单")
    else:
        yield event.plain_result(f"用户 {user_id} 已在黑名单中")


async def handle_remove_blacklist(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    args = event.message_str.removeprefix("nai黑名单移除").strip()
    if not args:
        yield event.plain_result("请指定用户ID，例如：nai黑名单移除 123456")
        return

    user_id = args.split()[0]
    if await asyncio.to_thread(plugin.user_manager.remove_from_blacklist, user_id):
        yield event.plain_result(f"已将用户 {user_id} 从黑名单移除")
    else:
        yield event.plain_result(f"用户 {user_id} 不在黑名单中")


async def handle_list_blacklist(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    blacklist = await asyncio.to_thread(plugin.user_manager.get_blacklist)
    if not blacklist:
        yield event.plain_result("黑名单为空")
    else:
        yield event.plain_result("黑名单用户：\n" + "\n".join(blacklist))


async def handle_add_whitelist(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    args = event.message_str.removeprefix("nai白名单添加").strip()
    if not args:
        yield event.plain_result("请指定用户ID，例如：nai白名单添加 123456")
        return

    user_id = args.split()[0]
    if await asyncio.to_thread(plugin.user_manager.add_to_whitelist, user_id):
        yield event.plain_result(f"已将用户 {user_id} 添加到白名单")
    else:
        yield event.plain_result(f"用户 {user_id} 已在白名单中")


async def handle_remove_whitelist(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    args = event.message_str.removeprefix("nai白名单移除").strip()
    if not args:
        yield event.plain_result("请指定用户ID，例如：nai白名单移除 123456")
        return

    user_id = args.split()[0]
    if await asyncio.to_thread(plugin.user_manager.remove_from_whitelist, user_id):
        yield event.plain_result(f"已将用户 {user_id} 从白名单移除")
    else:
        yield event.plain_result(f"用户 {user_id} 不在白名单中")


async def handle_list_whitelist(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    whitelist = await asyncio.to_thread(plugin.user_manager.get_whitelist)
    if not whitelist:
        yield event.plain_result("白名单为空")
    else:
        yield event.plain_result("白名单用户：\n" + "\n".join(whitelist))


async def handle_admin_query_user(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    args = event.message_str.removeprefix("nai查询用户").strip()
    if not args:
        yield event.plain_result("请指定用户ID，例如：nai查询用户 123456")
        return

    user_id = args.split()[0]
    quota = await asyncio.to_thread(plugin.user_manager.get_quota, user_id)

    status = ""
    if await asyncio.to_thread(plugin.user_manager.is_blacklisted, user_id):
        status = "（黑名单）"
    elif await asyncio.to_thread(plugin.user_manager.is_whitelisted, user_id):
        status = "（白名单）"

    yield event.plain_result(f"用户 {user_id}{status} 的额度：{quota} 次")


async def handle_set_quota(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    args = event.message_str.removeprefix("nai设置额度").strip().split()
    if len(args) < 2:
        yield event.plain_result("请指定用户ID和额度，例如：nai设置额度 123456 100")
        return

    user_id = args[0]
    try:
        quota = int(args[1])
    except ValueError:
        yield event.plain_result("额度必须是整数")
        return

    await asyncio.to_thread(plugin.user_manager.set_quota, user_id, quota)
    yield event.plain_result(f"已将用户 {user_id} 的额度设置为 {quota} 次")


async def handle_add_quota(plugin, event) -> AsyncIterator:
    if not plugin._check_permission(event):
        yield event.plain_result("权限不足，仅管理员可使用此命令")
        return

    args = event.message_str.removeprefix("nai增加额度").strip().split()
    if len(args) < 2:
        yield event.plain_result("请指定用户ID和额度，例如：nai增加额度 123456 10")
        return

    user_id = args[0]
    try:
        amount = int(args[1])
    except ValueError:
        yield event.plain_result("额度必须是整数")
        return

    new_quota = await asyncio.to_thread(plugin.user_manager.add_quota, user_id, amount)
    yield event.plain_result(
        f"已为用户 {user_id} 增加 {amount} 次额度，当前额度：{new_quota} 次"
    )
