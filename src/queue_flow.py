"""Shared queue/semaphore helpers.

These helpers centralize the repetitive queue reservation + release, and the
"wait finished" bookkeeping around the shared semaphore.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueueReservation:
    reserved_user: Any
    queue_total: int


class QueueRejectedError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Backward-compat alias (handlers import QueueRejected)
QueueRejected = QueueRejectedError


@asynccontextmanager
async def reserve_queue(
    plugin: Any,
    user_id: str,
    *,
    is_whitelisted: bool,
    consume_quota: Callable[[], Any] | None,
) -> AsyncIterator[QueueReservation]:
    res = await plugin._queue.reserve(
        user_id,
        is_whitelisted=is_whitelisted,
        max_queue_size=plugin.config.request.max_queue_size,
        max_concurrent=plugin.config.request.max_concurrent,
        consume_quota=consume_quota,
    )
    if not res.ok:
        raise QueueRejected(res.reason)

    try:
        yield QueueReservation(reserved_user=res.reserved_user, queue_total=res.queue_total)
    finally:
        await plugin._queue.release(
            user_id=user_id,
            reserved_user=res.reserved_user,
            max_concurrent=plugin.config.request.max_concurrent,
        )


@asynccontextmanager
async def acquire_generation_semaphore(plugin: Any) -> AsyncIterator[None]:
    sem = plugin._ensure_semaphore()
    async with sem:
        await plugin._queue.mark_wait_finished(
            max_concurrent=plugin.config.request.max_concurrent,
        )
        yield
