"""Shared queue state and helpers for request throttling."""

import asyncio
from asyncio import Semaphore
from dataclasses import dataclass
from typing import Callable, Literal

Reason = Literal["inflight", "queue_full", "quota"]


@dataclass
class ReserveResult:
    ok: bool
    reason: Reason | None
    queue_total: int
    reserved_user: bool


class SharedQueueState:
    def __init__(self):
        self.queue_count = 0
        self.waiting_count = 0
        self.user_inflight: set[str] = set()
        self.semaphore: Semaphore | None = None
        self.lock: asyncio.Lock | None = None
        self.max_concurrent: int | None = None

    def ensure(self, max_concurrent: int) -> tuple[Semaphore, asyncio.Lock]:
        if self.lock is None:
            self.lock = asyncio.Lock()
        if self.semaphore is None or self.max_concurrent != max_concurrent:
            self.semaphore = Semaphore(max_concurrent)
            self.max_concurrent = max_concurrent
        return self.semaphore, self.lock

    def _is_queue_full(self, max_queue_size: int) -> bool:
        if max_queue_size <= 0:
            return False
        return self.waiting_count >= max_queue_size

    async def reserve(
        self,
        user_id: str,
        *,
        is_whitelisted: bool,
        max_queue_size: int,
        max_concurrent: int,
        consume_quota: Callable[[], bool] | None = None,
    ) -> ReserveResult:
        _, lock = self.ensure(max_concurrent)
        async with lock:
            if (not is_whitelisted) and (user_id in self.user_inflight):
                return ReserveResult(False, "inflight", self.queue_count, False)
            if self._is_queue_full(max_queue_size):
                return ReserveResult(False, "queue_full", self.queue_count, False)
            if consume_quota and (not is_whitelisted):
                ok = await asyncio.to_thread(consume_quota)
                if not ok:
                    return ReserveResult(False, "quota", self.queue_count, False)

            reserved_user = False
            if not is_whitelisted:
                self.user_inflight.add(user_id)
                reserved_user = True

            self.queue_count += 1
            self.waiting_count += 1
            return ReserveResult(True, None, self.queue_count, reserved_user)

    async def release(self, *, user_id: str, reserved_user: bool, max_concurrent: int) -> None:
        _, lock = self.ensure(max_concurrent)
        async with lock:
            if self.queue_count > 0:
                self.queue_count -= 1
            if reserved_user:
                self.user_inflight.discard(user_id)

    async def mark_wait_finished(self, *, max_concurrent: int) -> None:
        _, lock = self.ensure(max_concurrent)
        async with lock:
            if self.waiting_count > 0:
                self.waiting_count -= 1

    def queue_status(self) -> int:
        return self.queue_count


_SHARED_QUEUE = SharedQueueState()


def get_shared_queue() -> SharedQueueState:
    return _SHARED_QUEUE
