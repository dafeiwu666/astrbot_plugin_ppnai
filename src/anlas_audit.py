"""Local NovelAI generation and Anlas balance audit."""
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from astrbot.api import logger
from astrbot.core.star.star_tools import StarTools

_PATH = "anlas_audit.jsonl"


def _path() -> Path:
    return StarTools.get_data_dir("astrbot_plugin_ppnai") / _PATH


def _read_sync(token: str, proxy: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(proxy=proxy or None, timeout=20, trust_env=False) as client:
            response = client.get("https://image.novelai.net/user/subscription", headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        if response.status_code != 200:
            return None
        data = response.json()
        steps = data.get("trainingStepsLeft") or {}
        return {"tier": data.get("tier"), "active": data.get("active"), "fixed": steps.get("fixedTrainingStepsLeft"), "purchased": steps.get("purchasedTrainingSteps")}
    except Exception:
        return None


async def read_balance(plugin, token: str) -> dict[str, Any] | None:
    return await asyncio.to_thread(_read_sync, token, getattr(plugin.config.request, "proxy", ""))


def official_rule(req, tier: int | None, batch_count: int = 1) -> str:
    try:
        width, height = (int(value) for value in str(req.size).lower().split("x", 1))
        has_base_image = bool(getattr(req.addition, "i2i", None))
        if tier == 3 and batch_count > 1:
            return "官方规则：Opus 批量生成会消耗 Anlas"
        if tier == 3 and width <= 1024 and height <= 1024 and int(req.steps) <= 28 and not has_base_image:
            return "官方规则：Opus 单张、无底图、≤28步、正常尺寸范围，预计免费"
    except Exception:
        pass
    return "官方未公开完整价格公式；以生图前后官方余额差为准"


def _append(entry: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


async def record_generation(plugin, *, token_index: int, token: str, req, user_id: str, sender_name: str, session: str, outcome: str, balance_before: dict[str, Any] | None, batch_count: int = 1) -> None:
    await asyncio.sleep(1.5)
    after = await read_balance(plugin, token)
    fixed_delta = None
    purchased_delta = None
    if balance_before and after:
        if isinstance(balance_before.get("fixed"), int) and isinstance(after.get("fixed"), int):
            fixed_delta = balance_before["fixed"] - after["fixed"]
        if isinstance(balance_before.get("purchased"), int) and isinstance(after.get("purchased"), int):
            purchased_delta = balance_before["purchased"] - after["purchased"]
    entry = {"at": datetime.now().astimezone().isoformat(timespec="seconds"), "token_index": token_index, "user_id": str(user_id), "sender_name": str(sender_name), "session": str(session), "outcome": outcome, "model": str(req.model), "size": str(req.size), "steps": str(req.steps), "sampler": str(req.sampler), "i2i": bool(getattr(req.addition, "i2i", None)), "batch_count": batch_count, "official_rule": official_rule(req, balance_before.get("tier") if balance_before else None, batch_count), "balance_before": balance_before, "balance_after": after, "actual_fixed_anlas_delta": fixed_delta, "actual_purchased_anlas_delta": purchased_delta}
    await asyncio.to_thread(_append, entry)


def recent_entries(limit: int = 20) -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    result = []
    for line in deque(path.read_text(encoding="utf-8").splitlines(), maxlen=limit):
        try:
            result.append(json.loads(line))
        except Exception:
            logger.debug("Skipping malformed Anlas audit entry")
    return list(reversed(result))
