"""Quark upload, login QR guidance, and expiring cleanup."""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.core.star.star_tools import StarTools

from ._vendor.quark_client import QuarkClient
from ._vendor.quark_client.auth.api_login import APILogin

_NAME = "astrbot_plugin_ppnai"


def _cookie_path() -> Path:
    return _data_dir() / "quark" / "session.cookie"


def _data_dir() -> Path:
    return StarTools.get_data_dir(_NAME)


def _registry_path() -> Path:
    return _data_dir() / "quark" / "upload_registry.json"


def _load_registry() -> list[dict]:
    try:
        value = json.loads(_registry_path().read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except Exception:
        return []


def _save_registry(records: list[dict]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def _upload_sync(cookie: str, image: bytes, filename: str, parent: str, expires: int) -> tuple[str, str]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="ppnai-", suffix=".png", delete=False) as stream:
            stream.write(image)
            temp_path = Path(stream.name)
        with QuarkClient(cookies=cookie, auto_login=False) as client:
            result = client.upload.upload_file(str(temp_path), parent_folder_id=parent)
            fid = result.get("finish_result", {}).get("fid")
            if not fid:
                raise RuntimeError("Quark upload finished without file ID")
            share = client.shares.create_share(file_ids=[str(fid)], title=filename, password=None, expired_at_ms=expires)
            url = share.get("share_url") or share.get("url")
            if not isinstance(url, str):
                raise RuntimeError("Quark did not return a share URL")
            return str(fid), url
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


async def upload_images_to_quark(
    plugin, images: Sequence[bytes], force: bool = False
) -> list[str]:
    config = getattr(plugin.config, "quark", None)
    if config is None or not (plugin.config.general.send_quark_link or force) or not images:
        return []
    try:
        cookie = _cookie_path().read_text(encoding="utf-8").strip()
        if not cookie:
            raise RuntimeError("session unavailable")
    except Exception:
        logger.warning("[ppnai] Quark cookie unavailable; login QR will be logged")
        await announce_quark_login(plugin, force=True)
        return []
    expires = int((time.time() + config.link_expiry_minutes * 60) * 1000)
    urls = []
    uploaded: list[tuple[str, str]] = []
    for index, image in enumerate(images, 1):
        try:
            fid, url = await asyncio.to_thread(_upload_sync, cookie, image, f"ppnai-{uuid.uuid4().hex[:12]}-{index}.png", str(config.parent_folder_id), expires)
            urls.append(url)
            uploaded.append((fid, url))
        except Exception as exc:
            logger.warning("[ppnai] Quark upload failed for image %s: %s", index, type(exc).__name__)
    if uploaded and config.delete_after_expiry:
        records = await asyncio.to_thread(_load_registry)
        records.extend({"fid": fid, "delete_after": expires / 1000} for fid, _ in uploaded)
        await asyncio.to_thread(_save_registry, records)
    return urls


def build_quark_result(event, urls: Sequence[str]):
    return event.chain_result([Plain("\n".join(urls))]) if urls else None


async def quark_cleanup_loop(plugin) -> None:
    while True:
        if plugin.config.general.send_quark_link and plugin.config.quark.delete_after_expiry:
            try:
                cookie = _cookie_path().read_text(encoding="utf-8").strip()
                records = await asyncio.to_thread(_load_registry)
                due = [item for item in records if item.get("delete_after", float("inf")) <= time.time()]
                pending = [item for item in records if item not in due]
                if due and cookie:
                    def delete_due() -> None:
                        with QuarkClient(cookies=cookie, auto_login=False) as client:
                            for item in due:
                                try:
                                    client.files.delete_files([str(item["fid"])])
                                except Exception as exc:
                                    logger.warning("[ppnai] Quark cleanup failed: %s", type(exc).__name__)
                    await asyncio.to_thread(delete_due)
                    await asyncio.to_thread(_save_registry, pending)
            except Exception as exc:
                logger.warning("[ppnai] Quark cleanup unavailable: %s", type(exc).__name__)
        await asyncio.sleep(max(300, int(plugin.config.quark.cleanup_interval_minutes) * 60))


def _ascii_qr(value: str) -> str:
    import qrcode

    qr = qrcode.QRCode(border=1)
    qr.add_data(value)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    lines = []
    for row_index in range(0, len(matrix), 2):
        upper = matrix[row_index]
        lower = matrix[row_index + 1] if row_index + 1 < len(matrix) else [False] * len(upper)
        line = "".join(
            "█" if top and bottom else "▀" if top else "▄" if bottom else " "
            for top, bottom in zip(upper, lower)
        )
        lines.append(line.rstrip())
    return "\n".join(lines)


def _login_and_save_cookie() -> str:
    login = APILogin(timeout=300)
    token, url = login.get_qr_code()
    logger.info("[ppnai] 未配置夸克 Cookie，请使用夸克 APP 扫描以下二维码登录：\n%s\n登录链接：%s", _ascii_qr(url), url)
    if not login.wait_for_login(token):
        raise RuntimeError("Quark QR login timed out or failed")
    cookies = "; ".join(
        f"{cookie.name}={cookie.value}"
        for cookie in login.client.cookies.jar
        if cookie.domain and "quark.cn" in cookie.domain
    )
    if not cookies:
        raise RuntimeError("Quark login returned no cookies")
    path = _cookie_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cookies, encoding="utf-8")
    logger.info("[ppnai] 夸克登录成功，Cookie 已保存")
    return cookies


async def announce_quark_login(plugin, force: bool = False) -> None:
    """Log a login QR once while sharing is enabled and no cookie exists."""
    if not (plugin.config.general.send_quark_link or force):
        return
    path = _cookie_path()
    if path.is_file() and path.read_text(encoding="utf-8").strip():
        return
    task = getattr(plugin, "_quark_login_task", None)
    if task is not None and not task.done():
        return
    plugin._quark_login_task = plugin._create_background_task(
        asyncio.to_thread(_login_and_save_cookie), name="nai:quark_login"
    )
