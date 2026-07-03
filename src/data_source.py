"""
NovelAI 官方 API 数据源模块

适配官方 NovelAI API (https://image.novelai.net)
"""

import hashlib
import io
import asyncio
import base64
import json
import random
import time
import zipfile
from collections.abc import Awaitable, Callable
from typing import Any

from httpx import AsyncClient, Timeout

from astrbot.api import logger

from .config import Config
from .models import Req

from .text_sanitize import shorten_base64_segments

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)


def create_client_from_config(config: "Config", token: str = ""):
    """创建 HTTP 客户端，配置官方 API 认证"""
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/zip",
        # 必须添加以下请求头，否则 NovelAI 服务器可能拒绝请求
        "Origin": "https://novelai.net",
        "Referer": "https://novelai.net",
    }
    
    # 注意：Bearer Token 建议按“每次请求”附带，避免共享 Client 时混用。
    return AsyncClient(
        base_url=config.request.base_url,
        headers=headers,
        timeout=Timeout(
            config.request.connect_timeout, read=config.request.read_timeout
        ),
    )


class GenerateError(Exception):
    def __init__(self, message: str = "", status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body


# ============================================================
# Vibe 编码缓存 —— 避免重复调 /ai/encode-vibe 被重复扣 Anlas
# ============================================================

class VibeCacheManager:
    """Vibe 编码缓存管理器 — 与其他 Manager 一致的 data_dir 注入模式

    缓存格式 (v2):
        {"hash123": {"v": "base64value", "ts": 1234567890.0}, ...}
    兼容旧格式 (v1):
        {"hash123": "base64value", ...}  → 加载时自动迁移至 v2
    """

    def __init__(self, data_dir: "Path", ttl_days: int = 7) -> None:
        self._data_file = data_dir / "vibe_encode_cache.json"
        self._data_dir = data_dir
        self._ttl_days = ttl_days
        self._cache: dict[str, dict[str, object]] | None = None

    @property
    def ttl_days(self) -> int:
        return self._ttl_days

    @ttl_days.setter
    def ttl_days(self, value: int) -> None:
        self._ttl_days = value

    def _ensure_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict[str, object]]:
        if self._cache is not None:
            return self._cache
        self._ensure_dir()
        try:
            if self._data_file.exists():
                raw = json.loads(self._data_file.read_text("utf-8"))
                migrated: dict[str, dict[str, object]] = {}
                needs_save = False
                for key, val in raw.items():
                    if isinstance(val, dict) and "v" in val and "ts" in val:
                        # v2 格式 — 直接使用
                        migrated[key] = val
                    elif isinstance(val, str):
                        # v1 格式 — 自动迁移，时间戳设为当前时间
                        migrated[key] = {"v": val, "ts": time.time()}
                        needs_save = True
                    else:
                        logger.warning(f"[nai] vibe 缓存条目格式异常，已跳过: {key}")
                self._cache = migrated
                logger.info(f"[nai] vibe 编码缓存已加载: {len(self._cache)} 条")
                if needs_save:
                    self._save()
                    logger.info("[nai] vibe 缓存已从 v1 迁移至 v2 格式")
            else:
                self._cache = {}
        except Exception as e:
            logger.warning(f"[nai] vibe 编码缓存加载失败，将使用空缓存: {e}")
            self._cache = {}
        return self._cache

    def _save(self) -> None:
        if self._cache is None:
            return
        try:
            self._ensure_dir()
            self._data_file.write_text(json.dumps(self._cache, ensure_ascii=False), "utf-8")
        except Exception as e:
            logger.error(f"[nai] vibe 编码缓存保存失败: {e}")

    def _is_expired(self, entry: dict[str, object]) -> bool:
        ts = entry.get("ts")
        if not isinstance(ts, (int, float)):
            return True
        age_secs = time.time() - float(ts)
        return age_secs > self._ttl_days * 86400

    @staticmethod
    def make_key(img_b64: str, info_extract: float) -> str:
        h = hashlib.sha256()
        h.update(img_b64.encode("utf-8"))
        h.update(f"{info_extract:.6f}".encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> str | None:
        cache = self._load()
        entry = cache.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            logger.info(
                f"[nai] vibe 缓存已过期 (TTL={self._ttl_days}天), 已删除: {key[:16]}..."
            )
            del cache[key]
            self._save()
            return None
        return str(entry["v"])

    def put(self, key: str, value: str) -> None:
        self._load()[key] = {"v": value, "ts": time.time()}
        self._save()

    def remove(self, key: str) -> bool:
        """删除单条缓存，返回是否成功删除"""
        cache = self._load()
        if key in cache:
            del cache[key]
            self._save()
            logger.info(f"[nai] vibe 缓存已删除: {key[:16]}...")
            return True
        return False

    def reset(self) -> None:
        """一键清空所有缓存"""
        self._cache = {}
        try:
            if self._data_file.exists():
                self._data_file.unlink()
            logger.info("[nai] vibe 编码缓存已重置（全部清空）")
        except Exception as e:
            logger.error(f"[nai] vibe 缓存重置失败: {e}")

    def cleanup_expired(self) -> int:
        """清理所有过期条目，返回清理数量"""
        cache = self._load()
        expired_keys = [k for k, v in cache.items() if self._is_expired(v)]
        for k in expired_keys:
            del cache[k]
        if expired_keys:
            self._save()
            logger.info(f"[nai] vibe 缓存过期清理: {len(expired_keys)} 条")
        return len(expired_keys)

    def __str__(self) -> str:
        cache = self._load() if self._cache is not None else {}
        return (
            f"VibeCacheManager({len(cache)} entries, "
            f"TTL={self._ttl_days} days, "
            f"file={self._data_file})"
        )


def _sanitize_for_log(obj: Any) -> Any:
    """递归处理对象，隐藏敏感信息但保留完整内容供排查。"""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "token" and isinstance(v, str) and v:
                result[k] = f"{v[:8]}...{v[-4:]}" if len(v) > 12 else "***"
            elif k == "base64" and isinstance(v, str) and v:
                if v.startswith("data:"):
                    mime_end = v.find(";")
                    mime_type = v[5:mime_end] if mime_end > 5 else "unknown"
                    result[k] = f"<{mime_type}, {len(v)} chars>"
                else:
                    result[k] = f"<{len(v)} chars>"
            else:
                result[k] = _sanitize_for_log(v)
        return result
    elif isinstance(obj, list):
        return [_sanitize_for_log(item) for item in obj]
    elif isinstance(obj, str) and obj:
        # 对任意字段里的长 base64 做统一缩短，避免官方 API 请求体把图片整段打进日志
        return shorten_base64_segments(obj)
    else:
        return obj


def _extract_base64_data(data_uri: str) -> str:
    """从 data URI 中提取纯 base64 数据"""
    if data_uri.startswith("data:"):
        if "," not in data_uri:
            logger.warning(
                "[nai] Malformed data URI (no comma separator); keep as-is. len=%s prefix=%r",
                len(data_uri),
                data_uri[:32],
            )
            return data_uri
        # 格式: data:image/jpeg;base64,xxxxx
        if ",base64," in data_uri:
            return data_uri.split(",base64,", 1)[1]
        elif "," in data_uri:
            return data_uri.split(",", 1)[1]
    return data_uri


# Opus 免费模式的限制
OPUS_FREE_MAX_PIXELS = 1024 * 1024  # 最大像素数（1024x1024）
OPUS_FREE_MAX_STEPS = 28  # 最大步数


def _adjust_size_for_opus_free(width: int, height: int) -> tuple[int, int]:
    """
    调整尺寸以符合 Opus 免费模式的限制（总像素 ≤ 1024x1024）
    保持宽高比，向下取整到最接近的 64 的倍数
    """
    total_pixels = width * height
    if total_pixels <= OPUS_FREE_MAX_PIXELS:
        return width, height
    
    # 计算缩放比例
    scale = (OPUS_FREE_MAX_PIXELS / total_pixels) ** 0.5
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 对齐到 64 的倍数（NovelAI 要求）
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    
    # 确保最小尺寸
    new_width = max(64, new_width)
    new_height = max(64, new_height)
    
    return new_width, new_height


def _convert_req_to_official_format(req: Req, opus_free_mode: bool = False) -> dict:
    """
    将内部请求格式转换为官方 NovelAI API 格式
    
    官方 API 格式:
    {
        "input": "正向提示词",
        "model": "模型名",
        "action": "generate",
        "parameters": {
            "width": 832,
            "height": 1216,
            "steps": 23,
            ...
        }
    }
    
    Args:
        req: 请求对象
        opus_free_mode: 是否开启 Opus 免费模式（小图模式）
    """
    # 解析尺寸
    width, height = [int(x) for x in req.size.split("x")]
    
    # Opus 免费模式：调整尺寸
    if opus_free_mode:
        original_size = f"{width}x{height}"
        width, height = _adjust_size_for_opus_free(width, height)
        if f"{width}x{height}" != original_size:
            logger.info(f"[nai] Opus免费模式: 尺寸调整 {original_size} → {width}x{height}")
    
    # 生成种子（如果未指定则随机）
    seed = int(req.seed) if req.seed else random.randint(0, 2**32 - 1)
    
    # 解析步数
    steps = int(req.steps)
    
    # Opus 免费模式：限制步数
    if opus_free_mode and steps > OPUS_FREE_MAX_STEPS:
        logger.info(f"[nai] Opus免费模式: 步数限制 {steps} → {OPUS_FREE_MAX_STEPS}")
        steps = OPUS_FREE_MAX_STEPS
    
    # 解析 SMEA 设置 (other 字段)
    # 0: 不使用, 1: Auto, 2: SMEA, 3: SMEA+DYN, 4: Auto+SMEA, 5: Auto+SMEA+DYN
    other_val = int(req.other)
    sm = other_val in (2, 3, 4, 5)  # SMEA
    sm_dyn = other_val in (3, 5)    # DYN
    
    # 构建正向提示词（包含画师串）
    prompt = req.tag
    if req.artist:
        prompt = f"{req.artist}, {prompt}" if prompt else req.artist
    
    # 构建参数对象
    # 注意：NovelAI v4 模型需要 v4_prompt 和 v4_negative_prompt 参数
    parameters: dict[str, Any] = {
        # 基础参数
        "params_version": 3,
        "width": width,
        "height": height,
        "steps": steps,
        "scale": float(req.scale),
        "seed": seed,
        "sampler": req.sampler,
        "noise_schedule": req.noise_schedule,
        "negative_prompt": req.negative,
        "cfg_rescale": float(req.cfg),
        "sm": sm,
        "sm_dyn": sm_dyn,
        "n_samples": 1,
        "ucPreset": 0,  # 使用自定义负面提示词
        "qualityToggle": False,  # 关闭质量切换（已在提示词中处理）
        # v4 模型必需参数
        "dynamic_thresholding": False,
        "controlnet_strength": 1,
        "normalize_reference_strength_multiple": True,
        "legacy": False,
        "add_original_image": True,
        "legacy_v3_extend": False,
        "skip_cfg_above_sigma": None,
        "use_coords": False,
        "characterPrompts": [],
        # v4 提示词格式
        "v4_prompt": {
            "caption": {
                "base_caption": prompt,
                "char_captions": []
            },
            "use_coords": False,
            "use_order": True
        },
        "v4_negative_prompt": {
            "caption": {
                "base_caption": req.negative,
                "char_captions": []
            }
        },
        # 参考图片相关（默认为空）
        "reference_image_multiple": [],
        "reference_information_extracted_multiple": [],
        "reference_strength_multiple": [],
    }
    
    # 确定 action 类型和处理高级功能
    action = "generate"
    
    # 处理图生图 (img2img)
    if req.addition.image_to_image_base64:
        action = "img2img"
        parameters["image"] = _extract_base64_data(req.addition.image_to_image_base64)
        parameters["strength"] = float(req.i2i_force)
        parameters["noise"] = 0
    
    # 处理氛围转移 (vibe transfer)
    if req.addition.vibe_transfer_list:
        vibe_images = []
        vibe_info_extracts = []
        vibe_strengths = []
        
        for vibe in req.addition.vibe_transfer_list:
            if vibe.base64:
                vibe_images.append(_extract_base64_data(vibe.base64))
                vibe_info_extracts.append(float(vibe.info_extract))
                vibe_strengths.append(float(vibe.ref_strength))
        
        if vibe_images:
            parameters["reference_image_multiple"] = vibe_images
            parameters["reference_information_extracted_multiple"] = vibe_info_extracts
            parameters["reference_strength_multiple"] = vibe_strengths
    
    # 处理角色保持 (character reference / director tools)
    # 注意：官方 API 使用不同的参数名
    if req.addition.character_keep and req.addition.character_keep.base64:
        ck = req.addition.character_keep
        # 官方 API 使用 director_reference_* 参数
        parameters["director_reference_images"] = [_extract_base64_data(ck.base64)]
        parameters["director_reference_strength_values"] = [float(ck.strength)]
        parameters["director_reference_information_extracted"] = [1.0]
        
        # 设置描述类型
        if ck.keep_vibe:
            # 保持角色+氛围
            parameters["director_reference_descriptions"] = [{
                "caption": {
                    "base_caption": "character&style",
                    "char_captions": []
                },
                "use_coords": False,
                "use_order": False,
                "legacy_uc": False
            }]
        else:
            # 仅保持角色
            parameters["director_reference_descriptions"] = [{
                "caption": {
                    "base_caption": "character",
                    "char_captions": []
                },
                "use_coords": False,
                "use_order": False,
                "legacy_uc": False
            }]
        
        # 设置 Fidelity（映射到 secondary_strength）
        parameters["director_reference_secondary_strength_values"] = [float(ck.strength)]
    
    # 处理多角色控制 - 转换为 v4_prompt 格式
    # 注意：这个功能在官方API中实现方式较复杂，暂时简化处理
    if req.addition.multi_role_list:
        # 多角色控制需要使用 v4_prompt 格式
        # 这里暂时将多角色提示词合并到主提示词中
        char_prompts = []
        for role in req.addition.multi_role_list:
            if role.prompt:
                char_prompts.append(f"[{role.position}: {role.prompt}]")
        if char_prompts:
            # 追加到主提示词后面
            prompt = f"{prompt}, {', '.join(char_prompts)}" if prompt else ', '.join(char_prompts)
            logger.warning("[nai] 多角色控制功能在官方API中暂使用简化处理")
    
    # 构建最终请求
    request_body = {
        "input": prompt,
        "model": req.model,
        "action": action,
        "parameters": parameters
    }
    
    return request_body


async def _encode_single_vibe(
    cli: AsyncClient,
    idx: int,
    total: int,
    img_b64: str,
    info_extract: float,
    model: str,
    token: str = "",
) -> str:
    """调用 NovelAI /ai/encode-vibe 将参考图编码为 vibe 向量。

    encode-vibe 是 vibe transfer 链路的前置步骤：
    将原始图片 base64 POST 到 /ai/encode-vibe，服务端返回二进制编码向量；
    客户端将其 base64 后再放入 generate-image 请求的 reference_image_multiple。

    若编码失败（网络/服务端错误），安全降级为返回原始 base64 作为 fallback，
    避免因编码步骤失败而中断整个生图流程。

    Returns:
        成功时返回编码向量的 base64 字符串；失败时返回原始 img_b64。
    """
    try:
        encode_resp = await cli.post(
            "/ai/encode-vibe",
            json={
                "image": img_b64,
                "information_extracted": info_extract,
                "model": model,
            },
            headers={"Authorization": f"Bearer {token}"} if token else None,
        )
        if encode_resp.status_code == 200:
            encoded_b64 = base64.b64encode(encode_resp.content).decode("utf-8")
            logger.info(
                f"[nai] vibe 编码成功 ({idx+1}/{total})"
            )
            return encoded_b64
        else:
            logger.error(
                f"[nai] vibe 编码失败 ({idx+1}/{total}): "
                f"status={encode_resp.status_code}, body={encode_resp.text[:200]}"
            )
            return img_b64  # fallback
    except Exception as e:
        logger.error(
            f"[nai] vibe 编码异常 ({idx+1}/{total}): {e}"
        )
        return img_b64  # fallback


async def _do_generate_post(
    cli: AsyncClient,
    request_body: dict,
    token: str = "",
    start_time: int | None = None,
) -> bytes:
    """向 NovelAI /ai/generate-image 发送生图请求并解压 ZIP 响应。

    这是 generate_image 中最核心的一步 —— 将组装好的请求体 POST 到官方 API，
    解析返回的 ZIP 包提取首张图片。任何非 2xx 响应或 ZIP 损坏均抛出 GenerateError。

    Returns:
        解压后的 PNG 图片字节数据。
    Raises:
        GenerateError: API 返回非 2xx 状态码，或 ZIP 解析失败。
    """

    # 记录请求日志（隐藏敏感信息）
    sanitized_body = _sanitize_for_log(request_body)
    logger.info(
        f"[nai] 发送请求: {json.dumps(sanitized_body, ensure_ascii=False, indent=2)}"
    )

    # 发送请求
    headers = {"Authorization": f"Bearer {token}"} if token else None
    response = await cli.post("/ai/generate-image", json=request_body, headers=headers)
    if start_time is not None:
        logger.debug(
            f"[nai] {start_time} -> {response.status_code}: "
            f"{response.headers.get('content-type', '') or 'unknown'}"
        )

    # 处理错误响应
    if response.status_code != 200 and response.status_code != 201:
        error_msg = "API请求失败"
        try:
            error_data = response.json()
            if "message" in error_data:
                error_msg = error_data["message"]
            elif "error" in error_data:
                error_msg = error_data["error"]
        except Exception:
            error_msg = response.text[:500] if response.text else f"HTTP {response.status_code}"

        logger.error(f"[nai] 官方API返回错误: status={response.status_code}, body={response.text[:500]}")
        raise GenerateError(error_msg, response.status_code, response.text)

    # 官方 API 返回 ZIP 文件，需要解压获取图片
    try:
        def _extract_image_from_zip(payload: bytes) -> bytes:
            zip_data = io.BytesIO(payload)
            with zipfile.ZipFile(zip_data, "r") as zf:
                file_list = zf.namelist()
                if not file_list:
                    raise GenerateError("返回的 ZIP 文件为空")
                image_filename = file_list[0]
                return zf.read(image_filename)

        return await asyncio.to_thread(_extract_image_from_zip, response.content)
    except zipfile.BadZipFile as e:
        logger.error(f"[nai] 无法解析返回的 ZIP 文件: {e}")
        raise GenerateError("返回的数据不是有效的 ZIP 文件") from e


async def generate_image(
    cli: AsyncClient,
    req: Req,
    opus_free_mode: bool = False,
    start_time: int | None = None,
    token: str = "",
    *,
    vibe_cache: "VibeCacheManager | None" = None,
) -> bytes:
    """
    调用官方 NovelAI API 生成图片
    
    Args:
        cli: HTTP 客户端
        req: 请求对象
        opus_free_mode: 是否开启 Opus 免费模式
    
    Returns:
        生成的图片字节数据
    """
    # 转换请求格式
    request_body = _convert_req_to_official_format(req, opus_free_mode=opus_free_mode)
    
    # Vibe Transfer: v4+ 模型需要先通过 /ai/encode-vibe 编码参考图
    # track: (index, img_b64, info_extract, cache_key) for entries that came from cache
    cached_vibe_entries: list[tuple[int, str, float, str]] = []
    
    if request_body["parameters"].get("reference_image_multiple"):
        if vibe_cache is not None:
            vibe_cache._load()  # ensure cache is loaded
        model = request_body.get("model", "")
        vibe_images = request_body["parameters"]["reference_image_multiple"]
        vibe_info_extracts = request_body["parameters"].get(
            "reference_information_extracted_multiple", []
        )
        encoded_vibes = []
        
        for i, img_b64 in enumerate(vibe_images):
            info_extract = vibe_info_extracts[i] if i < len(vibe_info_extracts) else 0.8
            cache_key = VibeCacheManager.make_key(img_b64, info_extract)
            
            # 检查缓存
            if vibe_cache is not None and (cached := vibe_cache.get(cache_key)) is not None:
                encoded_vibes.append(cached)
                cached_vibe_entries.append((i, img_b64, info_extract, cache_key))
                logger.info(
                    f"[nai] vibe 编码缓存命中 ({i+1}/{len(vibe_images)}, "
                    f"info_extract={info_extract})"
                )
                continue
            
            # 缓存未命中 → 调 encode-vibe
            encoded = await _encode_single_vibe(cli, i, len(vibe_images), img_b64, info_extract, model, token)
            encoded_vibes.append(encoded)
            if vibe_cache is not None and encoded != img_b64:
                vibe_cache.put(cache_key, encoded)
        
        request_body["parameters"]["reference_image_multiple"] = encoded_vibes
    
    # ── POST /ai/generate-image（带缓存失效重试）─────────────────────
    # NovelAI 的 vibe 编码向量会在服务端缓存一段时间，但官方未公开 TTL。
    # 如果缓存中的向量已过期，生图 API 会返回错误。
    #
    # 策略：最多尝试 2 次。首次失败时，若命中了本地缓存，
    # 则清除这些缓存条目并重新调 encode-vibe 获取新鲜向量，再 POST 一次。
    # 若第二次仍失败（或首次就没用到缓存），直接向上抛异常。
    # ────────────────────────────────────────────────────────────────
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            return await _do_generate_post(cli, request_body, token, start_time)
        except GenerateError as e:
            can_retry = (
                attempt == 0
                and len(cached_vibe_entries) > 0
                and vibe_cache is not None
            )
            if not can_retry:
                raise
            
            logger.warning(
                f"[nai] 生图失败 (status={e.status_code}), "
                f"正在清除 {len(cached_vibe_entries)} 条 vibe 缓存并重新编码..."
            )
            
            # 逐条清除过期缓存并重新编码
            model = request_body.get("model", "")
            vibe_images = request_body["parameters"]["reference_image_multiple"]
            vibe_info_extracts = request_body["parameters"].get(
                "reference_information_extracted_multiple", []
            )
            
            for idx, img_b64, info_extract, cache_key in cached_vibe_entries:
                vibe_cache.remove(cache_key)
                encoded = await _encode_single_vibe(cli, idx, len(vibe_images), img_b64, info_extract, model, token)
                vibe_images[idx] = encoded
                if encoded != img_b64:
                    vibe_cache.put(cache_key, encoded)
                    logger.info(
                        f"[nai] vibe 缓存已清除并重新编码 ({idx+1}/{len(vibe_images)})"
                    )
            
            request_body["parameters"]["reference_image_multiple"] = vibe_images
            cached_vibe_entries = []  # 清空追踪列表，避免第二次失败时重复清理
            logger.info("[nai] 使用重新编码的 vibe 数据重试生图...")
            
        except Exception as e:
            # 非 GenerateError 的异常（如网络超时、连接重置）也适用同样的重试逻辑
            can_retry = (
                attempt == 0
                and len(cached_vibe_entries) > 0
                and vibe_cache is not None
            )
            if not can_retry:
                raise
            
            logger.warning(
                f"[nai] 生图异常 ({type(e).__name__}: {e}), "
                f"正在清除 {len(cached_vibe_entries)} 条 vibe 缓存并重新编码..."
            )
            
            model = request_body.get("model", "")
            vibe_images = request_body["parameters"]["reference_image_multiple"]
            vibe_info_extracts = request_body["parameters"].get(
                "reference_information_extracted_multiple", []
            )
            
            for idx, img_b64, info_extract, cache_key in cached_vibe_entries:
                try:
                    vibe_cache.remove(cache_key)
                except Exception:
                    pass
                encoded = await _encode_single_vibe(cli, idx, len(vibe_images), img_b64, info_extract, model, token)
                vibe_images[idx] = encoded
                try:
                    if encoded != img_b64:
                        vibe_cache.put(cache_key, encoded)
                except Exception:
                    pass
            
            request_body["parameters"]["reference_image_multiple"] = vibe_images
            cached_vibe_entries = []
            logger.info("[nai] 使用重新编码的 vibe 数据重试生图...")
    
    # 不应到达这里，但保留作为兜底
    raise GenerateError("重试次数已耗尽")


async def wrapped_generate(
    req: Req,
    config: Config,
    token: str = "",
    *,
    client_getter: Callable[[], Awaitable[AsyncClient]] | None = None,
    vibe_cache: "VibeCacheManager | None" = None,
) -> bytes:
    """生成图片
    
    Args:
        req: 请求对象
        config: 配置
        token: 使用的 Token（Bearer Token）
    
    Returns:
        生成的图片字节数据
    """
    start_time = time.time_ns()
    opus_free_mode = config.request.opus_free_mode

    logger.debug(f"[nai] {start_time} -> start")
    
    close_after = False
    if client_getter is not None:
        cli = await client_getter()
    else:
        # 回退为“每次请求一个临时 client”，避免模块级全局状态。
        # 若你希望复用连接池，应在上层（Plugin 实例）注入 client_getter。
        cli = create_client_from_config(config)
        close_after = True

    try:
        image = await generate_image(
            cli,
            req,
            opus_free_mode=opus_free_mode,
            start_time=start_time,
            token=token,
            vibe_cache=vibe_cache,
        )
    finally:
        if close_after:
            await cli.aclose()
    
    consumed_time_s = (time.time_ns() - start_time) / 1e9
    logger.debug(f"[nai] {start_time} -> end ({consumed_time_s} s)")
    logger.info(f"[nai] 图片生成完成 ({consumed_time_s:.2f}s)")
    
    return image
