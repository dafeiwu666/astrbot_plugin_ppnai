"""
NovelAI 官方 API 数据源模块

适配官方 NovelAI API (https://image.novelai.net)
"""

import io
import asyncio
import json
import random
import re
import time
import zipfile
from typing import Any

from httpx import AsyncClient, Timeout

from astrbot.api import logger

from .config import Config
from .models import Req

BASE64_BLOB_RE = re.compile(r"(?:data:[^;]+;base64,)?[A-Za-z0-9+/]{512,}={0,2}")


def _shorten_base64_segments(text: str) -> str:
    """Replace long base64 blobs with placeholders for readability (log-safe)."""

    def _replace(match: re.Match[str]) -> str:
        chunk = match.group(0)
        if chunk.startswith("data:"):
            prefix, _, payload = chunk.partition(",")
            mime = prefix[5:].split(";")[0] if len(prefix) > 5 else "unknown"
            return f"<base64:{mime},len={len(payload)}>"
        return f"<base64:len={len(chunk)}>"

    return BASE64_BLOB_RE.sub(_replace, text)

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


_shared_client: AsyncClient | None = None
_shared_client_sig: tuple[str, float, float] | None = None
_shared_client_lock = asyncio.Lock()


def _client_signature(config: "Config") -> tuple[str, float, float]:
    return (
        str(config.request.base_url),
        float(config.request.connect_timeout),
        float(config.request.read_timeout),
    )


async def get_shared_client(config: "Config") -> AsyncClient:
    """Get a process-wide shared client for connection pooling."""
    global _shared_client, _shared_client_sig
    sig = _client_signature(config)
    async with _shared_client_lock:
        if _shared_client is not None and _shared_client_sig == sig:
            return _shared_client
        if _shared_client is not None:
            try:
                await _shared_client.aclose()
            except Exception:  # noqa: BLE001
                pass
        _shared_client = create_client_from_config(config)
        _shared_client_sig = sig
        return _shared_client


async def aclose_shared_client() -> None:
    global _shared_client, _shared_client_sig
    async with _shared_client_lock:
        if _shared_client is None:
            return
        try:
            await _shared_client.aclose()
        finally:
            _shared_client = None
            _shared_client_sig = None


class GenerateError(Exception):
    def __init__(self, message: str = "", status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        return f"{self.message} (status={self.status_code})"


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
        return _shorten_base64_segments(obj)
    else:
        return obj


def _extract_base64_data(data_uri: str) -> str:
    """从 data URI 中提取纯 base64 数据"""
    if data_uri.startswith("data:"):
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


async def generate_image(
    cli: AsyncClient,
    req: Req,
    opus_free_mode: bool = False,
    start_time: int | None = None,
    token: str = "",
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
        error_msg = f"API请求失败"
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
        zip_data = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_data, 'r') as zf:
            # ZIP 中通常只有一个图片文件
            file_list = zf.namelist()
            if not file_list:
                raise GenerateError("返回的 ZIP 文件为空")
            
            # 获取第一个图片文件
            image_filename = file_list[0]
            image_data = zf.read(image_filename)

            return image_data
            
    except zipfile.BadZipFile as e:
        logger.error(f"[nai] 无法解析返回的 ZIP 文件: {e}")
        raise GenerateError("返回的数据不是有效的 ZIP 文件") from e


async def wrapped_generate(req: Req, config: Config, token: str = "") -> bytes:
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
    
    cli = await get_shared_client(config)
    image = await generate_image(
        cli,
        req,
        opus_free_mode=opus_free_mode,
        start_time=start_time,
        token=token,
    )
    
    consumed_time_s = (time.time_ns() - start_time) / 1e9
    logger.debug(f"[nai] {start_time} -> end ({consumed_time_s} s)")
    logger.info(f"[nai] 图片生成完成 ({consumed_time_s:.2f}s)")
    
    return image
