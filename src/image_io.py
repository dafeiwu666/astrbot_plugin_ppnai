"""Image/base64 helpers for request assembly."""

import asyncio
import base64
import io

from PIL import Image as PILImage

from astrbot.api import logger
from astrbot.api.message_components import Image

from .utils import get_base64_mime


async def resolve_image(image: Image) -> str:
    """Normalize AstrBot image to data URI base64 string."""
    b64 = await image.convert_to_base64()
    if isinstance(b64, str) and b64.startswith("base64://"):
        b64 = b64.removeprefix("base64://")
    return f"data:{get_base64_mime(b64, 'image/jpeg')};base64,{b64}"


# Allowed target sizes for character-keep feature (server allowlist)
CHARACTER_KEEP_TARGET_SIZES: list[tuple[int, int]] = [
    (1472, 1472),
    (1536, 1024),
    (1024, 1536),
]


def _select_best_target_size(width: int, height: int) -> tuple[int, int]:
    """Pick target size with closest aspect ratio to source."""
    original_ratio = width / height
    best_size = CHARACTER_KEEP_TARGET_SIZES[0]
    best_ratio_diff = float("inf")
    for target_w, target_h in CHARACTER_KEEP_TARGET_SIZES:
        target_ratio = target_w / target_h
        ratio_diff = abs(original_ratio - target_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_size = (target_w, target_h)
    return best_size


def convert_to_jpeg_for_character_keep(image_b64: str) -> str:
    """Resize/pad image to allowed sizes and return JPEG data URI.

    This function is CPU-bound (PIL decode/resize/encode) and should NOT be called
    directly from the event loop. Prefer `aconvert_to_jpeg_for_character_keep()`.
    """
    # Parse input data URI or raw base64
    if image_b64.startswith("data:"):
        header, b64_data = image_b64.split(",", 1)
        original_mime = header.split(";")[0].replace("data:", "")
    else:
        b64_data = image_b64
        original_mime = "unknown"

    image_bytes = base64.b64decode(b64_data)
    pil_image = PILImage.open(io.BytesIO(image_bytes))
    original_width, original_height = pil_image.size
    original_mode = pil_image.mode

    logger.debug(
        f"[nai] 角色保持图片处理: MIME={original_mime}, "
        f"尺寸={original_width}x{original_height}, 模式={original_mode}"
    )

    target_width, target_height = _select_best_target_size(original_width, original_height)
    logger.debug(f"[nai] 选择目标尺寸: {target_width}x{target_height}")

    if pil_image.mode in ("RGBA", "LA", "P"):
        if pil_image.mode == "P":
            pil_image = pil_image.convert("RGBA")
        rgb_image = PILImage.new("RGB", pil_image.size, (0, 0, 0))
        if pil_image.mode == "RGBA":
            rgb_image.paste(pil_image, mask=pil_image.split()[3])
        else:
            rgb_image.paste(pil_image)
        pil_image = rgb_image
        logger.debug("[nai] 已将透明背景转换为黑色背景")
    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
        logger.debug(f"[nai] 已将 {original_mode} 模式转换为 RGB")

    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    pil_image = pil_image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
    logger.debug(f"[nai] 缩放后尺寸: {new_width}x{new_height}")

    canvas = PILImage.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    canvas.paste(pil_image, (paste_x, paste_y))

    if paste_x > 0 or paste_y > 0:
        logger.debug(
            f"[nai] 添加黑边填充: 左右各{paste_x}px, 上下各{paste_y}px"
        )

    output_buffer = io.BytesIO()
    canvas.save(output_buffer, format="JPEG", quality=90)
    output_buffer.seek(0)

    jpeg_b64 = base64.b64encode(output_buffer.read()).decode("utf-8")
    result = f"data:image/jpeg;base64,{jpeg_b64}"
    logger.debug(
        f"[nai] 角色保持图片处理完成: 最终尺寸={target_width}x{target_height}, "
        f"输出大小={len(result)} chars"
    )
    return result


async def aconvert_to_jpeg_for_character_keep(image_b64: str) -> str:
    """Async wrapper for `convert_to_jpeg_for_character_keep` (runs in a thread)."""
    return await asyncio.to_thread(convert_to_jpeg_for_character_keep, image_b64)


async def resolve_image_as_jpeg(image: Image) -> str:
    """Fetch image, resize/pad, and convert to JPEG for character-keep."""
    b64 = await image.convert_to_base64()
    if isinstance(b64, str) and b64.startswith("base64://"):
        b64 = b64.removeprefix("base64://")
    original_mime = get_base64_mime(b64, "image/jpeg")
    original_data_uri = f"data:{original_mime};base64,{b64}"
    logger.debug(
        f"[nai] 角色保持: 接收到图片, 原始MIME={original_mime}, 原始大小={len(b64)} chars"
    )
    return await aconvert_to_jpeg_for_character_keep(original_data_uri)
