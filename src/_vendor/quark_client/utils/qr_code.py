#!/usr/bin/env python3
from pathlib import Path

from .logger import get_logger


def print_ascii_qr(text: str):
    logger = get_logger(__name__)
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)  # 边框小一些，方便在终端放大显示
        qr.add_data(text)
        qr.make(fit=True)
        # 反相打印，提升终端显示对比度（部分终端/主题需要关闭 invert）
        qr.print_ascii(invert=True)
    except Exception as e:
        logger.warning(f"ASCII QR render failed: {e}")


def display_qr_code(qr_image_path: str):
    """
    显示二维码到终端

    由于QR码现在是从URL直接生成的，我们可以通过读取图片元数据或
    直接使用生成时的URL来显示ASCII二维码，而不需要复杂的图像解码
    """
    logger = get_logger(__name__)
    qr_path = Path(qr_image_path)

    if not qr_path.exists():
        logger.error(f"二维码文件不存在: {qr_image_path}")
        return

    # 由于我们知道这是从URL生成的，但没有直接方式获取原始URL，
    # 我们可以尝试从文件名或配置中获取，或者简单提示用户
    logger.info("二维码已生成，请使用夸克APP扫描")
    print(f"二维码文件位置: {qr_image_path}")


def display_qr_from_url(url: str):
    """
    直接从URL生成并显示ASCII二维码到终端
    这是更直接的方式，不需要先生成图片文件
    """
    print_ascii_qr(url)
