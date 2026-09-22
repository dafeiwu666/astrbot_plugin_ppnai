"""Generate QR-code images for per-request draw output."""
import base64
from io import BytesIO

import qrcode


def build_qr_images(image: bytes) -> list[bytes]:
    """Encode an image as numbered QR chunks because one QR cannot hold a PNG."""
    encoded = base64.b64encode(image).decode("ascii")
    chunk_size = 2200
    chunks = [encoded[index:index + chunk_size] for index in range(0, len(encoded), chunk_size)]
    result = []
    for index, chunk in enumerate(chunks, 1):
        qr = qrcode.QRCode(box_size=8, border=4, error_correction=qrcode.constants.ERROR_CORRECT_L)
        qr.add_data(f"PPNAI-IMAGE {index}/{len(chunks)}\n{chunk}")
        qr.make(fit=True)
        output = BytesIO()
        qr.make_image(fill_color="black", back_color="white").save(output, format="PNG")
        result.append(output.getvalue())
    return result
