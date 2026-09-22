"""Generate one QR-code image for a hosted image URL."""
from io import BytesIO

import qrcode


def build_qr_image(url: str) -> bytes:
    """Encode a hosted image URL into one QR image."""
    qr = qrcode.QRCode(
        box_size=8,
        border=4,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
    )
    qr.add_data(url)
    qr.make(fit=True)
    output = BytesIO()
    qr.make_image(fill_color="black", back_color="white").save(output, format="PNG")
    return output.getvalue()
