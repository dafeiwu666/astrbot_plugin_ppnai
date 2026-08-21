"""Formatting for optional post-generation parameter records."""

import base64
import binascii

from .models import Req


def format_generation_report(raw_input: str, req: Req) -> str:
    params = [
        f"model={req.model}",
        f"artist={req.artist or '(none)'}",
        f"size={req.size}",
        f"steps={req.steps}",
        f"scale={req.scale}",
        f"cfg={req.cfg}",
        f"sampler={req.sampler}",
        f"noise_schedule={req.noise_schedule}",
        f"seed={req.seed or '(random)'}",
        f"other={req.other}",
        f"i2i_force={req.i2i_force}",
        f"i2i_cl={req.i2i_cl}",
    ]
    if req.addition.image_to_image_base64:
        params.append("i2i=true")
    if req.addition.vibe_transfer_list:
        params.append(f"vibe_transfer={len(req.addition.vibe_transfer_list)} image(s)")
        for index, vibe in enumerate(req.addition.vibe_transfer_list, 1):
            params.append(
                f"vibe[{index}].info_extract={vibe.info_extract}, "
                f"ref_strength={vibe.ref_strength}"
            )
    if req.addition.character_keep:
        params.append("character_keep=true")
        params.append(
            f"character_keep_vibe={req.addition.character_keep.keep_vibe}, "
            f"strength={req.addition.character_keep.strength}"
        )
    return (
        "画图参数记录\n"
        f"原始输入：\n{raw_input or '(empty)'}\n\n"
        f"最终正向：\n{req.tag}\n\n"
        f"最终负向：\n{req.negative}\n\n"
        "最终参数：\n"
        + "\n".join(params)
        + (
            "\n\n输入图片：见下方"
            if (
                req.addition.image_to_image_base64
                or req.addition.vibe_transfer_list
                or (
                    req.addition.character_keep
                    and req.addition.character_keep.base64
                )
            )
            else ""
        )
    )


def get_input_image_bytes(req: Req) -> list[bytes]:
    """Return the source images used by the request for the detail record."""
    data_uris: list[str] = []
    if req.addition.image_to_image_base64:
        data_uris.append(req.addition.image_to_image_base64)
    data_uris.extend(
        item.base64
        for item in req.addition.vibe_transfer_list
        if item.base64
    )
    if req.addition.character_keep and req.addition.character_keep.base64:
        data_uris.append(req.addition.character_keep.base64)

    images: list[bytes] = []
    for data_uri in data_uris:
        encoded = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
        try:
            images.append(base64.b64decode(encoded, validate=True))
        except (ValueError, binascii.Error):
            continue
    return images
