"""Formatting for optional post-generation parameter records."""

from .models import Req


def format_generation_report(raw_input: str, req: Req, resource_keys: list[str]) -> str:
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
    resources = ", ".join(resource_keys) if resource_keys else "(none)"
    return (
        "预设生成结果\n"
        f"原始输入：\n{raw_input or '(empty)'}\n\n"
        f"最终正向：\n{req.tag}\n\n"
        f"最终负向：\n{req.negative}\n\n"
        "最终参数：\n"
        + "\n".join(params)
        + f"\n\n关联资源：{resources}"
    )
