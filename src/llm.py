from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic.dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context
from astrbot.core.agent.message import ImageURLPart, Message, TextPart
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.astr_agent_context import AstrAgentContext

from .config import Config
from .data_source import wrapped_generate
from .llm_schema import (
    GENERATE_IMAGE_ADVANCED_SCHEMA_TXT,
    OrientationType,
    STNaiGenerateImageAdvancedArgs,
    STNaiGenerateImageI2IArgs,
    STNaiGenerateImageMultiRoleArgs,
    STNaiGenerateImageVibeTransferArgs,
)
from .llm_utils import apply_regex_replacements, format_readable_error
from .models import Req, ReqAdditionMultiRole
from .image_io import resolve_image
from .params import complete_defaults, post_check_limits

if TYPE_CHECKING:
    pass
PROMPTS_DIR = Path(__file__).parent / "prompts"
ADVANCED_PROMPT_PATH = PROMPTS_DIR / "advanced.txt"


def get_size_from_config(config: Config, orientation: OrientationType) -> str:
    if orientation == "portrait":
        return config.defaults.portrait_size
    elif orientation == "landscape":
        return config.defaults.landscape_size
    elif orientation == "square":
        return config.defaults.square_size
    return config.defaults.size


@dataclass
class ConfigNeededTool(FunctionTool[AstrAgentContext]):
    config_init: Config | None = None

    def __post_init__(self):
        if not self.config_init:
            raise ValueError("config not provided")
        self.config = self.config_init


class ReturnToLLMError(Exception):
    pass

async def llm_generate_prepare_req(
    message: str,
    config: Config,
    i2i_image: str | None = None,
    vibe_transfer_images: list[str] | None = None,
    skip_default_prompts: bool = False,
) -> Req:
    """
    准备生成图片的请求参数
    
    Args:
        message: LLM 输出的 JSON 字符串
        config: 配置
        i2i_image: 图生图的图片
        vibe_transfer_images: 氛围转移的图片列表
        skip_default_prompts: 是否跳过默认前置/后置提示词（使用预设时为 True）
    """
    if message.strip() == "SKIP":
        raise RuntimeError(
            "Inner LLM Skipped advanced param generation"
            ", there may have a internal error"
        )

    # 应用正则清洗
    message = apply_regex_replacements(message, config.llm.regex_replacements)

    try:
        args = STNaiGenerateImageAdvancedArgs.model_validate_json(message)
    except Exception as e:
        logger.debug("Advanced argument parsing failed", exc_info=e)
        raise ReturnToLLMError(
            f"Failed to parse advanced generation arguments"
            f", please ensure your response follows the specified schema"
            f"\n{format_readable_error(e)}"
        ) from e

    try:
        data = {}

        # 构建正向提示词：如果 AI 生成的提示词为空且未跳过默认，则用默认提示词；否则用 AI 生成的
        ai_tag = args.prompt.strip()
        if not ai_tag and not skip_default_prompts:
            data["tag"] = config.llm.default_prompt
        else:
            data["tag"] = ai_tag

        # 构建反向提示词：如果 AI 生成的反向提示词为空且未跳过默认，则用默认反向提示词；否则用 AI 生成的
        ai_negative = args.additional_negative_prompt.strip()
        if not ai_negative and not skip_default_prompts:
            data["negative"] = config.llm.default_negative_prompt
        else:
            data["negative"] = ai_negative

        data["size"] = get_size_from_config(config, args.orientation)

        data_addition = data.setdefault("addition", {})

        if i2i_image:
            data_addition["image_to_image_base64"] = i2i_image
            if args.i2i and args.i2i.repainting_strength is not None:
                data["i2i_force"] = args.i2i.repainting_strength

        if vibe_transfer_images:
            vt: list[dict] = [{"base64": img} for img in vibe_transfer_images]
            data_addition["vibe_transfer_list"] = vt
            if args.vibe_transfer:
                if len(args.vibe_transfer) != len(vibe_transfer_images):
                    raise ValueError(
                        "The number of vibe transfer settings"
                        " does not match the number of vibe transfer images."
                    )
                for i, vt_args in enumerate(args.vibe_transfer):
                    if vt_args.information_extraction_rate is not None:
                        vt[i]["info_extract"] = vt_args.information_extraction_rate
                    if vt_args.reference_strength is not None:
                        vt[i]["ref_strength"] = vt_args.reference_strength

        if args.multi_role_list:
            data_addition["multi_role_list"] = [
                ReqAdditionMultiRole.model_validate(x) for x in args.multi_role_list
            ]

        complete_defaults(data, config)
        req = Req.model_validate(data)

        post_check_limits(req, config, is_whitelisted=True)  # AI 调用不受白名单限制

    except Exception as e:
        logger.debug("Generation parameter preparation failed", exc_info=e)
        raise ReturnToLLMError(
            f"Failed to prepare generation parameters"
            f"\n(If this is unrelated with your input, this may be an internal error."
            f" You can output four-alphabet uppercase word `SKIP`"
            f" at next round to abort generation)"
            f"\n{format_readable_error(e)}"
        ) from e

    return req


async def llm_generate_image(
    instructions: str,
    config: Config,
    ctx: Context,
    event: AstrMessageEvent,
    i2i_image: str | None = None,
    vibe_transfer_images: list[str] | None = None,
    vision_images: list[Any] | None = None,
    skip_default_prompts: bool = False,
    token: str = "",
):
    """
    使用 LLM 生成高级参数并生成图片
    
    Args:
        instructions: 用户的描述指令
        config: 配置
        ctx: Context
        event: 消息事件
        i2i_image: 图生图的图片
        vibe_transfer_images: 氛围转移的图片列表
        skip_default_prompts: 是否跳过默认前置/后置提示词（使用预设时为 True）
        token: 使用的 Token
    """
    req = await llm_generate_advanced_req(
        instructions=instructions,
        config=config,
        ctx=ctx,
        event=event,
        i2i_image=i2i_image,
        vibe_transfer_images=vibe_transfer_images,
        vision_images=vision_images,
        skip_default_prompts=skip_default_prompts,
    )

    try:
        return await wrapped_generate(req, config, token=token)
    except Exception as e:
        logger.debug("Failed to generate image", exc_info=e)
        raise ReturnToLLMError(
            f"Failed to generate image: \n{format_readable_error(e)}"
        ) from e


async def llm_generate_advanced_req(
    instructions: str,
    config: Config,
    ctx: Context,
    event: AstrMessageEvent,
    i2i_image: str | None = None,
    vibe_transfer_images: list[str] | None = None,
    vision_images: list[Any] | None = None,
    skip_default_prompts: bool = False,
) -> Req:
    """只调用“高级参数生成模型”，返回可用于绘图的 Req。

    这个函数不做绘图请求。
    这样上层可以在绘图失败时用“同一组 Req 参数”重放重试，避免再次调用 LLM。
    """
    provider_id = config.llm.advanced_arg_generation_provider
    if not provider_id:
        logger.warning("未指定用于生成高级画图参数的模型，自动选择当前聊天使用的模型")
        provider_id = await ctx.get_current_chat_provider_id(event.unified_msg_origin)

    logger.info(
        "[nai][vision] enable_vision=%s, vision_images=%s, provider_id=%s",
        bool(config.llm.enable_vision),
        0 if not vision_images else len(vision_images),
        provider_id,
    )

    system_prompt = ADVANCED_PROMPT_PATH.read_text("u8")
    # 使用 replace 而不是 format，避免误解析提示词中的其他花括号内容
    system_prompt = (
        system_prompt
        .replace("{schema}", GENERATE_IMAGE_ADVANCED_SCHEMA_TXT)
        .replace("{default_size}", get_size_from_config(config, "default"))
        .replace("{portrait_size}", get_size_from_config(config, "portrait"))
        .replace("{landscape_size}", get_size_from_config(config, "landscape"))
        .replace("{square_size}", get_size_from_config(config, "square"))
        .replace("{has_i2i_image}", "Yes" if i2i_image else "No")
        .replace("{vibe_transfer_image_count}", str(
            len(vibe_transfer_images) if vibe_transfer_images else 0
        ))
    )

    user_content: str | list[Any]
    if config.llm.enable_vision and vision_images:
        parts: list[Any] = [TextPart(text=instructions)]
        limit = int(getattr(config.llm, "vision_image_limit", 0) or 0)
        selected = vision_images if limit <= 0 else vision_images[:limit]
        logger.info(
            "[nai][vision] attach_images provided=%s selected=%s limit=%s",
            len(vision_images),
            len(selected),
            limit,
        )
        for i, img in enumerate(selected):
            data_uri = await resolve_image(img)
            header = data_uri.split(",", 1)[0] if isinstance(data_uri, str) else "<non-str>"
            logger.info(
                "[nai][vision] attach_image id=%s header=%s uri_len=%s",
                f"img{i}",
                header,
                len(data_uri) if isinstance(data_uri, str) else "?",
            )
            if isinstance(data_uri, str) and "base64://" in data_uri:
                logger.warning(
                    "[nai][vision] image data contains unexpected 'base64://' prefix; provider may ignore it"
                )
            parts.append(
                ImageURLPart(image_url=ImageURLPart.ImageURL(url=data_uri, id=f"img{i}"))
            )
        user_content = parts
    else:
        if config.llm.enable_vision and not vision_images:
            logger.debug(
                "[nai][vision] enable_vision is on but no images provided; using text-only"
            )
        user_content = instructions

    contexts = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]

    for _ in range(3):
        try:
            try:
                llm_resp = await ctx.llm_generate(
                    chat_provider_id=provider_id,
                    contexts=contexts,
                )
            except Exception:
                # 若 provider 不支持多模态，回退到纯文本
                if isinstance(user_content, list):
                    logger.warning(
                        "[nai][vision] provider failed with multimodal input; falling back to text-only (provider_id=%s)",
                        provider_id,
                    )
                    contexts = [
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=instructions),
                    ]
                    user_content = instructions
                    llm_resp = await ctx.llm_generate(
                        chat_provider_id=provider_id,
                        contexts=contexts,
                    )
                else:
                    raise
        except Exception as e:
            logger.debug("Inner LLM call failed", exc_info=e)
            raise ReturnToLLMError(
                f"Failed to call inner LLM for advanced parameter generation: \n"
                f"{format_readable_error(e)}"
            ) from e

        raw_output = llm_resp.completion_text or ""
        preview_limit = 500
        preview = (
            raw_output
            if len(raw_output) <= preview_limit
            else raw_output[:preview_limit] + "...(truncated)"
        )
        logger.info("[nai] inner llm output (%s chars): %s", len(raw_output), preview)

        try:
            return await llm_generate_prepare_req(
                raw_output,
                config,
                i2i_image,
                vibe_transfer_images,
                skip_default_prompts=skip_default_prompts,
            )
        except ReturnToLLMError as e:
            sys_m = f"Your response seems incorrect. Correct the error and try again.\n{e}"
            contexts.append(Message(role="assistant", content=llm_resp.completion_text))
            contexts.append(Message(role="user", content=sys_m))
            continue

    raise ReturnToLLMError(
        "Inner LLM failed to provide valid output after multiple attempts."
    )
