from collections.abc import Callable, Generator, Iterable, Mapping, MutableSequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    get_args,
)
from typing_extensions import TypedDict, Unpack

try:
    # Prefer public API exports when available
    from astrbot.api.message_components import BaseMessageComponent, Image, Reply
except Exception:  # pragma: no cover
    from astrbot.core.message.components import BaseMessageComponent, Image, Reply

try:
    # Newer AstrBot versions expose message components from astrbot.api
    from astrbot.api.message_components import Image as ApiImage
except Exception:  # pragma: no cover
    ApiImage = None  # type: ignore[assignment]


def _is_image_component(comp: BaseMessageComponent) -> bool:
    if isinstance(comp, Image):
        return True
    if ApiImage is not None and isinstance(comp, ApiImage):
        return True
    # Fallback for component type drift across AstrBot versions
    return (
        hasattr(comp, "convert_to_base64")
        and "image" in comp.__class__.__name__.lower()
    )


def _collect_images_with_replies(message: list) -> list[Image]:
    """Collect images from the current message and one-level Reply chains."""
    images: list[Image] = []
    for component in message:
        if _is_image_component(component):
            images.append(component)
            continue
        if isinstance(component, Reply):
            for reply_component in getattr(component, "chain", None) or []:
                if _is_image_component(reply_component):
                    images.append(reply_component)
    return images

from .models import (
    AVAILABLE_DOTH,
    AVAILABLE_MODELS,
    AVAILABLE_NOISE_SCHEDULERS,
    AVAILABLE_POSITIONS,
    AVAILABLE_SAMPLERS,
    CharacterKeepMode,
    Req,
    V5_CHARACTER_KEEP_MAX_STRENGTH,
)
from .image_io import resolve_image
from .image_params import (
    CHARACTER_KEEP_MODE_ALIASES,
    CHARACTER_KEEP_STRENGTH_ALIASES,
    FALSE_VALUES,
    character_keep_param,
    with_implicit_character_keep,
)

if TYPE_CHECKING:
    from .config import Config

TP = TypeVar("TP")  # TParam
TP_contra = TypeVar("TP_contra", contravariant=True)
TC = TypeVar("TC")  # TCtx
TC_contra = TypeVar("TC_contra", contravariant=True)
T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


class StartProcessHookArgs(TypedDict, Generic[T, TC]):
    assembler: "ParamAssembler[T, TC]"
    appliers: dict[str, "ApplierInfo[T, TC]"]
    input_params: list[tuple[str, str]]
    data: dict[str, Any]
    images: list[Image]
    ctx: TC


class StartProcessHook(Protocol[T, TC]):
    async def __call__(self, **kwargs: Unpack[StartProcessHookArgs[T, TC]]) -> Any: ...


class ProcessHookArgs(TypedDict, Generic[T, TC]):
    assembler: "ParamAssembler[T, TC]"
    input_params: list[tuple[str, str]]
    data: dict[str, Any]
    images: list[Image]
    ctx: TC


class ProcessHook(Protocol[T, TC]):
    async def __call__(self, **kwargs: Unpack[ProcessHookArgs[T, TC]]) -> Any: ...


class ParamApplierArgs(TypedDict, Generic[T, TC]):
    assembler: "ParamAssembler[T, TC]"
    value: str
    data: dict[str, Any]
    images: list[Image]
    ctx: TC
    reference_index: int


class ParamApplier(Protocol[T, TC]):
    async def __call__(self, **kwargs: Unpack[ParamApplierArgs[T, TC]]) -> Any: ...


class ParamTransformerArgs(TypedDict, Generic[T, TC]):
    assembler: "ParamAssembler[T, TC]"
    data: dict[str, Any]


class ParamTransformer(Protocol[T, TC]):
    async def __call__(self, **kwargs: Unpack[ParamTransformerArgs[T, TC]]) -> T: ...


@dataclass(frozen=True)
class ApplierInfo(Generic[T, TC]):
    id: str
    keywords: list[str]
    applier: ParamApplier[T, TC]
    doc_gen: Callable[[TC], str] | None = None


class ParamAssembler(Generic[T, TC]):
    def __init__(self):
        self.appliers: dict[str, ApplierInfo[T, TC]] = {}
        self.start_process_hooks: list[StartProcessHook[T, TC]] = []
        self.end_process_hooks: list[ProcessHook[T, TC]] = []
        self.transformer_func: ParamTransformer[T, TC] | None = None

    def to_appliers_map(
        self, appliers: dict[str, ApplierInfo[T, TC]]
    ) -> dict[str, list[ApplierInfo[T, TC]]]:
        appliers_map: dict[str, list[ApplierInfo[T, TC]]] = {}
        for x in appliers.values():
            for kw in (x.id, *x.keywords):
                if kw not in appliers_map:
                    appliers_map[kw] = []
                appliers_map[kw].append(x)
        return appliers_map

    @property
    def appliers_map(self) -> dict[str, list[ApplierInfo[T, TC]]]:
        return self.to_appliers_map(self.appliers)

    def add_applier(self, applier: ApplierInfo) -> None:
        self.appliers[applier.id] = applier

    def applier(
        self,
        id: str,
        keywords: list[str],
        doc_gen: Callable[[TC], str] | None = None,
    ):
        def decorator(func: ParamApplier) -> ParamApplier:
            self.add_applier(ApplierInfo(id, keywords, func, doc_gen))
            return func

        return decorator

    def start_process_hook(
        self, func: StartProcessHook[T, TC]
    ) -> StartProcessHook[T, TC]:
        self.start_process_hooks.append(func)
        return func

    def end_process_hook(self, func: ProcessHook[T, TC]) -> ProcessHook[T, TC]:
        self.end_process_hooks.append(func)
        return func

    def transformer(self, func: ParamTransformer[T, TC]) -> ParamTransformer[T, TC]:
        self.transformer_func = func
        return func

    def copy(self) -> "ParamAssembler[T, TC]":
        assembler = ParamAssembler()
        assembler.appliers = self.appliers.copy()
        assembler.start_process_hooks = self.start_process_hooks.copy()
        assembler.end_process_hooks = self.end_process_hooks.copy()
        return assembler

    async def apply(
        self, input_params: list[tuple[str, str]], images: list[Any], ctx: TC, is_whitelisted: bool = False
    ) -> T:
        if not self.transformer_func:
            raise RuntimeError("No transformer function defined")

        data: dict[str, Any] = {"_is_whitelisted": is_whitelisted}
        appliers = self.appliers

        for hook in self.start_process_hooks:
            await hook(
                assembler=self,
                appliers=appliers,
                input_params=input_params,
                data=data,
                images=images,
                ctx=ctx,
            )

        appliers_map = self.to_appliers_map(appliers)

        resolved_params = []
        for key, value in input_params:
            param, reference_index = character_keep_param(key) or (key, 1)
            if param not in appliers_map:
                raise KeyError(f"Unknown parameter key: {key}")
            resolved_params.append((appliers_map[param], value, reference_index))
        for selected_appliers, value, reference_index in resolved_params:
            for applier in selected_appliers:
                await applier.applier(
                    assembler=self, value=value, data=data, images=images, ctx=ctx,
                    reference_index=reference_index,
                )

        for hook in self.end_process_hooks:
            await hook(
                assembler=self,
                input_params=input_params,
                data=data,
                images=images,
                ctx=ctx,
            )

        model = await self.transformer_func(assembler=self, data=data)
        return model


def parse_params(raw_params: str) -> Generator[tuple[str, str], None, None]:
    for line in raw_params.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            yield key.strip(), value.strip()
        else:
            # 强制键值对格式
            raise ValueError(f"参数格式错误：'{line}'，请使用键值对格式，例如：tag=xxx")



def set_param(data: dict[str, Any], key: str, value: Any) -> None:
    if key in data:
        raise ValueError(f"Param `{key}` already set")
    data[key] = value


def set_param_if_not_exist(data: dict[str, Any], key: str, value: Any) -> None:
    if key not in data:
        data[key] = value


def pop_from_images(images: list[Image]) -> Image:
    if not images:
        raise ValueError("No items left in images queue")
    return images.pop(0)


async def resolve_first_image(images: list[Image]) -> str:
    return await resolve_image(pop_from_images(images))


req_model_assembler = ParamAssembler[Req, "Config"]()

PORTRAIT_KEYWORDS = ["portrait", "竖图"]
LANDSCAPE_KEYWORDS = ["landscape", "横图"]
SQUARE_KEYWORDS = ["square", "方图"]
CHARACTER_KEEP_MODE_SHORTCUTS: dict[str, CharacterKeepMode] = {
    "ctst": "character&style",
    "ct": "character",
    "st": "style",
    "oft": "outfit",
}
IMG_FIELDS = ["i2i", "vibe_transfer", "character_keep"]
RELATED_FIELDS_MAP = {
    "i2i": ["i2i_force", "i2i_cl"],
    "vibe_transfer": ["vibe_transfer_info_extract", "vibe_transfer_ref_strength"],
    "character_keep": ["character_keep_vibe", "character_keep_mode", "character_keep_strength"],
}
AVAILABLE_MODEL_NAME_MAP = {
    "nai-diffusion-3": "NAI3 标准模型",
    "nai-diffusion-furry-3": "NAI3 Furry模型",
    "nai-diffusion-4-full": "NAI4 完整版",
    "nai-diffusion-4-curated-preview": "NAI4 精选预览版",
    "nai-diffusion-4-5-curated": "NAI4.5 精选版",
    "nai-diffusion-4-5-full": "NAI4.5 完整版",
    "nai-diffusion-5-full": "NAI5 完整版",
    "nai-diffusion-5-curated": "NAI5 精选版",
}
# 模型简写到完整名称的映射
MODEL_ALIAS_MAP = {
    "nai3": "nai-diffusion-3",
    "nai3_furry": "nai-diffusion-furry-3",
    "nai4_full": "nai-diffusion-4-full",
    "nai4_c_p": "nai-diffusion-4-curated-preview",
    "nai4.5_c": "nai-diffusion-4-5-curated",
    "nai4.5_full": "nai-diffusion-4-5-full",
    "nai5_full": "nai-diffusion-5-full",
    "nai5_c": "nai-diffusion-5-curated",
}
AVAILABLE_DOTH_NAME_MAP = {
    "0": "不使用",
    "1": "Auto",
    "2": "SMEA",
    "3": "SMEA+DYN",
    "4": "Auto+SMEA",
    "5": "Auto+SMEA+DYN",
}


def format_separate_list(
    items: Iterable[str],
    name_map: Mapping[str, Iterable[str]] | None = None,
    wrap_ticks: bool = True,
) -> str:
    res = []
    for item in items:
        text = f"`{item}`" if wrap_ticks else item
        if name_map and item in name_map:
            desc = name_map[item]
            if isinstance(desc, str):
                text += f"（{desc}）"
            else:
                text += f"（{'；'.join(desc)}）"
        res.append(text)
    return "、".join(res)


def format_list(
    items: Iterable[str],
    name_map: Mapping[str, Iterable[str]] | None = None,
    wrap_ticks: bool = True,
) -> str:
    res = []
    for item in items:
        text = f"`{item}`" if wrap_ticks else item
        if name_map and item in name_map:
            desc = name_map[item]
            if isinstance(desc, str):
                text += f"（{desc}）"
            else:
                text += f"（{'；'.join(desc)}）"
        res.append(f"- {text}")
    return "\n".join(res)


# 核心提示词参数，始终允许使用，不受权限配置限制
CORE_PROMPT_FIELDS = {
    "tag", "negative", 
    "prepend_tag", "append_tag", 
    "prepend_negative", "append_negative"
}


@req_model_assembler.start_process_hook
async def start_process(
    appliers: dict[str, ApplierInfo[Req, "Config"]],
    input_params: list[tuple[str, str]],
    data: dict[str, Any],
    ctx: "Config",
    **_,
):
    # 从 data 中获取 is_whitelisted 标志
    is_whitelisted = data.get("_is_whitelisted", False)
    
    # 如果用户不是白名单，检查是否使用了白名单专用字段
    if not is_whitelisted:
        # 获取用户实际使用的参数键（去重）
        used_keys = {(character_keep_param(key) or (key, 1))[0] for key, _ in input_params}
        
        # 检查是否使用了白名单专用字段
        for key in used_keys:
            # 核心提示词参数始终允许
            if key in CORE_PROMPT_FIELDS:
                continue
            # 检查该参数是否在白名单专用字段列表中
            if key in ctx.permission.whitelist_only_fields:
                raise ValueError(f"参数 `{key}` 仅限白名单用户使用")


@req_model_assembler.applier(
    "model",
    ["模型"],
    lambda ctx: (
        f"默认 `{ctx.defaults.model}`。可用模型\n"
        f"{format_list(AVAILABLE_MODELS, AVAILABLE_MODEL_NAME_MAP)}\n"
        f"模型简写及其对应值 {format_separate_list(MODEL_ALIAS_MAP, MODEL_ALIAS_MAP)}。"
    ),
)
async def apply_model(value: str, data: dict[str, Any], **_):
    # 处理模型简写别名
    resolved_value = MODEL_ALIAS_MAP.get(value, value)
    set_param(data, "model", resolved_value)


@req_model_assembler.applier(
    "negative",
    ["反向提示词", "ne"],
    lambda _: "反向提示词，描述不想出现的内容",
)
async def apply_negative(value: str, data: dict[str, Any], **_):
    set_param(data, "negative", value)


@req_model_assembler.applier(
    "tag",
    ["正向提示词"],
    lambda _: "正向提示词，描述期望生成的图片内容",
)
async def apply_tag(value: str, data: dict[str, Any], **_):
    set_param(data, "tag", value)


@req_model_assembler.applier(
    "prepend_tag",
    ["前置正向", "前置正向提示词", "a_tag"],
    lambda _: "前置正向提示词，将被添加到所有正向提示词的最前方",
)
async def apply_prepend_tag(value: str, data: dict[str, Any], **_):
    set_param(data, "prepend_tag", value)


@req_model_assembler.applier(
    "append_tag",
    ["后置正向", "后置正向提示词", "b_tag"],
    lambda _: "后置正向提示词，将被添加到所有正向提示词的最后方",
)
async def apply_append_tag(value: str, data: dict[str, Any], **_):
    set_param(data, "append_tag", value)


@req_model_assembler.applier(
    "prepend_negative",
    ["前置负面", "前置负面提示词", "a_ne"],
    lambda _: "前置负面提示词，将被添加到所有负面提示词的最前方",
)
async def apply_prepend_negative(value: str, data: dict[str, Any], **_):
    set_param(data, "prepend_negative", value)


@req_model_assembler.applier(
    "append_negative",
    ["后置负面", "后置负面提示词", "b_ne"],
    lambda _: "后置负面提示词，将被添加到所有负面提示词的最后方",
)
async def apply_append_negative(value: str, data: dict[str, Any], **_):
    set_param(data, "append_negative", value)


@req_model_assembler.applier(
    "artist",
    ["画师", "画师串"],
    lambda _: "指定画师风格串，用于生成特定画师风格的图片",
)
async def apply_artist(value: str, data: dict[str, Any], **_):
    set_param(data, "artist", value)




@req_model_assembler.applier(
    "size",
    ["画面尺寸"],
    lambda ctx: (
        f"默认尺寸为 `{ctx.defaults.size}`。"
        f"\n"
        f"\n可以在参数值填入以下关键词来使用预置尺寸："
        f"\n"
        f"\n- {format_separate_list(PORTRAIT_KEYWORDS)}（`{ctx.defaults.portrait_size}`）"
        f"\n- {format_separate_list(LANDSCAPE_KEYWORDS)}（`{ctx.defaults.landscape_size}`）"
        f"\n- {format_separate_list(SQUARE_KEYWORDS)}（`{ctx.defaults.square_size}`）"
        f"\n"
        f"\n白名单用户可填写自定义尺寸，例如 `1024x768`。"
        f"\n宽高上限为 `{ctx.permission.width_limit}x{ctx.permission.height_limit}`。"
    ),
)
async def apply_size(value: str, data: dict[str, Any], ctx: "Config", **_):
    if value in PORTRAIT_KEYWORDS:
        value = ctx.defaults.portrait_size
    elif value in LANDSCAPE_KEYWORDS:
        value = ctx.defaults.landscape_size
    elif value in SQUARE_KEYWORDS:
        value = ctx.defaults.square_size
    set_param(data, "size", value)


@req_model_assembler.applier("seed", ["种子"], lambda _: "留空使用随机种子")
async def apply_seed(value: str, data: dict[str, Any], **_):
    set_param(data, "seed", value)


@req_model_assembler.applier(
    "steps",
    ["采样步数"],
    lambda ctx: (
        f"整数，默认 `{ctx.defaults.steps}`，范围为 1–{ctx.permission.steps_limit}。"
        "非白名单用户最多使用 28 步。"
    ),
)
async def apply_steps(value: str, data: dict[str, Any], **_):
    set_param(data, "steps", value)


@req_model_assembler.applier(
    "scale",
    ["提示词引导值"],
    lambda ctx: f"小数，默认为 `{ctx.defaults.scale}`",
)
async def apply_scale(value: str, data: dict[str, Any], **_):
    set_param(data, "scale", value)


@req_model_assembler.applier(
    "cfg",
    ["缩放引导值"],
    lambda ctx: f"小数，默认为 `{ctx.defaults.cfg}`",
)
async def apply_cfg(value: str, data: dict[str, Any], **_):
    set_param(data, "cfg", value)


@req_model_assembler.applier(
    "sampler",
    ["采样器"],
    lambda ctx: (
        f"默认为 `{ctx.defaults.sampler}`"
        f"\n"
        f"\n可选：{format_separate_list(AVAILABLE_SAMPLERS)}"
    ),
)
async def apply_sampler(value: str, data: dict[str, Any], **_):
    set_param(data, "sampler", value)


@req_model_assembler.applier(
    "noise_schedule",
    ["噪声调度", "n_s"],
    lambda ctx: (
        f"默认为 `{ctx.defaults.noise_schedule}`"
        f"\n"
        f"\n可选：{format_separate_list(AVAILABLE_NOISE_SCHEDULERS)}"
    ),
)
async def apply_noise_schedule(value: str, data: dict[str, Any], **_):
    set_param(data, "noise_schedule", value)


@req_model_assembler.applier(
    "other",
    ["高级配置"],
    lambda ctx: (
        f"默认为 `{ctx.defaults.other}`"
        f"\n"
        f"\n可选：{format_separate_list(AVAILABLE_DOTH, AVAILABLE_DOTH_NAME_MAP)}"
    ),
)
async def apply_other(value: str, data: dict[str, Any], **_):
    set_param(data, "other", value)


@req_model_assembler.applier(
    "i2i",
    ["图生图"],
    lambda _: "引用一张图重绘。`i2i=true` 使用附图或回复图，`i2i=图库名称` 使用指定图库图。",
)
async def apply_i2i(value: str, data: dict[str, Any], images: list[Image], **_):
    if value.lower() in FALSE_VALUES:
        return
    if "addition" not in data:
        data["addition"] = {}
    if "image_to_image_base64" in data["addition"]:
        raise ValueError("Param `i2i` already set")
    data["addition"]["image_to_image_base64"] = await resolve_first_image(images)


@req_model_assembler.applier(
    "data",
    ["输出参数记录"],
    lambda _: "`data=true` 单次输出绘图参数记录和参考图，`data=false` 按插件全局设置输出。",
)
async def apply_data(value: str, data: dict[str, Any], **_):
    value_lower = value.lower().strip()
    if value_lower in ("true", "1", "on", "yes", "是"):
        data["data"] = True
    elif value_lower in ("false", "0", "off", "no", "否", "关"):
        data["data"] = False
    else:
        raise ValueError(f"无效的值 `{value}`，请使用 true 或 false")


@req_model_assembler.applier(
    "i2i_force",
    ["重绘力度", "i_f"],
    lambda ctx: f"搭配 `i2i`，范围 0–1，默认 `{ctx.defaults.i2i_force}`。值越大，改动越多。",
)
async def apply_i2i_force(value: str, data: dict[str, Any], **_):
    set_param(data, "i2i_force", value)


@req_model_assembler.applier(
    "i2i_cl",
    ["图片处理"],
    lambda ctx: f"旧版图片处理参数，默认 `{ctx.defaults.i2i_cl}`。当前会显示在参数记录中。",
)
async def apply_i2i_cl(value: str, data: dict[str, Any], **_):
    set_param(data, "i2i_cl", value)


@req_model_assembler.applier(
    "vibe_transfer",
    ["氛围转移", "v_t"],
    lambda _: (
        "参考图片氛围。`v_t=true` 使用附图或回复图，`v_t=图库名称` 使用指定图库图。"
        "多张参考可重复填写 `v_t`，各自的提取度和强度紧跟在对应图片之后。"
        "与 `c_k` 精准参考分开使用。"
    ),
)
async def apply_vibe_transfer(
    value: str, data: dict[str, Any], images: list[Image], ctx: "Config", **_
):
    if value.lower() in FALSE_VALUES:
        return
    if "addition" not in data:
        data["addition"] = {}
    if "vibe_transfer_list" not in data["addition"]:
        data["addition"]["vibe_transfer_list"] = []
    li = data["addition"]["vibe_transfer_list"]
    if len(li) >= ctx.permission.vibe_transfer_image_limit:
        raise ValueError("Exceeded vibe transfer image limit")
    li.append({"base64": await resolve_first_image(images)})


@req_model_assembler.applier(
    "vibe_transfer_info_extract",
    ["氛围转移信息提取度", "v_t_i_e"],
    lambda ctx: (
        f"写在 `v_t` 之后，信息提取度范围 0–1，默认 `{ctx.defaults.vibe_transfer_info_extract}`。"
    ),
)
async def apply_vibe_transfer_info_extract(value: str, data: dict[str, Any], **_):
    if ("addition" not in data) or (
        not (li := data["addition"].get("vibe_transfer_list"))
    ):
        raise ValueError("No images referenced for vibe transfer")
    if (li[-1].get("info_extract")) is not None:
        raise ValueError(
            "Param `vibe_transfer_info_extract` already set for last image"
        )
    li[-1]["info_extract"] = value


@req_model_assembler.applier(
    "vibe_transfer_ref_strength",
    ["氛围转移参考强度", "v_t_r_s"],
    lambda ctx: (
        f"写在 `v_t` 之后，参考强度范围 0–1，默认 `{ctx.defaults.vibe_transfer_ref_strength}`。"
    ),
)
async def apply_vibe_transfer_ref_strength(value: str, data: dict[str, Any], **_):
    if ("addition" not in data) or (
        not (li := data["addition"].get("vibe_transfer_list"))
    ):
        raise ValueError("No images referenced for vibe transfer")
    if (li[-1].get("ref_strength")) is not None:
        raise ValueError(
            "Param `vibe_transfer_ref_strength` already set for last image"
        )
    li[-1]["ref_strength"] = value


def format_position_grid() -> str:
    """生成位置网格说明图"""
    return """\
位置网格（5x5）：
```
     A    B    C    D    E
  ┌────┬────┬────┬────┬────┐
1 │ A1 │ B1 │ C1 │ D1 │ E1 │
  ├────┼────┼────┼────┼────┤
2 │ A2 │ B2 │ C2 │ D2 │ E2 │
  ├────┼────┼────┼────┼────┤
3 │ A3 │ B3 │ C3 │ D3 │ E3 │
  ├────┼────┼────┼────┼────┤
4 │ A4 │ B4 │ C4 │ D4 │ E4 │
  ├────┼────┼────┼────┼────┤
5 │ A5 │ B5 │ C5 │ D5 │ E5 │
  └────┴────┴────┴────┴────┘
```
A-E 为横向（左→右），1-5 为纵向（上→下）
C3 为正中间"""


@req_model_assembler.applier(
    "role",
    ["角色", "多角色"],
    lambda _: (
        "格式 `role=位置|正向提示词|反向提示词`，最后一项可省略。"
        "当前官方请求将角色的位置和正向提示词拼入总提示词。\n"
        "示例 `role=C3|1girl, smile`。\n" + format_position_grid()
    ),
)
async def apply_role(value: str, data: dict[str, Any], ctx: "Config", **_):
    # 解析格式：位置|正向提示词|反向提示词(可选)
    parts = value.split("|")
    if len(parts) < 2:
        raise ValueError(
            "role 参数格式错误，正确格式：role=位置|正向提示词|反向提示词(可选)\n"
            "例如：role=C3|1girl, cute"
        )
    
    position = parts[0].strip().upper()
    prompt = parts[1].strip()
    negative_prompt = parts[2].strip() if len(parts) > 2 else ""
    
    # 验证位置
    if position not in AVAILABLE_POSITIONS:
        raise ValueError(
            f"位置 `{position}` 无效，可用位置：A1-E5\n"
            "A-E 为横向（左→右），1-5 为纵向（上→下）"
        )
    
    # 验证提示词
    if not prompt:
        raise ValueError("角色的正向提示词不能为空")
    
    # 初始化 addition 和 multi_role_list
    if "addition" not in data:
        data["addition"] = {}
    if "multi_role_list" not in data["addition"]:
        data["addition"]["multi_role_list"] = []
    
    # 添加角色
    data["addition"]["multi_role_list"].append({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "position": position,
    })


@req_model_assembler.applier(
    "character_keep",
    ["角色保持", "ck", "c_k"],
    lambda _: (
        "NAI4.5 / NAI5 精准参考功能，支持按编号使用多张参考图\n"
        "\n附图时可写 `c_k=true` 或直接指定 `c_k_m`，图库图片用 `c_k=图库名称`\n"
        "\n`c_k` 等同 `c_k1`。用 `c_k2` 等指定其他图片。"
        "附图按已启用的参考编号从小到大分配，图库图不占用附图。\n"
        "\n相关参数：\n"
        "- `c_k_m` 可填 `ct`、`ctst`、`st`、`oft`，默认 `ctst`（角色和画风）\n"
        "- `c_k_s` 参考强度，省略时读取插件默认值设置"
    ),
)
async def apply_character_keep(
    value: str, data: dict[str, Any], images: list[Image], reference_index: int, **_
):
    if value.lower() in FALSE_VALUES:
        return
    reference = _character_keep_slot(data, reference_index)
    if reference.get("base64"):
        raise ValueError(f"第 {reference_index} 张精准参考图已设置")
    if not images:
        raise ValueError(f"第 {reference_index} 张精准参考需要上传图片或指定图库图")
    reference["base64"] = await resolve_first_image(images)


def _character_keep_slot(data: dict[str, Any], index: int) -> dict[str, Any]:
    addition = data.setdefault("addition", {})
    if index == 1:
        return addition.setdefault("character_keep", {})
    return addition.setdefault("character_keep_extra", {}).setdefault(index, {})


@req_model_assembler.applier(
    "character_keep_vibe",
    ["角色保持氛围", "ck_vibe", "c_k_v"],
    lambda _: (
        "> 角色保持的附加参数\n"
        "\n兼容旧指令。角色和画风请使用 `c_k_m=ctst`。"
    ),
)
async def apply_character_keep_vibe(value: str, data: dict[str, Any], **_):
    # 解析布尔值
    value_lower = value.lower().strip()
    if value_lower in ("true", "1", "yes", "是"):
        keep_vibe = True
    elif value_lower in ("false", "0", "no", "否", ""):
        keep_vibe = False
    else:
        raise ValueError(
            f"无效的值 `{value}`，请使用 true 或 false"
        )
    
    reference = data.setdefault("addition", {}).setdefault("character_keep", {})
    reference["keep_vibe"] = keep_vibe


@req_model_assembler.applier(
    "character_keep_mode",
    CHARACTER_KEEP_MODE_ALIASES,
    lambda _: (
        "精准参考模式。省略 `c_k` 时，使用本条消息或被回复消息里的图片。\n"
        "\n`c_k_m` 等同 `c_k_m1`。用 `c_k_m2` 等设置下一张图。\n"
        "\n`ct` / `character` 参考角色，`ctst` / `character&style` 参考角色和画风，"
        "`st` / `style` 参考画风，`oft` / `outfit` 参考服装（NAI5）\n"
        "\n省略时默认 `character&style`"
    ),
)
async def apply_character_keep_mode(value: str, data: dict[str, Any], reference_index: int, **_):
    normalized = value.lower().replace(" ", "")
    mode = CHARACTER_KEEP_MODE_SHORTCUTS.get(normalized, normalized)
    modes = get_args(CharacterKeepMode)
    if mode not in modes:
        options = ", ".join(
            f"{short} / {full}" for short, full in CHARACTER_KEEP_MODE_SHORTCUTS.items()
        )
        raise ValueError(f"无效的参考模式 `{value}`，可选值为 {options}")

    reference = _character_keep_slot(data, reference_index)
    reference["mode"] = mode


@req_model_assembler.applier(
    "character_keep_strength",
    CHARACTER_KEEP_STRENGTH_ALIASES,
    lambda ctx: (
        "参考强度，NAI4.5 为 0–1，NAI5 为 0–5。"
        "`c_k_s` 等同 `c_k_s1`，用 `c_k_s2` 等设置下一张图。"
        f"省略时使用插件设置 `{ctx.defaults.character_keep_strength}`。"
        "值越大，生成结果越接近参考图。"
    ),
)
async def apply_character_keep_strength(value: str, data: dict[str, Any], reference_index: int, **_):
    try:
        strength = float(value)
    except ValueError as exc:
        raise ValueError(f"无效的角色参考强度 `{value}`，请输入数字") from exc
    
    if not 0 <= strength <= V5_CHARACTER_KEEP_MAX_STRENGTH:
        raise ValueError("角色参考强度须在 0–5 之间，NAI4.5 的上限为 1")
    
    reference = _character_keep_slot(data, reference_index)
    reference["strength"] = strength

def complete_defaults(data: dict[str, Any], ctx: "Config"):
    # token 由调用方在 wrapped_generate 时动态设置，不在这里设置
    set_param_if_not_exist(data, "model", ctx.defaults.model)
    set_param_if_not_exist(data, "size", ctx.defaults.size)
    set_param_if_not_exist(data, "negative", ctx.defaults.negative_prompt)
    set_param_if_not_exist(data, "steps", ctx.defaults.steps)
    set_param_if_not_exist(data, "scale", ctx.defaults.scale)
    set_param_if_not_exist(data, "cfg", ctx.defaults.cfg)
    set_param_if_not_exist(data, "sampler", ctx.defaults.sampler)
    set_param_if_not_exist(data, "noise_schedule", ctx.defaults.noise_schedule)
    set_param_if_not_exist(data, "other", ctx.defaults.other)
    set_param_if_not_exist(data, "i2i_force", ctx.defaults.i2i_force)
    set_param_if_not_exist(data, "i2i_cl", ctx.defaults.i2i_cl)

    if "addition" in data and "vibe_transfer_list" in data["addition"]:
        for vibe_transfer in data["addition"]["vibe_transfer_list"]:
            if "info_extract" not in vibe_transfer:
                vibe_transfer["info_extract"] = ctx.defaults.vibe_transfer_info_extract
            if "ref_strength" not in vibe_transfer:
                vibe_transfer["ref_strength"] = ctx.defaults.vibe_transfer_ref_strength

    for _, reference in _character_keep_slots(data):
        if reference.get("base64"):
            reference.setdefault("strength", ctx.defaults.character_keep_strength)


def _character_keep_slots(data: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    addition = data.get("addition", {})
    return [(1, addition.get("character_keep", {})), *addition.get("character_keep_extra", {}).items()]


@req_model_assembler.end_process_hook
async def end_process(data: dict[str, Any], ctx: "Config", **_):
    for index, reference in _character_keep_slots(data):
        if reference and not reference.get("base64"):
            raise ValueError(f"请为第 {index} 张精准参考上传图片或指定图库图")

    # 获取前置/后置提示词（这些是临时存储在 data 中的）
    prepend_tag = data.pop("prepend_tag", "").strip()
    append_tag = data.pop("append_tag", "").strip()
    prepend_negative = data.pop("prepend_negative", "").strip()
    append_negative = data.pop("append_negative", "").strip()
    
    # 拼接正向提示词：prepend_tag + tag + append_tag
    tag_parts = []
    if prepend_tag:
        tag_parts.append(prepend_tag)
    if data.get("tag", "").strip():
        tag_parts.append(data["tag"].strip())
    if append_tag:
        tag_parts.append(append_tag)
    
    # 如果没有任何正向提示词，则使用默认值
    if tag_parts:
        data["tag"] = ", ".join(tag_parts)
    else:
        data["tag"] = ctx.defaults.prompt
    
    # 拼接负面提示词：prepend_negative + negative + append_negative
    negative_parts = []
    if prepend_negative:
        negative_parts.append(prepend_negative)
    if data.get("negative", "").strip():
        negative_parts.append(data["negative"].strip())
    if append_negative:
        negative_parts.append(append_negative)
    
    # 如果没有任何负面提示词，则使用默认值
    if negative_parts:
        data["negative"] = ", ".join(negative_parts)
    else:
        data["negative"] = ctx.defaults.negative_prompt
    
    # 填充其他默认值
    complete_defaults(data, ctx)


@req_model_assembler.transformer
async def transform_req(data: dict[str, Any], **_) -> Req:
    # 移除内部使用的标志，避免 Pydantic 验证错误
    req_data = {k: v for k, v in data.items() if not k.startswith("_")}
    return Req.model_validate(req_data)


def post_check_limits(model: Req, ctx: "Config", is_whitelisted: bool = False):
    """
    检查请求参数是否超出限制
    
    Args:
        model: 请求模型
        ctx: 配置
        is_whitelisted: 用户是否在白名单中（白名单用户不受步数28和自定义尺寸限制）
    """
    w, h = [int(x) for x in model.size.split("x")]
    
    # 检查是否是预设尺寸（竖图、横图、方图）
    preset_sizes = [
        ctx.defaults.portrait_size,
        ctx.defaults.landscape_size,
        ctx.defaults.square_size,
    ]
    
    # 非白名单用户不能使用自定义尺寸
    if not is_whitelisted and model.size not in preset_sizes:
        raise ValueError("暂无权限使用自定义尺寸，请使用预设的竖图、横图或方图")
    
    # 宽高上限检查（所有人都要遵守）
    if w > ctx.permission.width_limit or h > ctx.permission.height_limit:
        raise ValueError(
            f"尺寸超出限制 {ctx.permission.width_limit}x{ctx.permission.height_limit}"
        )

    steps = int(model.steps)
    
    # 非白名单用户不能使用超过28步
    if not is_whitelisted and steps > 28:
        raise ValueError("暂无权限使用超过28步的采样步数")
    
    # 步数上限检查（所有人都要遵守）
    if steps > ctx.permission.steps_limit:
        raise ValueError(
            f"步数超出限制 {ctx.permission.steps_limit}"
        )

    model.validate_character_keep()


USER_DEFINABLE_FIELDS = list(req_model_assembler.appliers.keys())


def format_nai_parameter_reference(config: "Config") -> str:
    """Render the registered /nai parameters and their help text for the draw tool."""
    entries: list[str] = []
    for applier in req_model_assembler.appliers.values():
        names = format_separate_list((applier.id, *applier.keywords))
        entries.append(f"- {names}")
        if applier.doc_gen:
            entries.extend(f"  {line}" for line in applier.doc_gen(config).splitlines())
    return "\n".join(entries)


async def parse_req(
    raw_params: str,
    message: list[BaseMessageComponent],
    config: "Config",
    is_whitelisted: bool = False,
) -> Req:
    """
    解析用户输入的参数并生成请求模型
    
    Args:
        raw_params: 原始参数字符串
        message: 消息组件列表（用于提取图片）
        config: 配置
        is_whitelisted: 用户是否在白名单中（白名单用户不受步数28和自定义尺寸限制）
    """
    input_params = with_implicit_character_keep(parse_params(raw_params))
    images = _collect_images_with_replies(message)
    req = await req_model_assembler.apply(input_params, images, config, is_whitelisted)
    
    # 进行权限检查
    post_check_limits(req, config, is_whitelisted)
    
    return req


async def parse_req_with_remaining_images(
    raw_params: str,
    message: list[BaseMessageComponent],
    config: "Config",
    is_whitelisted: bool = False,
) -> tuple[Req, list[Image]]:
    """Like parse_req, but also returns remaining (unused) images.

    The returned images are the leftover queue after i2i/vibe_transfer/character_keep
    and similar image-consuming parameters have popped from it.
    """
    input_params = with_implicit_character_keep(parse_params(raw_params))
    images: list[Image] = _collect_images_with_replies(message)
    req = await req_model_assembler.apply(input_params, images, config, is_whitelisted)
    post_check_limits(req, config, is_whitelisted)
    return req, images


def delete_if_exists(v: MutableSequence[T], items: Iterable[T]):
    for item in items:
        if item in v:
            v.remove(item)
    return v


def delete_unused_related_fields_validator(v: list[str]):
    for field, related_fields in RELATED_FIELDS_MAP.items():
        if field not in v:
            delete_if_exists(v, related_fields)
    return v
