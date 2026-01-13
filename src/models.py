from collections.abc import Iterable
from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from cookit import camel_case
from cookit.pyd import model_with_model_config
from pydantic import (
    AfterValidator,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)

AVAILABLE_MODELS = [
    "nai-diffusion-3",
    "nai-diffusion-furry-3",
    "nai-diffusion-4-full",
    "nai-diffusion-4-curated-preview",
    "nai-diffusion-4-5-curated",
    "nai-diffusion-4-5-full",
]
AVAILABLE_SAMPLERS = [
    "k_euler_ancestral",
    "k_euler",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_2m_sde",
    "k_dpmpp_2m",
    "k_dpmpp_sde",
]
AVAILABLE_NOISE_SCHEDULERS = [
    "karras",
    "native",
    "exponential",
    "polyexponential",
]
AVAILABLE_DOTH = ["0", "1", "2", "3", "4", "5"]
AVAILABLE_POSITIONS = [
    "A1", "B1", "C1", "D1", "E1",
    "A2", "B2", "C2", "D2", "E2",
    "A3", "B3", "C3", "D3", "E3",
    "A4", "B4", "C4", "D4", "E4",
    "A5", "B5", "C5", "D5", "E5",
]  # fmt: skip
AvailablePositionsType: TypeAlias = Literal[
    "A1", "B1", "C1", "D1", "E1",
    "A2", "B2", "C2", "D2", "E2",
    "A3", "B3", "C3", "D3", "E3",
    "A4", "B4", "C4", "D4", "E4",
    "A5", "B5", "C5", "D5", "E5",
]  # fmt: skip


def make_item_exists_validator(existing_items: Iterable):
    def validator(value: str) -> str:
        if value not in existing_items:
            raise ValueError(f"Value `{value}` does not exist in {existing_items}")
        return value

    return validator


def make_inner_item_exists_validator(existing_items: Iterable):
    inner_validator = make_item_exists_validator(existing_items)

    def validator(value: list[str]) -> list[str]:
        for v in value:
            inner_validator(v)
        return value

    return validator


def make_number_string_validator(
    is_int: bool = False,
    lt: float | None = None,
    le: float | None = None,
    gt: float | None = None,
    ge: float | None = None,
    eq: float | None = None,
):
    if (eq is not None) and any(v is not None for v in (lt, le, gt, ge)):
        raise ValueError("eq cannot be used with lt, le, gt, or ge")
    if (lt is not None) and (le is not None):
        raise ValueError("lt and le cannot be used together")
    if (gt is not None) and (ge is not None):
        raise ValueError("gt and ge cannot be used together")

    def validator(value: str) -> str:
        try:
            num = int(value) if is_int else float(value)
        except ValueError:
            raise ValueError(
                f"Value `{value}` is not a valid"
                f" {'integer' if is_int else 'float'} string"
            )

        if eq is not None and num != eq:
            raise ValueError(f"Value `{value}` must be equal to {eq}")
        if lt is not None and not (num < lt):
            raise ValueError(f"Value `{value}` must be less than {lt}")
        if le is not None and not (num <= le):
            raise ValueError(f"Value `{value}` must be less than or equal to {le}")
        if gt is not None and not (num > gt):
            raise ValueError(f"Value `{value}` must be greater than {gt}")
        if ge is not None and not (num >= ge):
            raise ValueError(f"Value `{value}` must be greater than or equal to {ge}")

        return value

    return validator


def size_validator(v: str):
    if (comps := v.split("x")) and len(comps) == 2 and all(c.isdigit() for c in comps):
        return v
    raise ValueError(
        "Wrong format for size parameter, should be something like 1024x768"
    )


REQ_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    alias_generator=AliasGenerator(serialization_alias=camel_case),
    coerce_numbers_to_str=True,
)


@model_with_model_config(REQ_MODEL_CONFIG)
class ReqAdditionVibeTransfer(BaseModel):  # 氛围转移
    base64: str | None = None  # 参考图片 (data uri)
    info_extract: Annotated[float, Field(ge=0, le=1)] = 1  # 信息提取度
    ref_strength: Annotated[float, Field(ge=0, le=1)] = 0.5  # 参考强度


@model_with_model_config(REQ_MODEL_CONFIG)
class ReqAdditionMultiRole(BaseModel):  # 多角色控制
    prompt: str = ""  # 正向提示词
    negative_prompt: str = ""  # 反向提示词
    position: AvailablePositionsType = "C3"  # 位置


@model_with_model_config(REQ_MODEL_CONFIG)
class ReqAdditionCharacterKeep(BaseModel):  # 角色保持
    base64: str | None = None  # 参考图片 (data uri)
    keep_vibe: bool = False  # 保持氛围
    strength: Annotated[float, Field(ge=0, le=1)] = 0.5  # 参考强度


@model_with_model_config(REQ_MODEL_CONFIG)
class ReqAddition(BaseModel):
    image_to_image_base64: str | None = None  # 图生图
    vibe_transfer_list: list[ReqAdditionVibeTransfer] = []  # 氛围转移
    multi_role_list: list[ReqAdditionMultiRole] = []  # 多角色控制
    character_keep: ReqAdditionCharacterKeep | None = None  # 角色保持


@model_with_model_config(REQ_MODEL_CONFIG)
class Req(BaseModel):
    token: str = ""  # Key，经测试填在 Novel 官方密钥或授权 Key 处没有区别
    model: Annotated[  # 模型
        str, AfterValidator(make_item_exists_validator(AVAILABLE_MODELS))
    ] = ""
    tag: str = ""  # 正向提示词
    negative: str = ""  # 反向提示词
    artist: str = ""  # 画师串
    size: Annotated[  # 画面尺寸
        str,
        AfterValidator(size_validator),
    ] = "832x1216"
    seed: Annotated[  # 种子（留空随机）
        str,
        AfterValidator(make_number_string_validator(is_int=True)),
    ] = ""
    steps: Annotated[  # 采样步数
        str, AfterValidator(make_number_string_validator(is_int=True, ge=1, le=50))
    ] = "23"
    scale: Annotated[  # 提示词引导值
        str, AfterValidator(make_number_string_validator(is_int=False))
    ] = "5"
    cfg: Annotated[  # 缩放引导值
        str,
        AfterValidator(make_number_string_validator(is_int=False)),
    ] = "0"
    sampler: Annotated[  # 采样器
        str, AfterValidator(make_item_exists_validator(AVAILABLE_SAMPLERS))
    ] = AVAILABLE_SAMPLERS[0]
    noise_schedule: Annotated[  # 噪声调度
        str,
        Field(serialization_alias="noise_schedule"),
        AfterValidator(make_item_exists_validator(AVAILABLE_NOISE_SCHEDULERS)),
    ] = AVAILABLE_NOISE_SCHEDULERS[0]
    other: Annotated[  # 高级配置
        str, AfterValidator(make_item_exists_validator(AVAILABLE_DOTH))
    ] = AVAILABLE_DOTH[0]
    i2i_force: Annotated[str, Field(serialization_alias="i2iforce")] = "0.6"  # 重绘力度
    i2i_cl: Annotated[str, Field(serialization_alias="i2icl")] = "1"  # 图片处理
    addition: ReqAddition = ReqAddition()
    ch: bool = False
    nocache: int = 1
    stream: int = 1


class Resp(BaseModel):
    status: str
    data: str | None = None
    url: str | None = None


class QueryKeyErrorResp(BaseModel):
    status: str
    type: str
    data: str


class QueryKeySuccessRespData(BaseModel):
    desc: str
    value: int
    total: int
    created_at: datetime
    used_at: datetime
    lines: int


class QueryKeySuccessResp(BaseModel):
    status: str
    type: str
    data: dict[str, int]


QueryKeyResp = RootModel[QueryKeyErrorResp | QueryKeySuccessResp]


class ForceCleanQueueResp(BaseModel):
    status: str
