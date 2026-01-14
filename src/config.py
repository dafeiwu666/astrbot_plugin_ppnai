from typing import Annotated, Any, cast

import jsonref
from pydantic import AfterValidator, BaseModel, Field

from .models import (
    AVAILABLE_DOTH,
    AVAILABLE_MODELS,
    AVAILABLE_NOISE_SCHEDULERS,
    AVAILABLE_SAMPLERS,
    make_inner_item_exists_validator,
    make_item_exists_validator,
    size_validator,
)
from .params import USER_DEFINABLE_FIELDS, delete_unused_related_fields_validator

DEFAULT_PREPEND_PROMPT = "best quality, very aesthetic, absurdres"
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality"
    ", jpeg artifacts, very displeasing, chromatic aberration, multiple views"
    ", logo, too many watermarks"
)


def _coerce_max_concurrent(value: int) -> int:
    value = int(value)
    if value < 1:
        return 1
    return value


class GeneralConfig(BaseModel):
    help_t2i: Annotated[
        bool,
        Field(description="将画图帮助信息生成为图片"),
    ] = True


class RequestConfig(BaseModel):
    base_url: Annotated[
        str,
        Field(description="画图接口地址"),
    ] = "https://image.novelai.net"
    tokens: Annotated[
        list[str],
        Field(
            description="授权 Token 列表",
            json_schema_extra={
                "hint": "支持配置多个 Token，系统会轮询使用。请使用 NovelAI 官方 Persistent API Token（以 pst- 开头）",
            },
        ),
    ] = []
    connect_timeout: Annotated[
        float,
        Field(
            description="请求画图接口时的连接超时时间（秒）",
            gt=0,
            json_schema_extra={"hint": "值需大于 0"},
        ),
    ] = 5
    read_timeout: Annotated[
        float,
        Field(
            description="请求画图接口时的读取超时时间（秒）",
            gt=0,
            json_schema_extra={
                "hint": "值需大于 0，相当于从连接到后端起，到等待画图完成的超时时间",
            },
        ),
    ] = 120
    max_concurrent: Annotated[
        int,
        AfterValidator(_coerce_max_concurrent),
        Field(
            description="最大并发请求数",
            ge=0,
            le=10,
            json_schema_extra={
                "hint": "同时处理的最大请求数量，超出的请求会排队等待。建议设置为 1-3；若填 0 会自动按 1 处理",
            },
        ),
    ] = 2
    max_queue_size: Annotated[
        int,
        Field(
            description="最大队列长度",
            ge=1,
            le=50,
            json_schema_extra={
                "hint": "排队等待的最大请求数量，超出时新请求会被拒绝。设置为 0 表示不限制",
            },
        ),
    ] = 10

    retry_times: Annotated[
        int,
        Field(
            description="画图失败重试次数",
            ge=0,
            le=10,
            json_schema_extra={
                "hint": "当 /nai、nai画图、自动画图 的生成请求出现错误时，内部自动重试的次数（不外显）。0 表示不重试。",
            },
        ),
    ] = 0

    retry_wait: Annotated[
        float,
        Field(
            description="重试等待时间（秒）",
            ge=0,
            le=60,
            json_schema_extra={
                "hint": "每次重试前等待的时间（秒）。仅在 retry_times>0 时生效。",
            },
        ),
    ] = 1.0
    opus_free_mode: Annotated[
        bool,
        Field(
            description="Opus 免费模式（小图模式）",
            json_schema_extra={
                "hint": "开启后强制使用符合 Opus 免费生成条件的参数：分辨率 ≤1024x1024、步数 ≤28。适用于 Opus 订阅用户，可无限免费生成",
            },
        ),
    ] = False


class LLMConfig(BaseModel):
    advanced_arg_generation_provider: Annotated[
        str,
        Field(
            description="用于生成高级画图参数的模型",
            json_schema_extra={"_special": "select_provider"},
        ),
    ] = ""
    default_prompt: Annotated[
        str,
        Field(
            description="AI 自主画图时的默认正向提示词",
            json_schema_extra={
                "type": "text",
                "hint": "AI 生成的提示词会与此默认提示词合并",
            },
        ),
    ] = DEFAULT_PREPEND_PROMPT
    default_negative_prompt: Annotated[
        str,
        Field(
            description="AI 自主画图时的默认反向提示词",
            json_schema_extra={"type": "text"},
        ),
    ] = DEFAULT_NEGATIVE_PROMPT
    allow_i2i: Annotated[bool, Field(description="允许 AI 使用图生图功能")] = True
    allow_vibe_transfer: Annotated[
        bool, Field(description="允许 AI 使用氛围转移功能")
    ] = True
    regex_replacements: Annotated[
        list[str],
        Field(
            description="正则清洗表达式",
            json_schema_extra={
                "type": "text",
                "hint": (
                    "用于清洗 AI 生成内容的正则表达式，每行一个\n"
                    "格式：正则表达式|||替换内容（如果省略替换内容则删除匹配项）\n"
                    "例如：```markdown.*?```|||  会删除所有markdown代码块"
                ),
            },
        ),
    ] = []

    enable_vision: Annotated[
        bool,
        Field(
            description="启用视觉输入：在画图参数生成时附带用户图片",
            json_schema_extra={
                "hint": "开启后，nai画图/自动画图会将用户消息中的图片以多模态 image_url 方式传给用于高级参数生成的模型（需要模型支持 vision）。",
            },
        ),
    ] = False

    vision_image_limit: Annotated[
        int,
        Field(
            description="视觉输入图片数量上限（0 表示不限制）",
            ge=0,
            json_schema_extra={
                "hint": "用于高级参数生成模型的多模态图片数量上限。设置过大可能导致请求体/Token 过大、速度变慢或模型拒绝。",
            },
        ),
    ] = 0

    vision_provider: Annotated[
        str,
        Field(
            description="用于视觉输入的模型（可选）",
            json_schema_extra={"_special": "select_provider"},
        ),
    ] = ""



class PermissionConfig(BaseModel):
    whitelist_only_fields: Annotated[
        list[str],
        Field(
            description="白名单用户专用字段",
            json_schema_extra={
                "options": list(USER_DEFINABLE_FIELDS),
                "hint": (
                    "此处填写的参数只有白名单用户可以使用，"
                    "若不在此处填写的参数，则所有用户都可以使用"
                ),
            },
        ),
        AfterValidator(make_inner_item_exists_validator(USER_DEFINABLE_FIELDS)),
        AfterValidator(delete_unused_related_fields_validator),
    ] = []
    vibe_transfer_image_limit: Annotated[
        int, Field(description="氛围转移图片数量上限", gt=0)
    ] = 2
    width_limit: Annotated[int, Field(description="图片宽度上限")] = 1920
    height_limit: Annotated[int, Field(description="图片高度上限")] = 1920
    steps_limit: Annotated[
        int,
        Field(
            description="采样步数上限",
            ge=1,
            le=50,
            json_schema_extra={"hint": "值需在 1 到 50 之间"},
        ),
    ] = 23


class QuotaConfig(BaseModel):
    enable_quota: Annotated[
        bool,
        Field(
            description="启用额度系统",
            json_schema_extra={
                "hint": "关闭后所有用户均可无限使用（黑名单仍然生效）"
            },
        ),
    ] = True
    checkin_min_quota: Annotated[
        int,
        Field(
            description="签到获得的最少次数",
            ge=0,
            json_schema_extra={"hint": "值需大于等于 0"},
        ),
    ] = 1
    checkin_max_quota: Annotated[
        int,
        Field(
            description="签到获得的最多次数",
            ge=1,
            json_schema_extra={"hint": "值需大于等于 1，且不小于最少次数"},
        ),
    ] = 3
    checkin_quota_limit: Annotated[
        int,
        Field(
            description="签到可累积的额度上限",
            ge=1,
            json_schema_extra={
                "hint": "用户额度达到此上限后签到不再获得次数（管理员设置的额度不受限制）"
            },
        ),
    ] = 10



class DefaultsConfig(BaseModel):
    model: Annotated[
        str,
        Field(
            description="默认模型",
            json_schema_extra={"options": list(AVAILABLE_MODELS)},
        ),
        AfterValidator(make_item_exists_validator(AVAILABLE_MODELS)),
    ] = AVAILABLE_MODELS[-1]
    prompt: Annotated[
        str,
        Field(
            description="默认正向提示词",
            json_schema_extra={
                "type": "text",
                "hint": "使用 nai 命令时的默认正向提示词",
            },
        ),
    ] = DEFAULT_PREPEND_PROMPT
    negative_prompt: Annotated[
        str,
        Field(
            description="默认反向提示词",
            json_schema_extra={"type": "text"},
        ),
    ] = DEFAULT_NEGATIVE_PROMPT
    size: Annotated[
        str,
        Field(
            description="默认出图尺寸",
            json_schema_extra={"hint": "需按照 832x1216 这样的格式填写"},
        ),
        AfterValidator(size_validator),
    ] = "832x1216"
    portrait_size: Annotated[
        str,
        Field(
            description="预置竖图尺寸",
            json_schema_extra={"hint": "需按照 832x1216 这样的格式填写"},
        ),
        AfterValidator(size_validator),
    ] = "832x1216"
    landscape_size: Annotated[
        str,
        Field(
            description="预置横图尺寸",
            json_schema_extra={"hint": "需按照 1216x832 这样的格式填写"},
        ),
        AfterValidator(size_validator),
    ] = "1216x832"
    square_size: Annotated[
        str,
        Field(
            description="预置方图尺寸",
            json_schema_extra={"hint": "需按照 1024x1024 这样的格式填写"},
        ),
        AfterValidator(size_validator),
    ] = "1024x1024"
    steps: Annotated[
        int,
        Field(
            description="默认采样步数",
            ge=1,
            le=50,
            json_schema_extra={
                "hint": "值需在 1 到 50 之间，且请不要大于上面设置的步数上限"
            },
        ),
    ] = 23
    scale: Annotated[float, Field(description="默认提示词引导值")] = 5
    cfg: Annotated[float, Field(description="默认缩放引导值")] = 0
    sampler: Annotated[
        str,
        Field(
            description="默认采样器",
            json_schema_extra={"options": list(AVAILABLE_SAMPLERS)},
        ),
        AfterValidator(make_item_exists_validator(AVAILABLE_SAMPLERS)),
    ] = AVAILABLE_SAMPLERS[0]
    noise_schedule: Annotated[
        str,
        Field(
            description="默认噪声调度",
            json_schema_extra={"options": list(AVAILABLE_NOISE_SCHEDULERS)},
        ),
        AfterValidator(make_item_exists_validator(AVAILABLE_NOISE_SCHEDULERS)),
    ] = AVAILABLE_NOISE_SCHEDULERS[0]
    other: Annotated[
        str,
        Field(
            description="默认高级配置",
            json_schema_extra={"options": list(AVAILABLE_DOTH)},
        ),
        AfterValidator(make_item_exists_validator(AVAILABLE_DOTH)),
    ] = AVAILABLE_DOTH[0]
    i2i_force: Annotated[float, Field(description="默认重绘力度")] = 0.6
    i2i_cl: Annotated[int, Field(description="默认图片处理")] = 1
    vibe_transfer_info_extract: Annotated[
        float,
        Field(
            description="默认氛围转移信息提取度",
            ge=0,
            le=1,
            json_schema_extra={"hint": "值需在 0 到 1 之间"},
        ),
    ] = 1
    vibe_transfer_ref_strength: Annotated[
        float,
        Field(
            description="默认氛围转移参考强度",
            ge=0,
            le=1,
            json_schema_extra={"hint": "值需在 0 到 1 之间"},
        ),
    ] = 0.5
    character_keep_vibe: Annotated[
        bool,
        Field(description="默认角色保持氛围"),
    ] = False
    character_keep_strength: Annotated[
        float,
        Field(
            description="默认角色保持参考强度",
            ge=0,
            le=1,
            json_schema_extra={"hint": "值需在 0 到 1 之间"},
        ),
    ] = 0.5

class Config(BaseModel):
    general: Annotated[
        GeneralConfig,
        Field(description="通用设置"),
    ] = GeneralConfig()
    request: Annotated[
        RequestConfig,
        Field(description="请求设置"),
    ] = RequestConfig()
    llm: Annotated[
        LLMConfig,
        Field(description="LLM 设置"),
    ] = LLMConfig()
    permission: Annotated[
        PermissionConfig,
        Field(description="权限设置"),
    ] = PermissionConfig()
    quota: Annotated[
        QuotaConfig,
        Field(description="额度设置"),
    ] = QuotaConfig()

    defaults: Annotated[
        DefaultsConfig,
        Field(description="默认值设置"),
    ] = DefaultsConfig()


if __name__ == "__main__":
    # 开发辅助：生成 _conf_schema.json。
    # 为避免误触发（例如有人手滑执行了本文件），需要显式环境变量开关。
    import os

    if os.environ.get("ASTRBOT_GEN_CONF_SCHEMA") != "1":
        raise SystemExit(
            "Set env ASTRBOT_GEN_CONF_SCHEMA=1 to generate _conf_schema.json"
        )
    import json
    from pathlib import Path

    _type_transform_map = {
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
    }

    def _remap_type(schema: dict[str, Any]):
        if "type" in schema and (ot := schema["type"]) in _type_transform_map:
            schema["type"] = _type_transform_map[ot]

        if "items" in schema:
            schema["items"] = _remap_type(schema["items"])
        elif "properties" in schema:
            it = schema.pop("properties")
            for k, v in it.items():
                it[k] = _remap_type(v)
            schema["items"] = it

        return schema

    def _post_process_schema(schema: dict[str, Any]):
        schema = cast(Any, jsonref.replace_refs(schema, merge_props=True))
        schema = _remap_type(schema)
        return schema["items"]

    (Path(__file__).parent.parent / "_conf_schema.json").write_text(
        json.dumps(
            _post_process_schema(Config.model_json_schema()),
            ensure_ascii=False,
            indent=2,
        ),
        "u8",
    )
