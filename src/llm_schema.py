"""Pydantic schemas for advanced image generation."""

import json
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

from .models import AvailablePositionsType, ConfigDict, ReqAdditionMultiRole, model_with_model_config

OrientationType: TypeAlias = Literal["portrait", "landscape", "square", "default"]


@model_with_model_config(ConfigDict(extra="forbid"))
class STNaiGenerateImageI2IArgs(BaseModel):
    repainting_strength: Annotated[
        float | None,
        Field(
            description=(
                "Optional."
                " The repainting strength for image-to-image generation."
                " A float between 0.0 and 1.0."
                " The higher the value"
                ", the more the generated image will differ from the original."
            ),
            ge=0.0,
            le=1.0,
        ),
    ] = None


@model_with_model_config(ConfigDict(extra="forbid"))
class STNaiGenerateImageVibeTransferArgs(BaseModel):
    information_extraction_rate: Annotated[
        float | None,
        Field(
            description=(
                "Optional."
                " The information extraction rate for vibe/style transfer."
                " A float between 0.0 and 1.0."
                " The higher the value"
                ", the more details will be extracted from the reference image."
            ),
            ge=0.0,
            le=1.0,
        ),
    ] = None
    reference_strength: Annotated[
        float | None,
        Field(
            description=(
                "Optional."
                " The reference strength for vibe/style transfer."
                " A float between 0.0 and 1.0."
                " The higher the value"
                ", the more the generated image will resemble the reference image."
            ),
            ge=0.0,
            le=1.0,
        ),
    ] = None


@model_with_model_config(ConfigDict(extra="forbid"))
class STNaiGenerateImageMultiRoleArgs(BaseModel):
    prompt: Annotated[
        str,
        Field(
            description=(
                "Positive prompt for this character."
                " Comma separated short English phrases"
                " describing the desired appearance of this character."
            )
        ),
    ]
    negative_prompt: Annotated[
        str,
        Field(
            description=(
                "Negative prompt for this character."
                " Comma separated short English phrases"
                " describing the undesired appearance of this character."
                " Note this field will not apply default prompt."
            )
        ),
    ]
    position: Annotated[
        AvailablePositionsType,
        Field(
            description=(
                "The position of this character in the image."
                " Two character format like `C3`."
                " The first character (A-E) is X-axis (left to right),"
                " the second character (1-5) is Y-axis (top to bottom)."
            )
        ),
    ]


@model_with_model_config(ConfigDict(extra="forbid"))
class STNaiGenerateImageAdvancedArgs(BaseModel):
    orientation: Annotated[
        OrientationType,
        Field(description="The desired orientation of the generated image."),
    ]
    prompt: Annotated[
        str,
        Field(
            description=(
                "Positive prompt."
                " Comma separated short English phrases describing the desired image."
            )
        ),
    ]
    additional_negative_prompt: Annotated[
        str,
        Field(
            description=(
                "Optional."
                " Negative prompt."
                " Comma separated short English phrases describing the undesired things."
            )
        ),
    ] = ""
    i2i: Annotated[
        STNaiGenerateImageI2IArgs | None,
        Field(description=("Optional. Settings for image-to-image generation.")),
    ] = None
    vibe_transfer: Annotated[
        list[STNaiGenerateImageVibeTransferArgs] | None,
        Field(description=("Optional. Settings for vibe/style transfer.")),
    ] = None
    multi_role_list: Annotated[
        list[STNaiGenerateImageMultiRoleArgs] | None,
        Field(description=("Optional. List of multi-role control settings.")),
    ] = None


GENERATE_IMAGE_ADVANCED_SCHEMA_TXT = json.dumps(
    STNaiGenerateImageAdvancedArgs.model_json_schema(),
    indent=2,
    ensure_ascii=False,
)
