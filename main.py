import asyncio
import os
import uuid
from asyncio import Semaphore
from pathlib import Path
from typing import Annotated

from cookit.pyd import model_with_model_config
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
from typing_extensions import override

from astrbot import logger
from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, MessageChain, filter as event_filter
from astrbot.api.provider import LLMResponse
from astrbot.api.message_components import Image, Reply
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from .src.config import Config
from .src.data_source import GenerateError, wrapped_generate
from .src.llm import (
    ConfigNeededTool,
    ReturnToLLMError,
    llm_generate_advanced_req,
    llm_generate_image,
)
from .src.llm_utils import format_readable_error
from .src.models import Req
from .src.params import parse_req
from .src.image_io import resolve_image
from .src.user_manager import UserManager
from .src.preset_manager import PresetManager
from .src.queue_manager import get_shared_queue
from .src.handlers_nai import handle_cmd_nai, handle_nai_draw
from .src.handlers_auto import (
    handle_auto_draw,
    handle_auto_draw_off,
    handle_auto_draw_on,
    handle_llm_response_auto_draw,
)

COMMAND = "nai"

# region help

# 帮助文档路径
USAGE_MD_PATH = Path(__file__).parent / "docs" / "USAGE.md"


def load_usage_md() -> str:
    """读取 USAGE.md 文件内容作为帮助信息"""
    try:
        if USAGE_MD_PATH.exists():
            return USAGE_MD_PATH.read_text(encoding="utf-8")
        else:
            logger.warning(f"帮助文档不存在: {USAGE_MD_PATH}")
            return "# 泡泡画图\n\n帮助文档暂不可用，请联系管理员。"
    except Exception as e:
        logger.exception(f"读取帮助文档失败: {e}")
        return "# 泡泡画图\n\n帮助文档加载失败，请联系管理员。"


# endregion

WAITING_REPLIES = [
    "少女绘画中……",
    "在画了在画了",
    "你就在此地不要走动，等我给你画一幅",
]


@model_with_model_config(ConfigDict(extra="forbid"))
class STNaiGenerateImageArgsNoImage(BaseModel):
    instructions: Annotated[
        str,
        Field(
            description=(
                "Natural-language instructions for the image-generation agent"
                " that precisely describe the desired image"
                ", as detailed as possible."
            )
        ),
    ]


@model_with_model_config(ConfigDict(extra="forbid"))
class STNaiGenerateImageArgs(BaseModel):
    instructions: Annotated[
        str,
        Field(
            description=(
                "Natural-language instructions for the image-generation agent"
                " that precisely describe the desired image"
                ", as detailed as possible."
                " Don't use the original index number in image list here"
                ', instead, use sentences like "image referenced for image-to-image" or'
                '"the first image referenced in vibe transfer".'
            )
        ),
    ]
    i2i_image: Annotated[
        int | None,
        Field(
            description=(
                "Optional. The index of image you want to use"
                " as the base for image-to-image generation."
            )
        ),
    ] = None
    vibe_transfer_images: Annotated[
        list[int] | None,
        Field(
            description=(
                "Optional. The indices of images you want to"
                " use as the base for vibe/style transfer (in apply order)."
            )
        ),
    ] = None


@dataclass
class STNaiGenerateImageTool(ConfigNeededTool):
    name: str = "stnai_generate_image"
    description: str = (
        "Generate an anime-style image and send it to user."
        " Use when user wants you to draw an image."
    )
    parameters: dict = Field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        allow_image = self.config.llm.allow_i2i or self.config.llm.allow_vibe_transfer
        if not allow_image:
            self.parameters = STNaiGenerateImageArgsNoImage.model_json_schema()
        else:
            self.description += (
                " Images (in the latest user message ONLY) are gathered into an ordered list"
                "; refer to them by zero-based index in tool parameters."
            )
            parameters = STNaiGenerateImageArgs.model_json_schema()
            props = parameters["properties"]
            if not self.config.llm.allow_i2i:
                del props["i2i_image"]
            if not self.config.llm.allow_vibe_transfer:
                del props["vibe_transfer_images"]
            self.parameters = parameters

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        try:
            args = STNaiGenerateImageArgs.model_validate(kwargs)
        except Exception as e:
            tip = "Invalid arguments for STNaiGenerateImageTool"
            logger.debug(tip, exc_info=e)
            return format_readable_error(e)

        ctx = context.context.context
        event = context.context.event

        images = [x for x in event.message_obj.message if isinstance(x, Image)]
        sem = Semaphore(4)

        async def _get_image(index: int) -> str:
            try:
                img = images[index]
            except Exception as e:
                tip = f"Image index {index} is out of range (only {len(images)} images available)"
                logger.debug(tip)
                raise ReturnToLLMError(tip) from e
            try:
                async with sem:
                    return await resolve_image(img)
            except Exception as e:
                tip = f"Failed to fetch image at index {index}"
                logger.debug(tip, exc_info=e)
                raise ReturnToLLMError(f"{tip}:\n{format_readable_error(e)}") from e

        async def _resolve_i2i_image():
            return (
                (await _get_image(args.i2i_image))
                if args.i2i_image is not None
                else None
            )

        async def _resolve_vibe_transfer_images():
            if args.vibe_transfer_images is None:
                return None
            res: list[str] = []
            for idx in args.vibe_transfer_images:
                img_str = await _get_image(idx)
                res.append(img_str)
            return res

        try:
            i2i_image, vibe_transfer_images = await asyncio.gather(
                _resolve_i2i_image(),
                _resolve_vibe_transfer_images(),
            )
        except ReturnToLLMError as e:
            logger.debug(f"{e}")
            return f"{e}"

        # 视觉输入仅使用“未被 i2i/vibe 占用”的图片
        used_indices: set[int] = set()
        if args.i2i_image is not None:
            used_indices.add(args.i2i_image)
        if args.vibe_transfer_images:
            used_indices.update(args.vibe_transfer_images)
        vision_images = [img for idx, img in enumerate(images) if idx not in used_indices]

        try:
            # 在指令前添加"画一张图"
            instructions_with_prefix = f"画一张图\n\n{args.instructions}"
            image = await llm_generate_image(
                instructions_with_prefix,
                self.config,
                ctx,
                event,
                i2i_image,
                vibe_transfer_images,
                vision_images=vision_images,
            )
        except ReturnToLLMError as e:
            logger.debug(f"{e}")
            return f"{e}"
        except Exception as e:
            logger.exception("Internal error during image generation")
            return (
                f"Internal error during image generation: \n{format_readable_error(e)}"
            )

        try:
            await ctx.send_message(
                event.unified_msg_origin,
                MessageChain([Image.fromBytes(image)]),
            )
        except Exception as e:
            logger.exception("Send image failed")
            return (
                f"Failed to send image, "
                f"please report this error to user rather than retry"
                f": \n{format_readable_error(e)}"
            )

        return "Image successfully sent"


class Plugin(Star):
    """使用指令 nai 查看详细帮助"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = Config.model_validate(config)
        
        # 初始化用户管理器和预设管理器，数据存储在插件目录下的 data 文件夹
        data_dir = Path(__file__).parent / "data"
        self.user_manager = UserManager(data_dir)
        self.preset_manager = PresetManager(data_dir)
        
        # 自动画图状态（按会话存储）
        # key: unified_msg_origin
        # value: None 表示关闭，AutoDrawState 表示开启
        #   - enabled: 是否开启
        #   - presets: 预设名列表，按优先级排序 [s1, s2, ...]
        #   - opener_user_id: 开启者的用户ID，用于扣额度
        self.auto_draw_info: dict[str, dict | None] = {}
        
        # Token 轮询索引
        self._token_index = 0
        
        # 画图队列（进程内共享，避免多实例导致并发翻倍）
        self._queue = get_shared_queue()

        self.context.add_llm_tools(STNaiGenerateImageTool(config_init=self.config))

    @override
    async def initialize(self):
        # 在事件循环中初始化信号量（共享队列状态）
        self._queue.ensure(self.config.request.max_concurrent)
        logger.info(
            f"[nai] 队列系统初始化 pid={os.getpid()} instance={id(self)}: "
            f"最大并发={self.config.request.max_concurrent}, 最大队列={self.config.request.max_queue_size}"
        )

    @override
    async def terminate(self):
        pass

    def generate_help(self, umo: str) -> str:
        """读取 USAGE.md 文件内容作为帮助信息"""
        return load_usage_md()
    
    async def _render_markdown_to_images(self, markdown_content: str) -> list[str]:
        """使用 pillowmd 将 Markdown 渲染为图片列表
        
        Args:
            markdown_content: Markdown 内容
            
        Returns:
            图片文件路径列表
        """
        try:
            import pillowmd
            
            # 样式路径
            style_path = Path("data/styles/夏日冲浪")
            
            if style_path.exists():
                # 使用自定义样式
                style = pillowmd.LoadMarkdownStyles(str(style_path))
            else:
                # 使用默认样式
                logger.warning(f"样式路径不存在: {style_path}，使用默认样式")
                style = pillowmd.MdStyle()
            
            # 使用异步接口渲染
            # autoPage=True 支持长图分页
            render_result = await style.AioRender(
                text=markdown_content,
                useImageUrl=True,
                autoPage=True
            )
            
            # MdRenderResult 对象包含 images 列表
            if hasattr(render_result, 'images'):
                images = render_result.images
            elif isinstance(render_result, list):
                images = render_result
            else:
                # 回退处理
                images = [render_result]
            
            # 保存到本地缓存目录
            cache_dir = Path(__file__).parent / "data" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            saved_paths = []
            session_id = uuid.uuid4().hex[:8]
            
            for i, img in enumerate(images):
                # 生成唯一文件名
                image_path = cache_dir / f"help_{session_id}_{i}.png"
                img.save(str(image_path), format="PNG")
                saved_paths.append(str(image_path))

            return saved_paths
            
        except ImportError:
            logger.warning("pillowmd 未安装，回退到远程渲染")
            return []
        except Exception as e:
            logger.exception(f"pillowmd 渲染失败: {e}")
            return []
    
    def _get_user_id(self, event: AstrMessageEvent) -> str:
        """从事件中获取用户ID"""
        return event.get_sender_id()
    
    def _check_permission(self, event: AstrMessageEvent) -> bool:
        """检查是否是管理员"""
        # 这里简单判断，可以根据 AstrBot 的实际权限系统调整
        return event.is_admin if hasattr(event, 'is_admin') else False
    
    def _get_next_token(self) -> str:
        """轮询获取下一个可用的 Token"""
        tokens = self.config.request.tokens
        if not tokens:
            return ""
        token = tokens[self._token_index % len(tokens)]
        self._token_index = (self._token_index + 1) % len(tokens)
        return token

    async def _run_with_retry(self, func):
        """内部重试包装器（不外显）。

        func: 一个无参 async callable
        """
        retries = max(0, int(getattr(self.config.request, "retry_times", 0) or 0))
        wait_s = float(getattr(self.config.request, "retry_wait", 0.0) or 0.0)

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await func()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                if attempt >= retries:
                    raise
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
        assert last_exc is not None
        raise last_exc
    
    def _get_queue_status(self) -> str:
        """获取当前队列状态字符串"""
        queue_total = self._queue.queue_status()
        if queue_total > 1:
            return f"（当前队列：{queue_total}）"
        return ""

    def _ensure_semaphore(self) -> Semaphore:
        """确保并发信号量已初始化（兼容极端情况下 initialize 尚未执行）"""
        sem, _ = self._queue.ensure(self.config.request.max_concurrent)
        return sem
    
    def _get_reply_text(self, event: AstrMessageEvent) -> str:
        """获取引用消息的文本内容"""
        try:
            # 检查消息链中是否有Reply组件
            for component in event.message_obj.message:
                if isinstance(component, Reply):
                    # Reply组件包含被引用消息的信息
                    # 尝试获取Reply组件的文本属性
                    if hasattr(component, 'text') and component.text:
                        return component.text
                    
                    # 如果Reply有content属性
                    if hasattr(component, 'content') and component.content:
                        return str(component.content)
                    
                    # 如果有message属性（某些实现）
                    if hasattr(component, 'message'):
                        msg = component.message
                        if isinstance(msg, str):
                            return msg
                        elif hasattr(msg, 'get_plain_text'):
                            return msg.get_plain_text()
                    
                    # 尝试从event的原始消息中获取
                    if hasattr(event.message_obj, 'reply') and event.message_obj.reply:
                        reply_msg = event.message_obj.reply
                        if hasattr(reply_msg, 'message') and isinstance(reply_msg.message, str):
                            return reply_msg.message
                        elif hasattr(reply_msg, 'text') and isinstance(reply_msg.text, str):
                            return reply_msg.text
                    
                    return ""
            
            return ""
        except Exception:
            return ""

    async def _parse_args(self, event: AstrMessageEvent, is_whitelisted: bool = False) -> Req | None:
        """解析命令参数，支持多预设
        
        预设格式：s1=xxx, s2=xxx, ...
        优先级：直接参数 > s1 > s2 > ...
        tag 和 negative 是累加，其他参数是覆盖
        """
        raw_params = event.message_str.removeprefix(COMMAND).strip()
        if not raw_params:
            return None
        
        # 解析所有参数行
        lines = raw_params.split('\n')
        direct_params: list[tuple[str, str]] = []  # 直接参数
        preset_params_list: list[list[tuple[str, str]]] = []  # 按预设编号排序的预设参数
        preset_numbers: list[int] = []  # 预设编号列表
        
        import re
        preset_pattern = re.compile(r'^s(\d+)$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 检查是否是预设参数
                match = preset_pattern.match(key)
                if match:
                    preset_num = int(match.group(1))
                    preset = self.preset_manager.get_preset(value)
                    if preset is None:
                        raise ValueError(f"预设 {value} 不存在，使用 nai预设列表 查看可用预设")
                    
                    # 解析预设内容
                    preset_lines = preset.content.split('\n')
                    preset_params: list[tuple[str, str]] = []
                    for pl in preset_lines:
                        pl = pl.strip()
                        if not pl:
                            continue
                        if '=' in pl:
                            pk, pv = pl.split('=', 1)
                            preset_params.append((pk.strip(), pv.strip()))
                        else:
                            # 没有 = 号的行视为 tag
                            preset_params.append(('tag', pl))
                    
                    preset_numbers.append(preset_num)
                    preset_params_list.append(preset_params)
                else:
                    direct_params.append((key, value))
            else:
                # 强制键值对格式，不接受无等号的行
                raise ValueError(f"参数格式错误：'{line}'，请使用键值对格式，例如：tag=xxx")
        
        # 按预设编号排序（1, 2, 3, ...）
        sorted_presets = sorted(zip(preset_numbers, preset_params_list), key=lambda x: x[0])
        
        # 合并参数
        # - tag 和 negative 是累加的（按优先级顺序）
        # - prepend_* 是累加的（高优先级在前）
        # - append_* 是累加的（高优先级在后）
        # - 其他参数是覆盖的
        merged: dict[str, str] = {}
        tag_parts: list[str] = []
        negative_parts: list[str] = []
        prepend_tag_parts: list[str] = []
        append_tag_parts: list[str] = []
        prepend_negative_parts: list[str] = []
        append_negative_parts: list[str] = []
        
        # 从最低优先级到最高：sN, ..., s2, s1, 直接参数
        all_params_groups = [p for _, p in reversed(sorted_presets)] + [direct_params]
        
        for params in all_params_groups:
            for key, value in params:
                if key == 'tag':
                    tag_parts.append(value)
                elif key in ('negative', '反向提示词'):
                    negative_parts.append(value)
                elif key in ('prepend_tag', '前置正向', '前置正向提示词'):
                    # 高优先级在前，所以后遍历的插入到列表开头
                    prepend_tag_parts.insert(0, value)
                elif key in ('append_tag', '后置正向', '后置正向提示词'):
                    # 高优先级在后，所以后遍历的追加到列表末尾
                    append_tag_parts.append(value)
                elif key in ('prepend_negative', '前置负面', '前置负面提示词'):
                    # 高优先级在前
                    prepend_negative_parts.insert(0, value)
                elif key in ('append_negative', '后置负面', '后置负面提示词'):
                    # 高优先级在后
                    append_negative_parts.append(value)
                else:
                    # 其他参数直接覆盖
                    merged[key] = value
        
        # 构建最终参数字符串
        final_params: list[str] = []
        
        # 合并 tag（按优先级顺序）
        if tag_parts:
            final_params.append(f'tag={", ".join(tag_parts)}')
        
        # 合并 prepend/append 提示词
        if prepend_tag_parts:
            final_params.append(f'prepend_tag={", ".join(prepend_tag_parts)}')
        if append_tag_parts:
            final_params.append(f'append_tag={", ".join(append_tag_parts)}')
        if prepend_negative_parts:
            final_params.append(f'prepend_negative={", ".join(prepend_negative_parts)}')
        if append_negative_parts:
            final_params.append(f'append_negative={", ".join(append_negative_parts)}')
        
        # 添加其他参数
        for key, value in merged.items():
            final_params.append(f'{key}={value}')
        
        # 合并 negative
        if negative_parts:
            final_params.append(f'negative={", ".join(negative_parts)}')
        
        final_raw = '\n'.join(final_params)
        
        return await parse_req(final_raw, event.message_obj.message, self.config, is_whitelisted)

    # ========== 签到命令 ==========
    
    @event_filter.command("nai签到")
    async def cmd_checkin(self, event: AstrMessageEvent):
        """每日签到获取画图额度"""
        user_id = self._get_user_id(event)
        success, gained, message = self.user_manager.checkin(user_id, self.config)
        yield event.plain_result(message)
    
    @event_filter.command("nai队列")
    async def cmd_queue_status(self, event: AstrMessageEvent):
        """查询当前队列状态"""
        max_concurrent = self.config.request.max_concurrent
        max_queue = self.config.request.max_queue_size
        
        # 计算当前处理中的数量
        processing = max(self._queue.queue_count - self._queue.waiting_count, 0)
        waiting = self._queue.waiting_count
        
        status_lines = [
            "📊 当前队列状态",
            f"• 正在处理：{processing}/{max_concurrent}",
            f"• 排队等待：{waiting}/{max_queue if max_queue > 0 else '∞'}",
        ]
        
        if self._queue.queue_count == 0:
            status_lines.append("\n✅ 队列空闲，可以立即开始画图")
        elif max_queue > 0 and waiting >= max_queue:
            status_lines.append("\n⚠️ 队列已满，新请求将被拒绝")
        else:
            if max_queue > 0:
                status_lines.append(f"\n📝 还可加入 {max_queue - waiting} 个请求")
        
        yield event.plain_result("\n".join(status_lines))
    
    @event_filter.command("查询额度")
    async def cmd_query_quota(self, event: AstrMessageEvent):
        """查询自己的画图额度"""
        user_id = self._get_user_id(event)
        
        if self.user_manager.is_blacklisted(user_id):
            yield event.plain_result("你已被加入黑名单")
            return
        
        if self.user_manager.is_whitelisted(user_id):
            yield event.plain_result("你在白名单中，可无限使用画图功能")
            return
        
        if not self.config.quota.enable_quota:
            yield event.plain_result("当前未启用额度系统，可无限使用画图功能")
            return
        
        quota = self.user_manager.get_quota(user_id)
        yield event.plain_result(f"你当前剩余 {quota} 次画图额度")

    # ========== 管理员命令 ==========
    
    @event_filter.command("nai黑名单添加")
    async def cmd_add_blacklist(self, event: AstrMessageEvent):
        """将用户添加到黑名单（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        args = event.message_str.removeprefix("nai黑名单添加").strip()
        if not args:
            yield event.plain_result("请指定用户ID，例如：nai黑名单添加 123456")
            return
        
        user_id = args.split()[0]
        if self.user_manager.add_to_blacklist(user_id):
            yield event.plain_result(f"已将用户 {user_id} 添加到黑名单")
        else:
            yield event.plain_result(f"用户 {user_id} 已在黑名单中")
    
    @event_filter.command("nai黑名单移除")
    async def cmd_remove_blacklist(self, event: AstrMessageEvent):
        """将用户从黑名单移除（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        args = event.message_str.removeprefix("nai黑名单移除").strip()
        if not args:
            yield event.plain_result("请指定用户ID，例如：nai黑名单移除 123456")
            return
        
        user_id = args.split()[0]
        if self.user_manager.remove_from_blacklist(user_id):
            yield event.plain_result(f"已将用户 {user_id} 从黑名单移除")
        else:
            yield event.plain_result(f"用户 {user_id} 不在黑名单中")
    
    @event_filter.command("nai黑名单列表")
    async def cmd_list_blacklist(self, event: AstrMessageEvent):
        """查看黑名单列表（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        blacklist = self.user_manager.get_blacklist()
        if not blacklist:
            yield event.plain_result("黑名单为空")
        else:
            yield event.plain_result("黑名单用户：\n" + "\n".join(blacklist))
    
    @event_filter.command("nai白名单添加")
    async def cmd_add_whitelist(self, event: AstrMessageEvent):
        """将用户添加到白名单（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        args = event.message_str.removeprefix("nai白名单添加").strip()
        if not args:
            yield event.plain_result("请指定用户ID，例如：nai白名单添加 123456")
            return
        
        user_id = args.split()[0]
        if self.user_manager.add_to_whitelist(user_id):
            yield event.plain_result(f"已将用户 {user_id} 添加到白名单")
        else:
            yield event.plain_result(f"用户 {user_id} 已在白名单中")
    
    @event_filter.command("nai白名单移除")
    async def cmd_remove_whitelist(self, event: AstrMessageEvent):
        """将用户从白名单移除（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        args = event.message_str.removeprefix("nai白名单移除").strip()
        if not args:
            yield event.plain_result("请指定用户ID，例如：nai白名单移除 123456")
            return
        
        user_id = args.split()[0]
        if self.user_manager.remove_from_whitelist(user_id):
            yield event.plain_result(f"已将用户 {user_id} 从白名单移除")
        else:
            yield event.plain_result(f"用户 {user_id} 不在白名单中")
    
    @event_filter.command("nai白名单列表")
    async def cmd_list_whitelist(self, event: AstrMessageEvent):
        """查看白名单列表（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        whitelist = self.user_manager.get_whitelist()
        if not whitelist:
            yield event.plain_result("白名单为空")
        else:
            yield event.plain_result("白名单用户：\n" + "\n".join(whitelist))
    
    @event_filter.command("nai查询用户")
    async def cmd_admin_query_user(self, event: AstrMessageEvent):
        """查询用户额度（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        args = event.message_str.removeprefix("nai查询用户").strip()
        if not args:
            yield event.plain_result("请指定用户ID，例如：nai查询用户 123456")
            return
        
        user_id = args.split()[0]
        quota = self.user_manager.get_quota(user_id)
        
        status = ""
        if self.user_manager.is_blacklisted(user_id):
            status = "（黑名单）"
        elif self.user_manager.is_whitelisted(user_id):
            status = "（白名单）"
        
        yield event.plain_result(f"用户 {user_id}{status} 的额度：{quota} 次")
    
    @event_filter.command("nai设置额度")
    async def cmd_set_quota(self, event: AstrMessageEvent):
        """设置用户额度（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        args = event.message_str.removeprefix("nai设置额度").strip().split()
        if len(args) < 2:
            yield event.plain_result("请指定用户ID和额度，例如：nai设置额度 123456 100")
            return
        
        user_id = args[0]
        try:
            quota = int(args[1])
        except ValueError:
            yield event.plain_result("额度必须是整数")
            return
        
        self.user_manager.set_quota(user_id, quota)
        yield event.plain_result(f"已将用户 {user_id} 的额度设置为 {quota} 次")
    
    @event_filter.command("nai增加额度")
    async def cmd_add_quota(self, event: AstrMessageEvent):
        """增加用户额度（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        args = event.message_str.removeprefix("nai增加额度").strip().split()
        if len(args) < 2:
            yield event.plain_result("请指定用户ID和额度，例如：nai增加额度 123456 10")
            return
        
        user_id = args[0]
        try:
            amount = int(args[1])
        except ValueError:
            yield event.plain_result("额度必须是整数")
            return
        
        new_quota = self.user_manager.add_quota(user_id, amount)
        yield event.plain_result(f"已为用户 {user_id} 增加 {amount} 次额度，当前额度：{new_quota} 次")

    # ========== 预设命令 ==========
    
    @event_filter.command("nai预设列表")
    async def cmd_preset_list(self, event: AstrMessageEvent):
        """查看预设列表"""
        presets = self.preset_manager.list_presets()
        if not presets:
            yield event.plain_result("暂无预设，管理员可使用 nai预设添加 命令添加预设")
            return
        
        result = "📝 预设列表：\n" + "\n".join(f"• {title}" for title in presets)
        result += "\n\n使用方式：\nnai\ns1=预设名"
        yield event.plain_result(result)
    
    @event_filter.command("nai预设查看")
    async def cmd_preset_view(self, event: AstrMessageEvent):
        """查看预设详细内容"""
        args = event.message_str.removeprefix("nai预设查看").strip()
        if not args:
            yield event.plain_result("请指定预设名称，例如：nai预设查看 猫娘")
            return
        
        title = args.split()[0]
        preset = self.preset_manager.get_preset(title)
        
        if preset is None:
            yield event.plain_result(f"预设 #{title} 不存在")
            return
        
        # 使用代码块包裹以防平台解析错误或截断
        yield event.plain_result(f"📝 预设 #{title}\n\n```\n{preset.content}\n```")
    
    @event_filter.command("nai预设添加")
    async def cmd_preset_add(self, event: AstrMessageEvent):
        """添加预设（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        # 解析：第一行是 "nai预设添加 标题"，后面的行是内容
        full_text = event.message_str
        lines = full_text.split('\n', 1)
        
        # 从第一行提取标题
        first_line = lines[0].removeprefix("nai预设添加").strip()
        if not first_line:
            yield event.plain_result(
                "请指定预设标题和内容，格式：\n"
                "nai预设添加 标题名\n"
                "这里是预设内容..."
            )
            return
        
        title = first_line
        
        # 获取内容（第二行开始）
        if len(lines) < 2 or not lines[1].strip():
            yield event.plain_result(
                f"请在标题后换行添加预设内容，格式：\n"
                f"nai预设添加 {title}\n"
                f"这里是预设内容..."
            )
            return
        
        content = lines[1]
        
        # 检查是否已存在
        if self.preset_manager.get_preset(title) is not None:
            yield event.plain_result(
                f"预设 #{title} 已存在，如需修改请先删除再添加"
            )
            return
        
        self.preset_manager.add_preset(title, content)
        yield event.plain_result(f"✅ 预设 #{title} 添加成功！\n\n预览：\n{content[:200]}{'...' if len(content) > 200 else ''}")
    
    @event_filter.command("nai预设删除")
    async def cmd_preset_delete(self, event: AstrMessageEvent):
        """删除预设（管理员）"""
        if not self._check_permission(event):
            yield event.plain_result("权限不足，仅管理员可使用此命令")
            return
        
        args = event.message_str.removeprefix("nai预设删除").strip()
        if not args:
            yield event.plain_result("请指定预设名称，例如：nai预设删除 猫娘")
            return
        
        title = args.split()[0]
        
        if self.preset_manager.delete_preset(title):
            yield event.plain_result(f"✅ 预设 #{title} 已删除")
        else:
            yield event.plain_result(f"预设 #{title} 不存在")

    # ========== nai画图命令（直接调用插件AI） ==========
    
    def _parse_presets_from_params(self, raw_params: str) -> tuple[list[str], dict[str, str]]:
        """从参数中解析预设列表和其他参数
        
        Returns:
            (预设名列表按优先级排序, 其他参数字典)
        """
        import re
        preset_pattern = re.compile(r'^s(\d+)$')
        
        presets: list[tuple[int, str]] = []  # (编号, 预设名)
        other_params: dict[str, str] = {}
        
        for line in raw_params.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                match = preset_pattern.match(key)
                if match:
                    preset_num = int(match.group(1))
                    presets.append((preset_num, value))
                else:
                    other_params[key] = value
        
        # 按编号排序
        presets.sort(key=lambda x: x[0])
        return [name for _, name in presets], other_params
    
    @event_filter.command("nai画图")
    async def cmd_nai_draw(self, event: AstrMessageEvent):
        """使用插件 AI 直接画图"""
        async for result in handle_nai_draw(self, event, WAITING_REPLIES):
            yield result

    # ========== 自动画图命令 ==========
    
    @event_filter.command("nai自动画图关")
    async def cmd_auto_draw_off(self, event: AstrMessageEvent):
        """关闭自动画图"""
        async for result in handle_auto_draw_off(self, event):
            yield result
    
    @event_filter.command("nai自动画图开")
    async def cmd_auto_draw_on(self, event: AstrMessageEvent):
        """开启自动画图
        
        格式：
        nai自动画图开
        s1=xxx
        s2=xxx
        """
        async for result in handle_auto_draw_on(self, event):
            yield result
    
    @event_filter.command("nai自动画图")
    async def cmd_auto_draw(self, event: AstrMessageEvent):
        """查看或设置自动画图状态
        
        不带参数：显示当前状态
        带参数：设置预设并开启
        
        格式：
        nai自动画图             → 显示状态
        nai自动画图             → 设置预设（同时开启）
        s1=xxx
        """
        async for result in handle_auto_draw(self, event):
            yield result

    # ========== 画图命令 ==========

    @event_filter.command(COMMAND)
    async def cmd_nai(self, event: AstrMessageEvent):
        """泡泡画图"""
        async for result in handle_cmd_nai(self, event, WAITING_REPLIES):
            yield result

    # ========== 自动画图钩子 ==========
    
    @event_filter.on_llm_response(priority=50)
    async def on_llm_response_auto_draw(self, event: AstrMessageEvent, resp: LLMResponse):
        """监听主 AI 回复，自动生成图片"""
        await handle_llm_response_auto_draw(self, event, resp)

