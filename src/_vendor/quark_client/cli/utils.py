"""
CLI 工具函数
"""

import sys
from datetime import datetime
from typing import Optional

from rich import print as rprint
from rich.console import Console

from ..client import QuarkClient

console = Console()


def get_client(auto_login: bool = True) -> QuarkClient:
    """获取客户端实例"""
    try:
        return QuarkClient(auto_login=auto_login)
    except Exception as e:
        rprint(f"[red]❌ 创建客户端失败: {e}[/red]")
        sys.exit(1)


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.2f} {size_names[i]}"


def format_timestamp(timestamp) -> str:
    """格式化时间戳"""
    try:
        # 夸克网盘的时间戳可能是毫秒级
        if isinstance(timestamp, (int, float)):
            if timestamp > 1000000000000:  # 毫秒级时间戳
                timestamp = timestamp / 1000

            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        else:
            return str(timestamp)
    except:
        return str(timestamp)


def confirm_action(message: str, default: bool = False) -> bool:
    """确认操作"""
    suffix = " [Y/n]" if default else " [y/N]"
    response = console.input(f"[yellow]{message}{suffix}[/yellow] ")

    if not response:
        return default

    return response.lower().startswith('y')


def print_error(message: str):
    """打印错误信息"""
    rprint(f"[red]❌ {message}[/red]")


def print_success(message: str):
    """打印成功信息"""
    rprint(f"[green]✅ {message}[/green]")


def print_warning(message: str):
    """打印警告信息"""
    rprint(f"[yellow]⚠️ {message}[/yellow]")


def print_info(message: str):
    """打印信息"""
    rprint(f"[blue]ℹ️ {message}[/blue]")


def handle_api_error(e: Exception, operation: str = "操作"):
    """处理API错误"""
    error_msg = str(e)

    if "认证" in error_msg or "login" in error_msg.lower():
        print_error(f"{operation}失败: 认证过期，请重新登录")
        rprint("使用 [cyan]quarkpan auth login[/cyan] 重新登录")
    elif "网络" in error_msg or "network" in error_msg.lower():
        print_error(f"{operation}失败: 网络连接错误")
    elif "not found" in error_msg.lower():
        print_error(f"{operation}失败: 文件或文件夹不存在")
    elif "capacity limit" in error_msg.lower() or "容量不足" in error_msg:
        print_error(f"{operation}失败: 网盘容量不足")
        rprint("💡 解决方案:")
        rprint("  1. 清理网盘中的无用文件")
        rprint("  2. 删除回收站中的文件")
        rprint("  3. 升级网盘容量")
    elif "share expired" in error_msg.lower() or "分享过期" in error_msg:
        print_error(f"{operation}失败: 分享链接已过期")
    elif "share not found" in error_msg.lower() or "分享不存在" in error_msg:
        print_error(f"{operation}失败: 分享链接无效或已被删除")
    else:
        print_error(f"{operation}失败: {error_msg}")


def validate_file_id(file_id: str) -> bool:
    """验证文件ID格式"""
    if not file_id:
        return False

    # 夸克网盘的文件ID通常是32位十六进制字符串
    if len(file_id) == 32 and all(c in '0123456789abcdef' for c in file_id.lower()):
        return True

    # 根目录ID
    if file_id == "0":
        return True

    return False


def truncate_text(text: str, max_length: int = 50) -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def get_file_type_icon(file_name: str, is_folder: bool = False) -> str:
    """根据文件名获取图标"""
    if is_folder:
        return "📁"

    name_lower = file_name.lower()

    # 图片文件
    if any(name_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
        return "🖼️"

    # 视频文件
    if any(name_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']):
        return "🎬"

    # 音频文件
    if any(name_lower.endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']):
        return "🎵"

    # 文档文件
    if any(name_lower.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.txt', '.rtf']):
        return "📄"

    # 表格文件
    if any(name_lower.endswith(ext) for ext in ['.xls', '.xlsx', '.csv']):
        return "📊"

    # 演示文件
    if any(name_lower.endswith(ext) for ext in ['.ppt', '.pptx']):
        return "📽️"

    # 压缩文件
    if any(name_lower.endswith(ext) for ext in ['.zip', '.rar', '.7z', '.tar', '.gz']):
        return "📦"

    # 代码文件
    if any(name_lower.endswith(ext) for ext in ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']):
        return "💻"

    # 默认文件图标
    return "📄"


class FolderNavigator:
    """文件夹导航器"""

    def __init__(self):
        self.path_stack = []  # 路径栈，存储 (folder_id, folder_name) 元组
        self.current_folder_id = "0"
        self.current_folder_name = "根目录"

    def enter_folder(self, folder_id: str, folder_name: str):
        """进入文件夹"""
        # 保存当前位置到栈中
        self.path_stack.append((self.current_folder_id, self.current_folder_name))
        self.current_folder_id = folder_id
        self.current_folder_name = folder_name

    def go_back(self) -> bool:
        """返回上级目录"""
        if self.path_stack:
            self.current_folder_id, self.current_folder_name = self.path_stack.pop()
            return True
        return False

    def get_breadcrumb(self) -> str:
        """获取面包屑导航"""
        if not self.path_stack:
            return self.current_folder_name

        # 构建路径
        path_parts = [name for _, name in self.path_stack]
        path_parts.append(self.current_folder_name)

        # 如果路径太长，只显示最后几级
        if len(path_parts) > 3:
            return "... > " + " > ".join(path_parts[-2:])
        else:
            return " > ".join(path_parts)

    def get_current_folder(self) -> tuple:
        """获取当前文件夹信息"""
        return self.current_folder_id, self.current_folder_name

    def can_go_back(self) -> bool:
        """是否可以返回上级目录"""
        return len(self.path_stack) > 0


def get_folder_name_by_id(client, folder_id: str) -> str:
    """根据文件夹ID获取文件夹名称"""
    if folder_id == "0":
        return "根目录"

    try:
        file_info = client.get_file_info(folder_id)
        if file_info:
            return file_info.get('file_name', f'文件夹 {folder_id[:8]}...')
        else:
            return f'文件夹 {folder_id[:8]}...'
    except Exception as e:
        # 如果获取失败，尝试从父目录查找
        return f'文件夹 {folder_id[:8]}...'


def select_folder_from_list(file_list: list) -> tuple:
    """从文件列表中选择文件夹"""
    folders = []
    for i, file_info in enumerate(file_list):
        if file_info.get('file_type', 0) == 0:  # 是文件夹
            folders.append((i + 1, file_info.get('fid'), file_info.get('file_name')))

    if not folders:
        return None, None

    rprint("\n[cyan]可进入的文件夹:[/cyan]")
    for seq, fid, name in folders:
        rprint(f"  {seq}. 📁 {name}")

    rprint("\n[dim]输入序号进入文件夹，输入 'b' 返回上级，输入 'q' 退出[/dim]")

    try:
        choice = console.input("[cyan]请选择: [/cyan]").strip()

        if choice.lower() == 'q':
            return 'quit', None
        elif choice.lower() == 'b':
            return 'back', None
        else:
            seq = int(choice)
            for folder_seq, fid, name in folders:
                if folder_seq == seq:
                    return fid, name

            print_error("无效的序号")
            return None, None
    except (ValueError, KeyboardInterrupt):
        return None, None
