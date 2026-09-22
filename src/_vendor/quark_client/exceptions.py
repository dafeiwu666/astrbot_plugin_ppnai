"""
异常定义模块
"""

from typing import Optional


class QuarkClientError(Exception):
    """夸克客户端基础异常"""
    pass


class AuthenticationError(QuarkClientError):
    """认证相关异常"""
    pass


class ConfigError(QuarkClientError):
    """配置相关异常"""
    pass


class APIError(QuarkClientError):
    """API调用异常"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class NetworkError(QuarkClientError):
    """网络相关异常"""
    pass


class FileNotFoundError(QuarkClientError):
    """文件未找到异常"""
    pass


class ShareLinkError(QuarkClientError):
    """分享链接相关异常"""
    pass


class DownloadError(QuarkClientError):
    """下载相关异常"""
    pass
