# -*- coding: utf-8 -*-
"""
夸克网盘 Python 客户端

一个功能完整的夸克网盘API客户端，支持文件管理、分享转存等功能。
"""

# 初始化日志系统
from .utils.logger import setup_logger

setup_logger()

# 认证相关
from .auth.login import QuarkAuth, get_auth_cookies
# 主要客户端类
from .client import QuarkClient, create_client
# 核心API客户端
from .core.api_client import QuarkAPIClient
# 异常类
from .exceptions import (APIError, AuthenticationError, ConfigError,
                         DownloadError, FileNotFoundError, NetworkError,
                         QuarkClientError, ShareLinkError)
# 服务类
from .services.file_service import FileService
from .services.share_service import ShareService

__version__ = "0.1.0"
__author__ = "QuarkPan Team"
__email__ = "contact@quarkpan.dev"

__all__ = [
    # 主要客户端
    'QuarkClient',
    'create_client',

    # 认证
    'QuarkAuth',
    'get_auth_cookies',

    # 服务类
    'FileService',
    'ShareService',
    'QuarkAPIClient',

    # 异常
    'QuarkClientError',
    'AuthenticationError',
    'ConfigError',
    'APIError',
    'NetworkError',
    'FileNotFoundError',
    'ShareLinkError',
    'DownloadError',
]
