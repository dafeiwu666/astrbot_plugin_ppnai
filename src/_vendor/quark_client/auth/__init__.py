"""
认证模块
"""

from .login import QuarkAuth, get_auth_cookies

# 导入具体的登录实现
try:
    from .api_login import APILogin, api_login
    _api_available = True
except ImportError:
    _api_available = False

try:
    from .simple_login import SimpleLogin, simple_login
    _simple_available = True
except ImportError:
    _simple_available = False

# 构建导出列表
__all__ = ['QuarkAuth', 'get_auth_cookies']

if _api_available:
    __all__.extend(['APILogin', 'api_login'])

if _simple_available:
    __all__.extend(['SimpleLogin', 'simple_login'])
