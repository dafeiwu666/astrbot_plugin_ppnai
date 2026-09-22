#!/usr/bin/env python3
"""
QuarkPan CLI 模块主入口点
支持 python -m quark_client.cli 调用
"""

from .main import app

if __name__ == "__main__":
    app()