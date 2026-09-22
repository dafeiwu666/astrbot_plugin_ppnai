"""
简化登录模块
提供手动登录指导
"""

import json
import time
from pathlib import Path
from typing import Optional

from ..config import get_config_dir
from ..exceptions import AuthenticationError
from ..utils.logger import get_logger


class SimpleLogin:
    """简化登录管理器"""

    def __init__(self):
        """初始化简化登录"""
        self.config_dir = get_config_dir()
        self.cookies_file = self.config_dir / "cookies.json"
        self.logger = get_logger(__name__)

        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def manual_login(self) -> str:
        """
        手动登录指导

        Returns:
            Cookie字符串
        """
        print("\n" + "=" * 60)
        print("🌟 夸克网盘手动登录指导")
        print("=" * 60)

        print("\n📋 步骤1: 打开浏览器")
        print("   请在浏览器中访问: https://pan.quark.cn")

        print("\n📋 步骤2: 登录账号")
        print("   • 点击页面上的登录按钮")
        print("   • 使用夸克APP扫描二维码登录")
        print("   • 或使用手机号+验证码登录")

        print("\n📋 步骤3: 获取Cookies")
        print("   登录成功后，请按以下步骤获取Cookies:")
        print("   1. 按F12打开开发者工具")
        print("   2. 点击'Network'(网络)标签")
        print("   3. 刷新页面(F5)")
        print("   4. 在网络请求中找到任意一个请求")
        print("   5. 右键点击 -> Copy -> Copy as cURL")
        print("   6. 从cURL命令中复制Cookie部分")

        print("\n📋 步骤4: 输入Cookies")
        print("   请将获取到的Cookie字符串粘贴到下面:")
        print("   (格式类似: __kps=xxx; __uid=xxx; ...)")

        print("\n" + "-" * 60)

        # 等待用户输入
        while True:
            try:
                cookie_input = input("\n请输入Cookie字符串 (输入'help'查看详细帮助): ").strip()

                if not cookie_input:
                    print("❌ Cookie不能为空，请重新输入")
                    continue

                if cookie_input.lower() == 'help':
                    self._show_detailed_help()
                    continue

                if cookie_input.lower() in ['quit', 'exit', 'q']:
                    raise AuthenticationError("用户取消登录")

                # 验证Cookie格式
                if not self._validate_cookie_format(cookie_input):
                    print("❌ Cookie格式不正确，请检查后重新输入")
                    continue

                # 保存Cookie
                self._save_cookies(cookie_input)

                print("✅ Cookie保存成功!")
                return cookie_input

            except KeyboardInterrupt:
                raise AuthenticationError("用户取消登录")
            except Exception as e:
                print(f"❌ 输入处理失败: {e}")
                continue

    def _show_detailed_help(self):
        """显示详细帮助"""
        print("\n" + "=" * 60)
        print("📖 详细获取Cookie教程")
        print("=" * 60)

        print("\n🌐 方法1: 从开发者工具获取")
        print("   1. 在https://pan.quark.cn登录成功后")
        print("   2. 按F12打开开发者工具")
        print("   3. 点击'Application'(应用)标签")
        print("   4. 左侧展开'Storage' -> 'Cookies'")
        print("   5. 点击'https://pan.quark.cn'")
        print("   6. 复制所有Cookie，格式: name1=value1; name2=value2")

        print("\n🌐 方法2: 从网络请求获取")
        print("   1. 按F12打开开发者工具")
        print("   2. 点击'Network'(网络)标签")
        print("   3. 刷新页面(F5)")
        print("   4. 找到任意一个对quark.cn的请求")
        print("   5. 点击该请求，查看'Request Headers'")
        print("   6. 找到'Cookie:'行，复制其值")

        print("\n🌐 方法3: 从cURL命令获取")
        print("   1. 在Network标签中右键任意请求")
        print("   2. 选择'Copy' -> 'Copy as cURL'")
        print("   3. 从cURL命令中找到-H 'cookie: ...'部分")
        print("   4. 复制cookie:后面的内容")

        print("\n✅ Cookie示例:")
        print("   __kps=AASPtC4Ty9ciIswLbjZNOB9M; __uid=AASPtC4Ty9ciIswLbjZNOB9M; ...")

        print("\n⚠️  注意事项:")
        print("   • Cookie包含敏感信息，请勿泄露给他人")
        print("   • Cookie有有效期，过期后需要重新获取")
        print("   • 确保Cookie来自https://pan.quark.cn域名")

        print("\n" + "=" * 60)

    def _validate_cookie_format(self, cookie_string: str) -> bool:
        """验证Cookie格式"""
        if not cookie_string:
            return False

        # 检查是否包含夸克相关的关键Cookie
        required_cookies = ['__kps', '__uid']

        for required in required_cookies:
            if required not in cookie_string:
                print(f"⚠️  警告: 未找到必需的Cookie '{required}'")
                return False

        # 检查基本格式
        if '=' not in cookie_string:
            return False

        return True

    def _save_cookies(self, cookie_string: str):
        """保存Cookie到文件"""
        try:
            # 解析Cookie字符串为字典
            cookie_dict = {}
            for pair in cookie_string.split(';'):
                pair = pair.strip()
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    cookie_dict[key.strip()] = value.strip()

            # 保存到文件
            cookie_data = {
                'cookies': cookie_dict,
                'cookie_string': cookie_string,
                'timestamp': int(time.time()),
                'source': 'manual_input'
            }

            with open(self.cookies_file, 'w', encoding='utf-8') as f:
                json.dump(cookie_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Cookie已保存到: {self.cookies_file}")

        except Exception as e:
            raise AuthenticationError(f"保存Cookie失败: {e}")

    def load_saved_cookies(self) -> Optional[str]:
        """加载已保存的Cookie"""
        try:
            if not self.cookies_file.exists():
                return None

            with open(self.cookies_file, 'r', encoding='utf-8') as f:
                cookie_data = json.load(f)

            # 检查Cookie是否过期（7天）
            timestamp = cookie_data.get('timestamp', 0)
            if time.time() - timestamp > 7 * 24 * 3600:
                self.logger.warning("已保存的Cookie已过期")
                return None

            cookie_string = cookie_data.get('cookie_string', '')
            if cookie_string and self._validate_cookie_format(cookie_string):
                self.logger.info("加载已保存的Cookie")
                return cookie_string

            return None

        except Exception as e:
            self.logger.warning(f"加载Cookie失败: {e}")
            return None

    def login(self, force_relogin: bool = False) -> str:
        """
        执行登录流程

        Args:
            force_relogin: 是否强制重新登录

        Returns:
            Cookie字符串
        """
        # 如果不是强制重新登录，先尝试使用已保存的Cookie
        if not force_relogin:
            saved_cookies = self.load_saved_cookies()
            if saved_cookies:
                print("✅ 使用已保存的登录信息")
                return saved_cookies

        # 执行手动登录
        return self.manual_login()

    def logout(self):
        """登出并清除本地Cookie"""
        try:
            if self.cookies_file.exists():
                self.cookies_file.unlink()
                print("✅ 已清除本地登录信息")
            else:
                print("ℹ️ 没有找到本地登录信息")
        except Exception as e:
            print(f"❌ 清除登录信息失败: {e}")


def simple_login(force_relogin: bool = False) -> str:
    """
    简化登录便捷函数

    Args:
        force_relogin: 是否强制重新登录

    Returns:
        Cookie字符串
    """
    login_manager = SimpleLogin()
    return login_manager.login(force_relogin)
