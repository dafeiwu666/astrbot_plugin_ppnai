"""
基于API的登录模块
"""

import json
import sys
import threading
import time
import uuid
from typing import Dict, Optional, Tuple

import httpx

from ..config import get_config_dir
from ..exceptions import AuthenticationError
from ..utils.logger import get_logger
from ..utils.qr_code import display_qr_from_url


class APILogin:
    """基于API的登录管理器"""

    def __init__(self, timeout: int = 300):
        """
        初始化API登录

        Args:
            timeout: 登录超时时间（秒）
        """
        self.timeout = timeout
        self.config_dir = get_config_dir()
        self.client = httpx.Client(timeout=30.0)
        self.logger = get_logger(__name__)

        # 倒计时相关
        self._countdown_thread = None
        self._stop_countdown = False

        # 设置基本headers
        self.client.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Ch-Ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site'
        })

        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _show_countdown(self, total_seconds: int):
        """
        显示倒计时

        Args:
            total_seconds: 总倒计时秒数
        """
        def countdown():
            remaining = total_seconds - 1

            while remaining > 0 and not self._stop_countdown:
                minutes = remaining // 60
                seconds = remaining % 60

                # 清除当前行并显示倒计时
                sys.stdout.write(f"\r⏰ 二维码有效期剩余: {minutes:02d}:{seconds:02d}")
                sys.stdout.flush()

                time.sleep(1)
                remaining -= 1

            if not self._stop_countdown:
                sys.stdout.write("\r⏰ 二维码已过期，请重新获取\n")
                sys.stdout.flush()

        self._countdown_thread = threading.Thread(target=countdown, daemon=True)
        self._countdown_thread.start()

    def _stop_countdown_display(self):
        """停止倒计时显示"""
        self._stop_countdown = True
        if self._countdown_thread and self._countdown_thread.is_alive():
            self._countdown_thread.join(timeout=1)
        # 清除倒计时行
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()

    def get_qr_code(self) -> Tuple[str, str]:
        """
        获取登录二维码

        Returns:
            (qr_token, qr_url) 二维码token和URL
        """
        try:
            # 使用发现的真实API
            api_url = 'https://uop.quark.cn/cas/ajax/getTokenForQrcodeLogin'

            # 生成随机request_id
            request_id = str(uuid.uuid4())

            params = {
                'client_id': '532',
                'v': '1.2',
                'request_id': request_id
            }

            self.logger.debug(f"获取二维码token: {api_url}")
            response = self.client.get(api_url, params=params)

            if response.status_code == 200:
                data = response.json()

                if data.get('status') == 2000000:
                    # 成功获取token
                    token = data.get('data', {}).get('members', {}).get('token')
                    if token:
                        self.logger.debug(f"获取到二维码token: {token}")

                        # 构造真实的二维码URL
                        # 基于实际抓包分析的URL格式
                        qr_url = self._build_qr_url(token)

                        return token, qr_url
                    else:
                        raise AuthenticationError("响应中未找到token")
                else:
                    raise AuthenticationError(f"获取token失败: {data.get('message', '未知错误')}")
            else:
                raise AuthenticationError(f"API请求失败: {response.status_code}")

        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError(f"获取二维码失败: {e}")

    def _build_qr_url(self, token: str) -> str:
        """
        构造二维码URL

        Args:
            token: 从API获取的token

        Returns:
            完整的二维码URL
        """
        # 基于实际抓包分析的URL格式
        base_url = "https://su.quark.cn/4_eMHBJ"

        # 构造查询参数
        params = {
            'token': token,
            'client_id': '532',
            'ssb': 'weblogin',
            'uc_param_str': '',
            'uc_biz_str': 'S:custom|OPT:SAREA@0|OPT:IMMERSIVE@1|OPT:BACK_BTN_STYLE@0'
        }

        # 构造完整URL
        import urllib.parse
        query_string = urllib.parse.urlencode(params)
        qr_url = f"{base_url}?{query_string}"

        self.logger.debug(f"构造二维码URL: {qr_url}")
        return qr_url

    def _is_valid_qr_response(self, data: Dict) -> bool:
        """检查响应是否包含有效的二维码数据"""
        if not isinstance(data, dict):
            return False

        # 检查常见的二维码字段
        qr_fields = ['qr_code', 'qrcode', 'qr_url', 'qrUrl', 'code_url', 'url', 'token', 'ticket']
        return any(field in data for field in qr_fields)

    def _process_qr_response(self, data: Dict) -> Tuple[str, str, str]:
        """处理二维码响应数据"""
        # 提取二维码相关信息
        qr_token = None
        qr_url = None

        # 尝试不同的字段名
        for token_field in ['token', 'ticket', 'qr_token', 'id']:
            if token_field in data:
                qr_token = data[token_field]
                break

        for url_field in ['qr_code', 'qrcode', 'qr_url', 'qrUrl', 'code_url', 'url']:
            if url_field in data:
                qr_url = data[url_field]
                break

        if not qr_url:
            raise AuthenticationError("响应中未找到二维码URL")

        return qr_token or str(uuid.uuid4()), qr_url

    def _fallback_get_qr_code(self) -> Tuple[str, str]:
        """备用方案：访问登录页面获取二维码"""
        try:
            # 访问夸克网盘登录页面
            login_urls = [
                'https://pan.quark.cn/account/login',
                'https://account.quark.cn/passport/login',
                'https://passport.quark.cn/login'
            ]

            for login_url in login_urls:
                try:
                    self.logger.info(f"访问登录页面: {login_url}")
                    response = self.client.get(login_url)

                    if response.status_code == 200:
                        # 在页面中查找二维码相关的API调用
                        html = response.text

                        # 查找可能的二维码API URL
                        import re
                        api_patterns = [
                            r'["\']([^"\']*qr[^"\']*)["\']',
                            r'["\']([^"\']*code[^"\']*)["\']',
                            r'url\s*:\s*["\']([^"\']*)["\']'
                        ]

                        for pattern in api_patterns:
                            matches = re.findall(pattern, html, re.IGNORECASE)
                            for match in matches:
                                if 'http' in match and ('qr' in match.lower() or 'code' in match.lower()):
                                    self.logger.info(f"找到可能的二维码API: {match}")
                                    # 尝试调用这个API
                                    try:
                                        api_response = self.client.get(match)
                                        if api_response.status_code == 200:
                                            data = api_response.json()
                                            if self._is_valid_qr_response(data):
                                                return self._process_qr_response(data)
                                    except:
                                        continue

                except Exception as e:
                    self.logger.debug(f"访问 {login_url} 失败: {e}")
                    continue

            raise AuthenticationError("无法通过任何方式获取二维码")

        except Exception as e:
            raise AuthenticationError(f"备用获取二维码方案失败: {e}")

    def check_login_status(self, qr_token: str) -> Optional[Dict]:
        """
        检查登录状态

        Args:
            qr_token: 二维码token

        Returns:
            登录成功时返回用户信息，否则返回None
        """
        try:
            # 使用发现的真实API
            api_url = 'https://uop.quark.cn/cas/ajax/getServiceTicketByQrcodeToken'

            # 生成随机request_id
            request_id = str(uuid.uuid4())

            params = {
                'client_id': '532',
                'v': '1.2',
                'token': qr_token,
                'request_id': request_id
            }

            self.logger.debug(f"检查登录状态: {api_url}")
            self.logger.debug(f"请求参数: {params}")

            response = self.client.get(api_url, params=params)
            self.logger.debug(f"响应状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"API响应: {data}")

                # 检查登录状态
                if self._is_login_success(data):
                    self.logger.debug("登录成功检测通过")
                    return data
                elif self._is_login_failed(data):
                    self.logger.debug("登录失败检测通过")
                    return None
                else:
                    # 还在等待扫码
                    self.logger.debug("仍在等待扫码")
                    return None
            else:
                self.logger.error(f"API请求失败: {response.status_code}")

            return None

        except Exception as e:
            self.logger.error(f"检查登录状态失败: {e}")
            return None

    def _is_login_success(self, data: Dict) -> bool:
        """检查是否登录成功"""
        if not isinstance(data, dict):
            return False

        # 根据实际API响应检查成功标志
        status = data.get('status')
        message = data.get('message', '')

        # 登录成功的标志：status=2000000, message="ok", 且有data.members.service_ticket
        if (status == 2000000 and
            message == "ok" and
                data.get('data', {}).get('members', {}).get('service_ticket')):
            return True

        return False

    def _is_login_failed(self, data: Dict) -> bool:
        """检查是否登录失败"""
        if not isinstance(data, dict):
            return False

        # 检查失败标志
        status = data.get('status')
        message = data.get('message', '')

        # 明确的失败状态（不包括50004001，那是等待状态）
        fail_indicators = [
            status in [50004002, 50004003, 50004004],  # 明确的失败状态码
            'expired' in message.lower(),
            'failed' in message.lower(),
            'error' in message.lower(),
            'timeout' in message.lower(),
            'invalid' in message.lower()
        ]

        return any(fail_indicators)

    def wait_for_login(self, qr_token: str) -> bool:
        """
        等待用户扫码登录

        Args:
            qr_token: 二维码token

        Returns:
            登录是否成功
        """
        start_time = time.time()
        check_count = 0

        self.logger.debug(f"开始等待登录，超时时间: {self.timeout}秒")

        # 启动倒计时显示
        self._show_countdown(self.timeout)

        try:
            while time.time() - start_time < self.timeout:
                try:
                    check_count += 1
                    elapsed = int(time.time() - start_time)
                    self.logger.debug(f"第{check_count}次检查登录状态 (已等待{elapsed}秒)...")

                    result = self.check_login_status(qr_token)

                    if result is not None:
                        self.logger.debug(f"收到API响应: {result}")

                        if self._is_login_success(result):
                            self._stop_countdown_display()
                            self.logger.info("登录成功")
                            # 登录成功，保存cookies
                            self._save_login_result(result)
                            return True
                        elif self._is_login_failed(result):
                            self._stop_countdown_display()
                            self.logger.error("登录失败")
                            return False
                        else:
                            self.logger.debug("登录状态未知，继续等待...")
                    else:
                        self.logger.debug("等待扫码中...")

                    # 等待一段时间后再次检查
                    time.sleep(2)

                except Exception as e:
                    self.logger.error(f"等待登录时出错: {e}")
                    time.sleep(2)

            self._stop_countdown_display()
            self.logger.error(f"登录超时")
            return False

        finally:
            # 确保倒计时停止
            self._stop_countdown_display()

    def _save_login_result(self, result: Dict):
        """保存登录结果"""
        try:
            # 保存原始结果
            result_file = self.config_dir / "login_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            self.logger.debug(f"登录结果已保存到: {result_file}")

            # 提取登录token (service ticket)
            service_ticket = result.get('data', {}).get('members', {}).get('service_ticket')
            if service_ticket:
                self.logger.debug(f"获取到service ticket: {service_ticket}")

                # 使用service ticket获取用户信息和Cookie
                self._get_user_info_and_cookies(service_ticket)

        except Exception as e:
            self.logger.error(f"保存登录结果失败: {e}")

    def _get_user_info_and_cookies(self, service_ticket: str):
        """
        使用service ticket获取用户信息和设置Cookie

        Args:
            service_ticket: 登录成功后获取的service ticket
        """
        try:
            # 调用用户信息API，这会设置登录Cookie
            api_url = 'https://pan.quark.cn/account/info'
            params = {
                'st': service_ticket,
                'lw': 'scan'  # 登录方式为扫码
            }

            self.logger.debug(f"获取用户信息: {api_url}")
            response = self.client.get(api_url, params=params)

            if response.status_code == 200:
                user_info = response.json()
                self.logger.debug("用户信息获取成功")

                # 保存用户信息
                user_info_file = self.config_dir / "user_info.json"
                with open(user_info_file, 'w', encoding='utf-8') as f:
                    json.dump(user_info, f, ensure_ascii=False, indent=2)

                self.logger.debug(f"用户信息已保存到: {user_info_file}")

                # 此时Cookie应该已经被自动设置到client中
                self.logger.debug("登录Cookie已设置")

            else:
                raise Exception(f"获取用户信息失败: {response.status_code}")

        except Exception as e:
            self.logger.error(f"获取用户信息失败: {e}")
            raise

    def login(self) -> str:
        """
        执行API登录流程

        Returns:
            Cookie字符串
        """
        try:
            # 获取二维码
            qr_token, qr_url = self.get_qr_code()

            print("请使用夸克APP扫描二维码登录...")

            # 显示二维码 - 直接从URL生成ASCII二维码
            try:
                display_qr_from_url(qr_url)
            except Exception as e:
                self.logger.warning(f"显示ASCII二维码失败: {e}")
                print(f"请在浏览器中打开: {qr_url}")

            # 等待登录
            if self.wait_for_login(qr_token):
                # 从client中提取cookies
                cookies = []
                for cookie in self.client.cookies.jar:
                    # 只提取夸克相关的重要Cookie
                    if cookie.domain and 'quark.cn' in cookie.domain:
                        cookies.append(f"{cookie.name}={cookie.value}")

                cookie_string = "; ".join(cookies)
                self.logger.debug(f"提取到Cookie: {len(cookies)}个")

                return cookie_string
            else:
                raise AuthenticationError("登录超时或失败")

        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError(f"API登录失败: {e}")


def api_login(timeout: int = 300) -> str:
    """
    API登录便捷函数

    Args:
        timeout: 超时时间（秒）

    Returns:
        Cookie字符串
    """
    login_manager = APILogin(timeout=timeout)
    return login_manager.login()
