"""
批量分享服务
"""

import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..core.api_client import QuarkAPIClient
from ..exceptions import APIError
from ..utils.logger import get_logger
from .file_service import FileService
from .share_service import ShareService


class BatchShareService:
    """批量分享服务"""

    def __init__(self, client: QuarkAPIClient):
        """
        初始化批量分享服务

        Args:
            client: API客户端实例
        """
        self.client = client
        self.file_service = FileService(client)
        self.share_service = ShareService(client)
        self.logger = get_logger(__name__)

    def collect_target_directories(self,
                                   exclude_patterns: Optional[List[str]] = None,
                                   target_dir: Optional[str] = None,
                                   depth: int = 3,
                                   share_level: str = "folders") -> List[Dict[str, Any]]:
        """
        收集所有需要分享的目标目录/文件（统一入口）

        Args:
            exclude_patterns: 排除的目录名称模式列表
            target_dir: 指定的起始目录路径（None表示根目录）
            depth: 扫描深度（默认3表示四级目录）
            share_level: 分享类型（folders/files/both）

        Returns:
            目标目录/文件列表，每个元素包含目录信息和完整路径
        """
        if exclude_patterns is None:
            exclude_patterns = ["来自：分享"]

        # 根据参数选择不同的收集策略
        if target_dir:
            # 指定目录模式
            return self.collect_directories_by_path(target_dir, depth, share_level, exclude_patterns)
        else:
            # 默认模式：保持向后兼容
            if depth == 3 and share_level == "folders":
                # 使用原有逻辑（四级目录扫描）
                return self._collect_legacy_target_directories(exclude_patterns)
            else:
                # 使用新的深度模式
                return self.collect_directories_by_depth(depth, share_level, exclude_patterns)

    def _collect_legacy_target_directories(self, exclude_patterns: List[str]) -> List[Dict[str, Any]]:
        """
        原有的四级目录收集逻辑（保持向后兼容）

        Args:
            exclude_patterns: 排除的目录名称模式列表

        Returns:
            目标目录列表，每个元素包含目录信息和完整路径
        """
        target_directories = []

        self.logger.info("开始收集目标目录...")

        # 第一级：获取根目录下的所有文件夹（二级目录）
        try:
            root_response = self.file_service.list_files(folder_id="0", size=200)
            if not root_response.get('status') == 200:
                raise APIError("无法获取根目录文件列表")

            second_level_dirs = []
            root_files = root_response.get('data', {}).get('list', [])

            for item in root_files:
                if item.get('dir', False):  # 只处理文件夹
                    dir_name = item.get('file_name', '')
                    # 检查是否需要排除
                    if not any(pattern in dir_name for pattern in exclude_patterns):
                        second_level_dirs.append({
                            'fid': item.get('fid'),
                            'name': dir_name,
                            'path': f"/{dir_name}"
                        })
                        self.logger.info(f"找到二级目录: {dir_name}")
                    else:
                        self.logger.info(f"跳过排除目录: {dir_name}")

            # 第二级：遍历每个二级目录，获取三级目录
            for second_dir in second_level_dirs:
                try:
                    second_response = self.file_service.list_files(
                        folder_id=second_dir['fid'],
                        size=200
                    )
                    if not second_response.get('status') == 200:
                        self.logger.warning(f"无法获取二级目录文件列表: {second_dir['name']}")
                        continue

                    third_level_dirs = []
                    second_files = second_response.get('data', {}).get('list', [])

                    for item in second_files:
                        if item.get('dir', False):  # 只处理文件夹
                            dir_name = item.get('file_name', '')
                            third_level_dirs.append({
                                'fid': item.get('fid'),
                                'name': dir_name,
                                'path': f"{second_dir['path']}/{dir_name}"
                            })
                            self.logger.info(f"找到三级目录: {second_dir['name']}/{dir_name}")

                    # 第三级：遍历每个三级目录，获取四级目录（目标目录）
                    for third_dir in third_level_dirs:
                        try:
                            third_response = self.file_service.list_files(
                                folder_id=third_dir['fid'],
                                size=200
                            )
                            if not third_response.get('status') == 200:
                                self.logger.warning(f"无法获取三级目录文件列表: {third_dir['name']}")
                                continue

                            third_files = third_response.get('data', {}).get('list', [])

                            for item in third_files:
                                if item.get('dir', False):  # 只处理文件夹（目标目录）
                                    target_name = item.get('file_name', '')
                                    target_path = f"{third_dir['path']}/{target_name}"

                                    target_info = {
                                        'fid': item.get('fid'),
                                        'name': target_name,
                                        'full_path': target_path,
                                        'second_level': second_dir['name'],
                                        'third_level': third_dir['name'],
                                        'file_info': item
                                    }

                                    target_directories.append(target_info)
                                    self.logger.info(f"找到目标目录: {target_path}")

                        except Exception as e:
                            self.logger.error(f"处理三级目录时出错 {third_dir['name']}: {e}")
                            continue

                except Exception as e:
                    self.logger.error(f"处理二级目录时出错 {second_dir['name']}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"获取根目录时出错: {e}")
            raise

        self.logger.info(f"总共找到 {len(target_directories)} 个目标目录")
        return target_directories

    def collect_directories_by_path(
            self, target_dir: str, depth: int, share_level: str, exclude_patterns: List[str]) -> List[
            Dict[str, Any]]:
        """
        根据指定目录路径收集子目录/文件

        Args:
            target_dir: 目标目录路径
            depth: 扫描深度（相对于目标目录的深度）
            share_level: 分享类型（folders/files/both）
            exclude_patterns: 排除模式列表

        Returns:
            目标目录/文件列表
        """
        self.logger.info(f"开始扫描指定目录: {target_dir}，深度: {depth}，类型: {share_level}")

        # 解析目标目录路径，获取目录ID
        try:
            target_folder_id = self._resolve_path_to_folder_id(target_dir)
            if not target_folder_id:
                self.logger.error(f"无法找到目录: {target_dir}")
                return []

            # 处理路径格式，确保以/开头
            normalized_path = target_dir if target_dir.startswith('/') else '/' + target_dir

            # 从指定目录开始递归收集
            return self._collect_items_recursive(
                folder_id=target_folder_id,
                current_path=normalized_path,
                current_depth=0,  # 从指定目录开始，深度重新计算
                max_depth=depth,
                share_level=share_level,
                exclude_patterns=exclude_patterns
            )

        except Exception as e:
            self.logger.error(f"扫描指定目录失败 {target_dir}: {e}")
            return []

    def collect_directories_by_depth(self, depth: int, share_level: str, exclude_patterns: List[str]) -> List[Dict[str, Any]]:
        """
        根据指定深度从根目录收集目录/文件

        Args:
            depth: 扫描深度
            share_level: 分享类型（folders/files/both）
            exclude_patterns: 排除模式列表

        Returns:
            目标目录/文件列表
        """
        self.logger.info(f"开始扫描根目录，深度: {depth}，类型: {share_level}")

        try:
            # 从根目录开始递归收集
            return self._collect_items_recursive(
                folder_id="0",
                current_path="/",
                current_depth=0,
                max_depth=depth,
                share_level=share_level,
                exclude_patterns=exclude_patterns
            )

        except Exception as e:
            self.logger.error(f"按深度扫描失败: {e}")
            return []

    def _collect_items_recursive(self, folder_id: str, current_path: str, current_depth: int,
                                 max_depth: int, share_level: str, exclude_patterns: List[str]) -> List[Dict[str, Any]]:
        """
        递归收集目录/文件

        Args:
            folder_id: 当前文件夹ID
            current_path: 当前路径
            current_depth: 当前深度
            max_depth: 最大深度
            share_level: 分享类型
            exclude_patterns: 排除模式列表

        Returns:
            收集到的项目列表
        """
        items = []

        if current_depth >= max_depth:
            # 达到指定深度，收集该层的项目
            try:
                response = self.file_service.list_files(folder_id=folder_id, size=200)
                if response.get('status') != 200:
                    self.logger.warning(f"无法获取文件夹内容: {current_path}")
                    return items

                file_list = response.get('data', {}).get('list', [])

                for item in file_list:
                    item_name = item.get('file_name', '')
                    is_folder = item.get('dir', False)

                    # 检查排除模式
                    if any(pattern in item_name for pattern in exclude_patterns):
                        continue

                    # 根据分享类型过滤
                    if share_level == "folders" and not is_folder:
                        continue
                    elif share_level == "files" and is_folder:
                        continue

                    # 构造项目信息
                    item_path = f"{current_path.rstrip('/')}/{item_name}"
                    if current_path == "/":
                        item_path = f"/{item_name}"

                    item_info = {
                        'fid': item.get('fid'),
                        'name': item_name,
                        'full_path': item_path,
                        'is_folder': is_folder,
                        'file_info': item,
                        'depth': current_depth
                    }

                    items.append(item_info)
                    self.logger.info(f"找到{'文件夹' if is_folder else '文件'}: {item_path}")

            except Exception as e:
                self.logger.error(f"处理文件夹时出错 {current_path}: {e}")

        else:
            # 还未达到指定深度，继续递归
            try:
                response = self.file_service.list_files(folder_id=folder_id, size=200)
                if response.get('status') != 200:
                    self.logger.warning(f"无法获取文件夹内容: {current_path}")
                    return items

                file_list = response.get('data', {}).get('list', [])

                # 只处理文件夹，继续递归
                for item in file_list:
                    if not item.get('dir', False):
                        continue  # 跳过文件

                    folder_name = item.get('file_name', '')

                    # 检查排除模式
                    if any(pattern in folder_name for pattern in exclude_patterns):
                        self.logger.info(f"跳过排除文件夹: {folder_name}")
                        continue

                    # 构造子文件夹路径
                    sub_path = f"{current_path.rstrip('/')}/{folder_name}"
                    if current_path == "/":
                        sub_path = f"/{folder_name}"

                    # 递归处理子文件夹
                    sub_items = self._collect_items_recursive(
                        folder_id=item.get('fid'),
                        current_path=sub_path,
                        current_depth=current_depth + 1,
                        max_depth=max_depth,
                        share_level=share_level,
                        exclude_patterns=exclude_patterns
                    )

                    items.extend(sub_items)

            except Exception as e:
                self.logger.error(f"递归处理文件夹时出错 {current_path}: {e}")

        return items

    def _resolve_path_to_folder_id(self, path: str) -> Optional[str]:
        """
        将路径解析为文件夹ID

        Args:
            path: 文件夹路径（支持绝对路径和相对路径）

        Returns:
            文件夹ID，如果未找到返回None
        """
        if path == "/" or path == "":
            return "0"  # 根目录

        # 处理相对路径：如果不以/开头，则添加/
        if not path.startswith('/'):
            path = '/' + path

        # 移除开头和结尾的斜杠，然后分割
        path = path.strip('/')
        if not path:  # 如果处理后为空，说明是根目录
            return "0"

        path_parts = path.split('/')

        current_folder_id = "0"  # 从根目录开始

        for i, part in enumerate(path_parts):
            if not part:
                continue

            try:
                # 获取当前文件夹的内容
                response = self.file_service.list_files(folder_id=current_folder_id, size=200)
                if response.get('status') != 200:
                    self.logger.error(f"无法访问文件夹: {'/' if i == 0 else '/'.join(path_parts[:i])}")
                    return None

                file_list = response.get('data', {}).get('list', [])

                # 查找匹配的子文件夹
                found = False
                for item in file_list:
                    if item.get('dir', False) and item.get('file_name', '') == part:
                        current_folder_id = item.get('fid')
                        found = True
                        self.logger.info(f"找到路径段: {part} -> {current_folder_id}")
                        break

                if not found:
                    self.logger.error(f"路径中找不到文件夹: {part} (在 {'/' if i == 0 else '/' + '/'.join(path_parts[:i])})")
                    return None

            except Exception as e:
                self.logger.error(f"解析路径时出错在: {part}, 错误: {e}")
                return None

        self.logger.info(f"成功解析路径 {path} -> {current_folder_id}")
        return current_folder_id

    def create_batch_shares(self, target_directories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量创建分享链接

        Args:
            target_directories: 目标目录列表

        Returns:
            分享结果列表
        """
        share_results = []
        total = len(target_directories)

        self.logger.info(f"开始批量创建分享，共 {total} 个目录")

        for i, target_dir in enumerate(target_directories, 1):
            try:
                self.logger.info(f"正在创建分享 ({i}/{total}): {target_dir['full_path']}")

                # 创建分享
                share_result = self.share_service.create_share(
                    file_ids=[target_dir['fid']],
                    title=target_dir['name'],  # 使用目录名作为分享标题
                    expire_days=0,  # 永久
                    password=None   # 无密码
                )

                if share_result:
                    # 添加额外信息到结果中
                    share_info = {
                        'target_directory': target_dir,
                        'share_result': share_result,
                        'share_title': target_dir['name'],
                        'share_url': share_result.get('share_url', ''),
                        'share_id': share_result.get('pwd_id', ''),
                        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'success': True
                    }
                    share_results.append(share_info)
                    self.logger.info(f"分享创建成功: {target_dir['name']} -> {share_result.get('share_url', '')}")
                else:
                    # 分享失败
                    share_info = {
                        'target_directory': target_dir,
                        'share_result': None,
                        'share_title': target_dir['name'],
                        'share_url': '',
                        'share_id': '',
                        'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'success': False,
                        'error': '分享创建失败'
                    }
                    share_results.append(share_info)
                    self.logger.error(f"分享创建失败: {target_dir['name']}")

            except Exception as e:
                # 记录错误并继续
                share_info = {
                    'target_directory': target_dir,
                    'share_result': None,
                    'share_title': target_dir['name'],
                    'share_url': '',
                    'share_id': '',
                    'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'success': False,
                    'error': str(e)
                }
                share_results.append(share_info)
                self.logger.error(f"创建分享时出错 {target_dir['name']}: {e}")

        successful = sum(1 for result in share_results if result['success'])
        self.logger.info(f"批量分享完成: 成功 {successful}/{total}")

        return share_results

    def export_to_csv(self, share_results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        导出分享结果到CSV文件

        Args:
            share_results: 分享结果列表
            filename: CSV文件名，如果不指定则自动生成

        Returns:
            CSV文件路径
        """
        if filename is None:
            today = datetime.now().strftime('%Y-%m-%d')
            filename = f"shares_{today}.csv"

        # 确保文件名以.csv结尾
        if not filename.endswith('.csv'):
            filename += '.csv'

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # 写入标题行
                headers = ['分享标题', '分享链接', '完整路径', '创建时间']
                writer.writerow(headers)

                # 写入数据行
                for result in share_results:
                    if result['success']:
                        row = [
                            result['share_title'],
                            result['share_url'],
                            result['target_directory']['full_path'],
                            result['created_time']
                        ]
                        writer.writerow(row)
                    else:
                        # 对于失败的分享，也记录到CSV中，但链接为空
                        row = [
                            result['share_title'],
                            f"创建失败: {result.get('error', '未知错误')}",
                            result['target_directory']['full_path'],
                            result['created_time']
                        ]
                        writer.writerow(row)

            self.logger.info(f"CSV文件已保存: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"保存CSV文件时出错: {e}")
            raise

    def batch_share_and_export(self, csv_filename: Optional[str] = None, exclude_patterns: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], str]:
        """
        一站式批量分享和导出服务

        Args:
            csv_filename: CSV文件名
            exclude_patterns: 排除的目录名称模式列表

        Returns:
            (分享结果列表, CSV文件路径)
        """
        # 1. 收集目标目录
        self.logger.info("🔍 开始收集目标目录...")
        target_directories = self.collect_target_directories(exclude_patterns)

        if not target_directories:
            self.logger.warning("没有找到任何目标目录")
            return [], ""

        # 2. 批量创建分享
        self.logger.info("📤 开始批量创建分享...")
        share_results = self.create_batch_shares(target_directories)

        # 3. 导出到CSV
        self.logger.info("📊 开始导出CSV文件...")
        csv_path = self.export_to_csv(share_results, csv_filename)

        # 4. 统计信息
        successful = sum(1 for result in share_results if result['success'])
        failed = len(share_results) - successful

        self.logger.info(f"✅ 批量分享完成!")
        self.logger.info(f"   总计: {len(share_results)} 个目录")
        self.logger.info(f"   成功: {successful} 个")
        self.logger.info(f"   失败: {failed} 个")
        self.logger.info(f"   CSV文件: {csv_path}")

        return share_results, csv_path
