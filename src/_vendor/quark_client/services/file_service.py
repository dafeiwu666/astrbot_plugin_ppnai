# -*- coding: utf-8 -*-
"""
文件管理服务
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from ..core.api_client import QuarkAPIClient
from ..exceptions import APIError, FileNotFoundError


class FileService:
    """文件管理服务"""

    def __init__(self, client: QuarkAPIClient):
        """
        初始化文件服务

        Args:
            client: API客户端实例
        """
        self.client = client

    def list_files(
        self,
        folder_id: str = "0",
        page: int = 1,
        size: int = 50,
        sort_field: str = "file_name",
        sort_order: str = "asc"
    ) -> Dict[str, Any]:
        """
        获取文件列表

        Args:
            folder_id: 文件夹ID，"0"表示根目录
            page: 页码，从1开始
            size: 每页数量
            sort_field: 排序字段 (file_name, file_size, updated_at等)
            sort_order: 排序方向 (asc, desc)

        Returns:
            包含文件列表的字典
        """
        params = {
            'pdir_fid': folder_id,
            '_page': page,
            '_size': size,
            '_sort': f"{sort_field}:{sort_order}"
        }

        try:
            response = self.client.get('file/sort', params=params)
            return response
        except APIError as e:
            if 'not found' in str(e).lower():
                raise FileNotFoundError(f"文件夹不存在: {folder_id}")
            raise

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        获取文件详细信息

        Args:
            file_id: 文件ID

        Returns:
            文件信息字典
        """
        if not file_id or file_id == "0":
            raise ValueError("无效的文件ID")

        params = {'fids': file_id}

        try:
            response = self.client.get('file', params=params)

            # 检查响应格式
            if isinstance(response, dict) and 'data' in response:
                data = response['data']
                if isinstance(data, dict) and 'list' in data:
                    file_list = data['list']
                    if file_list and len(file_list) > 0:
                        # 查找匹配的文件ID
                        for file_info in file_list:
                            if file_info.get('fid') == file_id:
                                return file_info

                        # 如果没有找到精确匹配，返回第一个
                        return file_list[0]
                elif isinstance(data, list) and len(data) > 0:
                    # 兼容旧格式
                    return data[0]

            raise FileNotFoundError(f"文件不存在: {file_id}")

        except APIError as e:
            if 'not found' in str(e).lower():
                raise FileNotFoundError(f"文件不存在: {file_id}")
            raise

    def create_folder(self, folder_name: str, parent_id: str = "0") -> Dict[str, Any]:
        """
        创建文件夹

        Args:
            folder_name: 文件夹名称
            parent_id: 父文件夹ID，"0"表示根目录

        Returns:
            创建结果
        """
        data = {
            'pdir_fid': parent_id,
            'file_name': folder_name,
            'dir_init_lock': False,
            'dir_path': ''
        }

        # 添加查询参数
        params = {
            'pr': 'ucpro',
            'fr': 'pc',
            'uc_param_str': ''
        }

        response = self.client.post('file', json_data=data, params=params)
        return response

    def delete_files(self, file_ids: List[str]) -> Dict[str, Any]:
        """
        删除文件/文件夹

        Args:
            file_ids: 文件ID列表

        Returns:
            删除结果
        """
        data = {
            'action_type': 2,  # 删除操作
            'filelist': file_ids,
            'exclude_fids': []
        }

        # 添加查询参数
        params = {
            'pr': 'ucpro',
            'fr': 'pc',
            'uc_param_str': ''
        }

        response = self.client.post('file/delete', json_data=data, params=params)
        return response

    def rename_file(self, file_id: str, new_name: str) -> Dict[str, Any]:
        """
        重命名文件/文件夹

        Args:
            file_id: 文件ID
            new_name: 新名称

        Returns:
            重命名结果
        """
        data = {
            'fid': file_id,
            'file_name': new_name
        }

        # 添加查询参数
        params = {
            'pr': 'ucpro',
            'fr': 'pc',
            'uc_param_str': ''
        }

        response = self.client.post('file/rename', json_data=data, params=params)
        return response

    def search_files(
        self,
        keyword: str,
        folder_id: str = "0",
        page: int = 1,
        size: int = 50,
        sort_field: str = "file_type",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        搜索文件

        Args:
            keyword: 搜索关键词
            folder_id: 搜索范围文件夹ID，"0"表示全盘搜索（暂不支持）
            page: 页码
            size: 每页数量
            sort_field: 排序字段
            sort_order: 排序方向

        Returns:
            搜索结果
        """
        params = {
            'q': keyword,
            '_page': page,
            '_size': size,
            '_fetch_total': 1,
            '_sort': f'{sort_field}:{sort_order},updated_at:desc',
            '_is_hl': 1  # 启用高亮
        }

        # 注意：夸克网盘的搜索API似乎不支持文件夹范围限制
        _ = folder_id  # folder_id参数暂时不使用

        response = self.client.get('file/search', params=params)
        return response

    def get_folder_tree(self, folder_id: str = "0", max_depth: int = 3) -> Dict[str, Any]:
        """
        获取文件夹树结构

        Args:
            folder_id: 根文件夹ID
            max_depth: 最大深度

        Returns:
            文件夹树结构
        """
        params = {
            'pdir_fid': folder_id,
            'max_depth': max_depth
        }

        response = self.client.get('file/tree', params=params)
        return response

    def get_storage_info(self) -> Dict[str, Any]:
        """
        获取存储空间信息

        Returns:
            存储空间信息
        """
        response = self.client.get('capacity')
        return response

    def list_files_with_details(
        self,
        folder_id: str = "0",
        page: int = 1,
        size: int = 50,
        sort_field: str = "file_name",
        sort_order: str = "asc",
        include_folders: bool = True,
        include_files: bool = True
    ) -> Dict[str, Any]:
        """
        获取文件列表（增强版，支持过滤）

        Args:
            folder_id: 文件夹ID，"0"表示根目录
            page: 页码，从1开始
            size: 每页数量
            sort_field: 排序字段
            sort_order: 排序方向
            include_folders: 是否包含文件夹
            include_files: 是否包含文件

        Returns:
            包含文件列表的字典
        """
        response = self.list_files(folder_id, page, size, sort_field, sort_order)

        # 如果需要过滤，则处理响应数据
        if not include_folders or not include_files:
            if isinstance(response, dict) and 'data' in response:
                file_list = response['data'].get('list', [])
                filtered_list = []

                for file_info in file_list:
                    file_type = file_info.get('file_type', 0)
                    is_folder = file_type == 0

                    if (is_folder and include_folders) or (not is_folder and include_files):
                        filtered_list.append(file_info)

                response['data']['list'] = filtered_list
                response['data']['filtered_total'] = len(filtered_list)

        return response

    def search_files_advanced(
        self,
        keyword: str,
        folder_id: str = "0",
        page: int = 1,
        size: int = 50,
        file_extensions: Optional[List[str]] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        sort_field: str = "file_type",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        高级文件搜索（客户端过滤）

        Args:
            keyword: 搜索关键词
            folder_id: 搜索范围文件夹ID（暂不支持）
            page: 页码
            size: 每页数量
            file_extensions: 文件扩展名过滤 (如: ['pdf', 'doc', 'txt'])
            min_size: 最小文件大小（字节）
            max_size: 最大文件大小（字节）
            sort_field: 排序字段
            sort_order: 排序方向

        Returns:
            搜索结果
        """
        # 如果没有过滤条件，直接返回基础搜索结果
        if not file_extensions and min_size is None and max_size is None:
            return self.search_files(keyword, folder_id, page, size, sort_field, sort_order)

        # 获取更多结果用于客户端过滤
        search_size = max(size * 3, 100)
        response = self.search_files(keyword, folder_id, 1, search_size, sort_field, sort_order)

        # 应用客户端过滤器
        if isinstance(response, dict) and 'data' in response:
            file_list = response['data'].get('list', [])
            filtered_list = []

            for file_info in file_list:
                # 文件扩展名过滤
                if file_extensions:
                    file_name = file_info.get('file_name', '').lower()
                    file_ext = file_name.split('.')[-1] if '.' in file_name else ''
                    if file_ext not in [ext.lower() for ext in file_extensions]:
                        continue

                # 文件大小过滤
                file_size = file_info.get('size', 0)
                if min_size is not None and file_size < min_size:
                    continue
                if max_size is not None and file_size > max_size:
                    continue

                filtered_list.append(file_info)

            # 应用分页到过滤后的结果
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paginated_list = filtered_list[start_idx:end_idx]

            response['data']['list'] = paginated_list
            response['data']['filtered_total'] = len(filtered_list)
            # 更新metadata中的总数
            if 'metadata' in response:
                response['metadata']['_total'] = len(filtered_list)
                response['metadata']['_count'] = len(paginated_list)

        return response

    def get_file_path(self, file_id: str) -> str:
        """
        获取文件的完整路径

        Args:
            file_id: 文件ID

        Returns:
            文件路径字符串
        """
        try:
            file_info = self.get_file_info(file_id)
            # 这里需要根据实际API响应结构来获取路径
            # 可能需要递归获取父文件夹信息来构建完整路径
            return file_info.get('file_path', file_info.get('file_name', ''))
        except Exception:
            return ""

    def move_files(
        self,
        file_ids: List[str],
        target_folder_id: str,
        exclude_fids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        移动文件到指定文件夹

        Args:
            file_ids: 要移动的文件ID列表
            target_folder_id: 目标文件夹ID
            exclude_fids: 排除的文件ID列表

        Returns:
            移动结果
        """
        data = {
            'action_type': 1,  # 移动操作
            'to_pdir_fid': target_folder_id,
            'filelist': file_ids,
            'exclude_fids': exclude_fids or []
        }

        response = self.client.post('file/move', json_data=data)

        if not response.get('status') == 200:
            raise APIError(f"移动文件失败: {response.get('message', '未知错误')}")

        data = response.get('data', {})
        task_id = data.get('task_id')
        finish = data.get('finish', False)

        if finish:
            # 同步完成，直接返回结果
            return response
        elif task_id:
            # 异步任务，需要轮询状态
            return self._wait_for_move_task(task_id, response.get('metadata', {}).get('tq_gap', 500))
        else:
            raise APIError("移动任务创建失败，无法获取任务ID")

    def _wait_for_move_task(self, task_id: str, poll_interval: int = 500) -> Dict[str, Any]:
        """
        等待移动任务完成

        Args:
            task_id: 任务ID
            poll_interval: 轮询间隔（毫秒）

        Returns:
            任务完成结果
        """
        import time

        max_retries = 30  # 最多等待15秒
        retry_count = 0

        while retry_count < max_retries:
            try:
                task_response = self.client.get(
                    'task',
                    params={
                        'task_id': task_id,
                        'retry_index': retry_count
                    }
                )

                if task_response.get('status') == 200:
                    task_data = task_response.get('data', {})

                    # 检查任务状态
                    if task_data.get('status') == 2:  # 任务完成
                        return task_response
                    elif task_data.get('status') == 3:  # 任务失败
                        raise APIError(f"移动任务失败: {task_data.get('message', '任务失败')}")

                retry_count += 1
                time.sleep(poll_interval / 1000)  # 转换为秒

            except Exception as e:
                if retry_count >= max_retries - 1:
                    raise APIError(f"移动任务轮询失败: {e}")
                retry_count += 1
                time.sleep(poll_interval / 1000)

        raise APIError("移动任务超时")

    def resolve_path(self, path: str, current_dir_id: str = "0") -> Tuple[str, bool]:
        """
        解析文件路径，返回文件ID和是否为文件夹

        Args:
            path: 文件路径，如 "/L2-2/L23-1/民间秘术绝招大观.pdf" 或 "/L2-2/L23-1/"
            current_dir_id: 当前目录ID，默认为根目录

        Returns:
            (file_id, is_directory) 元组

        Raises:
            FileNotFoundError: 文件或路径不存在
        """
        # 标准化路径
        path = path.strip()
        if not path:
            return current_dir_id, True

        # 处理根目录
        if path == "/":
            return "0", True

        # 移除开头的斜杠并分割路径
        if path.startswith("/"):
            path = path[1:]
            current_dir_id = "0"  # 绝对路径从根目录开始

        if path.endswith("/"):
            path = path[:-1]
            is_target_dir = True  # 明确指定为目录
        else:
            is_target_dir = False  # 可能是文件或目录

        if not path:
            return current_dir_id, True

        path_parts = path.split("/")

        # 逐级查找路径
        for i, part in enumerate(path_parts):
            is_last_part = (i == len(path_parts) - 1)

            # 获取当前目录的文件列表
            response = self.list_files(current_dir_id)
            if response.get('status') != 200:
                raise FileNotFoundError(f"无法访问目录: {'/'.join(path_parts[:i+1])}")

            files_data = response.get('data', {})
            files_list = files_data.get('list', [])

            # 查找匹配的文件或文件夹
            found = False
            for file_info in files_list:
                if file_info['file_name'] == part:
                    found = True
                    current_dir_id = file_info['fid']

                    if is_last_part:
                        # 最后一个部分，确定是文件还是目录
                        if is_target_dir:
                            # 明确指定为目录（路径以/结尾）
                            if not file_info.get('dir', False):
                                raise FileNotFoundError(f"路径 '{path}' 指向的是文件，不是目录")
                            return current_dir_id, True
                        else:
                            # 可能是文件或目录
                            return current_dir_id, file_info.get('dir', False)
                    else:
                        # 中间路径，必须是目录
                        if not file_info.get('dir', False):
                            raise FileNotFoundError(f"路径 '{'/'.join(path_parts[:i+1])}' 不是目录")
                    break

            if not found:
                raise FileNotFoundError(f"路径不存在: {'/'.join(path_parts[:i+1])}")

        return current_dir_id, True

    def find_files_by_pattern(self, pattern: str, dir_id: str = "0") -> List[Dict[str, Any]]:
        """
        在指定目录中查找匹配模式的文件

        Args:
            pattern: 文件名模式（支持通配符）
            dir_id: 搜索的目录ID

        Returns:
            匹配的文件列表
        """
        import fnmatch

        response = self.list_files(dir_id)
        if response.get('status') != 200:
            return []

        files_data = response.get('data', {})
        files_list = files_data.get('list', [])

        matched_files = []
        for file_info in files_list:
            if fnmatch.fnmatch(file_info['file_name'], pattern):
                matched_files.append(file_info)

        return matched_files

    def get_download_urls(self, file_ids: List[str]) -> Dict[str, Any]:
        """
        获取文件下载链接

        Args:
            file_ids: 文件ID列表

        Returns:
            包含下载链接的响应数据
        """
        data = {'fids': file_ids}

        params = {
            'pr': 'ucpro',
            'fr': 'pc',
            'sys': 'win32',
            've': '2.5.56',
            'ut': '',
            'guid': '',
        }

        # 使用正确的 API 端点
        response = self.client.post(
            'file/download',
            json_data=data,
            params=params
        )

        return response

    def _generate_safe_filename(self, filepath: str) -> str:
        """
        生成安全的文件名，处理文件冲突

        Args:
            filepath: 原始文件路径

        Returns:
            安全的文件路径
        """
        if not os.path.exists(filepath):
            return filepath

        # 分离文件名和扩展名
        dir_path = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if '.' in filename:
            name, ext = os.path.splitext(filename)
        else:
            name, ext = filename, ''

        # 自动递增文件名
        counter = 1
        while True:
            new_filename = f"{name}{counter}{ext}"
            new_filepath = os.path.join(dir_path, new_filename)
            if not os.path.exists(new_filepath):
                return new_filepath
            counter += 1

    def _build_folder_path_map(self, files_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        构建文件夹路径映射

        Args:
            files_data: 文件数据列表

        Returns:
            文件夹ID到路径的映射
        """
        folder_map = {}

        for file_info in files_data:
            if file_info.get('dir'):
                fid = file_info['fid']
                pdir_fid = file_info.get('pdir_fid', '0')
                file_name = file_info['file_name']

                # 构建完整路径
                if pdir_fid == '0' or pdir_fid not in folder_map:
                    folder_map[fid] = file_name
                else:
                    parent_path = folder_map[pdir_fid]
                    folder_map[fid] = os.path.join(parent_path, file_name)

        return folder_map

    def download_file(self, file_path: str, save_dir: str = ".", progress_callback=None) -> str:
        """
        下载单个文件（支持路径）

        Args:
            file_path: 文件路径或文件ID
            save_dir: 保存目录
            progress_callback: 进度回调函数

        Returns:
            下载后的本地文件路径
        """
        # 解析路径获取文件ID
        if len(file_path) == 32 and file_path.isalnum():
            # 看起来是文件ID
            file_id = file_path
            # 获取文件信息
            file_info = self.get_file_info(file_id)
            if not file_info:
                raise FileNotFoundError(f"文件ID不存在: {file_id}")
            filename = file_info.get('file_name', 'unknown')
        else:
            # 路径格式
            file_id, is_dir = self.resolve_path(file_path)
            if is_dir:
                raise APIError(f"路径指向目录，请使用 download_folder 方法: {file_path}")

            # 获取文件信息
            file_info = self.get_file_info(file_id)
            if not file_info:
                raise FileNotFoundError(f"文件不存在: {file_path}")
            filename = file_info['file_name']

        # 获取下载链接
        download_response = self.get_download_urls([file_id])
        if download_response.get('status') != 200:
            raise APIError(f"获取下载链接失败: {download_response.get('message', '未知错误')}")

        download_data = download_response.get('data', [])
        if not download_data:
            raise APIError("未获取到下载链接")

        file_data = download_data[0]
        download_url = file_data.get('download_url')
        if not download_url:
            raise APIError("下载链接为空")

        # 准备保存路径
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        safe_save_path = self._generate_safe_filename(save_path)

        # 下载文件
        self._download_file_stream(download_url, safe_save_path, progress_callback)

        return safe_save_path

    def download_folder(self, folder_path: str, save_dir: str = ".", progress_callback=None) -> str:
        """
        下载文件夹（递归下载所有内容）

        Args:
            folder_path: 文件夹路径或文件夹ID
            save_dir: 保存目录
            progress_callback: 进度回调函数

        Returns:
            下载后的本地文件夹路径
        """
        # 解析路径获取文件夹ID
        if len(folder_path) == 32 and folder_path.isalnum():
            # 看起来是文件夹ID
            folder_id = folder_path
            # 获取文件夹信息
            folder_info = self.get_file_info(folder_id)
            if not folder_info:
                raise FileNotFoundError(f"文件夹ID不存在: {folder_id}")
            folder_name = folder_info.get('file_name', 'unknown')
        else:
            # 路径格式
            folder_id, is_dir = self.resolve_path(folder_path)
            if not is_dir:
                raise APIError(f"路径指向文件，请使用 download_file 方法: {folder_path}")

            # 获取文件夹信息
            folder_info = self.get_file_info(folder_id)
            if not folder_info:
                raise FileNotFoundError(f"文件夹不存在: {folder_path}")
            folder_name = folder_info['file_name']

        # 创建本地文件夹
        local_folder_path = os.path.join(save_dir, folder_name)
        os.makedirs(local_folder_path, exist_ok=True)

        # 递归下载文件夹内容
        self._download_folder_recursive(folder_id, local_folder_path, progress_callback)

        return local_folder_path

    def _download_folder_recursive(self, folder_id: str, local_path: str, progress_callback=None):
        """
        递归下载文件夹内容

        Args:
            folder_id: 文件夹ID
            local_path: 本地保存路径
            progress_callback: 进度回调函数
        """
        # 获取文件夹内容
        response = self.list_files(folder_id)
        if response.get('status') != 200:
            if progress_callback:
                progress_callback('error', f"无法访问文件夹: {folder_id}")
            return

        files_data = response.get('data', {})
        files_list = files_data.get('list', [])

        if not files_list:
            return

        # 分离文件和文件夹
        files = [f for f in files_list if not f.get('dir', False)]
        folders = [f for f in files_list if f.get('dir', False)]

        # 下载文件
        if files:
            file_ids = [f['fid'] for f in files]

            # 获取下载链接
            download_response = self.get_download_urls(file_ids)
            if download_response.get('status') == 200:
                download_data = download_response.get('data', [])

                for file_data in download_data:
                    filename = file_data.get('file_name', 'unknown')
                    download_url = file_data.get('download_url')

                    if download_url:
                        file_save_path = os.path.join(local_path, filename)
                        safe_file_path = self._generate_safe_filename(file_save_path)

                        try:
                            self._download_file_stream(download_url, safe_file_path, progress_callback)
                            if progress_callback:
                                progress_callback('file_complete', safe_file_path)
                        except Exception as e:
                            if progress_callback:
                                progress_callback('error', f"下载文件失败 {filename}: {e}")

        # 递归处理子文件夹
        for folder in folders:
            folder_name = folder['file_name']
            folder_id = folder['fid']

            # 创建子文件夹
            sub_folder_path = os.path.join(local_path, folder_name)
            os.makedirs(sub_folder_path, exist_ok=True)

            if progress_callback:
                progress_callback('folder_start', sub_folder_path)

            # 递归下载
            self._download_folder_recursive(folder_id, sub_folder_path, progress_callback)

    def _download_file_stream(self, download_url: str, save_path: str, progress_callback=None):
        """
        流式下载文件

        Args:
            download_url: 下载链接
            save_path: 保存路径
            progress_callback: 进度回调函数
        """
        import requests
        from tqdm import tqdm

        # 使用与 reference.py 完全相同的请求头和格式
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko)'
                          ' Chrome/94.0.4606.71 Safari/537.36 Core/1.94.225.400 QQBrowser/12.2.5544.400',
            'origin': 'https://pan.quark.cn',
            'referer': 'https://pan.quark.cn/',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cookie': self.client.cookies,  # 直接设置 cookie 头部
        }

        try:
            response = requests.get(download_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            filename = os.path.basename(save_path)

            # 创建进度条
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=filename,
                ncols=80
            ) as pbar:

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                            if progress_callback:
                                progress_callback('progress', {
                                    'filename': filename,
                                    'downloaded': pbar.n,
                                    'total': total_size,
                                    'percentage': (pbar.n / total_size * 100) if total_size > 0 else 0
                                })

            if progress_callback:
                progress_callback('complete', save_path)

        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)  # 清理不完整的文件
            raise APIError(f"下载失败: {e}")
