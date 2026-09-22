"""
文件名解析服务
"""

from typing import Dict, List, Optional, Tuple

from ..exceptions import APIError


class NameResolver:
    """文件名到ID的解析器"""

    def __init__(self, file_service):
        self.file_service = file_service
        self._cache = {}  # 缓存文件列表
        self._cache_folder_id = None
        self._name_cache = {}  # 缓存文件ID到真实名称的映射

    def resolve_path(self, path: str, current_folder_id: str = "0") -> Tuple[str, str]:
        """
        解析路径到文件ID和类型

        Args:
            path: 文件路径，支持以下格式：
                  - "文件名.txt" (在当前目录查找)
                  - "/文件夹/文件名.txt" (从根目录开始的绝对路径)
                  - "文件夹/文件名.txt" (相对路径)
                  - "文件夹名/" (文件夹，末尾带/)
            current_folder_id: 当前文件夹ID

        Returns:
            (file_id, file_type) - file_type为'file'或'folder'
        """
        if not path:
            raise ValueError("路径不能为空")

        # 处理绝对路径
        if path.startswith('/'):
            current_folder_id = "0"
            path = path[1:]  # 移除开头的/

        # 如果路径为空（只有/），返回根目录
        if not path:
            return "0", "folder"

        # 分割路径
        parts = [p for p in path.split('/') if p]

        # 逐级解析路径
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)

            # 如果是最后一部分且以/结尾，说明是文件夹
            if is_last and path.endswith('/'):
                file_id = self._find_in_folder(part, current_folder_id, 'folder')
                return file_id, 'folder'

            # 查找当前部分
            if is_last:
                # 最后一部分，可能是文件或文件夹
                try:
                    # 先尝试作为文件查找
                    file_id = self._find_in_folder(part, current_folder_id, 'file')
                    return file_id, 'file'
                except APIError:
                    # 如果找不到文件，尝试作为文件夹
                    file_id = self._find_in_folder(part, current_folder_id, 'folder')
                    return file_id, 'folder'
            else:
                # 中间部分，必须是文件夹
                current_folder_id = self._find_in_folder(part, current_folder_id, 'folder')

        raise APIError(f"无法解析路径: {path}")

    def _find_in_folder(self, name: str, folder_id: str, expected_type: Optional[str] = None) -> str:
        """
        在指定文件夹中查找文件或文件夹

        Args:
            name: 文件/文件夹名称
            folder_id: 文件夹ID
            expected_type: 期望的类型 ('file', 'folder', None表示任意)

        Returns:
            文件ID
        """
        # 总是刷新缓存以确保数据最新
        self._refresh_cache(folder_id)

        # 在缓存中查找
        for file_info in self._cache.get(folder_id, []):
            file_name = file_info.get('file_name', '')
            file_type = 'folder' if file_info.get('file_type') == 0 else 'file'
            file_id = file_info.get('fid', '')

            if file_name == name:
                if expected_type is None or file_type == expected_type:
                    # 缓存真实的文件名
                    self._name_cache[file_id] = file_name
                    return file_id

        # 如果没找到，抛出异常
        type_desc = f"{expected_type}类型的" if expected_type else ""
        raise APIError(f"在文件夹中找不到{type_desc}文件: {name}")

    def _refresh_cache(self, folder_id: str):
        """刷新指定文件夹的缓存"""
        try:
            # 获取文件夹内容
            result = self.file_service.list_files(folder_id, size=1000)
            file_list = result.get('data', {}).get('list', [])

            # 更新缓存
            self._cache[folder_id] = file_list
            self._cache_folder_id = folder_id

        except Exception as e:
            raise APIError(f"无法获取文件夹内容: {e}")

    def resolve_multiple_paths(self, paths: List[str], current_folder_id: str = "0") -> List[Tuple[str, str, str]]:
        """
        解析多个路径

        Args:
            paths: 路径列表
            current_folder_id: 当前文件夹ID

        Returns:
            [(file_id, file_type, original_path), ...] 列表
        """
        results = []
        for path in paths:
            try:
                file_id, file_type = self.resolve_path(path, current_folder_id)
                results.append((file_id, file_type, path))
            except Exception as e:
                raise APIError(f"解析路径 '{path}' 失败: {e}")

        return results

    def get_file_info_by_name(self, name: str, folder_id: str = "0") -> Dict:
        """
        根据文件名获取文件信息

        Args:
            name: 文件名
            folder_id: 文件夹ID

        Returns:
            文件信息字典
        """
        file_id = self._find_in_folder(name, folder_id)
        return self.file_service.get_file_info(file_id)

    def list_folder_contents(self, folder_id: str = "0") -> List[str]:
        """
        列出文件夹内容的名称列表

        Args:
            folder_id: 文件夹ID

        Returns:
            文件名列表
        """
        if self._cache_folder_id != folder_id:
            self._refresh_cache(folder_id)

        names = []
        for file_info in self._cache.get(folder_id, []):
            file_name = file_info.get('file_name', '')
            file_type = file_info.get('file_type', 1)

            # 文件夹名称后面加/
            if file_type == 0:
                names.append(f"{file_name}/")
            else:
                names.append(file_name)

        return sorted(names)

    def get_real_name(self, file_id: str) -> Optional[str]:
        """
        获取文件的真实名称（从列表缓存中获取）

        Args:
            file_id: 文件ID

        Returns:
            真实的文件名，如果没有缓存则返回None
        """
        return self._name_cache.get(file_id)

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        self._cache_folder_id = None
        self._name_cache.clear()
