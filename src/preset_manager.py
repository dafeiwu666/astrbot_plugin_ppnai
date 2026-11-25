"""预设提示词管理模块"""

import asyncio
import json
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, Field

from .drawing_recipe import DrawingRecipe, write_recipe_file
from .utils import match_resource_names


class Preset(DrawingRecipe):
    """单个预设"""
    title: str = Field(description="预设标题")
    owner_id: str = Field(default="system", description="创建者标识")


class PresetStore(BaseModel):
    """预设数据存储"""
    presets: dict[str, Preset] = Field(default_factory=dict, description="预设列表")


class PresetManager:
    """预设管理器"""
    
    def __init__(self, data_dir: Path):
        self.data_file = data_dir / "presets.json"
        self.data_dir = data_dir
        self._store: PresetStore | None = None
        self._lock = RLock()
    
    def _ensure_dir(self):
        """确保数据目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load(self) -> PresetStore:
        """加载预设数据"""
        with self._lock:
            if self._store is not None:
                return self._store

            self._ensure_dir()
            if self.data_file.exists():
                data = json.loads(self.data_file.read_text("utf-8"))
                self._store = PresetStore.model_validate(data)
            else:
                self._store = PresetStore()
            return self._store
    
    def _save(self, presets: dict[str, Preset]) -> None:
        store = PresetStore(presets=presets)
        write_recipe_file(self.data_file, store.model_dump_json(indent=2, exclude_unset=True))
        self._store = store
    
    def list_presets(self, owner_id: str | None = None) -> list[str]:
        """获取所有预设标题列表"""
        return [
            title
            for title, preset in self._load().presets.items()
            if owner_id is None or preset.owner_id == owner_id
        ]

    def list_grouped(self, owner_id: str | None = None) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for title, preset in self._load().presets.items():
            if owner_id is not None and preset.owner_id not in {owner_id, "system"}:
                continue
            grouped.setdefault(preset.owner_id, []).append(title)
        return {key: sorted(value) for key, value in sorted(grouped.items())}

    def find_matching(self, query: str, owner_id: str | None = None) -> list[str]:
        titles = (title for group in self.list_grouped(owner_id).values() for title in group)
        return match_resource_names(titles, query)
    
    def get_preset(
        self, title: str, owner_id: str | None = None
    ) -> Preset | None:
        """获取预设；指定 owner_id 时只允许读取本人或系统预设。"""
        store = self._load()
        preset = store.presets.get(title)
        if (
            preset is not None
            and owner_id is not None
            and preset.owner_id not in {owner_id, "system"}
        ):
            return None
        return preset
    
    def add_preset(self, title: str, content: str | DrawingRecipe, owner_id: str = "system") -> bool:
        """
        添加预设
        返回: True 表示添加成功，False 表示已存在同名预设
        """
        with self._lock:
            presets = self._load().presets
            if title in presets:
                return False
            recipe = DrawingRecipe(content=content) if isinstance(content, str) else content
            preset = Preset(title=title, owner_id=owner_id, **recipe.model_dump(exclude_unset=True))
            self._save({**presets, title: preset})
        return True
    
    def update_preset(
        self, title: str, content: str | DrawingRecipe, owner_id: str | None = None
    ) -> bool:
        """
        更新预设
        返回: True 表示更新成功，False 表示预设不存在
        """
        with self._lock:
            presets = self._load().presets
            preset = presets.get(title)
            if preset is None or (owner_id is not None and preset.owner_id != owner_id):
                return False
            recipe = DrawingRecipe(content=content) if isinstance(content, str) else content
            replacement = Preset(title=title, owner_id=preset.owner_id, **recipe.model_dump(exclude_unset=True))
            self._save({**presets, title: replacement})
        return True
    
    def delete_preset(self, title: str, owner_id: str | None = None) -> bool:
        """
        删除预设
        返回: True 表示删除成功，False 表示预设不存在
        """
        with self._lock:
            presets = dict(self._load().presets)
            preset = presets.get(title)
            if preset is None or (owner_id is not None and preset.owner_id != owner_id):
                return False
            del presets[title]
            self._save(presets)
        return True
    
    def reload(self):
        """重新加载数据"""
        with self._lock:
            self._store = None
            self._load()

    async def alist_presets(self) -> list[str]:
        return await asyncio.to_thread(self.list_presets)

    async def aget_preset(self, title: str) -> Preset | None:
        return await asyncio.to_thread(self.get_preset, title)

    async def aadd_preset(self, title: str, content: str) -> bool:
        return await asyncio.to_thread(self.add_preset, title, content)

    async def aupdate_preset(self, title: str, content: str) -> bool:
        return await asyncio.to_thread(self.update_preset, title, content)

    async def adelete_preset(self, title: str) -> bool:
        return await asyncio.to_thread(self.delete_preset, title)

    async def areload(self) -> None:
        await asyncio.to_thread(self.reload)
