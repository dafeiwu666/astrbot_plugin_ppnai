"""预设提示词管理模块"""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Preset(BaseModel):
    """单个预设"""
    title: str = Field(description="预设标题")
    content: str = Field(description="预设内容（提示词）")


class PresetStore(BaseModel):
    """预设数据存储"""
    presets: dict[str, Preset] = Field(default_factory=dict, description="预设列表")


class PresetManager:
    """预设管理器"""
    
    def __init__(self, data_dir: Path):
        self.data_file = data_dir / "presets.json"
        self.data_dir = data_dir
        self._store: PresetStore | None = None
    
    def _ensure_dir(self):
        """确保数据目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load(self) -> PresetStore:
        """加载预设数据"""
        if self._store is not None:
            return self._store
        
        self._ensure_dir()
        if self.data_file.exists():
            try:
                data = json.loads(self.data_file.read_text("utf-8"))
                self._store = PresetStore.model_validate(data)
            except Exception:
                self._store = PresetStore()
        else:
            self._store = PresetStore()
        return self._store
    
    def _save(self):
        """保存预设数据"""
        if self._store is None:
            return
        self._ensure_dir()
        self.data_file.write_text(
            self._store.model_dump_json(indent=2),
            "utf-8"
        )
    
    def list_presets(self) -> list[str]:
        """获取所有预设标题列表"""
        return list(self._load().presets.keys())
    
    def get_preset(self, title: str) -> Preset | None:
        """获取指定预设，不存在返回 None"""
        store = self._load()
        return store.presets.get(title)
    
    def add_preset(self, title: str, content: str) -> bool:
        """
        添加预设
        返回: True 表示添加成功，False 表示已存在同名预设
        """
        store = self._load()
        if title in store.presets:
            return False
        store.presets[title] = Preset(title=title, content=content)
        self._save()
        return True
    
    def update_preset(self, title: str, content: str) -> bool:
        """
        更新预设
        返回: True 表示更新成功，False 表示预设不存在
        """
        store = self._load()
        if title not in store.presets:
            return False
        store.presets[title] = Preset(title=title, content=content)
        self._save()
        return True
    
    def delete_preset(self, title: str) -> bool:
        """
        删除预设
        返回: True 表示删除成功，False 表示预设不存在
        """
        store = self._load()
        if title not in store.presets:
            return False
        del store.presets[title]
        self._save()
        return True
    
    def reload(self):
        """重新加载数据"""
        self._store = None
        self._load()
