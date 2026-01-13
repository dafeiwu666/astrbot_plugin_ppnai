"""用户额度管理模块"""

import json
import random
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .config import Config


class UserData(BaseModel):
    """单个用户的数据"""
    quota: int = Field(default=0, description="剩余额度")
    last_checkin_date: str | None = Field(default=None, description="上次签到日期")


class UserDataStore(BaseModel):
    """用户数据存储"""
    users: dict[str, UserData] = Field(default_factory=dict, description="用户数据")
    whitelist: list[str] = Field(default_factory=list, description="白名单用户")
    blacklist: list[str] = Field(default_factory=list, description="黑名单用户")


class UserManager:
    """用户管理器"""
    
    def __init__(self, data_dir: Path):
        self.data_file = data_dir / "user_data.json"
        self.data_dir = data_dir
        self._store: UserDataStore | None = None
    
    def _ensure_dir(self):
        """确保数据目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load(self) -> UserDataStore:
        """加载用户数据"""
        if self._store is not None:
            return self._store
        
        self._ensure_dir()
        if self.data_file.exists():
            try:
                data = json.loads(self.data_file.read_text("utf-8"))
                self._store = UserDataStore.model_validate(data)
            except Exception:
                self._store = UserDataStore()
        else:
            self._store = UserDataStore()
        return self._store
    
    def _save(self):
        """保存用户数据"""
        if self._store is None:
            return
        self._ensure_dir()
        self.data_file.write_text(
            self._store.model_dump_json(indent=2),
            "utf-8"
        )
    
    def _get_user(self, user_id: str) -> UserData:
        """获取用户数据，不存在则创建"""
        store = self._load()
        if user_id not in store.users:
            store.users[user_id] = UserData()
        return store.users[user_id]
    
    # ========== 黑白名单管理 ==========
    
    def is_blacklisted(self, user_id: str) -> bool:
        """检查用户是否在黑名单中"""
        return user_id in self._load().blacklist
    
    def is_whitelisted(self, user_id: str) -> bool:
        """检查用户是否在白名单中"""
        return user_id in self._load().whitelist
    
    def add_to_blacklist(self, user_id: str) -> bool:
        """添加用户到黑名单，返回是否成功（已存在则返回False）"""
        store = self._load()
        if user_id in store.blacklist:
            return False
        # 从白名单中移除
        if user_id in store.whitelist:
            store.whitelist.remove(user_id)
        store.blacklist.append(user_id)
        self._save()
        return True
    
    def remove_from_blacklist(self, user_id: str) -> bool:
        """从黑名单中移除用户，返回是否成功"""
        store = self._load()
        if user_id not in store.blacklist:
            return False
        store.blacklist.remove(user_id)
        self._save()
        return True
    
    def add_to_whitelist(self, user_id: str) -> bool:
        """添加用户到白名单，返回是否成功"""
        store = self._load()
        if user_id in store.whitelist:
            return False
        # 从黑名单中移除
        if user_id in store.blacklist:
            store.blacklist.remove(user_id)
        store.whitelist.append(user_id)
        self._save()
        return True
    
    def remove_from_whitelist(self, user_id: str) -> bool:
        """从白名单中移除用户，返回是否成功"""
        store = self._load()
        if user_id not in store.whitelist:
            return False
        store.whitelist.remove(user_id)
        self._save()
        return True
    
    def get_blacklist(self) -> list[str]:
        """获取黑名单列表"""
        return self._load().blacklist.copy()
    
    def get_whitelist(self) -> list[str]:
        """获取白名单列表"""
        return self._load().whitelist.copy()
    
    # ========== 额度管理 ==========
    
    def get_quota(self, user_id: str) -> int:
        """获取用户额度"""
        return self._get_user(user_id).quota
    
    def set_quota(self, user_id: str, quota: int) -> None:
        """设置用户额度（管理员用，不受限制）"""
        user = self._get_user(user_id)
        user.quota = max(0, quota)
        self._save()
    
    def add_quota(self, user_id: str, amount: int) -> int:
        """增加用户额度，返回增加后的额度"""
        user = self._get_user(user_id)
        user.quota = max(0, user.quota + amount)
        self._save()
        return user.quota
    
    def consume_quota(self, user_id: str) -> bool:
        """消耗一次额度，返回是否成功"""
        user = self._get_user(user_id)
        if user.quota <= 0:
            return False
        user.quota -= 1
        self._save()
        return True
    
    def can_use(self, user_id: str) -> tuple[bool, str]:
        """
        检查用户是否可以使用画图功能
        返回: (是否可用, 原因说明)
        """
        if self.is_blacklisted(user_id):
            return False, "你已被加入黑名单，无法使用画图功能"
        
        if self.is_whitelisted(user_id):
            return True, ""
        
        quota = self.get_quota(user_id)
        if quota <= 0:
            return False, "你的画图次数已用完，请/nai签到获取额度"
        
        return True, ""
    
    # ========== 签到系统 ==========
    
    def checkin(self, user_id: str, config: "Config") -> tuple[bool, int, str]:
        """
        用户签到
        返回: (是否签到成功, 获得次数, 提示消息)
        """
        if self.is_blacklisted(user_id):
            return False, 0, "你已被加入黑名单，无法签到"
        
        user = self._get_user(user_id)
        today = date.today().isoformat()
        
        # 检查是否已签到
        if user.last_checkin_date == today:
            return False, 0, "你今天已经签到过了，明天再来吧~"
        
        # 检查额度上限
        checkin_limit = config.quota.checkin_quota_limit
        if user.quota >= checkin_limit:
            user.last_checkin_date = today
            self._save()
            return True, 0, f"签到成功！但你的额度已达到签到上限({checkin_limit}次)，本次获得 0 次画图机会"
        
        # 计算获得的次数
        min_quota = config.quota.checkin_min_quota
        max_quota = config.quota.checkin_max_quota
        gained = random.randint(min_quota, max_quota)
        
        # 不超过上限
        if user.quota + gained > checkin_limit:
            gained = checkin_limit - user.quota
        
        user.quota += gained
        user.last_checkin_date = today
        self._save()
        
        return True, gained, f"签到成功！获得 {gained} 次画图机会，当前剩余 {user.quota} 次"
    
    def reload(self):
        """重新加载数据（用于外部修改后刷新）"""
        self._store = None
        self._load()
