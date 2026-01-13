## 📖 简介

本插件为 AstrBot 带来了强大的 **NovelAI** 绘图能力。它不仅仅是一个简单的 API 调用工具，还集成了 **LLM 辅助生图**、**自动绘图**、**额度管理**、**队列系统**以及 NovelAI 的高级功能（如氛围转移 Vibe Transfer、角色保持 Character Keep）。

支持 **Opus 免费模式**（小图模式），订阅用户可无限畅玩。

## ✨ 特性

- **🎨 多种绘图模式**：
    - **基础模式** (`nai`)：直接使用 Prompt 标签绘图。
    - **智能模式** (`nai画图`)：使用自然语言描述，由 LLM 自动生成专业标签和参数。
    - **自动模式** (`nai自动画图`)：监听对话上下文，自动为群友的聊天内容配图。
- **🖼️ 高级功能支持**：
    - **图生图 (i2i)**：支持引用图片进行重绘。
    - **氛围转移 (Vibe Transfer)**：提取参考图的风格/构图。
    - **角色保持 (Character Keep)**：保持角色特征一致性。
    - **多角色控制 (Role)**：指定不同区域角色的外貌。
    - **视觉识图**：支持多模态模型识别发送的图片并生成参数。
- **⚙️ 强大的系统管理**：
    - **队列系统**：并发控制，防止 API 过载，支持排队等待。
    - **额度经济**：支持每日签到、额度限制，防止滥用。
    - **黑白名单**：管理员可精细控制用户权限。
    - **预设管理**：保存常用的 Prompt 组合。
- **🏠 CozyNook 社区预设**：可在 CozyNook 社区找到同款/对应预设，再用本插件的“预设管理”命令一键落地到本地使用。
- **💸 Opus 免费模式**：自动调整参数至免费限额内（分辨率≤1024x1024，步数≤28），Opus 订阅者可无限生成。

## 📦 安装

1. 确保你已经安装了 [AstrBot](https://github.com/Soulter/AstrBot)。
2. 将本插件文件夹放入 AstrBot 的 `data/plugins/` 目录下。
3. 安装依赖：
   ```bash
   pip install cookit[pydantic] jsonref
   # 可选：如果需要将帮助文档渲染为精美图片
   pip install pillowmd
   ```
4. 重启 AstrBot。

## 🔧 配置

在 AstrBot 管理面板或插件配置文件中进行设置：

| 配置项 | 说明 | 必填 |
| :--- | :--- | :--- |
| `request.tokens` | **授权 Token 列表**。请使用 NovelAI 官网获取的 `pst-` 开头的 Persistent API Token。支持多个 Token 轮询。 | ✅ 是 |
| `request.opus_free_mode` | **Opus 免费模式**。开启后强制限制图片尺寸和步数，Opus 订阅账号不扣点数。 | ❌ 否 |
| `llm.advanced_arg_generation_provider` | 用于将自然语言转换为绘图参数的 LLM 模型提供商。 | ❌ 否 |
| `llm.enable_vision` | 是否启用视觉输入（需要模型支持 Vision），用于参考图分析。 | ❌ 否 |
| `quota.enable_quota` | 是否启用额度/签到系统。 | ❌ 否 |

## 💻 指令列表

### 🎨 绘图指令

| 指令 | 示例/说明 |
| :--- | :--- |
| `nai` | 基础绘图。<br>`nai tag=1girl, white hair` |
| `nai画图` | AI 辅助绘图。<br>`nai画图 画一个在海边玩耍的白发少女` |
| `nai自动画图` | 查看当前自动画图状态或设置。<br>`nai自动画图 s1=预设名` |
| `nai自动画图开` | 开启自动画图（消耗开启者额度）。<br>`nai自动画图开 s1=预设名` |
| `nai自动画图关` | 关闭自动画图。 |

### 🧩 预设与辅助

| 指令 | 说明 |
| :--- | :--- |
| `nai预设列表` | 查看所有可用预设。 |
| `nai预设查看` | 查看指定预设的详细内容。<br>`nai预设查看 预设名` |
| `nai队列` | 查看当前绘图队列状态（处理中/排队中）。 |
| `nai签到` | 每日签到获取绘图额度。 |
| `查询额度` | 查询当前剩余绘图次数。 |

### 👮 管理员指令

| 指令 | 说明 |
| :--- | :--- |
| `nai预设添加` | 添加新预设。<br>`nai预设添加 预设名`<br>`tag=...` |
| `nai预设删除` | 删除指定预设。 |
| `nai黑名单添加` | `nai黑名单添加 [用户ID]` |
| `nai白名单添加` | 白名单用户无限额度，无视部分限制。<br>`nai白名单添加 [用户ID]` |
| `nai设置额度` | `nai设置额度 [用户ID] [数量]` |

## 📝 高级参数示例

在 `nai` 或 `nai画图` 命令中，你可以使用以下高级参数（支持换行）：

**1. 氛围转移 (Vibe Transfer)**
```text
nai 1girl
vibe_transfer=true
vibe_transfer_info_extract=0.8
[附带一张图片]
```

**2. 角色保持 (Character Keep)**
```text
nai 1girl
character_keep=true
[附带一张图片]
```

**3. 多角色控制**
```text
nai 2girls
role=A2|1girl, pink hair|bad quality
role=D2|1girl, blue hair|bad quality
```
*(位置网格：A-E为横向，1-5为纵向，C3为中心)*

## 📄 License

MIT License