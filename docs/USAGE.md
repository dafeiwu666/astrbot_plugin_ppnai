# 泡泡画图插件 使用文档

## 📖 概述

泡泡画图是一个基于 NovelAI 官方 API 的 AI 绘图插件，支持文生图、图生图、氛围转移、角色保持等多种功能。

### 🆓 Opus 免费模式

本插件支持 **Opus 免费模式**（小图模式），开启后会自动调整参数以符合 NovelAI Opus 订阅的免费生成条件：
- 分辨率限制在 **1024×1024 像素**以内
- 步数限制在 **28 步**以内

> 💡 Opus 订阅用户在此条件下可以**无限免费生成**图片，无需消耗 Anlas！
> 
> ⚠️ 注意：部分高级功能（如高分辨率、高步数）在免费模式下会被自动调整。

---

## 🎨 画图命令

### 基础命令 `nai`

直接使用提示词绘图（所有参数必须使用 `key=value` 格式）：

```
nai
tag=1girl, coffee shop, smile
```

#### 批量生成（扣除对应额度）

```
nai
tag=1girl, coffee shop, smile
n=3
```

> 未填写 `n` 时默认生成 1 张；若开启额度系统，会按 `n` 扣除次数。
> `nai画图` / `nai自动画图` 的批量模式会为每张图重新调用 LLM 生成参数。

#### 使用预设

```
nai
s1=预设名
```

#### 使用角色保持（cs）

---

## 🧩 角色保持（cs）

### 新建角色保持

```
cs
na=角色名
aa=用户描述
```

或直接保存已有提示词：

```
cs
na=角色名
nn=NovelAI tag style 外貌提示词
```

> `aa` 模式会将“用户描述 + cssaying.txt”发送给 LLM 生成并保存。

### 删除角色保持

```
dcs 角色名
```

### 查看角色保持名称列表

```
cs
```

### 查询外貌提示词

```
scs 角色名
```

### 修改外貌提示词

```
ccs 角色名 新内容
```

> 修改会覆盖 `NovelAI tag style 外貌提示词` 内容。

创建 `cs` 或使用 `ccs` 修改角色保持时，在同一条消息中附带一张图片即可保存或更新预览图。

```
nai
cs=角色名
cs2=角色名
tag=1girl, coffee shop
```

> `cs` 会把该角色的外貌提示词追加到正向提示词后面。

#### 使用自定义参数

换行定义多个参数：

```
nai
tag=1girl, coffee shop
model=nai-diffusion-4-5-curated
画面尺寸=竖图
```

#### 使用提示词包装器

可以使用前置/后置提示词来包装主提示词：

```
nai
前置正向=best quality, masterpiece
tag=1girl, coffee shop
后置正向=solo, simple background
前置负面=lowres, bad anatomy
后置负面=extra limbs
```

最终生成的提示词：
- 正向：`best quality, masterpiece, 1girl, coffee shop, solo, simple background`
- 负面：`lowres, bad anatomy, extra limbs`

---

### AI 画图命令 `nai画图`

使用 AI 自动解析描述生成参数：

```
nai画图
s1=预设名
ds=画一个在咖啡店微笑的女孩
```

- `s1`、`s2` 等：按优先级使用多个预设
- `ds`：自然语言描述，AI 会自动转换为绘图参数

#### 使用 `nai` 参数进行个性化设置（推荐）

`nai画图` 除了支持自然语言 `ds` 让 AI 自动生成参数，也支持直接写 **与 `nai` 同款的 `key=value` 参数**来个性化控制；这些显式参数会优先生效/覆盖 AI 自动生成的结果。

示例：
```
nai画图
ds=把图里的角色改成冬装
model=nai-diffusion-4-5-curated
steps=28
size=竖图
前置正向=best quality, masterpiece
s=预设
s2=预设
cs=角色名
cs2=角色名
```

#### 识图（可选）

`nai画图` 和 `nai自动画图` 支持“把你发送的图片或引用的图片作为参考”交给**高级参数模型**进行识图（多模态）。

- 使用方式：发送命令时带上图片即可（同一条消息内）。

当你在同一条消息里发送多张图片时：

- 若启用了 `i2i` / `vibe_transfer` / `character_keep` 等参数，会按图片顺序先“消耗”对应数量。
- **剩余图片会作为识图参考**传给高级参数模型（需要开启 `llm.enable_vision=true`）。
- 可选：用 `llm.vision_image_limit` 限制传给模型的参考图片数量（0 表示不限制）。

---

### 自动画图

监听主 AI 回复，自动生成配图。

#### 查看状态
```
nai自动画图
```

#### 开启自动画图
```
nai自动画图开
s=预设名
s2=预设名
cs=角色名
cs2=角色名
```

#### 关闭自动画图
```
nai自动画图关
```

> ⚠️ 自动画图的额度由开启者承担

#### 在自动画图里使用 `nai` 参数个性化

`nai自动画图` 的预设内容支持写入与 `nai` 相同的 `key=value` 参数（例如 `model/size/steps/seed/role/i2i/vibe_transfer/character_keep/前置正向...`），用于个性化控制自动出图风格与参数。

---

## 📝 预设管理

### 查看预设列表
```
nai预设列表
```

### 查看预设内容
```
nai预设查看 预设名
```

### 修改预设（管理员或资源所有者）
```
nai预设修改 预设名
这里是新的预设内容
```

添加或修改预设时，在同一条消息中附带的第一张图片会保存为预览图。

### 添加预设（管理员）
```
nai预设添加 预设名
这里是预设内容
tag=1girl, cute
negative=bad quality
size=竖图
```

### 删除预设（管理员）
```
nai预设删除 预设名
```

## 🖼️ 图库管理

### 查看图库列表
```
nai图库
```

### 查看图库图片
```
nai图片查看 图片名
```

### 添加、修改或删除图库图片（管理员）
```
nai图片添加 图片名
nai图片修改 图片名
nai图片删除 图片名
```

> `nai图片添加` 和 `nai图片修改` 执行后，请在 60 秒内发送一张图片。
> 图库图片的添加、修改和删除仅限机器人管理员及配置中的额外图库管理员。

图库添加或修改时，后续消息可以发送两张图片：第一张是图库原图，第二张是预览图。
如果只发送一张图片，则只更新图库原图，并保留已有预览图（修改时）。

### 在画图参数中引用图库图片

图库图片可以直接作为 `i2i`、氛围转移或角色保持的参考图，不需要再次上传：
```text
nai
tag=1girl, coffee shop
i2i=草图
```

```text
nai
tag=1girl, coffee shop
vibe_transfer=画风参考
vibe_transfer_info_extract=0.8
```

```text
nai
tag=1girl, coffee shop
c_k=角色参考
```

对应的参数别名也可以使用：
- `i2i=图库名称`：使用图库图片进行图生图
- `vibe_transfer=图库名称` 或 `v_t=图库名称`：使用图库图片进行氛围转移；可重复填写以使用多张图片
- `character_keep=图库名称`、`c_k=图库名称` 或 `ck=图库名称`：使用图库图片进行角色保持

同一请求中仍不能同时使用 `vibe_transfer` 和 `character_keep`。
首次使用预设、角色保持或图库资源生成成功后，插件会自动保存一张预览图；使用
`nai图片查看 图片名` 时会一并发送图库原图和预览图。

## 💰 额度系统

### 每日签到
```
nai签到
```

### 查询额度
```
查询额度
```

---

## 📊 队列系统

当多个用户同时请求画图时，系统会自动进行排队管理。

### 查询队列状态
```
nai队列
```

显示当前正在处理的请求数、排队等待的请求数，以及队列是否已满。

> 💡 当队列已满时，新的画图请求会被暂时拒绝，请稍后重试。

---

## 🔧 管理员命令

### 黑名单管理
```
nai黑名单添加 用户ID
nai黑名单移除 用户ID
nai黑名单列表
```

### 白名单管理
```
nai白名单添加 用户ID
nai白名单移除 用户ID
nai白名单列表
```

### 额度管理
```
nai查询用户 用户ID
nai设置额度 用户ID 次数
nai增加额度 用户ID 次数
```

---

## 🖼️ 图片引用功能

引用图片要注意只能引用聊天记录的第一张图，无法多张引用并使用。

### 图生图(i2i)
```
nai 1girl
i2i=true

[图片]
```

### 氛围转移(vibe_transfer)
```
nai 1girl
vibe_transfer=true
vibe_transfer_info_extract=0.8

[图片]
```

> ⚠️ `vibe_transfer` 与 `character_keep` 不能在同一次请求中同时使用。NovelAI
> 官方接口不允许同时混合普通参考图和 Director Reference；如需使用角色保持，
> 请移除 `vibe_transfer` 及其对应的参考图片。

### 角色保持(character_keep)
```
nai 1girl
character_keep=true

[图片]
```


## 🖼️ 图片输出方式

默认会以**合并转发消息**的形式发送，避免图片在聊天列表中过于显眼。

如需直接发送图片，可在配置中关闭：

- `general.merge_draw_to_chat_record=false`

### 生成参数记录

开启后，每次生成完成时，插件都会额外发送一条合并聊天记录，内容包括：
- 原始指令
- 最终正向和反向提示词
- 最终模型、尺寸、步数、采样器、种子等参数
- 使用的输入图片（若有）
该功能默认关闭。


## 📋 可用模型

- `nai-diffusion-3` `nai3` - NAI3 标准模型
- `nai-diffusion-furry-3` `nai3_furry` - NAI3 Furry模型
- `nai-diffusion-4-full` `nai4_full` - NAI4 完整版
- `nai-diffusion-4-curated-preview` `nai4_c_p` - NAI4 精选预览版
- `nai-diffusion-4-5-curated` `nai4.5_c` - NAI4.5 精选版
- `nai-diffusion-4-5-full` `nai4.5_full` - NAI4.5 完整版
- `nai-diffusion-5-full` `nai5_full` - NAI5 完整版
- `nai-diffusion-5-curated` `nai5_c` - NAI5 精选版

---

## 🎯 多角色控制(role)

格式：`role=位置|正向提示词|反向提示词`

位置网格（5x5）：
```
     A    B    C    D    E
  ┌────┬────┬────┬────┬────┐
1 │ A1 │ B1 │ C1 │ D1 │ E1 │
  ├────┼────┼────┼────┼────┤
2 │ A2 │ B2 │ C2 │ D2 │ E2 │
  ├────┼────┼────┼────┼────┤
3 │ A3 │ B3 │ C3 │ D3 │ E3 │
  ├────┼────┼────┼────┼────┤
4 │ A4 │ B4 │ C4 │ D4 │ E4 │
  ├────┼────┼────┼────┼────┤
5 │ A5 │ B5 │ C5 │ D5 │ E5 │
  └────┴────┴────┴────┴────┘
```

示例：
```
nai 2girls
role=A2|1girl, cute, smile
role=D2|1girl, cool|bad anatomy
```

---

## ⚙️ 参数指令表

| 参数 | 别名 | 说明 |
|------|------|------|
| `nai` | 基础绘图指令 | 支持 `key=value` 参数 |
| `nai画图` | AI 辅助绘图指令 | 支持 `ds=` 与参数覆盖 |
| `nai自动画图` | 查看或设置自动画图 | 可附带预设参数 |
| `nai自动画图开` | 开启自动画图 | 消耗开启者额度 |
| `nai自动画图关` | 关闭自动画图 | |
| `s` `s1` `s2`... | 预设 | 按优先级使用预设内容（仅命令输入）s=s1 |
| `ds` | 描述 | 自然语言描述（`nai画图` 使用） |
| `cs` `cs1` `cs2`  | 角色保持 | 引用已保存的角色保持内容 cs=cs1 |
| `cs` | 创建角色保持或查看名称列表 |
| `dcs` | 删除角色保持 |
| `scs` | 查询外貌提示词 |
| `ccs` | 修改外貌提示词（覆盖） |
| `tag` | 正向提示词 | 期望生成的图片内容 |
| `negative` `ne` | 反向提示词 | 不想出现的内容 |
| `prepend_tag` `a_tag` | 前置正向/前置正向提示词 | 添加到正向提示词最前方 |
| `append_tag` `b_tag` | 后置正向/后置正向提示词 | 添加到正向提示词最后方 |
| `prepend_negative` `a_ne` | 前置负面/前置负面提示词 | 添加到负面提示词最前方 |
| `append_negative` `b_ne` | 后置负面/后置负面提示词 | 添加到负面提示词最后方 |
| `n` | 批量生成 | 批量出图并扣除对应额度 |
| `model` | 模型 | 选择绘图模型 |
| `artist` | 画师/画师串 | 指定画师风格 |
| `size` | 画面尺寸 | 竖图`portrait`/横图`landscape`/方图`square` 或 WxH(白名单专用) |
| `seed` | 种子 | 固定随机种子 |
| `steps` | 采样步数 | 1-50，默认23 (28以上为白名单专用)|
| `scale` | 提示词引导值 | 默认5 |
| `cfg` | 缩放引导值 | 默认0 |
| `sampler` | 采样器 | 选择采样方法 |
| `noise_schedule` `n_s` | 噪声调度 | karras等 |
| `other` | 高级配置 | SMEA等设置 |
| `i2i` | 图生图 | 引用图片进行重绘 |
| `i2i_force` `i_f` | 重绘力度 | 0-1，默认0.6 |
| `vibe_transfer` `v_t` | 氛围转移 | 参考图片风格 |
| `vibe_transfer_info_extract` `v_t_i_e` | 氛围转移信息提取度 | 0-1 |
| `vibe_transfer_ref_strength` `v_t_r_s` | 氛围转移参考强度 | 0-1 |
| `role` | 角色/多角色 | 多角色控制 |
| `character_keep` `c_k` | 角色保持/ck | 保持角色特征 |
| `character_keep_vibe` `c_k_v` | 角色保持氛围 | true/false |
| `character_keep_strength` `c_k_s` | 角色保持强度 | 0-1 |
| `nai预设列表` | 查看所有可用预设 |
| `nai预设查看` | 查看指定预设内容 |
| `nai预设添加` | 添加预设（管理员） |
| `nai预设修改` | 修改预设内容，并可更新预览图 |
| `nai预设删除` | 删除预设（管理员） |
| `nai图库` | 查看图库列表 |
| `nai图片查看` | 查看指定图库图片；不填写图片名时查看图库列表 |
| `nai图片添加` | 添加图库图片（管理员） |
| `nai图片修改` | 修改图库图片（管理员） |
| `nai图片删除` | 删除图库图片（管理员） |
| `nai队列` | 查询绘图队列状态 |
| `nai签到` | 每日签到获取额度 |
| `查询额度` | 查询剩余额度 |
| `nai黑名单添加` | 添加黑名单（管理员） |
| `nai黑名单移除` | 移除黑名单（管理员） |
| `nai黑名单列表` | 查看黑名单（管理员） |
| `nai白名单添加` | 添加白名单（管理员） |
| `nai白名单移除` | 移除白名单（管理员） |
| `nai白名单列表` | 查看白名单（管理员） |
| `nai查询用户` | 查询用户额度（管理员） |
| `nai设置额度` | 设置用户额度（管理员） |
| `nai增加额度` | 增加用户额度（管理员） |
---