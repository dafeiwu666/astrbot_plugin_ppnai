# 角色保持（Character Keep）图片处理问题分析

## 问题描述
- ✅ **氛围转移**（vibe_transfer）可以正常使用发送图片
- ❌ **角色参考指令**（character_keep）无法使用发送图片

## 根本原因

### 1. 差异在图片处理函数

**氛围转移** - 使用 `resolve_image()` 函数：
```python
# src/image_params.py, 第 93 行
result.vibe_transfer_images.append(await resolve_image(pop_image(key)))
```

**角色保持** - 使用 `resolve_image_as_jpeg()` 函数：
```python
# src/image_params.py, 第 88 行  
result.character_keep_image = await resolve_image_as_jpeg(pop_image(key))
```

### 2. JPEG 转换流程的问题

在 `src/image_io.py` 中，`resolve_image_as_jpeg()` 函数会：

```python
async def resolve_image_as_jpeg(image: Image) -> str:
    """Fetch image, resize/pad, and convert to JPEG for character-keep."""
    b64 = await image.convert_to_base64()
    if isinstance(b64, str) and b64.startswith("base64://"):
        b64 = b64.removeprefix("base64://")
    original_mime = get_base64_mime(b64, "image/jpeg")
    original_data_uri = f"data:{original_mime};base64,{b64}"
    
    # 调用 CPU 密集的 JPEG 转换 → 可能在这里失败！
    return await aconvert_to_jpeg_for_character_keep(original_data_uri)
```

### 3. JPEG 转换可能失败的原因

在 `convert_to_jpeg_for_character_keep()` 函数中（第 45-112 行）：

**第 59-62 行** - 图片解码
```python
image_bytes = base64.b64decode(b64_data)
pil_image = PILImage.open(io.BytesIO(image_bytes))  # ← 可能异常：格式不支持或损坏
original_width, original_height = pil_image.size
original_mode = pil_image.mode
```

**第 69-70 行** - 选择目标尺寸（仅支持3种固定尺寸）
```python
target_width, target_height = _select_best_target_size(original_width, original_height)
# 支持的尺寸: (1472, 1472), (1536, 1024), (1024, 1536)
```

**第 72-84 行** - 图片模式转换（复杂的格式处理）
```python
if pil_image.mode in ("RGBA", "LA", "P"):
    if pil_image.mode == "P":
        pil_image = pil_image.convert("RGBA")
    rgb_image = PILImage.new("RGB", pil_image.size, (0, 0, 0))
    # 透明通道处理 ← 复杂操作，更容易出错
```

## 对比两个功能的处理差异

| 功能 | 图片处理 | 错误处理 | 异常抛出 |
|------|--------|--------|--------|
| **氛围转移** | 简单的 base64 转换 | 最小化处理 | ❌ 无特殊处理 |
| **角色保持** | 复杂的 JPEG 转换+尺寸适配+格式转换 | 多步骤处理 | ⚠️ 任何步骤失败都会抛异常 |

## 可能的失败场景

1. **图片格式不兼容**
   - WebP、AVIF 等现代格式可能不被 PIL 支持
   - 损坏或不完整的图片数据

2. **透明背景处理问题**
   - PNG 图片转换成 RGB 时可能丢失 Alpha 通道信息
   - 某些图片库生成的特殊格式 PNG

3. **尺寸计算问题**
   - 某些极端宽高比的图片可能导致缩放异常

4. **I/O 错误**
   - BytesIO 操作异常
   - PIL 的 JPEG 编码失败

## 解决方案

### 方案 1：改用 `resolve_image()` 处理（快速修复）

在 `src/image_params.py` 第 88 行改为：

```python
# 改为简单的 resolve_image() 处理
result.character_keep_image = await resolve_image(pop_image(key))
```

**优点**：快速、稳定，与氛围转移保持一致  
**风险**：可能不符合 NovelAI API 对角色保持的具体格式要求

### 方案 2：添加详细的错误处理和日志（推荐）

修改 `src/image_io.py` 的 `convert_to_jpeg_for_character_keep()` 函数：

```python
def convert_to_jpeg_for_character_keep(image_b64: str) -> str:
    """Resize/pad image to allowed sizes and return JPEG data URI."""
    try:
        # 解析 base64
        if image_b64.startswith("data:"):
            header, b64_data = image_b64.split(",", 1)
            original_mime = header.split(";")[0].replace("data:", "")
        else:
            b64_data = image_b64
            original_mime = "unknown"

        # 添加详细日志
        logger.debug(f"[nai] Converting to JPEG: mime={original_mime}, b64_len={len(b64_data)}")
        
        try:
            image_bytes = base64.b64decode(b64_data)
        except Exception as e:
            logger.error(f"[nai] Base64 decode failed: {e}")
            raise ValueError(f"图片 base64 解码失败：{e}") from e
        
        try:
            pil_image = PILImage.open(io.BytesIO(image_bytes))
            pil_image.load()  # 强制加载以检测损坏的图片
        except Exception as e:
            logger.error(f"[nai] PIL image open/load failed: {e}")
            raise ValueError(f"图片格式不支持或已损坏（尝试过 PIL 处理）：{e}") from e
        
        # ... 后续处理保持不变 ...
        
    except ValueError:
        raise  # 重新抛出 ValueError
    except Exception as e:
        logger.exception(f"[nai] JPEG conversion failed: {e}")
        raise ValueError(f"角色保持图片处理失败：{e}") from e
```

### 方案 3：提供用户友好的错误提示

在异常捕获处添加帮助信息：

```python
except Exception as e:
    yield event.plain_result(
        f"❌ 角色保持图片处理失败\n\n"
        f"错误原因：{format_readable_error(e)}\n\n"
        f"请检查：\n"
        f"1. 图片格式（推荐 JPG、PNG、WebP）\n"
        f"2. 图片是否完整未损坏\n"
        f"3. 图片大小是否过大\n\n"
        f"如果问题持续，请尝试使用 vibe_transfer（氛围转移）代替"
    )
```

## 建议修复步骤

1. **立即** - 添加详细的错误日志（方案 2）以便诊断具体是哪个环节失败
2. **短期** - 根据日志结果，针对性地修复具体问题
3. **长期** - 考虑统一 `character_keep` 和 `vibe_transfer` 的图片处理流程

## 相关文件

- `src/image_io.py` - 图片转换函数（第 45-131 行）
- `src/image_params.py` - 参数解析（第 87-88 行）
- `src/params.py` - 参数应用（第 728-743 行）
- `src/handlers_nai.py` - 命令处理（第 117-124 行）
