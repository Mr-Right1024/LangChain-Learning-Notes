# 第五章、content 和 content_blocks

在 LangChain 的消息模型（`BaseMessage` / `AIMessage` 等）中，`content` 和 `content_blocks` 对应了文本传输与多模态/结构化块的不同数据抽象模式：

| **维度** | **content** | **content_blocks** |
| --- | --- | --- |
| **数据类型** | `str` 或 `List[Union[str, dict]]` | `List[ContentBlock]`（字典或标准化对象） |
| **主要用途** | 快速读取纯文本内容，向后兼容旧版 API | 结构化解析多模态输入（文本+图片）及思维链/工具调用块 |
| **包含内容** | 消息的原始文本或原始 dict 列表 | 规范化后的块（如 `text`、`image_url`、`reasoning_content` 等） |
| **应用场景** | 简单的 LLM 问答、日志打印、纯文本渲染 | 多模态输入解析、工具调用提取、Reasoning/Thinking 过程提取 |

**核心差异细节**

- **`content`（通用与兼容）**
    
    在常规单模态交互中，`content` 是最常用的属性，直接返回大模型输出的字符串文本。如果输入是多模态，`content` 可能是一个包含混合类型的原始列表（例如包含图片 URL 和文本的 dict），未做标准格式化。
    
- **`content_blocks`（结构化与多模态）**
    
    随着多模态与推理模型（如 Claude 3.5、DeepSeek-R1、OpenAI o3 等）的普及，输出不仅包含纯文本，还可能混杂推理思维链（Thinking Block）、工具调用签名（Tool Call Block）以及图像块。`content_blocks` 将这些内容统一拆解为标准化的 Block 列表，便于开发者按类型精准筛选提取。
    

例如，当处理包含推理过程与文本的响应时：

```python
# content 得到完整的原始字符串或混合结构
print(message.content)

# content_blocks 可以按类型遍历拆解
for block in message.content_blocks:
    if block["type"] == "reasoning":
        print("思考过程:", block["reasoning"])
    elif block["type"] == "text":
        print("最终回答:", block["text"])
```

示例代码

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

# 1. 初始化模型
model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

# 2. 正确构造多模态消息的内容块 (content)
message_content = [
    {"type": "text", "text": "这张图里有什么内容"},
    {
        "type": "image_url",
        "image_url": {
            "url": "https://img2.baidu.com/it/u=615045428,3170210785&fm=253&app=138&f=JPEG?w=500&h=500"
        }
    }
]

# 3. 将 content 封装进 HumanMessage 并放入列表传入 invoke
response = model.invoke([HumanMessage(content=message_content)])

# 4. 打印模型的回答
print(response.content)
```

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

# 1. 初始化模型
model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

# 2. 正确构造多模态消息的内容块 (content)
message_content = [
    {"type": "text", "text": "这张图里有什么内容"},
    {
        "type": "image_url",
        "image_url": {
            "url": "https://img2.baidu.com/it/u=615045428,3170210785&fm=253&app=138&f=JPEG?w=500&h=500"
        }
    }
]

# 3. 将 content_blocks 封装进 HumanMessage 并放入列表传入 invoke
response = model.invoke([HumanMessage(content_blocks=message_content)])

# 4. 打印模型的回答
# print(response.content_blocks)
for block in response.content_blocks:
    if block['type'] == "reasoning":
        print("思考过程:", block["reasoning"])
    elif block["type"] == "text":
        print("最终回答:", block["text"])
```