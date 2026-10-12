# 第六章、提示词模板 (PromptTemplate 和 ChatPromptTemplate )

在 LangChain 中，**提示词模板（Prompt Template）是一种将用户输入、上下文或动态参数注入到预设文本格式中的机制。它的核心作用是实现提示词的参数化、标准化与可复用**，帮助开发者从“手动拼字符串”升级为“通过模版动态生成输入”。

在 LangChain 1.x 体系中（基于 LCEL 链式语法和核心接口架构），提示词模板作为表达式中的重要组件，被设计为接受字典输入并输出 `PromptValue` 类型的对象（可自动兼容纯文本模型 LLM 与对话模型 ChatModel）。

# 主要类型与使用方式

## 1. 基础文本模板 (`PromptTemplate`)适用于纯文本生成模型，通过占位符（如 `{variable}`）将变量替换为具体的字符串。

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)

model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

prompt = PromptTemplate.from_template("将以下内容翻译成{language}: {text}")

fromatted_prompt = prompt.format(language="英文", text="你好，世界")

resp = model.invoke(fromatted_prompt)
print(resp.content)
```

## **2. 对话提示词模板 (`ChatPromptTemplate`)**

现代 LLM（如 GPT-4、Claude 等）通常采用 Message 结构的对话输入。`ChatPromptTemplate` 允许构建由不同角色消息（System, Human, AI）组成的结构化提示词。

```python
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)

model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深的{role}专家，请使用专业且简明的语言回答。"),
    ("human", "{question}")
])

# 转换为多角色消息结构
messages = chat_prompt.format_messages(role="Python", question="什么是装饰器？")

res = model.invoke(messages)
print(res.content)
```

## **3. 动态历史占位符 (`MessagesPlaceholder`)**

在构建支持多轮对话的 Agent 或 Chain 时，可以用它动态插入历史对话记录列表。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. 定义包含 MessagesPlaceholder 的模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的 AI 助手。"),
    # variable_name 对应的变量必须传入一个 BaseMessage 对象列表
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{input}")
])

# 2. 模拟从数据库或 Memory 中提取的历史对话列表
history = [
    HumanMessage(content="你好，我叫小明。"),
    AIMessage(content="你好，小明！很高兴认识你。有什么我可以帮你的吗？")
]

# 3. 格式化生成提示词，传入历史消息列表和当前用户输入
formatted_messages = prompt_template.format_messages(
    chat_history=history,
    input="还记得我的名字吗？"
)

# 4. 打印格式化后的结果
for msg in formatted_messages:
    print(f"[{msg.type}]: {msg.content}")
```

# PromptTemplate 和 ChatPromptTemplate 之间的区别

| 特性 | PromptTemplate | ChatPromptTemplate |
| --- | --- | --- |
| 输出格式 | 纯文本字符串 | 消息列表 |
| 角色支持 | ❌ 无 | ✅ system/user/assistant |
| 对话历史 | ❌ 不支持 | ✅ 支持 |
| 适用场景 | 简单提示 | 聊天、对话、多轮交互 |

# ChatPromptTemplate的使用

在LangChain 1.0中，ChatPromptTemplate 是用于生成消息列表的核心组件。

ChatPromptTemplate是创建 聊天消息列表 的提示模板。它比普通 PromptTemplate 更适合处理多角色、多轮次的对话场景。

支持 System / Human / AI 等不同角色的消息模板。

在 LangChain 1.x 中，实例化 `ChatPromptTemplate` 主要有以下 **5 种常用方式**，覆盖了从简单快速到复杂组合的不同场景：

### 1. `ChatPromptTemplate.from_messages()` **（最推荐/最常用）**

支持使用元组（Tuple）、消息对象（BaseMessage）或字典（Dict）列表来构建多角色对话模板。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# 方式 A：使用简短的元组 (role, text) —— 推荐，语法最简洁
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深的{role}专家。"),
    MessagesPlaceholder(variable_name="history"), # 可混合放置占位符
    ("human", "{question}")
])

# 方式 B：使用明确的消息类对象
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一名资深的 Python 专家。"),
    HumanMessage(content="{question}")
])
```

### 2. `ChatPromptTemplate.from_template()` **（快速单文本转化）**

将一段单文本转化为只包含一条 `HumanMessage` 的对话模板，适合不需要设置 System 角色等复杂结构的简单对话。

```python
from langchain_core.prompts import ChatPromptTemplate

# 自动封装为包含 1 条 HumanMessage 的 ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_template("请用一句话解释：{concept}")
```

### 3. 直接通过构造函数实例化 `ChatPromptTemplate()`

直接传入 `SystemMessagePromptTemplate`、`HumanMessagePromptTemplate` 等具体的 PromptTemplate 消息类列表，底层逻辑最明确。

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("你是一个{role}助手。"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

formatted_messages = chat_prompt.invoke({"role": "编程", "input": "什么是Python"})
res = model.invoke(formatted_messages)
print(res.content)
```

### 4. 加载外部文件/模板

将复杂的提示词与代码解耦，从模版文件或配置文件中直接加载。

```python
from langchain_core.prompts import ChatPromptTemplate

# 从 JSON / YAML 文件加载
chat_prompt = ChatPromptTemplate.from_file("prompts/chat_prompt.json")

# 从字符串（如 Jinja2 / f-string 样式的模版）加载
chat_prompt = ChatPromptTemplate.from_template_file(
    template_file="prompts/chat_prompt.txt",
    template_format="f-string"
)
```

### 5. 通过操作符加号 (`+`) 进行拼接组装

可以将多个 `ChatPromptTemplate` 或消息对象通过 `+` 号动态拼接，构建极其灵活的组合式提示词。

```python
from langchain_core.prompts import ChatPromptTemplate

system_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个遵循 Strict 规则的 AI。")
])

user_prompt = ChatPromptTemplate.from_messages([
    ("human", "请回答这个问题：{question}")
])

# 拼接为一个全新的 ChatPromptTemplate
combined_prompt = system_prompt + user_prompt
```

### 方法对比总结

| **实例化方法** | **适用场景** | **优势** |
| --- | --- | --- |
| **`from_messages()`** | 多角色对话、Agent 开发、带历史消息（Memory） | 最符合 LLM 对话规范，支持 `MessagesPlaceholder`，最常用 |
| **`from_template()`** | 快速原型开发、无 System 角色的单轮提示 | 语法极简 |
| **直接构造 `ChatPromptTemplate()`** | 需要高度自定义模板配置 | 显式声明，结构严谨 |
| **`from_file()`** | 企业级工程项目 | 提示词与业务代码解耦 |
| **`+` 操作符拼接** | 动态拼接不同模块的 Prompt | 灵活度高，可组合性强 |

# 模板变量预填充partial()

在 LangChain 1.x 中，提示词模板（如 `ChatPromptTemplate` 和 `PromptTemplate`）上的 `.partial()` 方法用于**偏格式化（Partial Formatting）**。

它的核心作用是：**预先填充模板中的一部分变量，生成一个新的模板对象**。这样调用方后续只需传入剩余的变量即可，无须一次性提供所有参数。

## **核心应用场景**

- **分阶段传入变量**：部分变量（如系统配置、语言设置）在初始化阶段就已经确定，而另一些变量（如用户输入）需要等到运行时才能获取。
- **绑定动态函数**：可以将函数作为变量传入（如获取当前时间、从数据库拉取状态），每次模板格式化时会自动执行该函数获取最新值。

### **一、 静态变量绑定（字符串/普通值）**

当模板包含多个变量，但某些变量在流程早期就已经确定时，可以使用 `.partial()` 先固定它们：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv(verbose=True)

model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

# 1. 定义包含 3 个变量的模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专门解答 {domain} 领域问题的助手。回答风格：{style}。"),
    ("human", "{question}")
])

# 2. 预先填充 domain 和 style 变量
domain_prompt = prompt.partial(domain="金融", style="专业严谨")

# 3. 只需传入剩余的 question 变量
response = domain_prompt.invoke({"question": "什么是量化交易?"})

model_resp = model.invoke(response)
print(model_resp.content)
```

### **二、 动态函数绑定（传递 Callable）**

`.partial()` 允许传入一个无参函数，该函数会在最终调用 `format()` 或 `invoke()` 的那一步被触发执行，非常适合注入实时变化的上下文：

```python
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

# 定义获取当前时间的函数
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

prompt = ChatPromptTemplate.from_messages([
    ("system", "当前时间是：{current_time}。请先推算该日期是星期几，然后再回答用户的问题。"),
    ("human", "{user_input}")
])

# 使用 partial 绑定动态函数
partial_prompt = prompt.partial(current_time=get_current_time)

# 后续调用时，不需要手动传 current_time，模板内部会自动调用 get_current_time()
messages = partial_prompt.format_messages(user_input="今天周几？")

response = model.invoke(messages)
print(response.content)
```

### **三、 直接在初始化时传入 `partial_variables`**

除了调用 `.partial()` 方法，也可以在实例化模板时通过 `partial_variables` 参数完成相同操作：

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="时间：{time}，用户：{user}，操作：{action}",
    input_variables=["user", "action"],  # 剩余需要传入的变量
    partial_variables={"time": get_current_time}  # 绑定的偏变量
)
```

**`partial()` 与 Python `functools.partial` 的关系**

LangChain 的 `.partial()` 借鉴了 Python 标准库 `functools.partial` 的思想，但它是针对 `BasePromptTemplate` 专有数据结构封装的方法。它返回的是一个新的 **LangChain 提示词模板实例**，因此完全兼容 LCEL（LangChain 表达式语言）管道和运行规范。
