# 第二章、LangChain 消息以及模型调用方法

# 1、消息

在 LangChain 中，Message（消息）是模型交互的最基本单元。它既代表模型接收到的 输入（Input） ，也代表模型生成的 输出（Output）

## 1.1 消息的内部结构

LangChain的消息（Message）对象包含三种字段

- Role：消息所属的角色或类型，如 system 、 user 、 assistant 。
- Content：消息内容
- Metadata：（可选）元数据，存储额外信息。如：消息ID、响应时间、token消耗量、消息标签等

## 1.2 消息的类型

LangChain定义了很多消息类型，通过 role 区分。常用的有四种。

1、系统消息：也称为系统提示词

2、用户消息：也称为用户提示词，在多轮对话中，它表示用户的一次输入。

3、助手(AI)消息：代表模型的回复，包括生成的文本、工具调用、元数据等

4、工具调用消息：工具调用结果匹配的消息类型。将此消息返回给模型，让模型基于这个结果继续生成回复

## 1.3 消息的格式

### Json格式的消息

1、系统消息：

```json
{"role": "system", "content": "你是个精通编程的软件架构师"}
```

2、用户消息：

```json
{"role": "user", "content": "介绍一下 Python 的主要特点~"}
```

3、助手(AI)消息：

```json
{"role": "assistant", "content": "Python 的主要特点可以概括为这几类：......"}
```

4、工具调用消息：工具调用结果匹配的消息类型。将此消息返回给模型，让模型基于这个结果继续生成回复

```json
{"role": "tool", "content": "今天天气很好", "tool_call_id": "call_00_nUD2NC9QRN5Cg1GaoIkBJQ4s"}
```

### 对象格式的消息

1、系统消息：

```json
SystemMessage(content="你是个编程助手")

相当于
SystemMessage("你是个编程助手")

content 可以省略
```

2、用户消息：

```json
HumanMessage(content="介绍Python主要特点~")
```

3、助手(AI)消息：

```json
AIMessage("Python 的主要特点可以概括为这几类：......")
```

4、工具调用消息：工具调用结果匹配的消息类型。将此消息返回给模型，让模型基于这个结果继续生成回复

```json
ToolMessage(
content="<工具输出>",
tool_call_id="call_00_nUD2NC9QRN5Cg1GaoIkBJQ4s" # 一定要和AI消息中的调用ID匹
配
)
```

# 2、消息示例代码

## 2.1、json格式案例

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv(override=True)

model = init_chat_model (
    model = "gpt-5.4",
    model_provider = "openai",
)

messages = [
    {"role": "system", "content": "你是一个资深程序员，帮助用户解决编程问题"},
    {"role": "user", "content": "什么是机器学习？"}
]

response = model.invoke(messages)
print(response.content)
```

```
机器学习是让计算机从数据中学习规律，而不是靠人把每一步规则都写死。

简单说：
- 你给它很多样本
- 它自己找出输入和输出之间的关系
- 之后遇到新数据时，能做预测、分类或决策

常见例子：
- 垃圾邮件识别
- 图片里识别人脸
- 推荐你可能喜欢的电影或商品
- 语音转文字

它通常分三类：
- 监督学习：有标准答案，学输入到输出的映射
- 无监督学习：没有标准答案，自己找结构
- 强化学习：通过试错和奖励来学习策略

一句话概括：机器学习就是“让机器通过数据自动改进表现”的方法。
```

## 2.2 对象格式案例

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv(override=True)

model = init_chat_model (
    model = "gpt-5.4",
    model_provider = "openai",
)

messages = [
    SystemMessage("你是一个资深程序员，帮助用户解决编程问题"),
    HumanMessage("什么是tcp三次握手？")
]

response = model.invoke(messages)
print(response.content)
```

```
TCP 三次握手是客户端和服务端在建立 TCP 连接前，互相确认“你能发、我能收、初始序号是多少”的过程。

流程分 3 步：

1. 客户端发 `SYN`
   客户端告诉服务端：我要建立连接，并发送自己的初始序号 `seq=x`。

2. 服务端回 `SYN + ACK`
   服务端收到后，回复客户端：我收到了你的请求，同时我也准备好了建立连接。
   这里会返回：
   - `ACK=x+1`，表示确认收到了客户端的 `SYN`
   - `seq=y`，表示服务端自己的初始序号

3. 客户端再回 `ACK`
   客户端收到后，再发一个确认：
   - `ACK=y+1`
   到这一步，双方都确认连接可以建立，TCP 连接正式建立。

可以简化理解为：

- 客户端：`我想连你`
- 服务端：`可以，我也能连你`
- 客户端：`收到，那我们开始`

为什么要三次，核心原因是：

- 确认双方的收发能力都正常
- 同步双方的初始序号
- 防止旧的、失效的连接请求突然到达，造成误建立连接

如果你愿意，我也可以继续讲：
- 为什么不是两次握手
- 四次挥手和三次握手的区别
- 用抓包角度看三次握手的报文结构
```

# 3、模型的调用

在 LangChain 中，模型调用（Invocation）是指通过特定方法触发大语言模型生成输出的过程。

根据不同的应用场景和需求，LangChain 提供了几种核心的调用方式，主要是 invoke() 、 stream() 和
batch() 方法，以及它们的异步版本 ainvoke() 、 astream() 和 abatch() ，下面将系统地介绍这些方
法。

---

**调用方式一览：**

- **invoke()：** 阻塞式，一次性返回完整结果问答、批处理任务、无需实时反馈的场景。

```python
response = model.invoke("你好，请介绍一下你自己。")
```

- **stream()：** 流式输出，实时返回每个token聊天机器人、长文本生成、需要提升用户体验的交互应用。

```python
for chunk in model.stream("写一首关于秋天的诗"):
    print(chunk.content, end="", flush=True)
```

- **batch()：** 批量处理多个输入高并发场景，需要同时处理大量请求。

```python
responses = model.batch(["介绍Python", "介绍Java", "介绍C++"])
```

- **ainvoke() / asteam() / abatch() ：** 非阻塞式，提高系统吞吐量高并发Web应用、IO密集型任务。

```python
response = await model.ainvoke("异步调用的提示词")

async for chunk in llm.astream("写一首关于秋天的诗"):
    print(chunk.content, end="", flush=True)
    

results = await llm.abatch(["介绍Python", "介绍Java", "介绍C++"])
# rprint(results)
contents = [res.content for res in results]
print(contents)
```

**一般常用 invoke 和 stream 方法**

## **stream 方法示例**

```python
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv(override=True)

llm = ChatOpenAI(
    model="gpt-4o-mini",
)

messages = [
    HumanMessage("请以雪为题眼，写一首诗词")
]

for i in llm.stream(messages):
    print(i.content, end="", flush=True)
```