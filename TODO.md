# Agent 文献搜索与管理 Demo — TODO

## 1. 项目目标

通过“交互式教学 + 实战”的方式，从零实现一个最小可运行的文献搜索与管理 Agent。

当前阶段重点不是直接使用 LangGraph、OpenAI Agents SDK 或 Multi-Agent 框架，而是先手写并理解 Agent 的基础机制：

```text
Agent = LLM + Tools + Loop + State
```

最终希望逐步完成：

```text
V1：最小 Agent Loop
    ↓
V2：文献搜索 + PDF 阅读 + 文献管理 + RAG
    ↓
V3：Planning + Memory + Graph / Multi-Agent
```

---

## 2. 学习与实现原则

- [ ] 先理解底层机制，再使用高级框架
- [ ] 不一次性生成整个项目
- [ ] 每次只实现一个小模块
- [ ] 每段代码都要明确“为什么存在”
- [ ] 每实现一个模块，都先验证再继续
- [ ] 必要时通过小问题或选择题确认理解
- [ ] 优先保证数据流清晰，而不是追求工程复杂度
- [ ] V1 不提前加入 Planning、Memory、RAG、Graph、Multi-Agent、MCP

---

## 3. V1：最小 Agent Demo

### 3.1 V1 目标

完成下面这条最小链路：

```text
main
  ↓
state
  ↓
agent
  ↓
llm
  ↓
tool_call
  ↓
runtime
  ↓
tool
  ↓
tool_result
  ↓
state
  ↓
llm
  ↓
final answer
```

第一阶段甚至不接真实 LLM，而是使用 Fake / Mock LLM，先验证 Agent Loop 是否正确。

### 3.2 V1 最小功能

- [ ] 用户输入一个文献相关请求
- [ ] 创建当前任务 State
- [ ] Agent 调用 LLM 接口
- [ ] LLM 返回统一格式的 `tool_call` 或 `final`
- [ ] Runtime 根据工具名查找 Tool Registry
- [ ] Runtime 执行 Tool
- [ ] Tool 返回结构化结果
- [ ] Tool Result 写入 State / messages
- [ ] Runtime 再次调用 LLM
- [ ] LLM 最终返回 Final Answer
- [ ] 设置 `max_steps`，避免无限循环
- [ ] 对工具不存在、参数错误、工具异常进行基本处理

---

## 4. V1 项目结构

```text
agent_demo/
├── main.py
├── agent.py
├── runtime.py
├── state.py
├── tools.py
└── llm.py
```

### `main.py`

职责：

- [ ] 程序入口
- [ ] 读取用户输入
- [ ] 初始化 State
- [ ] 创建 Agent / Runtime
- [ ] 启动 Agent Loop
- [ ] 输出最终结果

### `state.py`

第一版只使用普通 `dict`。

```text
state = {
    user_query,
    messages,
    step
}
```

职责：

- [ ] 创建初始 State
- [ ] 保存用户原始请求
- [ ] 保存 Agent 执行历史
- [ ] 保存当前 step

暂不使用：

- dataclass
- TypedDict
- Pydantic
- 长期 Memory

### `tools.py`

第一版只实现少量、单一职责的 Tool。

首个 Tool：

```text
search_paper(query)
```

建议输入：

```text
query: string
```

建议输出：

```text
{
    found: true / false,
    title: ...,
    pdf_url: ...
}
```

TODO：

- [ ] 实现 `search_paper`
- [ ] 保证输入结构化
- [ ] 保证输出结构化
- [ ] 区分正常业务失败和真正异常

正常业务结果示例：

```text
{
    found: false,
    title: null,
    pdf_url: null
}
```

真正异常包括：

- 网络超时
- API Key 无效
- 数据解析失败
- 服务不可用

后续再增加：

- [ ] `read_paper`
- [ ] `save_paper`
- [ ] `list_library`

### `llm.py`

目标：屏蔽不同模型厂商 API 的差异。

概念接口：

```text
call_llm(messages, tools)
```

统一返回：

```text
LLMResponse

{
    type: "final" | "tool_call",
    content: string | null,
    tool_name: string | null,
    tool_arguments: dict | null
}
```

Tool Call 示例：

```text
{
    type: "tool_call",
    content: null,
    tool_name: "search_paper",
    tool_arguments: {
        query: "TASA"
    }
}
```

Final 示例：

```text
{
    type: "final",
    content: "...",
    tool_name: null,
    tool_arguments: null
}
```

TODO：

- [ ] 定义统一 LLMResponse
- [ ] 实现 Fake / Mock LLM
- [ ] 第一次调用固定返回 `search_paper("TASA")`
- [ ] 第二次调用读取 Tool Result 后返回 Final Answer
- [ ] 保证 Runtime 不依赖任何厂商特有字段

以后再考虑：

```text
llm/
├── base.py
├── openai_adapter.py
├── local_adapter.py
└── mock_adapter.py
```

但 V1 不做这层工程化。

### `agent.py`

职责：

- [ ] 定义 Agent instructions
- [ ] 定义模型配置
- [ ] 定义 allowed tools
- [ ] 将 messages 和 Tool Schema 交给 `llm.py`
- [ ] 返回统一 LLMResponse

Agent 负责“决策接口”，不负责真正执行 Tool。

### `runtime.py`

Runtime 是 V1 中最关键的部分。

职责：

- [ ] 实现 Agent Loop
- [ ] 维护 Tool Registry
- [ ] 接收 Tool Call
- [ ] 检查工具是否存在
- [ ] 检查参数
- [ ] 调用真实 Python Tool
- [ ] 捕获工具异常
- [ ] 将 Tool Result 写入 messages
- [ ] 更新 `step`
- [ ] 控制 `max_steps`
- [ ] 在 Tool 执行后重新调用 LLM
- [ ] 收到 `final` 后退出 Loop

核心思想：

```text
LLM = 决策
Runtime = 执行与约束
Tool = 具体能力
```

---

## 5. V1 推荐实现顺序

不要按照文件在目录中的顺序机械实现，而是按照依赖关系逐步搭建。

### Step 1：明确最小需求和数据流

- [ ] 明确用户输入是什么
- [ ] 明确第一个 Tool 是什么
- [ ] 明确 Tool Call 格式
- [ ] 明确 Tool Result 格式
- [ ] 明确 State 中最少需要保存什么
- [ ] 画出一次完整 Agent Loop

验收标准：能够不看代码，口头解释一次完整执行过程。

### Step 2：实现 `state.py`

- [ ] 创建初始 State
- [ ] 验证 `user_query / messages / step`

验收标准：能够创建和打印一个完整初始 State。

### Step 3：实现 `tools.py`

- [ ] 先实现一个假的 `search_paper`
- [ ] 输入 `TASA`
- [ ] 固定返回结构化论文信息

验收标准：独立调用 Tool 能得到稳定、可序列化结果。

### Step 4：实现 `llm.py`

- [ ] 实现 Fake LLM
- [ ] 第一次返回 Tool Call
- [ ] 第二次返回 Final

验收标准：Fake LLM 的输出完全符合统一 LLMResponse 格式。

### Step 5：实现 `agent.py`

- [ ] 定义 instructions
- [ ] 定义 allowed tools
- [ ] 将 Tool Schema 传给 LLM
- [ ] 返回 LLMResponse

验收标准：Agent 层不执行 Tool，也不包含 Runtime 逻辑。

### Step 6：实现 `runtime.py`

- [ ] 建立 Tool Registry
- [ ] 实现 Tool dispatch
- [ ] 实现最小 Loop
- [ ] Tool Result 写回 State
- [ ] 实现 `max_steps`
- [ ] 实现最小错误处理

验收标准：Fake LLM + Fake Tool 可以完整跑通一次 Agent Loop。

### Step 7：实现 `main.py`

- [ ] 接收用户输入
- [ ] 创建 State
- [ ] 启动 Runtime
- [ ] 打印 Final Answer

验收标准：可以从命令行完整运行 V1 Demo。

### Step 8：理解验证

确认可以解释：

- [ ] 为什么 LLM 不能直接调用 Python 函数
- [ ] Tool Schema 与 Tool Registry 的关系
- [ ] 为什么 Tool 执行后要再次调用 LLM
- [ ] Agent 与固定 Workflow 的区别
- [ ] 为什么 Runtime 负责 `max_steps` 和 retry
- [ ] State 和 Memory 的区别
- [ ] 为什么模型厂商差异要收敛到 `llm.py`

---

## 6. V1 完成标准

只有下面这些全部完成后，再进入 V2：

- [ ] 不依赖任何 Agent 框架
- [ ] Fake LLM 能正常工作
- [ ] Tool Schema 与 Tool Registry 分离
- [ ] Agent Loop 可正常结束
- [ ] 每次 Tool Result 都进入 State
- [ ] Tool 执行后由 LLM 重新决策
- [ ] `max_steps` 可以阻止死循环
- [ ] Tool 错误不会直接导致整个程序无控制崩溃
- [ ] Runtime 不包含 OpenAI / Anthropic 等厂商特有解析逻辑
- [ ] 能清楚解释整个数据流

---

## 7. V2：文献搜索、阅读、管理与 RAG

V1 完全跑通后再开始。

目标：

```text
最小 Agent Loop
    ↓
真实文献搜索
    ↓
论文读取
    ↓
本地文献管理
    ↓
RAG / Chunk Retrieval
```

TODO：

- [ ] 将 Fake `search_paper` 替换成真实搜索能力
- [ ] 增加 `read_paper`
- [ ] 设计 PDF 文本存储方式
- [ ] 长 PDF 不直接全部塞入 messages
- [ ] 实现 chunking
- [ ] 实现相关段落检索
- [ ] 只把当前相关 chunks 放入 Context
- [ ] 实现 `save_paper`
- [ ] 实现 `list_library`
- [ ] 定义最小文献 metadata
- [ ] 区分 State、Memory、RAG、Context

V2 暂时仍不要求 Multi-Agent。

---

## 8. V3：高级 Agent 机制

在 V2 已经具备真实文献研究能力后，再逐步增加高级机制。

### Planning

- [ ] Planner
- [ ] Executor
- [ ] 结构化 Plan
- [ ] 必要时 Replan

原则：不是每次 Tool Call 后都 Replan，只有计划明显失效时才重新规划。

### Memory

- [ ] 跨任务长期信息存储
- [ ] 用户偏好
- [ ] 已读论文
- [ ] 文献状态
- [ ] 长期研究上下文

### Graph

理解并实现：

```text
Graph = State + Node + Edge
```

- [ ] Node
- [ ] Edge
- [ ] Conditional Edge
- [ ] State transition

再考虑是否引入 LangGraph。

### Multi-Agent

可能的结构：

```text
Orchestrator
├── Research Agent
├── Reader Agent
├── Library Agent
└── Writer Agent
```

TODO：

- [ ] 明确每个 Agent 的职责边界
- [ ] 为不同 Agent 设置最小 Tool 权限
- [ ] Agent 间使用结构化 Schema 通信
- [ ] 区分 Agent 与 Tool
- [ ] 再决定是否需要真正的 Multi-Agent

---

## 9. 当前明确不做的内容

在 V1 阶段，以下内容全部暂缓：

- [ ] LangGraph
- [ ] OpenAI Agents SDK
- [ ] Multi-Agent
- [ ] Planner / Replanner
- [ ] 长期 Memory
- [ ] 完整 RAG
- [ ] Vector Database
- [ ] MCP
- [ ] 复杂异步执行
- [ ] 并发 Tool Calling
- [ ] 复杂权限系统
- [ ] 复杂 Pydantic Schema
- [ ] Provider Adapter 抽象层级过度工程化

原则：只有当前一层机制完全理解并跑通后，才增加下一层复杂度。

---

## 10. 当前下一步

从这里继续实战：

- [ ] 明确 V1 文献 Agent 的最小用户需求
- [ ] 确定 V1 第一版 Tool 数量
- [ ] 确定一次完整的数据流
- [ ] 开始实现 `state.py`

当前建议只保留一个 Tool：

```text
search_paper
```

并使用 Fake LLM + Fake Search Tool 跑通第一条 Agent Loop。
