# AgentReader Demo — 开发路线

## 1. 项目目标

通过“交互式教学 + 实战”理解并实现文献 Agent 的基础机制：

```text
Agent = LLM + Tools + Loop + State
```

开发顺序保持为：

```text
V1：最小 Agent Loop
    ↓
V2：真实搜索 + 文献阅读 + 文献管理 + 最小检索
    ↓
V2.0.1：提取正确性 + 交互式终端
    ↓
V2.1：全文覆盖 + 检索质量评测
    ↓
V3：Planning + Replanning + Checkpoint
```

当前状态：V1、V2、V2.0.1 和 V2.1 均已完成。V3 已完成 Plan 纯数据层、Planner
输入边界、Runtime 可信 Plan 创建、最小 step Executor 和 DeepSeek thinking 多步
消息协议；尚未接入 Replan、blocked 或 Checkpoint。目标是让单 Agent 对多论文
任务进行显式规划、有限重规划，并在进程退出后恢复同一任务。

---

## 2. 开发原则

- 先理解底层机制，再考虑高级框架。
- 每次只实现一个小步骤，并解释信息流和设计原因。
- 每个步骤完成后运行相关测试。
- 优先保证数据流清晰，避免提前抽象和过度工程化。
- 不读取或提交 `.env`、API Key 和个人文献库。

---

## 3. V1：最小 Agent Loop（已完成）

### 核心链路

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
tool_result 写回 state
  ↓
llm
  ↓
final answer
```

### 已完成能力

- [x] `main.py` 接收用户输入并输出最终答案。
- [x] `state.py` 使用普通 `dict` 保存 `user_query`、`messages` 和 `step`。
- [x] `agent.py` 管理 instructions 和 allowed tools，不直接执行 Tool。
- [x] `llm.py` 将不同 Provider 的输出归一为 `tool_call` 或 `final`。
- [x] Fake LLM 可以离线、确定性地跑通完整 Agent Loop。
- [x] Runtime 维护 Tool Registry，并根据工具名分发 Python 函数。
- [x] Tool Result 会写回 messages，再交给 LLM 决策。
- [x] `max_steps` 可以阻止无限循环。
- [x] 未注册 Tool、参数绑定错误和 Tool 异常会转成结构化错误。
- [x] Provider 特有响应解析只存在于 `llm.py`。
- [x] `--debug` 可以显示 LLM 决策、Tool Call、Tool Result 和步数。

### V1 需要持续保持的边界

- 不引入 LangGraph 或 OpenAI Agents SDK。
- 不引入 Planning、长期 Memory、Graph 或 Multi-Agent。
- 不把完整 RAG、Vector Database 或 MCP 塞进基础 Agent Loop。
- Runtime 负责执行和约束，LLM 只负责提出下一步动作。

---

## 4. V2：已经完成的原型能力

### 真实文献搜索

- [x] `search_paper(query)` 使用结构化输入和输出。
- [x] 优先访问 `export.arxiv.org`。
- [x] arXiv 不可用或没有结果时回退到 Crossref。
- [x] 最多返回 3 篇候选论文。
- [x] 返回统一论文 metadata。
- [x] 区分“没有结果”和网络、解析、服务异常。
- [x] Agent instructions 要求比较候选，信息不足时展示歧义。
- [x] 默认测试 Mock 网络响应，不依赖外部服务。
- [x] 真实搜索测试通过环境变量显式启用。

### 模型 Provider

- [x] 支持 Fake LLM。
- [x] 支持 DeepSeek API。
- [x] 兼容 OpenRouter 的 OpenAI 风格接口。
- [x] 使用 Conda `environment.yml` 管理 Python 3.13 依赖。
- [x] DeepSeek 在线 Agent Loop 测试通过环境变量显式启用。

### 本地文献保存原型

- [x] 实现 `save_paper(paper)`。
- [x] 默认保存到 `data/library.json`。
- [x] 支持通过 `PAPER_LIBRARY_PATH` 修改保存位置。
- [x] 使用 arXiv ID、DOI 或稳定哈希生成论文 ID。
- [x] 对相同 arXiv ID 或 DOI 去重。
- [x] 使用临时文件替换方式写入 JSON。
- [x] 保存行为有离线测试，不污染真实个人文献库。

### 多轮命令行对话

- [x] Agent 每轮回答后继续等待用户输入。
- [x] 后续消息追加到同一个 State，保留候选和 Tool Result。
- [x] 用户可以回答“第 1 篇”来消除上一轮搜索歧义。
- [x] `max_steps` 按每轮用户输入单独计数，同时保留累计 step。
- [x] 空输入继续等待；支持 `/exit`、`exit`、`quit`、`退出` 或 EOF 结束会话。
- [x] Debug Trace 每轮只显示本轮产生的新步骤。

### 用户触发的重新搜索

- [x] 用户否定旧候选并提供新线索时，允许再次调用 `search_paper`。
- [x] Agent instructions 要求新的 query 保留用户补充的领域信息。
- [x] 要求模型每个响应最多提出一个 Tool Call。
- [x] Provider 仍返回多个 Tool Calls 时，只取第一个进入串行 Agent Loop。
- [x] query 包含明确 arXiv ID 时使用 `id_list` 定向查询。
- [x] arXiv HTTP 429 时退避重试一次，并记录最终回退原因。
- [x] 记录并拒绝同一轮内完全重复的 query。
- [x] 提供最低词法相关性状态，并限制单轮搜索次数。
- [x] 扩大初始候选池并根据标题、摘要做本地重排。

---

## 5. 可靠的保存边界（已完成）

当前已完成多轮澄清和保存目标的来源校验：用户可以继续选择上一轮候选，LLM
只提交 `candidate_id`，Runtime 只接受当前任务中 `search_paper` 真正返回过的
候选。Runtime 在调用存储函数前还会通过终端确认器展示确切目标；缺少确认器、
直接回车或用户拒绝时都不会写入。

### 目标流程

```text
search_paper 返回可信候选
    ↓
LLM 只选择 candidate_id
    ↓
Runtime 从历史 Tool Result 中解析对应论文
    ↓
终端展示待保存论文并请求确认
    ↓
save_paper 写入本地文献库
```

### 已完成

- [x] 为搜索候选定义稳定的 `candidate_id`。
- [x] LLM 调用保存 Tool 时只提交 `candidate_id`，不复制整篇 metadata。
- [x] Runtime 验证候选确实来自当前任务的历史搜索结果。
- [x] 测试不存在的候选和复制整份 metadata 的非法调用。
- [x] 在本地文件写入前增加明确的用户确认。
- [x] 用户拒绝时，把结构化拒绝结果写回 State，让 LLM 正常收尾。
- [x] 测试确认和拒绝流程。

完成标准：模型不能凭空构造一篇论文并写入文献库，用户可以在写入前看到并
确认确切目标。

---

## 6. 当前及后续 V2 路线

### Step 1：拆分 Tools 包（已完成）

- [x] 创建 `tools/` 包。
- [x] 将 Tool Schema、搜索实现和文献库实现分开。
- [x] 通过 `tools/__init__.py` 保持稳定的公共导入接口。
- [x] 只做结构重构，不同时改变行为。

### Step 2：查看文献库（已完成）

- [x] 实现只读的 `list_library` Tool。
- [x] 支持列出已保存论文的最小 metadata。
- [x] 默认返回最近 20 篇，单次最多返回 50 篇。
- [x] 为不存在、空库和损坏 JSON 增加测试。

### Step 3：读取论文（已完成）

- [x] 实现受限的 PDF 下载和本地缓存。
- [x] 明确文件大小、超时、保存路径和失败处理。
- [x] 只接受当前 State 中搜索或文献库查询返回过的稳定 ID。
- [x] Tool Result 不包含 PDF 二进制内容。
- [x] 实现 PDF 文本提取。
- [x] 只读取当前 State 中可信下载结果对应的缓存路径。
- [x] 默认限制为开头 5 页和 12000 字符，硬上限为 10 页和 20000 字符。
- [x] 用结构化结果区分空文本、页数截断和字符数截断。
- [x] 加密、损坏 PDF 和越界路径返回结构化错误。
- [x] 不把长论文全文直接放进 messages。

### Step 4：最小段落检索（已完成）

- [x] 实现按页切分的固定字符窗口 chunking。
- [x] 为每个 chunk 保留稳定 ID、页码和页内字符区间。
- [x] 限制 chunk 大小和 overlap，并继承源文本截断状态。
- [x] 使用无外部依赖的 TF-IDF 余弦相似度进行相关性排序。
- [x] 支持英文单词和中文双字片段，定义无匹配和确定性同分顺序。
- [x] 默认只选 top 3，硬上限为 top 5。
- [x] 注册 `retrieve_paper_chunks` Tool，只把相关 chunks 放入当前 Context。
- [x] 明确区分 State、长期 Memory、RAG 和 Context。

概念边界：State 是当前命令行会话内保留的消息记录；Context 是一次模型调用实际
接收的 instructions、历史消息和 Tool Result；长期 Memory 是跨任务保留并按需
召回的信息，本项目尚未实现；RAG 是“检索外部知识后再生成”的整体模式。V2 的
这一小步当时只有有界词法检索；V2.1 已增加持久化全文索引，但仍没有向量库、
语义检索或跨会话 Memory，因此仍不称为完整 RAG。

### Step 5：搜索相关性边界（已完成）

- [x] 记录并拒绝同一轮内经过空白规范化后完全重复的 query；下一用户轮次重置。
- [x] 搜索源最多取 10 个候选，基于标题和摘要做确定性词法重排。
- [x] 返回词法得分、命中位置和原始来源排名，最终仍只向模型返回前 3 个。
- [x] 精确标题命中优先于只包含完整标题短语的扩展标题。
- [x] 用四种状态表达最低相关性信号，不把无词法匹配伪装成相关结果。
- [x] 每个用户轮次最多执行 2 次不同搜索，第三次返回结构化错误。
- [x] 搜索 Debug Trace 使用有界摘要预览，不让长结果遮住最终回答和下一轮提示。

### Step 6：V2 端到端验收（已完成）

- [x] 使用真实 arXiv + DeepSeek 检查一次强匹配搜索。
- [x] 使用真实 arXiv + DeepSeek 检查一次弱匹配或无匹配后的澄清行为。
- [x] 检查搜索、保存、下载、提取和段落检索的 Debug Trace 与文档一致。
- [x] 汇总 V2 已知限制，并选择先改进全文检索再评估 V3。

### Step 7：V2.0.1 提取正确性与交互式终端（已完成）

- [x] 区分 PDF 的 `pages_scanned`、`pages_read`、`page_numbers` 和部分页状态。
- [x] 防止 PDF 正文中的 `[Page N]` 伪造结构页码，并校验提取 metadata。
- [x] 为字符边界、空白页、部分页和伪造页码增加回归测试。
- [x] Runtime 通过可选只读事件回调暴露 LLM/Tool 生命周期。
- [x] 隔离事件观察者异常和可变数据，避免 UI 改变 Agent 执行。
- [x] LLM 或 Tool 被中断时补齐内部消息协议，安全结束当前轮。
- [x] 实现 Prompt Toolkit + Rich 增强终端和无 ANSI 的 `--plain` 模式。
- [x] 支持进程内历史、`/help`、`/debug`、`/clear`、`/exit`。
- [x] Slash Commands 不进入 State；非交互失败返回非零状态码。
- [x] 完成 Plain 管道、真实 TTY 和真实 arXiv + DeepSeek + PDF 系统测试。

---

## 7. V2.1：全文覆盖与检索质量（已完成）

- [x] 选择首次检索时自动建立的持久化本地 JSON 索引。
- [x] 使用索引格式版本、PDF SHA-256 和提取上限判断缓存是否有效。
- [x] PDF 变化、索引损坏或契约过期时自动原子重建。
- [x] 限制为最多 200 页、200 万字符和 32 MiB 索引文件。
- [x] 保持 `retrieve_paper_chunks` 模型接口不变，只返回 top-k 片段。
- [x] 先保持 TF-IDF，把全文覆盖与排序算法升级分开。
- [x] 建立固定论文与问题组成的检索质量评测集。
- [x] 定义 Recall@K、Hit Rate@K、MRR 和无答案准确率，并把当前 Top-3 结果锁为
  回归快照；中英文与无答案类别分别报告。
- [x] 用隔离的合成 PDF 测量首次检索与缓存命中的时延、索引磁盘占用和新增
  Tool Result Context 载荷；不把机器相关时间设为测试阈值。
- [x] 在同一评测集上比较原始/带门槛的 TF-IDF 与 BM25。
- [x] 增加 50% 查询词覆盖门槛，使无答案准确率从 0.500 提升到 1.000，同时保持
  Recall@3、Hit Rate@3 和 MRR 为 0.875。
- [x] 根据指标持平的结果保留 TF-IDF 作为默认排序；BM25 保留为评测选项。
- [x] 判断 V2.1 暂不引入 embedding/hybrid；剩余跨语言失败留给后续查询改写或
  语义检索实验。
- [x] 全文索引、评测、资源测量和最终算法选择均已同步更新
  `docs/agent-development-process.md`。

---

## 8. V3：可恢复的单 Agent 计划执行器（最小 Executor 已完成）

目标任务示例：搜索两篇指定方向的代表论文，下载 PDF，分别检索方法和实验依据，
最后给出带论文与页码证据的比较。V3 不改变已有 Tool 的可信 ID、人工确认和有界
Context 原则。

### Step 1：Plan 数据契约和状态机

- [x] 新增版本化 Plan 结构，包含任务目标、任务状态、revision 和有序 steps。
- [x] 每个 step 包含稳定 ID、目标描述、状态、尝试次数和证据引用；不直接保存任意
  URL、路径或模型伪造的 Tool Result。
- [x] 支持 `pending → running → completed|blocked|failed`；用户补充必要信息后，
  `blocked` 可恢复为 `running`。
- [x] Plan 最多 8 步，所有转换返回新的已校验 Plan，不原地修改输入。
- [x] 使用内部结构化 `submit_plan` 契约取得模型计划，不把它注册为外部 Tool。
- [x] 独立 Planner 的初次模型决策可为复合任务选择 `submit_plan`，简单任务继续走现有
  `tool_call|final`；避免再增加一次分类模型调用或脆弱的关键词路由。
- [x] 为合法计划、非法结构、非法状态转换和越界步骤增加纯函数测试。

### Step 2：Planner、Executor 与有限 Replanning

- [x] Runtime 在没有活动 Plan 时使用 Planner 的首次决策，并从 `plan` 动作创建可信
  Plan；普通 `tool_call|final` 保持 V2 路径。
- [x] Runtime 独占 Plan 状态、任务预算和可信证据引用的写权限；模型返回的 Tool
  Call ID 不会直接成为 evidence reference。
- [ ] Planner 生成面向结果的高层步骤，不硬编码具体 Tool 调用序列。
- [x] Executor 复用现有 Agent Loop，每次仍只执行一个 Tool Call。
- [x] Runtime 启动第一个 pending step，将当前目标注入模型 Context；模型返回
  step-level final 后才完成该 step，而不是把任意一次 Tool 成功直接当作完成。
- [ ] Tool 失败、候选歧义或检索证据不足时才允许 Replan；普通成功步骤不重复规划。
- [x] Runtime 记录任务级 LLM/Tool 用量，并在执行前阻止超过 32 次 LLM 决策或
  24 次 Tool 执行。
- [ ] 每个任务最多 Replan 2 次。
- [ ] Replan 必须保留已完成步骤、可信结果和证据，不能重复有副作用的操作。
- [ ] 缺少关键用户选择时进入 `blocked` 并请求澄清，不得由模型擅自补全。
- [ ] 保存论文仍逐篇经过 `y/N` 确认；原始请求没有授权的副作用不能被 Plan 扩大。
- [ ] 为成功、失败恢复、阻塞、预算耗尽和重复副作用增加 Runtime/Fake LLM 测试。

当前最小 Executor 已打通 `pending → running → completed`：创建 Plan 后先返回预览，
下一条用户消息开始执行；每次 Tool Result 在对应 Tool 消息和当前 step 中写入同一个
Runtime-owned `tool-result-NNN` 引用，Tool 本身不会完成 step。多个 step 在同一个
现有 Loop 内串行推进，step-level final 才触发完成；达到单轮 `max_steps` 时保留活动
Plan，后续轮次可以继续。单轮默认值已由 5 调整为 20；如果活动 Plan 仍达到上限，
终端用中文说明任务只是暂停并提示输入“继续”。任务级 32 次 LLM 决策和 24 次 Tool
执行上限保持不变。新增 2 项 Executor 测试、4 项 Runtime 执行/预算测试和 1 项
evidence 转换测试；最小 Executor 阶段真实 arXiv + DeepSeek 全量 142 项通过、无
跳过。

### Step 2 前置：DeepSeek thinking 消息协议升级（已完成）

开始修改 thinking 协议前必须先完成以下 Git 边界，避免把已经完成的 Plan/Planner
工作与 Provider 消息协议混在同一个提交中：

- [x] Codex 先提醒用户提交当前工作，不得直接开始修改 `llm.py`。
- [x] 用户完成提交后，检查 `git status` 确认工作区干净。
- [x] 从已提交的当前分支创建并切换到新分支，例如：
  `git checkout -b feat/deepseek-thinking-protocol`。
- [x] 再次确认分支名和工作区状态，之后才能开始 thinking 协议修改。

thinking 协议的实现范围：

- [x] 在 `llm.py` 中显式配置 thinking 开关和 reasoning effort，不依赖模型默认行为。
- [x] `_normalize_api_response()` 提取 Provider 返回的 `reasoning_content`，同时兼容
  非 thinking 响应没有该字段的情况。
- [x] Runtime 将 `reasoning_content` 作为不透明 Assistant 消息字段写入 State；不
  解析、不修改，也不将它当作文献证据。
- [x] `_to_api_messages()` 在带 Tool 的后续 DeepSeek 请求中原样回传对应
  `reasoning_content`，保持 Tool Calling 消息协议完整。
- [x] 普通终端和 Debug 默认不显示 reasoning 内容；后续 Checkpoint 是否保存该字段
  必须在实现恢复前单独评估大小、隐私和 Provider 兼容性。
- [x] 增加响应解析、消息序列化、thinking Tool Call 多轮回传、非 thinking 兼容和
  OpenRouter 缺少或使用不同 reasoning 字段时的测试。
- [x] 完成后显式运行 DeepSeek 联网测试，不得把联网用例计为跳过，并同步更新
  `docs/agent-development-process.md`。

修复结果：DeepSeek 默认显式启用 thinking，并将每个模型 Assistant 响应的
`reasoning_content` 作为不透明字段写入 State、在所有后续请求中原样回传；
OpenRouter 同时兼容字符串 reasoning 和结构化 `reasoning_details`，但默认不强制
开启推理。普通与 Debug 终端都不显示该字段。新增 13 项协议、Planner、Runtime 和
真实 DeepSeek 集成测试；启用真实 arXiv 与 DeepSeek 后，全量 155 项通过、无跳过。
OpenRouter 兼容性当前由 Mock API 响应和请求参数覆盖，尚未使用真实 OpenRouter
Key 验收。下一步进入有限 Replan 与 `blocked`。

### Step 3：Checkpoint 与恢复

- [ ] 同一进程只允许一个活动计划任务，使用稳定 `task_id` 标识。
- [ ] 每次 Plan 或步骤状态变化后，将版本化 Checkpoint 原子写入
  `data/checkpoints/active.json`；文件最大 4 MiB，并被 Git 忽略。
- [ ] Checkpoint 不保存 API Key、PDF 二进制、全文索引或其他未进入 State 的数据。
- [ ] `python main.py --resume` 校验并恢复 `running|blocked` 任务；损坏或过期结构
  必须报错，不能静默覆盖。
- [ ] Ctrl+C 或可恢复错误后保留可恢复 Checkpoint；任务完成、失败或明确取消后清除
  活动 Checkpoint。Checkpoint 是同一任务恢复，不是跨任务长期 Memory。
- [ ] 启动时若发现活动 Checkpoint，必须提示使用 `--resume`，不能由新任务静默覆盖。
- [ ] 恢复时重新校验 PDF 缓存与可信 ID，已完成步骤不得再次执行。
- [ ] 为写入中断、损坏文件、版本失效、超限、恢复和幂等行为增加测试。

### Step 4：终端、事件与验收

- [ ] 增加 `/plan` 查看当前计划，增加 `/cancel` 明确取消活动任务，并同步 `/help`。
- [ ] 活动任务未取消时，`/clear` 不得静默丢弃其 Checkpoint。
- [ ] 活动任务执行期间，普通用户消息只用于补充或澄清该任务；开始无关目标前必须
  先完成或取消当前任务。
- [ ] Runtime 增加 UI 无关的计划创建、步骤转换、Replan、Checkpoint 和阻塞事件；
  Debug 输出保持有界。
- [ ] 建立确定性的两篇论文规划评测，记录完成率、Replan 次数、LLM/Tool 步数和
  Checkpoint 大小。
- [ ] 使用真实 arXiv、DeepSeek 和 PDF 完成一次多论文端到端任务；完整联网测试
  不得跳过。
- [ ] 将实现、限制、测试结果和关键决策同步到全部项目文档。

### V3 明确非目标

- 不实现长期跨任务 Memory。
- 不实现 Graph / LangGraph、Multi-Agent 或 MCP。
- 不实现 Embedding、Vector Database 或跨语言语义检索。
- 不实现并发 Tool Calling、后台任务或复杂异步调度。
- 不重写现有 Provider Adapter 或已有检索算法。

V3 完成后再根据实际评测决定下一项能力，而不是自动把所有高级 Agent 概念加入
项目。
