# Agent 开发流程与 AgentReader 项目复盘

## 1. 文档定位

本文是一份持续维护的 Agent 开发手册，也是 AgentReader Demo 的阶段复盘。
它不重复 README 的使用说明，也不替代需求和架构文档，而是回答四个问题：

1. 一个 Agent 项目应当按什么顺序开发？
2. 每个阶段为什么存在，完成标准是什么？
3. AgentReader 在 V1、V2 中做对了什么，又踩过哪些坑？
4. 哪些做法可以迁移到真实项目，哪些只是教学 Demo 的简化？

文档之间的职责如下：

| 文档 | 主要回答的问题 |
|---|---|
| `README.md` | 项目是什么、怎样安装、运行和测试 |
| `TODO.md` | 当前阶段、下一步任务和完成状态 |
| `docs/requirements.md` | 系统现在必须表现出什么行为 |
| `docs/architecture.md` | 当前代码怎样实现这些行为 |
| 本文 | 为什么按这个过程开发，以及经验如何复用 |

本文面向未来真实 Agent 项目，因此会保留当前不足，而不是把 Demo 描述成生产系统。

## 2. 持续维护规则

本文不是一次性总结。在 V2.1 和 V3 中，遇到以下情况必须同步更新：

- 开始一个新版本或里程碑；
- 修改核心信息流、State、Context 或 Tool 契约；
- 改变 Agent 与确定性 Runtime 的职责边界；
- 引入新的外部 Provider、数据源、持久化或副作用；
- 修复会影响设计判断的重要 Bug；
- 完成离线测试、真实集成测试或端到端验收；
- 原计划、非目标或技术选型发生变化。

更新时至少记录：目标、设计决策、验证证据、已知限制和下一步。小型格式调整、
注释修改等不改变设计认知的工作，不需要写入本文。

每个阶段使用同一模板：

```text
阶段目标
用户可见行为
非目标
信息流或契约变化
关键设计决策及备选方案
实现顺序
测试与真实验收证据
遇到的问题及根因
遗留限制
阶段结论
```

重要 Bug 使用下面的格式：

```text
现象 → 最小复现 → 根因 → 修复 → 回归测试 → 可复用经验
```

## 3. 推荐的 Agent 开发主流程

```text
定义用户行为和成功标准
        ↓
划定版本边界和非目标
        ↓
画清信息流与职责边界
        ↓
定义 State、消息协议和 Tool 契约
        ↓
用 Fake LLM 跑通最小垂直闭环
        ↓
接入真实 Provider 与外部服务
        ↓
增加权限、可信引用和资源限制
        ↓
控制 Context，并建立可观测性
        ↓
离线测试 + 质量评测 + 真实集成测试
        ↓
端到端验收、阶段性提交和复盘
        ↓
根据真实问题决定下一版本
```

顺序很重要。先引入框架、规划器或多 Agent，往往会掩盖消息如何流动、Tool 为何
被调用、错误由谁处理等根本问题。

### 3.1 定义用户可见行为和成功标准

先写用户故事，不要先列技术名词。例如：

```text
用户输入论文线索，系统返回真实候选；
用户明确要求保存时，确认后写入本地文献库；
用户针对已下载论文提问，系统返回带页码的相关片段。
```

然后为每个故事写可观察的完成标准：

- 会调用哪个 Tool；
- Tool Result 必须包含哪些字段；
- 没有结果和执行失败怎样区分；
- 什么操作需要用户确认；
- 输出必须披露哪些截断或不确定性。

真实项目还应在此阶段定义质量、时延、成本、成功率和安全指标。没有验收标准，
“模型看起来能回答”无法作为完成依据。

### 3.2 划定版本边界和非目标

每个版本只解决一个清晰层级的问题。AgentReader 的划分是：

- V1：看清 `LLM + Tools + Loop + State` 的最小机制；
- V2：加入真实搜索、文献管理、PDF 读取和有界检索；
- V2.1：优先解决全文覆盖与检索质量，但仍保持 Context 有界；
- V3：只有出现明确需求时，才评估 Planning、Memory 或 Graph 等能力。

非目标不是永久拒绝，而是防止尚未验证基础链路时过早增加复杂度。每次考虑新框架
时，应先回答：它解决了哪个已经复现的问题？不用它的代价是什么？如何验收？

### 3.3 画清信息流与职责边界

在写代码前画出一次完整循环：

```text
User
  → State
  → Agent 组装 instructions/messages/tools
  → LLM 返回 final 或 tool_call
  → Runtime 校验并执行
  → Tool Result 写回 State
  → LLM 观察结果后继续决策
```

关键原则是把概率性判断和确定性约束分开：

| 适合交给 LLM | 必须由代码保证 |
|---|---|
| 理解用户意图 | Tool 是否存在、参数是否合法 |
| 选择查询词或候选 | ID 是否来自可信历史结果 |
| 判断是否需要继续检索 | 次数、页数、字符数和文件大小上限 |
| 根据证据组织自然语言 | 写入确认、路径边界和错误结构 |

Prompt 可以引导行为，但不能代替权限、来源校验、幂等性和资源限制。

### 3.4 定义 State、Context、消息协议和 Tool 契约

State 是 Runtime 持有的任务状态；Context 是某一次模型调用实际接收的内容；长期
Memory 是跨任务保存并按需召回的信息。三者不能混用。

最小消息协议至少需要表达：

```python
{"role": "user" | "assistant", "content": "..."}
{"role": "assistant", "tool_call": {...}}
{"role": "tool", "tool_call_id": "...", "content": {...}}
```

Tool 契约应同时定义：

- 模型可见的名称、用途和 JSON Schema；
- Python 函数实际签名；
- 成功、正常空结果和异常的结构；
- 是否有副作用；
- 权限和可信输入来自哪里；
- 单次调用的时间、大小和结果数量上限；
- Tool Result 中哪些字段会进入模型 Context。

Schema 与实现分开有助于理解，但真实项目必须增加契约一致性测试，避免描述、Schema
和 Python 签名逐渐漂移。

### 3.5 先用 Fake LLM 跑通最小垂直闭环

Fake LLM 的目的不是模拟模型智能，而是确定性验证控制流：

```text
固定 user message
→ 固定 tool_call
→ Runtime 执行测试 Tool
→ Tool Result 写回 messages
→ Fake LLM 返回 final
```

这一阶段应验证：

- Tool Result 是否真的写回 State；
- `tool_call_id` 是否正确配对；
- Runtime 是否按注册表分发；
- 未知 Tool、非法参数和 Tool 异常是否可观察；
- `max_steps` 是否能停止循环；
- Provider 字段是否没有泄漏到 Agent 和 Runtime。

Fake 路径通过后再接真实 API，可以把“循环 Bug”和“模型行为不稳定”分开诊断。

### 3.6 接入真实 Provider 和外部服务

Provider 适配层只负责：

- 配置 base URL、API Key 和模型；
- 把内部消息和 Tool Schema 转为 Provider 格式；
- 把返回值归一为内部 `final` 或 `tool_call`；
- 将 SDK 异常转成稳定的项目错误。

不要让 `agent.py` 或 `runtime.py` 解析 Provider 特有字段。外部服务也应由独立 Tool
封装，并区分正常无结果、限流、超时、响应损坏和认证失败。

真实项目还需要补充超时、重试、退避、速率限制、熔断、成本预算和 Provider 降级
策略。重试次数等确定性策略应在代码中，而不是交给 Prompt 临场决定。

### 3.7 增加权限、可信引用和资源限制

任何写入、下载、发送消息或调用付费服务的操作，都要单独设计副作用边界。

AgentReader 使用了“模型只提交稳定 ID，Runtime 从可信 Tool Result 恢复真实对象”
的模式：

```text
模型提交 candidate_id / paper_id
→ Runtime 在历史 Tool Result 中查找
→ Runtime 恢复 metadata、URL 或本地路径
→ Tool 执行
```

这比允许模型直接提交任意 URL、路径或完整对象更安全。对于不可逆或个人数据写入，
还要在实际执行前取得明确确认。

真实项目应进一步考虑：用户和租户权限、操作审计、域名或服务 allowlist、DNS
解析后的网络边界、并发写入锁、幂等键、事务、密钥隔离和敏感信息脱敏。

### 3.8 控制 Context，并建立可观测性

外部数据不应无界进入 Context。应在 Tool 边界限制：

- 搜索候选数量；
- 文献库返回数量；
- PDF 页数和字符数；
- chunk 数量和单个 chunk 长度；
- Debug Trace 的预览长度。

需要分别记录“源数据是否完整”和“返回给模型的结果是否截断”。显示层预览不能
修改真正写回 State 的 Tool Result。

学习项目使用可读 Debug Trace 足够；真实项目应增加结构化事件，例如 request ID、
conversation ID、step、tool name、duration、result status、retry count、token usage
和 estimated cost，同时避免记录 API Key、完整私人文档或不必要的模型输入。

### 3.9 建立分层测试和质量评测

建议测试金字塔：

1. 纯函数单元测试：解析、校验、排序、chunking、打分；
2. Runtime 契约测试：可信引用、权限、错误和次数限制；
3. Fake LLM 流程测试：确定性验证多步 Agent Loop；
4. Mock 网络测试：验证请求参数和响应解析；
5. 可选真实集成测试：显式开关，避免默认产生网络和费用；
6. 端到端行为评测：固定任务集，检查正确性、引用、拒绝和失败恢复。

单元测试回答“代码是否按规则运行”，评测集回答“规则和模型组合后是否真的解决
用户问题”。真实 Agent 项目不能只依赖几个手工成功示例。

至少要覆盖：正常路径、空结果、歧义、外部服务失败、模型非法调用、重复调用、
达到步数或预算上限、用户拒绝副作用、恶意或越界输入，以及 Context 被截断时的
回答披露。

### 3.10 端到端验收、阶段性提交和复盘

版本完成前，用真实 Provider 和真实外部数据走完整用户链路，并记录：

- 输入和期望；
- 实际 Tool 序列；
- 关键结构化结果；
- 最终回答是否忠于证据；
- 是否进入下一轮交互；
- 测试日期、Provider、模型和已知环境差异。

每个可独立验收的小里程碑都应形成可回滚的 Git 提交。提交前检查 secrets、生成
文件、个人数据和缓存；提交后再进入下一步。避免把一个完整版本积累成一个巨大的
未提交工作区，否则 Review、定位回归和回滚都会变难。

阶段复盘必须回答：哪些假设被证实？哪些问题是设计问题而不是 Prompt 问题？哪些
能力真的需要进入下一版本？

## 4. AgentReader V1–V2 开发过程

| 阶段 | 主要产物 | 学到的核心机制 |
|---|---|---|
| V1 最小闭环 | State、Fake LLM、Tool Registry、同步 Loop | Tool Result 必须写回 State 后再次调用 LLM |
| Provider 接入 | DeepSeek、OpenRouter 兼容层、Conda 环境 | Provider 差异应封装在 `llm.py` |
| 真实搜索 | arXiv 优先、Crossref 备用 | 无结果是数据，网络或解析失败是错误 |
| Tool 拆包 | schemas/search/library/download/extract/retrieval | Tool 按职责组织，公共导入保持稳定 |
| 多轮与保存 | 会话 State、可信 candidate ID、人工确认 | Prompt 引导意图，Runtime 强制权限边界 |
| PDF 下载与读取 | 安全 URL、大小限制、缓存、文本提取 | 外部内容和本地路径都需要二次校验 |
| 最小检索 | 页内 chunk、TF-IDF、top-k、页码 | Tool 内处理大量数据，只把有界证据交给模型 |
| 搜索边界 | 本地重排、相关性状态、重搜限制 | 模型判断相关性，代码提供信号并限制资源 |
| V2 验收 | 真实搜索、保存、下载、提取、检索链路 | 单元测试之后仍需真实模型和真实数据验收 |

## 5. V1–V2 Review 结论

### 5.1 值得保留并迁移的做法

1. **先机制后框架**：显式 Loop 让 Tool Call、State 更新和停止条件都可观察。
2. **Provider 隔离**：Agent 和 Runtime 只认识统一响应，不依赖 DeepSeek 或
   OpenRouter 的原始字段。
3. **概率决策与确定约束分离**：LLM 选择动作，Runtime 校验参数、可信来源、确认
   和调用次数。
4. **稳定 ID 引用**：模型不直接控制文件路径和下载 URL。
5. **有界 Tool Result**：候选、列表、文本、chunk 和 Debug 显示都有上限。
6. **正常空结果结构化**：没有论文、没有可提取文本和没有匹配 chunk 不冒充异常。
7. **测试由内向外扩展**：先纯函数和 Fake Loop，再真实 arXiv、DeepSeek 与 PDF。
8. **重要 Bug 都补回归测试**：精确标题排序和长 Debug 输出均经过复现后修复。

### 5.2 已遇到的问题及可复用经验

#### 依赖没有先固化

```text
现象：运行时报 ModuleNotFoundError: openai
根因：代码依赖已经出现，但环境定义没有先成为项目契约
修复：使用 environment.yml 管理 Python 3.13、openai、dotenv 和 pypdf
经验：新增外部依赖时，同一步更新环境文件和安装验证命令
```

#### 测试启动方式依赖工作目录

```text
现象：直接运行 tests/test_arxiv_integration.py 找不到 tools
根因：脚本入口改变了 Python 的模块搜索路径
修复：统一从项目根目录使用 python -m unittest
经验：文档和 CI 必须使用同一种规范测试入口
```

#### 持久化文件尚未产生

```text
现象：json.tool 提示 data/library.json 不存在
根因：文献库采用惰性创建，只有首次成功保存才写文件
修复：解释创建时机，并让空库查询不制造副作用
经验：明确区分“功能未运行”与“文件损坏”，记录持久化生命周期
```

#### 搜索结果存在但不相关

```text
现象：短缩写可能得到词面相似但语义无关的论文
根因：搜索源排序不等于任务相关性，模型也缺少显式最低证据
修复：扩大候选池、本地词法重排、相关性状态、重搜去重与单轮次数上限
经验：召回、排序、选择和重试是不同层，必须分别定义和测试
```

#### 精确标题被扩展标题压过

```text
现象：Attention Is All You Need 搜索首先返回 Tool Attention Is All You Need
根因：旧评分奖励完整短语和摘要重叠，却没有区分标题完全相等与包含关系
修复：加入 exact_title_match 和确定性额外权重，真实 arXiv 回归验证
经验：用真实失败样本建立排序回归集，不能只检查 found=true
```

#### Debug 输出遮住下一轮交互

```text
现象：三个完整摘要形成超长单行，最终回答和输入提示像是被截断
根因：只限制了 PDF 文本预览，没有限制搜索摘要的显示层
修复：Debug 使用 abstract_preview，State 和模型 Context 仍保留原始摘要
经验：数据边界、Context 边界和显示边界要分别设计
```

#### PDF 页数元数据高估模型实际阅读范围

```text
现象：字符上限截断在第 1 页时，Tool Result 仍可能返回 pages_read=10
最小复现：对 10 页 PDF 使用 max_pages=10、max_chars=500，文本只有 [Page 1]
根因：旧实现先按 max_pages 提取全部页面，最后统一截断字符串，但 pages_read 仍取
      计划解析页数
修复：增量构建有界文本；区分 pages_scanned、pages_read 和 page_numbers，并记录
      last_page_partial；确认非空正文无法完整返回后停止
回归测试：断言 10 页输入只返回第 1 页时 pages_read=1、page_numbers=[1]
经验：资源处理量、返回数据覆盖范围和模型可用证据是三个不同指标，不能共用一个字段
```

#### PDF 正文伪造结构页码

```text
现象：正文中独立一行 [Page 99] 会被 Chunking 当作真实页码，产生错误引用
最小复现：实际第 1 页正文包含 [Page 99]，检索片段却返回 page=99
根因：提取器把结构标记和未转义正文拼成字符串，下游只靠正则重新解析
修复：提取时转义同形正文行；Chunking 校验文本页码与提取 metadata 完全一致
回归测试：断言正文标记被转义，并拒绝页码与 metadata 不一致的输入
经验：文本内控制标记也是协议，必须转义并在消费端再次校验
```

#### 终端观察者影响执行或留下不完整消息

```text
现象：事件回调可以修改 Tool 参数/结果；Ctrl+C 可能留下没有 Tool Result 的调用
根因：观察层收到共享可变对象，中断路径没有按内部消息协议收尾
修复：事件发出深拷贝并隔离回调异常；LLM/Tool 中断时补写 Assistant/Tool 收尾消息
回归测试：恶意回调不能修改 State；分别中断 LLM 和 Tool 后消息序列仍合法
经验：Observability 必须是单向只读边界，取消也属于状态机的一条正式路径
```

#### 管道运行失败却返回成功状态

```text
现象：非交互输入发生 Runtime 错误后，主循环读到 EOF 并以状态码 0 退出
根因：交互式“显示错误后继续”策略被无条件用于管道和 CI
修复：区分交互与非交互；后者显示错误并以非零状态码退出
回归测试：子进程注入失败 Provider，断言返回码为 1 且没有 Python traceback
经验：人机恢复策略和自动化退出语义必须分别定义
```

### 5.3 教学 Demo 中合理、真实项目中需要升级的部分

| 当前简化 | 为什么适合 Demo | 真实项目需要什么 |
|---|---|---|
| 普通 `dict` State | 信息流直观 | 类型化状态、版本迁移、持久化和并发控制 |
| 同步单 Agent Loop | 易于理解和调试 | 超时取消、任务恢复、异步或队列按需求引入 |
| instructions 决定大部分意图 | 展示 LLM 决策 | 策略层、权限层和高风险动作审批 |
| 完整消息历史进入 Context | 便于观察 | token 预算、摘要、裁剪、引用式状态和隔离 |
| 词法搜索与 TF-IDF | 可解释、无额外服务 | 查询分析、混合检索、重排和离线质量评测 |
| JSON 文献库 | 低成本验证持久化 | 数据库事务、锁、索引、迁移、备份和审计 |
| 本地 PDF 缓存 | 便于练习安全下载 | 流式下载、哈希、对象存储、配额和生命周期 |
| 手工 Debug Trace | 学习价值高 | 结构化 tracing、指标、日志脱敏和成本监控 |
| 自动化真实验收 | 真实发现 Provider、网络与 PDF 链路问题 | 固定高层场景、显式开关、结构断言和回归报告 |

### 5.4 当前 Review 发现的技术债务

这些不是 V2 完成的阻塞项，但复制到真实项目之前应处理：

- 当前 State 和模型 Context 会随多轮对话持续增长，没有 token 预算和压缩策略；
- 全文索引仍受 200 页 / 200 万字符约束，扫描 PDF 仍无法通过 OCR 覆盖；
- 搜索与检索只有词法信号，不支持同义词、跨语言或语义匹配；
- arXiv 一旦返回候选，即使词法相关性很弱也不会再比较 Crossref；
- Tool Schema、Python 签名和文档由人工同步，缺少自动契约一致性检查；
- `SEARCH_PAPER_SCHEMA` 的描述仍需随本地重排语义保持同步；
- 普通 DeepSeek 协议测试仍 Mock 搜索层，但已另有真实 arXiv + 双 PDF 系统用例；
- 尚无 OpenRouter 真实冒烟测试和 Provider 行为兼容矩阵；
- 已有小型离线段落检索基线，但还没有真实论文规模的搜索质量集、Prompt 回归评测
  或语义检索基线；
- Runtime Event 已包含 duration，但仍没有 token、cost、retry、持久 trace ID 和指标
  汇总；
- 文献库和 PDF 缓存的临时文件策略尚未为多进程并发写入设计锁或唯一临时名；
- PDF URL 检查拒绝显式私有 IP，但真实系统还要处理 DNS 解析和重绑定风险；
- V1、V2 和 V2.1 已形成独立提交；后续版本仍应保持小而可回滚的阶段提交。

V3 已完成显式任务状态、预算、恢复、终端控制、事件和规划评测。下一阶段应先根据
V3 真实使用结果选择一个明确问题，再评估 Context 压缩、检索升级或其他能力；不因
版本号自动引入长期 Memory、Graph 或 Multi-Agent。

## 6. 可复用检查清单

### 6.1 新阶段开始前

- [ ] 用户问题和成功标准是否具体？
- [ ] 是否写清非目标？
- [ ] 新能力解决的是已经观察到的问题吗？
- [ ] 信息流、State 和 Context 会怎样变化？
- [ ] 是否定义成本、时延和资源上限？
- [ ] 如何离线测试，如何真实验收？

### 6.2 新 Tool

- [ ] 一个 Tool 是否只有一个主要职责？
- [ ] Schema、函数签名和文档是否一致？
- [ ] 输入是否全部需要由模型直接控制？
- [ ] 能否改为稳定 ID，并由 Runtime 恢复可信对象？
- [ ] 正常空结果和异常是否分开？
- [ ] 是否有明确的超时、大小和结果数量上限？
- [ ] 是否幂等，失败后是否留下半成品？
- [ ] Tool Result 有多少内容会进入 Context？

### 6.3 Runtime 与副作用

- [ ] 未知 Tool 和非法参数是否被拒绝？
- [ ] 写入或外部动作是否真的获得用户授权？
- [ ] 模型能否伪造 URL、路径、ID 或对象？
- [ ] 是否限制循环次数、重试次数和预算？
- [ ] 错误是否结构化并重新交给模型观察？
- [ ] 是否需要幂等键、事务、锁或审计记录？

### 6.4 Context 与输出

- [ ] State、Context、长期 Memory 是否明确区分？
- [ ] 外部内容进入模型前是否有界？
- [ ] 是否保留来源、页码和截断状态？
- [ ] 没有证据时是否会停止而不是补写答案？
- [ ] Debug 显示是否有界且不改变原始 Tool Result？
- [ ] 敏感内容是否会进入日志或第三方 Provider？

### 6.5 测试与发布

- [ ] 纯函数、Runtime、Fake Loop、Mock 网络是否分别测试？
- [ ] 是否覆盖空结果、失败、越界、重复和拒绝路径？
- [ ] 是否有真实 Provider 和真实数据的显式测试？
- [ ] 是否有固定质量评测集，而不只是 `found=true`？
- [ ] 文档、环境定义和示例命令是否同步？
- [ ] 是否检查 secrets、缓存、个人数据和生成文件？
- [ ] 是否形成小而可回滚的提交并记录验收证据？

## 7. 阶段记录

### 7.1 V1：最小 Agent Loop

- **状态**：已完成。
- **目标**：不依赖 Agent 框架，理解一次 Tool Loop 的完整信息流。
- **结论**：显式 State、统一 LLM 响应和 Runtime 分发足以解释核心机制。
- **保留原则**：后续版本不能让 Provider 字段或 Tool 执行泄漏回 Agent 决策层。

### 7.2 V2：真实搜索、管理、读取与最小检索

- **状态**：实现和端到端验收已完成，尚未形成独立 Git 提交。
- **测试证据**：2026-09-12 最新完整回归运行 107 项测试，显式启用真实 arXiv 与
  DeepSeek，107 项全部通过，没有跳过。
- **真实证据**：使用真实 arXiv、DeepSeek 和真实 PDF 完成强匹配、无结果澄清、
  人工确认保存、文献库查询、PDF 下载、前 5 页提取及前 10 页 top-k 检索。
- **关键边界**：候选和页数有界、可信 ID 恢复、保存确认、搜索次数限制、截断披露。
- **遗留限制**：见 5.4。

### 7.3 V2.0.1：提取正确性与交互式终端

- **状态**：已完成。
- **阶段目标**：修复模型实际可见页数的错误元数据，并把批量式 `input() + print()`
  会话升级为保留显式 Agent Loop 的轻量交互式终端。
- **用户可见行为**：方向键会话历史、实时 LLM/Tool 状态、`/help`、`/debug`、
  `/clear`、`/exit`，以及适合 CI 和管道输入的 `--plain` 模式。
- **非目标**：全屏 TUI、逐 token 流式输出、后台任务、跨进程历史、任务检查点与
  恢复。
- **信息流变化**：Runtime 在不依赖终端模块的前提下，通过可选回调发出结构化执行
  事件；终端只观察事件并负责渲染，不执行 Tool、不修改 State。事件使用深拷贝，
  观察者异常被隔离。
- **关键设计决策**：交互输入和富文本渲染使用小型终端依赖；非 TTY 自动退回纯文本；
  保存确认仍由 Runtime 强制，只把具体输入界面交给终端层。
- **实现顺序**：页码 Bug 与回归测试 → 真实系统测试 → Runtime 事件 → 终端层 →
  Slash Commands → 终端系统测试。
- **验收标准**：旧调用在无事件回调时行为不变；事件顺序可测试；空输入不退出；命令
  不进入 State；Debug 仍有界；真实 TTY 能连续完成两轮交互。
- **测试证据**：最新联网完整回归 107 项测试全部通过，没有跳过；编译、依赖检查
  和补丁格式检查通过。
- **真实系统证据**：真实 arXiv + DeepSeek 下载并检索 `Attention Is All You Need`
  PDF，回答架构问题并引用第 1、3 页；真实 TTY 验证历史重放、Slash Commands、
  实时 Tool 状态和 Debug 切换；Plain 管道验证无增强终端也可连续运行。

### 7.4 V2.1：全文覆盖与检索质量

- **状态**：已完成。
- **观察到的问题**：旧检索只覆盖开头 10 页，方法、实验和附录可能无法召回；若
  直接更换排序算法，会无法判断改善来自覆盖范围还是评分方法。
- **已选方案**：`retrieve_paper_chunks` 首次调用时在 Tool 内部自动建立持久化 JSON
  索引，不新增模型可见 Tool；在相同数据集上比较 TF-IDF、BM25 和独立查询证据
  门槛，模型可见 Tool 签名保持不变。
- **索引生命周期**：格式版本、提取上限、论文 ID、PDF SHA-256 和文件大小全部匹配
  才复用；缺失时建立，PDF 变化、JSON 损坏或契约过期时原子重建。
- **资源边界**：最多 200 页、200 万字符和 32 MiB 索引文件；完整索引与全部 chunks
  留在 Tool 内部，只有 top-k 片段进入 State 和模型 Context。
- **离线证据**：生成的 12 页 PDF 能从第 12 页召回唯一证据；重复查询复用索引；
  PDF 改变和损坏索引会触发重建；页数和字符上限会显式披露。
- **真实系统证据**：DeepSeek 驱动搜索、缓存下载、全文检索和最终回答；15 页
  `Attention Is All You Need` 索引覆盖全部页面，区分第 9 页正文引用与第 12 页
  Penn Treebank 参考文献，重复检索命中缓存。
- **评测设计**：版本化固定 JSON 包含逐页文本、问题、期望页码和标签；评测直接
  复用正式 chunking/ranking 纯函数，不进入 Agent Loop，也不访问网络或模型。
- **质量基线**：原始 TF-IDF 和原始 BM25 的 10 问 Top-3 Recall、Hit Rate、MRR
  均为 0.875，无答案准确率均为 0.500；加入 50% 查询词覆盖门槛后前三项保持
  0.875，无答案准确率提升到 1.000。7 个英文词法问题三项指标均为 1.0；剩余失败
  是中文问题无法直接匹配英文正文。
- **算法决策**：BM25 没有改善当前指标，因此默认保持 TF-IDF + 50% 门槛，BM25
  只作为可复现实验配置。查询证据门槛与排序分离，方便以后独立替换其中一层。
- **语义检索决策**：当前唯一失败需要跨语言查询改写或语义匹配，单纯更换词法公式
  无效；考虑到评测集仍小，V2.1 不新增 embedding/hybrid 依赖。Agent instructions
  继续要求英文论文使用英文技术词，原始检索纯函数仍诚实保留跨语言限制。
- **回归规则**：测试锁定当前总体和分标签结果作为确定性快照，变化时必须显式
  审查；这不是生产质量声明，新算法必须使用同一评测集报告改善与退化。
- **资源测量**：隔离合成 12 页 / 36,000 字符 PDF，重复 5 次；首次完整检索中位数
  约 8.9 ms，缓存检索约 1.4 ms，37,196 B 索引产生 5,131 B top-3 Tool Result。
  计时包括哈希、索引加载/建立、chunking 和排序；Context 数字只代表本次新增结果，
  不包含历史消息和 tokenizer，机器相关耗时不作为跨机器阈值。
- **验收标准**：同数据集四配置报告可复现；默认结果有回归测试；弱通用词误命中
  被门槛拦截；正常终端解释证据不足；联网 arXiv 与 DeepSeek 完整测试不跳过。
- **测试证据**：107 项联网完整回归全部通过，没有跳过；真实 15 页 PDF 的查询词
  覆盖率为 1.0，首次建立索引后再次检索命中缓存并召回第 9、12 页。
- **下一小步**：进入 V3，先实现 Plan 数据契约、验证器和状态转换测试。
- **暂定非目标**：长期 Memory、Graph、Multi-Agent 和语义检索。
- **排序升级门槛**：先得到 TF-IDF 基线，再在完全相同的评测集上比较 BM25；只有
  词法方案仍无法满足已定义问题时才评估 embedding/hybrid retrieval。

### 7.5 V3：可恢复的单 Agent 计划执行器

- **状态**：已完成。Plan、Planner、Executor、DeepSeek thinking 协议、有限 Replan、
  blocked、Checkpoint/`--resume`、终端控制、生命周期事件、规划评测和真实多论文
  验收均已接入。
- **阶段目标**：让单 Agent 把多论文、多交付结果的用户目标拆成显式 Plan，串行执行
  当前步骤，在有限条件下 Replan，并在进程退出后恢复同一个未完成任务。
- **目标示例**：搜索两个方向的代表论文，下载 PDF，分别检索方法与实验结果，再
  输出带论文和页码证据的比较。
- **观察到的问题**：V2 Runtime 只对当前 LLM 决策做循环；复杂任务没有显式完成
  条件、步骤进度和任务级预算，进程退出后 State 也无法恢复。
- **信息流变化**：在用户目标与现有 Executor/Tool Loop 之间增加 Planner 和经过
  Runtime 校验的 Plan；在每个重要状态转换后由独立 Checkpoint 层原子保存 State。
- **保持不变**：`llm.py` 仍隔离 Provider 解析；Runtime 仍唯一执行 Tool 和修改可信
  状态；每次只执行一个 Tool Call；保存仍逐篇确认；全文和索引不进入 Context。
- **Plan 边界**：最多 8 步；模型只能提出高层步骤，不能伪造状态、计数器、URL、
  路径、论文 ID 或 evidence reference。
- **Replan 边界**：只在 Tool 失败、候选歧义、证据不足或前提失效时触发；最多 2
  次，且必须保留完成步骤和可信证据，不重复副作用。
- **任务预算**：普通模式累计最多 32 次 LLM 决策，Thinking 模式最多
  100 次；Tool 仍最多执行 24 次。达到上限后进入明确的 `blocked` 或 `failed`。
- **恢复边界**：第一版同一进程只有一个活动任务，Checkpoint 默认上限 4 MiB；
  `--resume` 只恢复同一任务。完成、不可恢复失败或取消后清除活动 Checkpoint；新
  任务不能静默覆盖尚未恢复的活动文件。
- **关键概念**：Checkpoint 是 operational state persistence；长期 Memory 是新任务
  中的选择性召回。V3 只实现前者，避免把“磁盘上有 JSON”误称为 Agent Memory。
- **终端行为**：`--resume`、活动任务 `/clear` 保护、`/plan` 有界查看和 `/cancel`
  原子取消均已完成；普通消息在活动期间只进入同一个 State。
- **非目标**：LangGraph、Multi-Agent、MCP、长期 Memory、Embedding/Vector DB、
  跨语言语义检索、并发 Tool Calling、后台任务和复杂异步调度。
- **验收标准**：离线覆盖成功、非法 Plan、阻塞、有限 Replan、预算、取消、损坏
  Checkpoint、恢复与幂等；真实 arXiv + DeepSeek + PDF 完成一个多论文任务，联网
  测试不得跳过。
- **实现顺序**：Plan 纯函数与状态机 → Planner/Fake → Runtime 单步执行 → Replan
  与 blocked → Checkpoint → Terminal/Events → 离线评测 → 真实验收。
- **Step 1 结果**：新增 `planning.py`，由 Runtime 输入原始目标和步骤描述后生成稳定
  step ID；严格拒绝额外字段、非法状态、越界预算和不可信 evidence reference；状态
  转换不原地修改旧 Plan。新增 13 项纯函数测试；启用真实 arXiv 与 DeepSeek 后，
  全量 120 项回归全部通过，没有跳过。
- **Planner 输入边界结果**：新增 `planner.py`，只在首次规划决策中临时提供内部
  `submit_plan` Schema；模型只能提交 1–8 条步骤描述。Planner 将其归一为 `plan`
  动作，普通 `tool_call|final` 保持不变，且 `submit_plan` 不进入 Tool Registry。
  新增 8 项 Fake LLM/契约测试；真实 arXiv + DeepSeek 全量 128 项通过，无跳过。
- **Runtime 创建结果**：第一次决策可以返回 `plan|tool_call|final`。Runtime 使用
  最新用户消息与本地 UUID 创建 Plan，记录首次 LLM 用量，并在完整校验后同时更新
  `State.task_id` 和 `State.plan`；`submit_plan` 不进入消息 Tool 协议。当前只返回
  有界 Plan 预览，活动 Plan 不会被下一轮静默替换。新增 6 项 Runtime Planning 测试
  和 1 项预算测试；真实 arXiv + DeepSeek 全量 135 项通过，无跳过。
- **最小 Executor 结果**：新增 `executor.py`，只注入整体目标、已完成步骤和当前
  running step，并继续使用现有串行 Tool Loop。Tool Result 获得 Runtime-owned
  `tool-result-NNN` 引用，但不会直接完成 step；step-level final 才推进状态。Runtime
  同时记录 LLM/Tool 预算并在调用前阻止越界。新增 2 项 Executor、4 项 Runtime
  执行/预算和 1 项 evidence 转换测试；真实 arXiv + DeepSeek 全量 142 项通过，无
  跳过。
- **thinking 协议观察**：真实多步任务在计划创建、一次 Tool Call 和 step-level
  final 后发起第三次模型请求时，DeepSeek 返回 HTTP 400，明确要求回传此前
  `reasoning_content`。这说明 reasoning 不是只用于显示的中间文本，而是 thinking
  Tool Calling 的会话协议字段。
- **thinking 协议结果**：在独立分支 `feat/deepseek-thinking-protocol` 中由 `llm.py`
  显式配置 thinking 和 reasoning effort，并把 DeepSeek `reasoning_content`、
  OpenRouter 字符串 reasoning 或结构化 `reasoning_details` 归一为不透明字段。
  Planner 内部化 `submit_plan` 时保留该字段，Runtime 将它写入对应 Assistant State
  消息，`llm.py` 在所有后续请求中原样回传。普通终端、Debug Trace 和 Runtime
  Event 都不显示 reasoning。新增 13 项协议、Planner、Runtime 与真实 DeepSeek
  集成测试；启用真实 arXiv 与 DeepSeek 后，全量 155 项通过、无跳过。OpenRouter
  当前完成离线协议兼容测试，未使用真实 OpenRouter Key 验收。
- **测试设计修正**：首次全量联网测试中，真实模型把最初测试提示判为简单任务，
  没有生成 Plan；这不是协议失败。集成测试随后使用仅暴露 `submit_plan` 的受控
  Planner 指令，稳定进入需要验证的多步协议路径，而 Executor 和 Provider 仍使用
  真实生产链路。
- **单轮上限调整**：复杂 Plan 在第 5 次 LLM 决策完成 PDF 下载后触发旧的 V1
  `max_steps=5`，状态虽然保留，但英文 `Agent stopped` 容易被理解成任务失败。
  默认值调整为 20，仍低于任务级 32 次 LLM 决策上限；活动 Plan 真正到限时改为
  中文说明本轮暂停并提示输入“继续”，不改变 Plan 状态和任务预算。
- **Replan/blocked 结果**：Runtime 只根据最新未处理的可信 Tool Result 开放一次
  Replan 选择；普通成功观察不能再次规划。`revise_plan()` 保留 completed/failed
  历史、attempts 和 evidence，只替换 pending 后缀，并把任务限制在 2 次 Replan 和
  8 个总步骤内。`request_clarification` 作为内部控制动作进入 blocked，只有新的用户
  消息才能恢复原步骤。单轮上限正好落在失败 Tool Result 后时，该失败信号也可跨
  “继续”保留。
- **副作用结果**：Replan 后相同参数的成功 `save_paper` / `download_paper` 调用由
  Runtime 复用可信旧结果，不再次执行 Tool；用户拒绝和错误结果不会被当作成功复用。
  首次保存的逐篇 `y/N` 确认与原始目标权限边界保持不变。
- **验证结果**：新增 2 项 Plan、3 项 Executor 和 9 项 Runtime 用例，覆盖重规划
  条件、2 次预算、动态容量、失败跨轮恢复、blocked 恢复、硬失败收敛、软歧义和
  重复下载及失效缓存保护。真实 arXiv + DeepSeek 全量 170 项通过，无跳过。
- **Checkpoint 结果**：新增独立 `checkpoint.py`，用版本 1、4 MiB 上限、同目录唯一
  临时文件、`fsync` 与原子替换保存活动 State。加载会重新验证 Plan、消息配对、
  evidence、稳定论文 ID 和 PDF 缓存；不同 task、损坏、超限或不兼容版本均拒绝且
  不覆盖旧文件。thinking reasoning 因 Provider 恢复协议需要而随 State 保存，但不
  进入终端输出。
- **恢复结果**：Runtime 在 Plan 创建和每个一致状态边界保存，在完成或不可恢复失败
  后清除；Ctrl+C、可恢复错误和退出保留最近快照。CLI 启动拒绝静默覆盖，`--resume`
  只显示有界摘要并继续同一 `task_id`。活动任务的 `/clear` 已被阻止。
- **Checkpoint 验证结果**：完整联网回归 190 项全部通过、无跳过；覆盖原子写失败、
  损坏/版本/超限、可信 ID/evidence 篡改、缓存失效、running/blocked 恢复、Ctrl+C、
  完成/失败清理和 CLI 冲突提示。
- **终端与事件结果**：`/plan` 有界显示 Runtime-owned Plan；`/cancel` 先清理
  Checkpoint 再提交取消状态，失败时不留下半取消。新增计划创建、步骤转换、Replan、
  blocked、cancelled 和 Checkpoint 保存事件；事件不含 reasoning、路径或完整 State，
  终端继续作为只读观察者。
- **规划评测结果**：`evaluate_planning.py` 用正式 Runtime、受控 Provider/Tool 执行
  固定两论文成功流程。当前机器可读快照为 completed=true、Replan=0、LLM=6、
  Tool=2、Checkpoint 保存 13 次、最大 4,572 B。
- **真实系统验收结果**：测试用受控四步 Plan 降低规划输出随机性，Executor 与
  DeepSeek、arXiv:1706.03762、arXiv:1810.04805、双 PDF 下载、全文索引、页码证据和
  最终比较均走真实路径。2026-09-13 全量联网 202 项全部通过、无跳过。
- **交互复盘发现**：真实执行“创建 Plan → 退出 → 拒绝覆盖 → 恢复 → 取消”时，
  State 和 Checkpoint 均正确，但恢复摘要把第一条 pending step 标成了“当前步骤”，
  而 `/plan` 根据 `current_step_id=None` 显示尚未开始。根因是两个 UI 入口使用了
  不同的展示推导；修复后都分别显示实际 current 和下一 pending step，并补回归测试。
- **Thinking 模式决策**：Provider reasoning、单轮 `max_steps` 和任务级预算是三个
  独立控制。`--thinking` 通过 `LLM_THINKING` 显式开启推理，同时把后两者都提高
  到 100。普通模式仍使用 20/32，Tool 24 次与 Replan 2 次边界不变，因此仍然
  不能无限循环。跨进程恢复时需再次传入 `--thinking --resume`。
- **Replan 搜索额度修正**：一次真实任务中，arXiv 429/超时使两次初始搜索
  回退 Crossref，Crossref 候选又没有 PDF URL。下载失败触发 Replan 后，原先按
  `run_agent()` 整轮计数的 2 次搜索额度阻止了新 revision 的 arXiv ID 定向恢复。
  Runtime 现改为每个有界搜索阶段最多 2 次，可信 Replan 被接受后在同一
  用户轮内为新 revision 开始新阶段；
  每任务最多 2 次 Replan，所以恢复能力仍然有界。
- **联网测试修正**：全量复测首次遇到一次真实系统任务进入可恢复 `blocked`，单独
  重跑通过，说明这是外部网络或模型决策的非确定性，而不是 CLI 回归。系统用例改为
  在最多 3 个执行轮次内按正式 blocked 恢复协议补充“信息已齐全、重试可信 ID”，
  连续不能恢复仍失败，且最终双论文、双 PDF、页码证据断言不放宽。
- **下一小步**：提交 V3 工作后进行一次版本复盘；只有从真实使用或评测中识别出
  明确瓶颈，再与用户共同定义下一版本需求。

## 8. 更新记录

| 日期 | 阶段 | 更新内容 | 证据 |
|---|---|---|---|
| 2026-09-13 | V3 Thinking/恢复搜索修正 | Thinking 单轮与任务级 LLM 上限提高到 100；Replan 开启新的有界搜索阶段 | 新增高预算与 Replan 后两次恢复搜索回归；真实 arXiv + DeepSeek 全量 207 项通过、0 跳过 |
| 2026-09-13 | V3 交互复盘修正 | current/next step 显示一致；新增 `--thinking`，单轮 30、任务级仍为 32；系统测试按协议有限恢复 blocked | 新增 3 项终端测试；真实 arXiv + DeepSeek 全量 205 项通过、无跳过 |
| 2026-09-13 | V3 完成 | `/plan`、`/cancel`；计划生命周期事件；确定性 Planning 评测；真实双论文 PDF 验收 | 规划快照 completed=true、Replan=0、LLM=6、Tool=2、Checkpoint 最大 4,572 B；真实 arXiv + DeepSeek 全量 202 项通过、无跳过 |
| 2026-09-13 | V3 Checkpoint/恢复 | 版本化原子快照；严格恢复校验；`--resume`；冲突保护；活动任务 `/clear` 保护 | 真实 arXiv + DeepSeek 全量 190 项通过、无跳过 |
| 2026-09-13 | V3 Replan/blocked | 可信失败条件下有限重规划；保留历史/evidence；用户澄清阻塞恢复；重复副作用复用 | 新增 14 项测试；真实 arXiv + DeepSeek 全量 170 项通过、无跳过 |
| 2026-09-12 | V3 单轮执行上限 | 默认 `max_steps` 从 5 调整为 20；活动 Plan 到限时保留状态并提示继续 | 新增 1 项默认值测试；真实 arXiv + DeepSeek 全量 156 项通过、无跳过；任务级 32/24 预算不变 |
| 2026-09-12 | V3 DeepSeek thinking 协议 | 显式 thinking 配置；跨 Plan、Tool Call 和 step final 原样回传 reasoning；OpenRouter 格式兼容；终端不显示推理字段 | 13 项新增测试；真实 arXiv + DeepSeek 全量 155 项通过、无跳过 |
| 2026-09-12 | V3 最小 Executor | 当前 step 控制 Context、串行 Tool Loop、step-level final、可信 evidence 和执行前预算边界 | 7 项新增测试；真实 arXiv + DeepSeek 全量 142 项通过、无跳过 |
| 2026-09-12 | V3 Runtime 创建 | 首次 Planner 决策创建 Runtime-owned task ID、Plan 和初始预算；简单任务保持 V2 路径 | 6 项 Runtime Planning + 1 项预算测试；真实 arXiv + DeepSeek 全量 135 项通过、无跳过 |
| 2026-09-12 | V3 Planner 边界 | 增加内部 `submit_plan` Schema、规划 instructions 和三类初始动作归一化 | 8 项 Planner 测试；真实 arXiv + DeepSeek 全量 128 项通过、无跳过 |
| 2026-09-12 | V3 Step 1 | 实现 Plan 纯数据契约、严格验证和不可变状态转换 | `planning.py`；13 项 Plan 测试；真实 arXiv + DeepSeek 全量 120 项通过、无跳过 |
| 2026-09-12 | V3 需求定义 | 将 V3 收敛为单 Agent Planning、有限 Replanning 和同任务 Checkpoint 恢复 | 明确 Plan/状态/预算/权限/恢复边界、非目标、实现顺序和验收标准；基线提交 `0a1ed33` |
| 2026-09-12 | V2.1 算法选择 | 同数据集比较 TF-IDF/BM25 与查询词覆盖门槛，默认采用 TF-IDF + 50% 门槛 | Recall/Hit/MRR=0.875；无答案准确率由 0.500 提升到 1.000；BM25 指标持平；107 项联网测试全部通过 |
| 2026-09-12 | V2.1 资源测量 | 增加隔离合成 PDF 的首次/缓存检索、磁盘和 Context 载荷基准 | 首次约 8.9 ms；缓存约 1.4 ms；索引 37,196 B；结果 5,131 B |
| 2026-09-12 | V2.1 检索评测 | 增加版本化 10 问离线评测、分标签结果和 JSON 报告，锁定 TF-IDF Top-3 基线 | Recall/Hit/MRR=0.875；无答案准确率=0.500；100 项测试，97 项通过、3 项跳过 |
| 2026-09-12 | V2.1 全文索引 | 增加可失效的持久化 JSON 索引，保持模型 Tool 接口和 TF-IDF 不变 | 96 项离线测试；12 页回归样本；DeepSeek + 真实 15 页 PDF 全文检索 |
| 2026-09-12 | V2.0.1 | 修复提取页码契约，增加 Runtime Event 和交互式终端 | 91 项离线测试；真实 arXiv + DeepSeek + PDF；Plain 与真实 TTY 冒烟测试 |
| 2026-09-12 | V1–V2 Review | 创建开发流程、项目复盘、检查清单和 V2.1/V3 维护模板 | 代码与文档 Review；真实端到端 Trace |
