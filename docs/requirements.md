# AgentReader Demo 需求说明

## 1. 文档目的

本文描述 AgentReader Demo 当前已经实现的行为、明确限制和下一阶段需求。
它记录的是项目现状，不是最终产品规格。

当前版本定位：V1 最小 Agent Loop、V2 真实搜索/管理/阅读/最小检索、V2.0.1
提取正确性和交互式终端，以及 V2.1 全文索引、离线评测、资源测量和排序方法选择
均已完成。V3 已完成 Plan 纯数据层、Planner 输入边界、Runtime 可信 Plan 创建、
最小 step Executor、DeepSeek thinking 多步消息协议、有限 Replan、blocked，以及
Checkpoint 与 `--resume`、终端任务控制、扩展事件、规划评测和真实多论文验收。
V3.1/V3.2 已补充分层运行健康分、LitSearch 同语料基线及结构化 Agent 任务成功率。

## 2. 学习目标

项目首先用于理解下面四个组成部分如何协作：

```text
Agent = LLM + Tools + Loop + State
```

学习过程中应当能够解释：

- LLM 为什么只能提出 Tool Call，不能直接执行 Python 函数。
- Tool Schema 与 Tool Registry 的区别。
- Tool Result 为什么必须写回 State 后再次调用 LLM。
- 为什么 Runtime 负责执行、错误处理和 `max_steps`。
- 为什么 Provider 的响应解析集中在 `llm.py`。

## 3. 当前功能需求

### FR-1：命令行入口

- 用户可以通过 `python main.py` 输入一条文献任务。
- 程序创建单次任务 State，运行 Agent Loop，并输出最终答案。
- 每轮回答后程序继续等待输入；后续消息追加到同一个 State。
- 用户可以通过后续消息消除候选歧义，不需要重新启动程序或重新搜索。
- 保存操作执行前，终端展示可信论文 metadata 并读取 `y/N` 确认。
- 空输入不得退出；`/exit`、`exit`、`quit`、`退出` 或输入流结束可以结束会话。
- `/help` 显示命令，`/debug [on|off]` 在会话内切换调试模式，`/clear` 清空当前
  State，`/plan` 查看有界计划，`/cancel` 取消活动任务；Slash Commands 不得进入
  State 或发送给模型。
- 交互式 TTY 提供当前进程内的输入历史和 Slash Command 补全，不把历史默认写盘。
- `python main.py --debug` 按实际发生顺序显示每步 LLM 决策、Tool Call、Tool Result
  和耗时，不需要等待整轮完成。
- `python main.py --plain` 禁用颜色、Spinner 和增强输入；非 TTY 自动使用 Plain
  模式，输出不得含 ANSI 控制符，运行失败时必须返回非零退出码。
- `python main.py --thinking` 必须显式强制当前进程的 Provider thinking，将单轮
  LLM 决策上限和任务级 LLM 预算都提高到 100，并在终端显示模式。
- 恢复 Thinking 任务时由用户再次传入 `--thinking --resume`；Checkpoint 继续保存
  Provider 协议所需的 reasoning 字段，但不把 CLI 模式当作长期配置。
- 搜索 Tool Result 的调试显示必须把每篇摘要限制为最多 400 字符的
  `abstract_preview`；完整摘要仍保留在 State 中，显示裁剪不得改变模型 Context。

### FR-2：统一 LLM 接口

- `call_llm(messages, tools)` 是 Agent 使用的唯一模型接口。
- 支持 `fake`、`deepseek` 和 `openrouter` Provider。
- Provider 输出必须归一为 `final` 或单个 `tool_call`。
- V1 每个 LLM step 最多处理一个 Tool Call。
- Provider 同时返回多个 Tool Calls 时，只将第一个归一化并交给 Runtime；其余
  调用不执行，模型观察第一个结果后再决定下一步。
- DeepSeek 必须显式配置 thinking 开关和 reasoning effort；不能依赖模型端可能变化
  的默认值。OpenRouter 的 reasoning 配置保持可选，只在所选模型支持时启用。
- Provider 返回的 `reasoning_content`、`reasoning` 字符串别名或结构化
  `reasoning_details` 必须在 `llm.py` 归一为至多一种内部表示。没有推理字段的响应
  必须保持原有 `final|tool_call` 契约。
- Runtime 必须把归一化 reasoning 作为对应 Assistant 消息的不透明协议元数据保存，
  不得解释、修改或当作文献证据；后续 Provider 请求必须原样回传。
- 普通终端、Debug Trace 和 Runtime Event 不得输出 reasoning 内容。活动任务的
  Checkpoint 必须保存该字段，确保恢复后的 Provider 协议连续；它仍受 Checkpoint
  结构校验和 4 MiB 总上限约束。

### FR-3：文献搜索

- `search_paper(query)` 接收非空字符串。
- 首先查询 arXiv Atom API。
- query 包含标准数字 arXiv ID 时，应使用 `id_list` 定向查询。
- arXiv 返回 HTTP 429 时，应等待服务建议或默认间隔后重试一次。
- 明确 arXiv ID 的 Atom API 请求不可用时，应从有界的官方 arxiv.org 摘要页读取
  citation metadata，并校验返回 ID 与请求 ID 一致；页面超限、缺字段或 ID 不一致
  时不得生成可信候选。
- arXiv 不可用或没有结果时查询 Crossref。
- 因 arXiv 不可用而回退时，结果应包含简短的 `arxiv_error` 诊断信息。
- 普通关键词搜索从来源最多获取 10 篇候选；明确 arXiv ID 时只获取 1 篇。
- 本地排序对 query、标题和摘要进行不区分大小写的英文/数字词项匹配；连续中文
  使用双字片段。
- 每个标题命中词计 4 分，每个摘要命中词计 1 分；query 的完整词项序列在标题中
  连续出现时，再增加 `query 词项数 × 2`；规范化标题与完整 query 完全相等时，
  再增加 `query 词项数 × 2`，使精确标题优先于带前后缀的扩展标题。
- 同分时保持搜索来源的原始顺序。最终仍最多返回 3 篇候选，不得把完整候选池放入
  State 或模型 Context。
- 结果必须包含 `candidate_count`、`truncated`、`ranking_method`，并为每篇返回
  `local_relevance`：总分、标题命中词、摘要命中词、标题短语命中状态、精确标题
  命中状态和来源排名。
- 顶层 `relevance_assessment` 必须包含 `status` 和 `best_score`。status 为
  `identifier_match`、`lexical_match`、`no_lexical_match` 或 `no_candidates`。
- 只有 arXiv ID 与返回记录稳定 ID 一致时才是 `identifier_match`；最高本地分数
  大于 0 才是 `lexical_match`。有候选但最高分为 0 时必须明确返回
  `no_lexical_match`，不能暗示候选一定相关。
- “没有结果”是正常结构化结果；网络失败、响应解析失败等属于异常。

每篇候选使用统一 metadata：

```text
source
candidate_id
arxiv_id
doi
title
authors
abstract
published
paper_url
pdf_url
```

### FR-4：候选选择

- Agent 应比较用户请求与候选标题、作者等信息。
- Agent 可以使用 `local_relevance` 作为可解释的词法证据，但不能把分数当成语义
  正确性的保证；仍须检查标题、摘要和作者。
- 收到 `no_lexical_match` 或 `no_candidates` 时，Agent 不得把候选描述为相关。
  如果当前用户上下文支持一个实质不同的 query，可以精炼后再搜索一次；否则应
  请求用户补充标题、作者或领域线索。
- 请求有歧义时，应展示相关候选，而不是假定第一篇一定正确。
- Agent 不应虚构搜索结果中不存在的 metadata。
- 用户否定旧候选并补充新上下文时，Agent 应使用新的具体 query 再次调用
  `search_paper`，而不是继续推荐已被否定的候选。

### FR-5：本地保存原型

- 用户明确要求保存时，Agent 可以调用 `save_paper`。
- LLM 只向保存 Tool 提交搜索结果中的 `candidate_id`。
- Runtime 必须在当前任务的历史 `search_paper` 结果中解析该 ID。
- 不存在的 ID 或完整论文 metadata 参数必须被拒绝，且不得写入文件。
- Runtime 必须在调用存储函数前取得显式确认；没有确认器时默认拒绝写入。
- 只有 `y`、`yes`、`是` 或 `确认` 表示同意；回车和其他输入均不写入。
- 用户拒绝是正常结构化 Tool Result，必须写回 State 让 LLM 正常收尾。
- 默认文献库路径为 `data/library.json`。
- 可以通过 `PAPER_LIBRARY_PATH` 指定其他路径。
- 文献库使用版本化 JSON 对象：`{"version": 1, "papers": [...]}`。
- 相同 arXiv ID（忽略版本号）或 DOI 不重复保存。
- 缺少 arXiv ID 和 DOI 时，使用标题与论文 URL 的稳定哈希作为 ID。
- 写入过程使用同目录临时文件替换，避免直接覆盖到一半。

### FR-6：查看本地文献库

- 用户询问已保存论文时，Agent 应调用只读的 `list_library`，而不是重新搜索。
- `limit` 是可选整数，默认为 20，允许范围为 1 到 50。
- 结果按最近保存优先排列，并返回 `count`、`total` 和 `truncated`。
- 单篇结果只包含 ID、标题、作者、来源、发表时间、链接和保存时间，不返回摘要。
- 文献库不存在或为空属于正常空结果，查询不得创建文件。
- 损坏的 JSON 或非法文献库结构属于异常，由 Runtime 转成结构化 Tool Result。

### FR-7：下载和缓存 PDF

- 只有用户明确要求下载时，Agent 才能调用 `download_paper`。
- LLM 只能提交 `search_paper` 的 `candidate_id` 或 `list_library` 的 `id`。
- Runtime 必须从当前 State 的可信 Tool Result 中解析 PDF URL；任意 ID 或模型
  直接提交的 URL 必须被拒绝。
- PDF URL 必须使用 HTTPS、不得包含凭据，也不得直接指向 localhost 或私有 IP。
- 下载超时为 20 秒，单个文件最大为 20 MiB。
- 响应 Content-Type 必须是 PDF 或通用二进制类型，文件内容必须以 `%PDF-` 开头。
- 默认缓存目录为 `data/pdfs/`，可通过 `PAPER_CACHE_DIR` 覆盖。
- 缓存文件名根据稳定论文 ID 确定；有效缓存不得重复下载。
- 写入必须先使用临时文件再原子替换，失败时不得留下不完整临时文件。
- Tool Result 只返回论文 ID、标题、URL、本地路径和大小，不包含 PDF 二进制。

### FR-8：PDF 文本提取

- 只有用户明确要求读取、分析或总结 PDF 时，Agent 才能调用
  `extract_paper_text`。
- LLM 只能提交此前 `download_paper` 返回的 `paper_id`，以及可选的
  `max_pages`、`max_chars`；不得提交本地路径。
- Runtime 必须从当前 State 的可信 `download_paper` Tool Result 中恢复本地路径。
- 提取函数必须再次校验解析后的真实路径位于 `PAPER_CACHE_DIR` 内，并校验 PDF
  文件头、大小和普通文件类型。
- 默认最多解析开头 5 页并返回 12000 个字符；硬上限为 10 页和 20000 个字符。
- Tool Result 必须分别返回总页数、解析器实际检查页数、返回文本实际覆盖页数与
  页码、最后一页是否为部分文本、实际字符数、是否截断和截断原因。
- `pages_read` 只能统计实际出现在返回文本中的 `[Page N]` 页面；一旦确认非空正文
  无法完整返回就停止解析，也不得把未返回给模型的页面计为已读取。若某页正好填满
  上限，可以继续检查后续页，以判断是否真的发生字符截断。
- 空文本属于正常结构化结果，并提示扫描 PDF 需要当前未实现的 OCR。
- 加密、损坏和缓存路径越界属于异常，由 Runtime 转成结构化 Tool Result。
- 默认测试必须在临时目录生成小型 PDF，不依赖真实网络或用户缓存。

### FR-9：内部文本 Chunking

- Chunking 只消费 `extract_paper_text` 的有界结构化结果，不直接读取任意文件。
- 输入文本必须使用 `[Page N]` 标记，页码严格递增；chunk 不得跨页。
- PDF 正文自身出现的同形独立行必须在提取时转义；Chunking 还必须验证解析出的
  页码与提取 metadata 一致，不能让正文伪造页码引用。
- 默认 `chunk_size=1200`、`overlap=200`；chunk 大小允许 200–4000，overlap
  不得超过 chunk 大小的一半。
- 每个 chunk 必须包含稳定 `chunk_id`、`paper_id`、页码、页内字符起止位置和
  文本。
- 结果必须继承源提取结果的截断状态和截断原因，不能暗示覆盖了论文全文。
- 空提取文本返回正常空 chunks；超出提取字符上限或页码标记非法属于异常。
- `chunk_paper_text` 是确定性的内部函数，不单独暴露为模型可见 Tool；全部 chunks
  不得写入 State。

### FR-10：内部 Chunk 相关性排序

- 排序只消费 `chunk_paper_text` 的结构化结果和非空查询，不读取文件或调用模型。
- 英文和数字按不区分大小写的 token 处理；连续中文使用双字片段，以便不增加
  分词依赖仍可进行基础中文匹配。
- 支持平滑 IDF 的 TF-IDF 余弦相似度和 BM25 两种 chunk 分数；分数只在相同方法
  内比较，不假定两种方法的绝对分数可互换。
- 默认使用 TF-IDF，并要求至少 50% 的不同查询词项在整篇索引中出现。门槛判断与
  chunk 排序分离；门槛不足时不得返回排名片段。
- 默认 `top_k=3`，允许范围为 1–5；只返回分数大于 0 的结果。
- 返回结果必须包含 score 和 matched_terms，同时保留 chunk ID、页码和字符区间。
- 没有共同检索词或查询词覆盖低于门槛属于正常空结果，`found=false`，不得退化为
  任意返回第一段。结果必须披露查询词、匹配词、覆盖率、最低门槛和是否因此拒绝；
  门槛前的正分 chunk 数可作为 `total_matches` 保留用于诊断。
- 同分结果必须依次按页码、页内起点和 chunk ID 排序，保证结果可重复。
- 排序结果必须继续保留源文本的截断状态和原因。
- `rank_paper_chunks` 仍是内部函数，不单独暴露为模型可见 Tool。

### FR-11：模型可见的最小段落检索

- 用户针对已下载论文提出方法、架构、数据集、实验、指标或其他具体问题时，Agent
  应优先调用 `retrieve_paper_chunks`，而不是返回一段连续开头文本。
- LLM 只能提交此前 `download_paper` 返回的 `paper_id`、非空检索 `query` 和可选
  `top_k`；不得提交本地路径。
- query 去除首尾空白后最多 500 字符；`top_k` 默认为 3，允许范围为 1–5。
- Runtime 必须从当前 State 的可信下载结果恢复本地 PDF，再由 Tool 在内部加载或
  建立持久化全文索引、执行页内 chunking、默认 TF-IDF 排序和查询词覆盖门槛。
- 首次检索自动建立索引；后续检索只有在索引格式版本、提取上限、论文 ID、PDF
  SHA-256 和文件大小都匹配时才允许复用。
- PDF 变化、索引损坏、版本或上限不匹配时必须自动重建。写入使用同目录唯一临时
  文件和原子替换，失败后不得留下半成品。
- 索引默认最多检查 200 页、保存 200 万字符，JSON 文件上限为 32 MiB；必须返回
  `coverage_complete`、实际覆盖页码、部分页状态和截断原因，不能把受限索引描述为
  完整论文。
- 索引默认保存到 `data/indexes/`，可通过 `PAPER_INDEX_DIR` 覆盖；索引文件属于
  可重新生成的本地缓存，不提交 Git。
- Tool Result 只包含 top-k 匹配 chunks 和必要 metadata，不得包含连续提取文本、
  完整索引、索引路径或全部 chunks。
- Tool Result 必须返回 `index_status=built|cached|rebuilt` 和索引格式版本；状态只
  用于可观察性，不构成论文内容证据。
- 无共同检索词或查询证据不足是正常结果，`found=false`；Agent 必须区分
  `rejected_low_query_coverage`，不得据此编造论文内容。
- 默认测试必须使用固定本地文本和 Fake LLM，不依赖真实网络或模型 API。

### FR-12：循环和错误边界

- Runtime 必须通过 Tool Registry 查找工具。
- Runtime 必须拒绝未知工具和无法绑定到 Python 签名的参数。
- Tool 异常必须转成结构化 Tool Result，交回 LLM 处理。
- `run_agent()` 接受可选的只读事件回调；没有回调时必须保持原有行为。回调异常
  不得控制或中断 Agent Loop，也不得通过可变事件内容修改 Tool 调用或 State。
- LLM 或 Tool 执行被用户中断时，Runtime 必须用 Assistant 收尾；Tool 已经产生
  Tool Call 时还必须补写 `cancelled` Tool Result，不能留下不完整消息协议。
- `max_steps` 必须在模型不能结束时终止当前轮；默认值为 20，每次用户新输入后重新
  计数。活动 Plan 到达该上限时必须保留 Plan，并明确提示用户可输入“继续”；不得
  把本轮暂停误报为整个任务失败。
- Runtime 必须在当前有界搜索阶段中记录已经尝试的有效搜索 query；比较前
  折叠首尾及连续空白，与 `search_paper` 的规范化一致。
- 同一搜索阶段的重复 query 必须返回 `duplicate_search_query` 结构化错误，不得
  再次调用搜索 Tool。精炼后的不同 query 仍可执行。
- 简单任务的去重集合只属于当前 `run_agent()` 调用；下一条用户消息开始时必须
  重置，以保留用户主动重新搜索同一 query 的能力。
- 每次 `run_agent()` 开始一个新的有界搜索阶段，最多允许 2 个不同的有效
  query。第三个必须返回 `search_limit_reached`，不得执行搜索 Tool。
- 只有可信失败触发的 Replan 被 Runtime 接受后，才能在同一用户轮内为新
  revision 开始新的搜索阶段；普通 LLM 决策不得自行重置。
- Runtime 不负责生成精炼 query；LLM 观察 Tool Result 后决定是否进行第二次搜索。

### FR-13：Runtime 事件与终端显示

- Runtime 只能发出 UI 无关的结构化事件，不得导入终端显示依赖。
- 最小事件包括 `llm_started`、`llm_finished`、`tool_started`、`tool_finished`、
  `turn_finished` 和 `run_failed`。
- Tool 完成事件必须在 Tool Result 写入 State 之后发出。
- 普通终端只显示 Tool 名、短参数和结构化摘要，不输出完整摘要、提取文本或全部
  chunk；Debug 结果继续使用最多 400 字符的预览。
- 交互模式中 LLM 或网络错误后必须停止活动状态并允许用户继续输入；非交互模式
  必须以失败退出码结束，不能让 CI 把失败判断为成功。
- 保存确认必须复用当前终端对象，但授权边界仍由 Runtime 强制执行。

### FR-14：离线检索质量评测

- 评测集必须是版本化的固定 JSON，包含论文 ID、逐页文本、查询、期望证据页码和
  分类标签；相同代码版本重复运行应得到相同结果。
- 评测必须直接复用正式的 `chunk_paper_index` 和 `rank_paper_chunks`，不得复制
  一套只为评测服务的排名算法。
- 默认评测不得调用 Agent、LLM、网络或本地 PDF/索引缓存，也不得修改会话 State。
- 对有答案问题分别计算宏平均 Recall@K、Hit Rate@K 和 MRR；对期望页码为空的
  问题单独计算无答案准确率，不能把两类问题混入同一个分母。
- 报告必须包含逐问题结果和按标签汇总，使跨语言失败、通用词误命中等短板可见。
- 必须能在完全相同的数据集上比较原始 TF-IDF、原始 BM25、带查询词覆盖门槛的
  TF-IDF 和 BM25，并记录选择规则和最终默认配置。
- 命令行同时支持简洁人类可读输出和 `--json` 机器可读输出。
- 当前默认 Top-3 回归快照为 Recall@3=0.875、Hit Rate@3=0.875、MRR=0.875、
  无答案准确率=1.000；确定性结果变化时必须显式审查和更新，不代表生产质量达标。

### FR-15：检索资源测量

- 资源基准必须使用临时目录中的确定性合成 PDF，不读取或污染个人 PDF 与索引缓存。
- 每次重复测量必须从空索引开始，先执行一次完整检索建立索引，再对同一 PDF 和
  query 执行缓存检索；两次都必须包含哈希、chunking 和排序实际成本。
- 报告必须分别给出首次检索与缓存检索的样本、中位数、最小值和最大值，并记录
  PDF、索引、索引正文与 chunk 数量。
- “Context 大小”在本阶段定义为一个 `retrieve_paper_chunks` Tool Result 使用
  `json.dumps(..., ensure_ascii=False)` 序列化后的字符数和 UTF-8 字节数；必须说明
  它不包含 instructions、Tool Schemas、历史消息和 Provider tokenizer 开销。
- 计时结果受机器和运行环境影响，只作参考观测，不得成为跨机器单元测试阈值；
  磁盘和载荷边界应通过结构性断言验证。

### FR-16：V3 Plan 与 Planner 输入契约（已实现）

- V3 面向需要多篇论文或多个交付结果的复合任务；简单搜索、保存或单篇问答继续
  使用已经稳定的 V2 Agent Loop。
- 新任务的第一次模型决策可以选择内部 `submit_plan`，不需要额外的关键词路由器；
  简单任务仍可直接返回现有 `tool_call` 或 `final`。活动 Plan 存在时，不得再次创建
  初始 Plan。
- LLM 的 `submit_plan` 参数只能包含有序的高层步骤描述。Runtime 根据原始用户目标
  创建版本化 Plan，并补充 `task_id`、任务 `status`、`revision`、稳定 step ID、
  尝试次数和空证据引用。
- 步骤描述不能被当作可信 URL、路径、论文 ID 或 Tool Result。
- 任务状态只允许 `running|blocked|completed|failed|cancelled`；步骤状态只允许
  `pending|running|completed|blocked|failed`。步骤可从 `blocked` 恢复到 `running`，
  其余状态不能绕过 Runtime 任意回退；所有状态转换由 Runtime 校验和写入。
- 单个 Plan 最多 8 步。模型返回非法字段、重复 step ID、非法状态或越界步骤时，
  计划不得进入执行阶段。
- `submit_plan` 是内部结构化输出契约；它不进入外部 Tool Registry、不计为 Tool
  执行，也不产生外部副作用。Runtime 只允许它用于初始规划或已经授权的 Replan。

当前 `planning.py` 已实现 Runtime 可调用的 `create_plan`、严格结构验证器和纯状态
转换：输入 Plan 不被原地修改，非法字段、非法状态、越界计数器、不可信 evidence
reference 和多个活动步骤都会在进入执行层前被拒绝。`planner.py` 已增加只包含
`steps: list[str]` 的内部 `submit_plan` Schema，将其归一为 `type=plan` 动作；普通
Tool Call 与 final 不变，而且该内部契约不在 Tool Registry。当前 Runtime 已能创建
可信 Plan；Executor 的 step 启动与执行才是下一小步。Plan 与 Planner 边界完成时共
21 项测试通过；启用真实 arXiv 与 DeepSeek 后，当时全量 128 项回归全部通过，没有
跳过。

### FR-17：V3 执行与有限 Replanning（已实现）

- Executor 必须复用现有串行 Agent Loop 和 Tool 安全边界，每次模型决策最多执行
  一个 Tool Call。
- Tool Result 由 Runtime 记录为当前步骤的可信 evidence reference，但不能直接完成
  高层步骤；模型返回 step-level final 后才由 Runtime 完成当前步骤。模型不能直接
  伪造完成状态、计数器或 evidence reference。
- 只有 Tool 失败、候选歧义、检索证据不足或计划前提不成立时才允许 Replan。
- 每个任务最多 Replan 2 次、Tool 执行 24 次。普通模式累计最多 32 次 LLM
  决策，Thinking 模式最多 100 次；达到当前模式的任一上限后必须进入
  `failed` 或 `blocked`，不能继续循环。
- Replan 必须保留已完成步骤和可信证据。已经成功完成或产生副作用的步骤不能仅因
  新计划而重复执行。
- 缺少论文选择、写入许可等关键用户输入时，任务必须进入 `blocked` 并请求澄清。
- 原始目标只能授予其明确包含的动作；Planning 与恢复不能扩大下载或保存权限，
  `save_paper` 仍须逐次人工确认。
- 最终回答必须区分 `completed`、`blocked`、`failed` 和 `cancelled`，并只引用实际
  Tool Result 中存在的论文与页码证据。

当前 Runtime 已接入完整的有限 Replan 与阻塞恢复路径。它对最新未处理的 Tool
Result 做确定性分类：Tool 错误、无候选、无词法匹配、检索证据不足、PDF 无可提取
文本和空文献库属于硬失败；无精确标题匹配的多候选搜索属于可处理歧义。只有存在
这些可信观察、尚有 Plan 容量且 Replan 预算未耗尽时，Executor 才会看到内部
`submit_plan` Schema。模型提交的替换步骤数量还会根据已保留历史动态收紧。

Replan 时 Runtime 先把当前步骤标为 failed，再由 `planning.revise_plan()` 保留全部
completed/failed 步骤、attempts 和 evidence reference，删除旧 pending 后缀并创建
新 step ID，同时增加 revision 和 `budget.replans`。每个任务最多 2 次 Replan；普通
成功观察不能触发重规划。若硬失败后模型改为请求更多信息，Runtime 将当前步骤和
任务置为 blocked；用户下一条消息恢复原步骤并增加 attempts。即使硬失败发生在
单轮 `max_steps` 的最后一次 Tool Call，其可信失败信号也会跨“继续”保留。

`save_paper` 和 `download_paper` 的相同参数已经产生成功结果后，后续 Plan/Replan
调用会复用可信旧 Tool Result，并标记 `runtime_reused=true`，不再次执行写入或下载。
首次 `save_paper` 仍通过现有 Runtime 人工确认边界逐篇询问 `y/N`。普通模式任务预算为
32 次 LLM 决策、24 次 Tool 调用和 2 次 Replan；Thinking 模式只将第一项
提高到 100。

DeepSeek thinking 多步协议也已接入：`llm.py` 提取每个 Assistant 响应的推理元数据，
Planner 在内部化 `submit_plan` 时保留它，Runtime 将其附着到 Plan 预览、Tool Call
和 step-level final 对应的 Assistant State 消息，下一次请求再由 `llm.py` 原样序列化。
这样第三次及之后的 thinking Tool Calling 请求不会因丢失历史
`reasoning_content` 被 Provider 拒绝。OpenRouter 的字符串与结构化 reasoning 格式
已有离线兼容测试。有限 Replan 与 blocked 已接入。本阶段
新增 14 项 Plan、Executor 与 Runtime 用例；启用真实 arXiv 与 DeepSeek 后全量
170 项通过，无跳过。

### FR-18：V3 Checkpoint 与恢复（核心已实现）

- 同一 CLI 进程同时只能有一个活动计划任务，并使用稳定 `task_id` 标识。
- 每次 Plan 创建、步骤状态转换、Replan 或阻塞后，Runtime 必须写入版本化
  Checkpoint。默认路径为 `data/checkpoints/active.json`，文件上限为 4 MiB。
- Checkpoint 必须使用同目录唯一临时文件和原子替换；写入失败不能破坏上一份有效
  Checkpoint。
- Checkpoint 可以保存恢复任务所需的 State、Plan、计数器和可信结果，但不得包含
  API Key、PDF 二进制、完整全文索引或 Tool 内部未返回的数据。
- `python main.py --resume` 只恢复结构有效的 `running|blocked` 任务；损坏、超限或
  版本不兼容必须给出明确错误，不能静默覆盖。
- 恢复时必须重新校验缓存文件、稳定 ID 和已有 Tool Result。完成步骤不得重复执行，
  未完成的写入操作不得被假定为已获授权。
- Ctrl+C 和可恢复 Runtime 错误保留活动 Checkpoint；任务完成或达到不可恢复的
  `failed` 后清除活动文件。用户明确取消时由 FR-19 `/cancel` 入口清除。
- 程序启动时发现活动 Checkpoint，必须提示使用 `--resume`；新任务不能静默覆盖旧
  Checkpoint。
- Checkpoint 只用于恢复同一个任务，不会被自动召回到新任务，因此不是长期 Memory。

实现采用同目录 `NamedTemporaryFile`、文件 `fsync` 和 `os.replace`。恢复时严格校验
Checkpoint 版本/时间、State/Plan 契约、Assistant Tool Call 与 Tool Result 配对、
Plan evidence reference、稳定论文 ID 和已下载 PDF 缓存；任一校验失败都拒绝恢复且
不覆盖原文件。CLI 恢复摘要不会显示 messages 或 reasoning。活动计划下 `/clear`
已被拒绝；显式 `/cancel` 已通过 FR-19 接入同一可信清理边界。

### FR-19：V3 终端、事件与评测（已实现）

- `/plan` 必须以有界形式显示目标、revision、任务状态、当前步骤和各步骤状态；
  `/cancel` 明确取消当前活动任务，并同步更新 `/help`。
- 创建 Plan 后尚未启动步骤时，`/plan` 和 `--resume` 必须显示“当前步骤：尚未启动”，
  并把第一条 pending step 单独标为“下一步骤”，不能把 pending 误称为 current。
- 活动任务存在时，`/clear` 不得静默删除 Plan 或 Checkpoint；应提示先继续或取消。
- 活动任务执行期间，普通用户消息只用于补充或澄清当前任务；开始无关任务前必须
  先完成或 `/cancel` 当前任务。
- Runtime Event 必须增加 UI 无关的计划创建、步骤转换、Replan、Checkpoint 保存和
  任务阻塞事件；终端仍只是观察者，事件回调不能修改 Runtime 状态。
- 离线测试必须覆盖两篇论文的成功流程、非法 Plan、歧义阻塞、Tool 失败后 Replan、
  预算耗尽、取消、Checkpoint 损坏、进程恢复和副作用幂等。
- 规划评测至少记录任务完成状态、Replan 次数、累计 LLM 决策、Tool 执行次数和
  Checkpoint 字节数；这些数据必须与普通 Debug 文本分离，能够机器读取。
- 最终验收必须显式启用真实 arXiv 与 DeepSeek，并使用真实 PDF 完成一次多论文、
  多步骤、带页码证据的任务；联网用例不得计为跳过。

当前实现中，`/plan` 只显示有界目标、revision、任务状态、当前步骤和各步骤状态；
`/cancel` 调用 Runtime 取消入口，在成功清除 Checkpoint 后才更新 State，清除失败时
两者均保持活动。计划生命周期新增 `plan_created`、`plan_step_changed`、
`plan_replanned`、`task_blocked`、`task_cancelled` 和 `checkpoint_saved` 事件；事件
不携带 reasoning 或 Checkpoint 路径，并继续使用深拷贝隔离观察者。

`evaluate_planning.py` 使用正式 Runtime、固定 Provider 响应和固定 Tool 数据执行
两论文成功场景，可输出人类报告或 JSON。当前确定性结果为 completed=true、
Replan=0、LLM 决策 6 次、Tool 执行 2 次、Checkpoint 保存 13 次、最大 4,572 B。
真实系统用例以固定四步高层计划减少 Planner 随机性，但 DeepSeek Executor、arXiv
搜索、两份 PDF 下载、全文索引、页码检索和最终比较均走正式路径。2026-09-13
完整联网验收运行 202 项测试，202 项全部通过，没有跳过。

提交后的交互复盘又增加 `--thinking` 和 current/next step 显示回归测试；最新一次
完整联网验收运行 207 项测试，207 项全部通过，没有跳过。

### FR-20：V3.1 分层评测（已实现）

- Debug 模式必须在每次 Runtime turn 结束后输出确定性的运行健康分；普通模式不
  增加这段输出。
- 健康分只能使用结构化 Runtime Event，不读取 reasoning，不再次调用模型，也不把
  用户查询、摘要、chunk 或完整 Tool Result 持久化到评分器。
- 评分必须分别暴露本轮控制、事件协议完整性、Tool 成功率和预算健康度；Tool 未被
  调用时对应维度为 `N/A`，用户取消时不评分。
- 输出必须明确声明健康分不等于答案正确率。一个控制流完整的错误答案不能得到
  “检索正确”的结论。
- 外部文献搜索质量采用 LitSearch 的官方记录格式和宏平均 Recall@K 定义；至少报告
  broad Recall@20、specific Recall@5 和 specific Recall@20。
- LitSearch scorer 必须读取 `.json`/`.jsonl`，支持官方文档级 corpus ID 以及段落
  结果中的 `(corpusid, paragraph_idx)` 形状，并提供 `--json` 机器可读报告。
- scorer 默认离线，不下载完整 LitSearch corpus、不调用 Agent、LLM 或网络；输入
  不完整、ID 非法或重复 gold ID 时必须失败，而不是产生貌似有效的分数。
- 只有在相同 LitSearch corpus 上生成、并已映射为 Semantic Scholar corpus ID 的
  `retrieved` 列表才能与论文结果直接比较。AgentReader 在线 arXiv/Crossref Top-3
  结果不得自动冒充官方 benchmark 输出。

运行健康分权重固定为：本轮控制 45、协议完整性 25、Tool 成功率 20、预算健康度
10。没有 Tool 时只对其余适用维度重新归一；达到 `max_steps`、Runtime guard、
Tool error 或 `run_failed` 只降低其对应的可解释维度。

2026-09-13 完整联网验收在项目 Conda 环境运行 219 项测试，真实 arXiv、DeepSeek
thinking 和双 PDF 系统用例全部执行，219 项全部通过、0 跳过。

### FR-21：V3.2 LitSearch 同语料检索基线（Step 1 已实现）

- benchmark adapter 必须只位于 `evals/`，不得替换或改变 production
  arXiv/Crossref `search_paper`。
- corpus 输入必须是 JSON/JSONL，包含唯一正整数 `corpusid`、字符串或 null
  `title` 和 `abstract`；官方空文本记录必须保留在固定 ID 空间，重复 ID、超限
  文本和非投影的大文件必须被拒绝。
- query 输入复用 LitSearch 的 `query`、`query_set`、`specificity`、`quality` 和
  gold `corpusids` 校验，但在检索前不要求 `retrieved`。
- 轻量 baseline 使用当前 Python SQLite 的 FTS5、Porter tokenizer 和 BM25 排序；
  它必须使用独立方法名，并显式声明不等于官方 NLTK + `rank_bm25` baseline。
- 每个 query 必须生成至少 Top-20、最多 Top-200，避免用候选深度不足的列表计算
  Recall@20；匹配不足时按固定 corpus 顺序确定性补齐。
- 输出必须保留 Semantic Scholar corpus ID，能够直接交给 FR-20 scorer；可选使用
  原子写入保存官方形状 JSON/JSONL。
- CLI 必须支持查询数量限制、结果输出和完整 JSON 报告；默认不访问网络、不调用
  LLM，也不读取 Agent State 或个人文献库。
- `data/evals/` 必须被 Git 忽略。官方数据下载、全 597-query 实跑和正式指标记录
  属于 V3.2 Step 2，在完成前不得声称已有 AgentReader 全量 LitSearch 基线。

2026-09-13 Step 1 验收在项目 Conda 环境运行 226 项完整联网测试，真实 arXiv、
DeepSeek、Thinking 和双 PDF 系统用例全部执行，226 项全部通过、0 跳过。

### FR-22：V3.2 官方数据准备与首次基线（已实现）

- 数据下载依赖必须位于 `environment-benchmark.yml` 的独立 Conda 环境，不得加入
  主应用 `environment.yml`。
- 数据集必须固定 revision；只下载 `query` 和 `corpus_clean`，不得下载更大的
  `corpus_s2orc`。
- Parquet 必须流式投影 query 契约及 `corpusid/title/abstract`，不得把 full paper
  写入评测输入。
- 投影必须原子写入、校验 597/64,183 记录并生成文件大小与 SHA-256 元数据；缓存、
  投影和结果都位于被 Git 忽略的 `data/evals/`。
- 首次基线必须先通过 20-query smoke test，再运行 597-query 全集并用 FR-20 scorer
  独立复算。
- 2026-09-13 固定基线为 Broad R@20=0.424、Specific R@5=0.523、Specific
  R@20=0.692、全体 R@5=0.465、全体 R@20=0.623；它是 AgentReader SQLite
  FTS5 基线，不是官方 BM25 数值。
- Step 2 完整离线验收为 231 项通过、5 项联网按开关跳过；完整联网验收为
  231 项通过、0 跳过。

### FR-23：V3.2 Agent 层结构化任务成功率（已实现）

- task case 必须是版本化 JSON，显式声明 goal、受控执行 fixture 和结构化期望。
- scorer 只能读取最终 Plan 状态、Assistant Tool Call 和可信 Tool Result；不得读取
  reasoning，不得用另一个 LLM 评分，也不得从自由文本推断论文标题是否正确。
- 至少分别报告任务完成、Tool 使用、gold 候选发现、下载选择、页码证据和无结果
  安全停止；不适用维度必须为 `N/A`，不能按失败计分。
- 报告只能保留有界 ID、Tool 名、计数和页码，不得复制摘要、chunk 正文、API Key
  或本地文件路径。
- 确定性 runner 必须经过正式 Runtime；只固定 Provider 决策和 Tool 外部数据，不得
  复制一套只为评测服务的 Agent Loop。
- 评测 CLI 必须支持按 case 过滤、`--json` 和 `--debug`；Debug 只能在 case 已提供
  gold 时显示任务正确性分数。普通 `main.py --debug` 仍只显示运行健康度。
- 真实 DeepSeek 双论文验收可以复用 scorer，但原有真实 arXiv、双 PDF、页码引用
  和最终回答断言不得被 scorer 替代或放宽。
- 最新固定 task suite 为 3/3、平均 100 分；完整离线测试共 247 项，其中 5 项联网
  按开关跳过；完整 arXiv + DeepSeek 联网验收 247 项全部通过、0 跳过。

## 4. 状态需求

V1 State 只保存当前任务需要的信息：

```python
{
    "user_query": str,
    "messages": list,
    "step": int,
}
```

同一次命令行会话的后续用户消息继续写入这份 State，因此模型能看到上一轮的
候选和回答。退出 CLI 后 State 会清空；磁盘上的文献库、PDF 和全文索引可在下次
运行时继续使用，但不会自动进入 State，因此仍不是跨任务 Agent Memory。

每次 `agent.py` 调用模型时，会把 instructions 和当前 State 中的 messages 组成
本次 Context。`retrieve_paper_chunks` 只限制这一次新增的 Tool Result 为少量
top-k 片段；旧消息仍保留在会话 State 中。本项目没有自动压缩 Context，也没有
跨会话长期 Memory。

V3 State 现已增加可为空的 `task_id` 和 `plan`；创建计划后，当前步骤、任务级预算
和证据引用由 Plan 保存。后续 step 执行会填充这些字段。
Checkpoint 是 State 的持久化快照；Context 仍是某一次模型调用选择发送的内容，
二者不能混为一谈。Plan 与任务预算用于限制 Context 持续增长，但 V3 不实现跨任务
语义召回或自动长期 Memory。

## 5. 配置和数据要求

- 使用 Conda 和 `environment.yml` 管理 Python 3.13 环境。
- 增强终端使用 `prompt-toolkit` 和 `rich`；Plain 模式不得依赖终端控制序列。
- `.env` 保存本地 Provider 配置和 API Key，不提交到 Git。
- `data/library.json` 是个人文献库数据，不提交到 Git。
- `data/pdfs/` 和 `data/indexes/` 是可重新生成的本地缓存，不提交到 Git。
- V3 的 `data/checkpoints/` 是可恢复的任务状态，不提交到 Git。
- 默认离线测试不得访问真实模型或外部文献 API。
- 在线测试必须通过环境变量显式启用。
- 完整验收必须同时设置 `RUN_LIVE_ARXIV_TESTS=1` 和 `RUN_LIVE_LLM_TESTS=1`，实际
  执行 arXiv 与 DeepSeek 测试；普通离线运行仍保留安全开关。

## 6. 当前已知限制

- “用户是否明确要求保存”仍由 Agent instructions 判断，但 Runtime 会在实际
  写入前独立请求确认，因此模型不能绕过人工授权产生本地写入。
- 已实现 PDF 下载、缓存、有界文本提取、持久化全文索引和模型可见的词法 Chunk
  Retrieval；索引仍受 200 页 / 200 万字符上限约束，也没有跨语言语义匹配、OCR
  或完整 RAG。
- 搜索和段落检索仍是简单词法信号，可能受同义词、跨语言表达和通用词影响；50%
  查询词覆盖门槛已拦截固定评测中的通用词无答案误命中，但阈值只经过小型教学
  数据集验证，并非通用置信度。中文查询英文正文仍无直接召回，当前没有 embedding
  或语义重排；Agent 只能通过生成英文技术词做轻量查询改写。
- 终端不是全屏 TUI，不支持逐 token 流式输出、后台任务和一般对话历史恢复；活动
  Plan 的 Ctrl+C 和进程退出可以通过 Checkpoint 恢复同一任务，但不是长期 Memory。

## 7. V2–V2.1 验收结果与下一里程碑

V2 端到端验收已完成：

- 强匹配搜索产生了可解释分数和 `exact_title_match`；
- 无匹配时模型请求补充信息，没有把无依据候选当作答案；
- 使用真实 arXiv、DeepSeek 和 PDF 验证了搜索、保存、下载、提取和段落检索；
- 使用真实 TTY 验证了历史、Slash Commands、实时 Tool 状态和连续对话；
- 生成的 12 页 PDF 能从第 12 页召回证据；DeepSeek 在真实 15 页 PDF 上区分第 9
  页正文引用和第 12 页参考文献，第二次查询复用同一索引；
- 固定 10 问评测的默认 TF-IDF + 门槛 Top-3 为 Recall@3=0.875、Hit Rate@3=
  0.875、MRR=0.875、无答案准确率=1.000；7 个英文词法问题的三项检索指标均为
  1.0；原始 TF-IDF 与 BM25 的无答案准确率均为 0.500，带门槛后均为 1.000，
  其余三项总体指标保持 0.875；
- 12 页 / 36,000 字符合成 PDF 的首次检索中位数约为 8.9 ms，缓存检索约为
  1.4 ms；37,196 B 索引只向 Context 新增 5,131 B top-3 Tool Result；
- 2026-09-12 联网完整验收运行 107 项测试，107 项全部通过，没有跳过。

V2.1 已完成：BM25 没有改善当前数据集指标，最低查询词覆盖门槛解决了已知的通用
词误命中，因此默认保留 TF-IDF + 50% 门槛。V3 已确定为显式 Planning、有限
Replanning 和同任务 Checkpoint 恢复；当前已完成 Plan、Planner、Executor、有限
Replan、blocked、Checkpoint、`--resume`、终端控制、Runtime Events、规划评测和
真实多论文系统验收，V3 已完成。

## 8. 当前非目标

V3 不实现 LangGraph、OpenAI Agents SDK、长期 Memory、Graph、Multi-Agent、
MCP、Vector Database、完整 RAG、并发 Tool Calling 或复杂异步执行。跨语言语义
检索也不与 Planning 同期实现。
