# AgentReader Demo 需求说明

## 1. 文档目的

本文描述 AgentReader Demo 当前已经实现的行为、明确限制和下一阶段需求。
它记录的是项目现状，不是最终产品规格。

当前版本定位：V1 最小 Agent Loop、V2 真实搜索/管理/阅读/最小检索、V2.0.1
提取正确性和交互式终端，以及 V2.1 全文索引、离线评测、资源测量和排序方法选择
均已完成。V3 尚未开始。

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
  State；Slash Commands 不得进入 State 或发送给模型。
- 交互式 TTY 提供当前进程内的输入历史和 Slash Command 补全，不把历史默认写盘。
- `python main.py --debug` 按实际发生顺序显示每步 LLM 决策、Tool Call、Tool Result
  和耗时，不需要等待整轮完成。
- `python main.py --plain` 禁用颜色、Spinner 和增强输入；非 TTY 自动使用 Plain
  模式，输出不得含 ANSI 控制符，运行失败时必须返回非零退出码。
- 搜索 Tool Result 的调试显示必须把每篇摘要限制为最多 400 字符的
  `abstract_preview`；完整摘要仍保留在 State 中，显示裁剪不得改变模型 Context。

### FR-2：统一 LLM 接口

- `call_llm(messages, tools)` 是 Agent 使用的唯一模型接口。
- 支持 `fake`、`deepseek` 和 `openrouter` Provider。
- Provider 输出必须归一为 `final` 或单个 `tool_call`。
- V1 每个 LLM step 最多处理一个 Tool Call。
- Provider 同时返回多个 Tool Calls 时，只将第一个归一化并交给 Runtime；其余
  调用不执行，模型观察第一个结果后再决定下一步。

### FR-3：文献搜索

- `search_paper(query)` 接收非空字符串。
- 首先查询 arXiv Atom API。
- query 包含标准数字 arXiv ID 时，应使用 `id_list` 定向查询。
- arXiv 返回 HTTP 429 时，应等待服务建议或默认间隔后重试一次。
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
- `max_steps` 必须在模型不能结束时终止当前轮；每次用户新输入后重新计数。
- Runtime 必须在每次 `run_agent()` 调用中记录已经尝试的有效搜索 query；比较前
  折叠首尾及连续空白，与 `search_paper` 的规范化一致。
- 同一用户轮次的重复 query 必须返回 `duplicate_search_query` 结构化错误，不得
  再次调用搜索 Tool。精炼后的不同 query 仍可执行。
- 这份去重集合只属于当前 `run_agent()` 调用；下一条用户消息开始时必须重置，
  以保留用户主动重新搜索同一 query 的能力。
- 每次 `run_agent()` 最多允许 2 个不同的有效搜索 query。第三个不同 query 必须
  返回 `search_limit_reached`，不得执行搜索 Tool；下一用户轮次重新计数。
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

## 5. 配置和数据要求

- 使用 Conda 和 `environment.yml` 管理 Python 3.13 环境。
- 增强终端使用 `prompt-toolkit` 和 `rich`；Plain 模式不得依赖终端控制序列。
- `.env` 保存本地 Provider 配置和 API Key，不提交到 Git。
- `data/library.json` 是个人文献库数据，不提交到 Git。
- `data/pdfs/` 和 `data/indexes/` 是可重新生成的本地缓存，不提交到 Git。
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
- 终端不是全屏 TUI，不支持逐 token 流式输出、后台任务和跨进程历史；Ctrl+C 能
  安全结束当前调用，但不提供中间检查点或恢复执行。

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
词误命中，因此默认保留 TF-IDF + 50% 门槛。剩余跨语言失败不是更换词法评分公式
能解决的问题；V2.1 不引入 embedding/hybrid。开始 V3 前必须先用真实任务选择一项
能力并定义验收标准。

## 8. 当前非目标

本阶段不实现 LangGraph、OpenAI Agents SDK、Planning、长期 Memory、Graph、
Multi-Agent、MCP、Vector Database、完整 RAG、并发 Tool Calling 或复杂异步
执行。
