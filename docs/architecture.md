# AgentReader Demo 架构说明

## 1. 当前架构

AgentReader 使用一个显式、同步、单 Agent 循环：

```text
terminal.py 收集用户输入
  ↓
main.py 解析 Slash Command，创建或复用 State
  ↓
runtime.py 启动循环
  ├─ 结构化 Runtime Event → terminal.py 实时渲染
  ├─ 无活动 Plan → planner.py 首次选择 plan | tool_call | final
  ├─ 活动 Plan → planning.py 启动可信 step → executor.py 决策
  └─ 简单任务后续决策 → agent.py
             ↓
  llm.py 调用 Fake / DeepSeek / OpenRouter
             ├─ tool_call → Runtime 校验并执行 Tool
             │                  ↓
             │            Tool Result 写入 State/Plan evidence
             │                  ↓
             │            回到下一次 LLM 决策
             └─ final → 完成当前 step 或本轮
                                      ↓
                           terminal.py 输出并等待输入
                                      ↓
                              追加到同一个 State
```

这个结构故意不使用 Agent 框架，便于直接观察每次状态变化。

## 2. 模块职责

### `main.py`

- 解析 `--debug`、`--plain` 和 `--resume`。
- 组合 Terminal、State 和 Runtime，但不包含具体显示实现。
- 启动时检测唯一活动 Checkpoint；普通启动拒绝覆盖，`--resume` 校验后恢复 State。
- 第一轮创建 State，后续轮次向同一个 State 追加用户消息。
- 处理 `/help`、`/debug`、`/clear`、`/plan`、`/cancel`、`/exit`；这些命令不会
  写入 State。
- 空输入重新提示；`/exit`、`exit`、`quit`、`退出` 或 EOF 结束会话。
- 交互模式的可恢复运行错误不会关闭会话；非交互模式以非零状态码退出。
- 不包含 Agent 决策或 Tool 实现。

### `terminal.py`

- 使用 Prompt Toolkit 提供当前进程内的历史和 Slash Command 补全。
- 使用 Rich 显示 Spinner、Markdown 回答和实时 LLM/Tool 状态。
- 非 TTY 或 `--plain` 时退回无 ANSI 控制序列的纯文本界面。
- 把 Tool Result 转成有界显示摘要；Debug 预览不修改 State 中的原始数据。
- 展示可信待保存论文并读取确定性的 `y/N` 确认。
- 有界显示 Checkpoint 冲突、恢复摘要和校验错误，不显示消息历史或 reasoning。
- 有界显示 Runtime-owned Plan，并渲染计划创建、步骤转换、Replan、阻塞、取消和
  Checkpoint 保存事件。
- 只负责输入与显示，不执行 Tool，也不决定授权策略。

### `events.py`

- 定义 UI 无关的 `AgentEvent` 和可选 `EventHandler` 类型。
- 事件描述 LLM/Tool 以及 Plan 创建、步骤转换、Replan、阻塞、取消和 Checkpoint
  保存生命周期，不包含 Rich 或 Prompt Toolkit 对象。
- Plan 生命周期事件只携带现有契约约束下的 metadata，不包含 reasoning、完整 State
  或 Checkpoint 路径；Tool 事件仍提供真实结果，终端负责有界显示。Runtime 向所有
  观察者发送深拷贝。

### `state.py`

- 创建单次任务的普通 `dict`。
- 保存原始请求、多轮消息历史和累计 LLM step 数量。
- Assistant 消息可带一份归一化的 `reasoning_content` 或
  `reasoning_details`；State 只为后续 Provider 协议回传保存它，不解释其内容。
- 提供追加后续用户消息的最小操作。
- 活动 Plan 可以通过 Checkpoint 跨进程恢复；当前没有跨任务 Memory。

### `agent.py`

- 定义文献助手 instructions。
- 声明 Agent 可以看到的 Tool Schemas。
- 将当前 messages 交给 `llm.py`。
- 不执行 Tool。

### `llm.py`

- 根据 `LLM_PROVIDER` 选择 Fake、DeepSeek 或 OpenRouter。
- 把内部 Tool Schema 转成 OpenAI Chat Completions 工具格式。
- 把内部 messages 转成 Provider API messages。
- 将 Provider 响应归一为内部的 `final` 或 `tool_call`。
- 显式生成 DeepSeek thinking / reasoning effort 请求选项，并按需生成 OpenRouter
  reasoning 选项。
- 将 DeepSeek 的 `reasoning_content`、OpenRouter 的 `reasoning` 字符串别名或
  `reasoning_details` 归一化，并在后续请求中原样序列化。
- Provider 返回多个 Tool Calls 时只归一化第一个，保持 Runtime 串行执行。
- Fake LLM 为离线测试提供确定性行为。

Provider 原始格式到此为止。Agent 和 Runtime 不解析厂商响应或 reasoning 内容；
Runtime 只保存 `llm.py` 已归一化的不透明协议字段。

### `runtime.py`

- 维护 Tool Registry。
- 循环请求 Agent 决策。
- 检查工具是否注册、参数是否为字典、参数是否匹配 Python 签名。
- 将 `save_paper` 的 `candidate_id` 解析为历史搜索结果中的可信 metadata。
- 在执行 `save_paper` 前调用确认器；没有确认器或用户拒绝时不写入。
- 将 `extract_paper_text` 的 `paper_id` 解析为历史下载结果中的可信缓存路径。
- 将 `retrieve_paper_chunks` 的 `paper_id` 解析为历史下载结果；只允许模型补充
  query 和有界 `top_k`，不接受本地路径。
- 在每次 `run_agent()` 调用中维护本轮搜索 query 集合，阻止相同有效 query 再次
  到达搜索 Tool，并把不同 query 的数量限制为 2；下一用户轮次重新创建集合。
- 执行 Tool，并将结果或结构化错误写回 State。
- 将每次归一化模型响应的 reasoning 元数据附着到对应 Assistant State 消息，使
  `llm.py` 能在下次请求中恢复完整的 thinking Tool Calling 历史。
- 通过可选回调发出 LLM、Tool、Plan、Checkpoint、结束和失败事件；发给观察者的是
  深拷贝，回调异常不得改变 Agent Loop。
- 在 LLM 或 Tool 被用户中断时补齐消息协议，并用 `cancelled` 状态结束当前轮。
- 更新累计 step，处理本轮 final，并用每轮默认 20 次的 `max_steps` 阻止无限循环；
  活动 Plan 到限时保留进度并提示用户输入“继续”。
- 在 Plan 创建、步骤转换、Tool Result、Replan、blocked 和暂停等一致边界保存活动
  Checkpoint；完成或不可恢复失败后清除。可恢复异常沿用最近有效快照。
- `/cancel` 通过独立 Runtime 入口执行：先清除活动 Checkpoint，再提交 cancelled
  Plan；清除失败时不修改 State。

Runtime 是执行和约束层。它限制保存、下载和提取操作只能引用当前任务中的可信
Tool Result，并强制保存操作必须经过人工确认。`terminal.py` 负责具体终端交互，
`main.py` 负责组合各层，Runtime 负责执行确认和来源校验策略；具体 Tool 不读取
用户输入。

### `checkpoint.py`

- 解析默认路径 `data/checkpoints/active.json` 或测试用路径覆盖。
- 只保存结构有效的 `running|blocked` State，格式版本为 1、上限为 4 MiB。
- 使用同目录唯一临时文件、`fsync` 和 `os.replace`，写入失败保留上一份文件。
- 加载时重新验证 Plan、消息配对、可信 ID、evidence reference 和 PDF 缓存。
- 拒绝不同 `task_id` 覆盖活动文件；提供幂等清理，但不决定何时取消任务。
- 保存 messages 中 Provider 恢复所需的 reasoning 元数据，不解析或显示它。

该模块不知道 LLM、Planner 或终端。Checkpoint 写入时机由 Runtime 决定，启动恢复
策略由 `main.py` 决定。

### `tools/`

Tool 包通过 `tools/__init__.py` 暴露稳定公共接口，调用方继续使用
`from tools import ...`。内部按职责拆分：

- `schemas.py`：全部模型可见的 Tool Schema。
- `search.py`：arXiv 优先、Crossref 备用的只读搜索、响应解析和候选本地重排。
- `library.py`：生成稳定论文 ID、校验 metadata、写入 JSON 和有界只读查询。
- `download.py`：校验可信 PDF URL，执行有界下载并管理本地缓存。
- `extract.py`：再次校验缓存路径，并对可信 PDF 执行有界文本提取。
- `index.py`：建立、校验、失效和原子写入持久化全文 JSON 索引。
- `retrieval.py`：封装索引页 chunking、确定性 TF-IDF/BM25 排序和查询证据门槛。

`search.py` 复用 `library.py` 的论文 ID 函数，使搜索返回的 `candidate_id` 与最终
文献库 ID 始终遵循同一规则。包内文件可以协作，但 Agent 和 Runtime 只依赖
`tools/__init__.py` 暴露的公共接口。

### `evals/` 与评测命令

- `evals/retrieval_cases.json` 保存版本化的逐页文本、查询、期望页码和标签。
- `evals/retrieval.py` 校验数据集，构造与正式全文索引相同的内存结构，并直接复用
  `chunk_paper_index` 和 `rank_paper_chunks`。
- `evaluate_retrieval.py` 只是命令行入口，提供人类可读报告、`--json` 输出和
  `--compare` 同数据集方法对比。
- `evals/resources.py` 在隔离目录生成合成 PDF，测量首次检索、缓存检索、磁盘索引
  和 top-k Tool Result；`measure_retrieval_resources.py` 是对应命令行入口。
- `evals/planning.py` 使用正式 Runtime 和 Checkpoint，固定 Provider 响应与 Tool
  数据执行确定性的两论文任务；`evaluate_planning.py` 输出人类报告或 JSON 指标。
- 检索质量评测不进入 Runtime 或 Agent Loop；Planning 评测则刻意经过正式 Runtime，
  但不调用真实 LLM 或网络。资源基准只使用自己在临时目录创建的缓存。

## 3. 内部消息协议

### 用户或普通 Assistant 消息

```python
{"role": "user" | "assistant", "content": "..."}
```

### Assistant Tool Call

```python
{
    "role": "assistant",
    "content": None,
    "reasoning_content": "...",  # 可选；不透明且不向终端显示
    "tool_call": {
        "id": "...",
        "name": "search_paper",
        "arguments": {"query": "TASA"},
    },
}
```

### Tool Result

```python
{
    "role": "tool",
    "tool_call_id": "...",
    "name": "search_paper",
    "content": {...},
}
```

`tool_call_id` 将某一次调用与结果配对。Tool Result 进入 messages 后，LLM 才能
看到真实执行结果并决定是继续调用工具还是返回 final。

thinking Provider 的 Assistant 消息还可以带 `reasoning_content`，或用
`reasoning_details` 保存 OpenRouter 的结构化格式。内部消息至多保留一种表示；这些
字段只随对应 Assistant 消息回传 Provider，不进入 Runtime Event 或终端输出。

### Runtime Event

Runtime 另外向可选观察者依次发出：

```text
llm_started → llm_finished
                  ├─ final → turn_finished
                  └─ tool_call → tool_started → Tool Result 写入 State
                                      ↓
                                tool_finished → 下一 step
```

活动 Plan 还会在可信状态边界发出：

```text
plan_created
plan_step_changed: pending → running → completed|blocked|failed
plan_replanned
task_blocked | task_cancelled
checkpoint_saved
```

未处理异常发出 `run_failed` 后仍向调用方抛出。事件是显示与执行之间的只读边界，
不是 State，也不会交给 LLM；终端回调失败或修改收到的字典都不能改变 Tool 参数和
State 中的结果。

## 4. LLM 响应协议

`llm.py` 对外只返回两种结构。这里的 `final` 表示当前用户输入已经得到一条
完整回答，不表示整个命令行会话必须结束。

### Tool Call

```python
{
    "type": "tool_call",
    "content": None,
    "tool_call_id": "...",
    "tool_name": "search_paper",
    "tool_arguments": {"query": "TASA"},
    "reasoning_content": "...",  # 可选
}
```

### Final

```python
{
    "type": "final",
    "content": "...",
    "tool_call_id": None,
    "tool_name": None,
    "tool_arguments": None,
    "reasoning_content": "...",  # 可选
}
```

上面可选的 `reasoning_content` 也可能由 `reasoning_details: list` 代替，二者不能
同时出现。无 thinking 响应继续保持原结构。Runtime 将它视为 Provider 协议元数据，
不会分析、裁剪或作为 evidence reference；`llm.py` 在下一次请求中把它复制回对应
Assistant 消息。DeepSeek thinking 模式要求完整回传此前轮次，因此 Plan 预览、
Tool Call 和 step-level final 都必须经过同一条保存链路。

V1 每一步只接受一个 Tool Call。如果 Provider 同时返回多个 Tool Calls，
`llm.py` 只保留第一个。Runtime 执行并写回结果后，LLM 可以在下一 step 再提出
另一个调用。这样不会并发执行，也不会因为模型返回多个调用而中断会话。

## 5. Tool 契约

### `search_paper`

输入：

```python
{"query": str}
```

输出：

```python
{
    "found": bool,
    "source": "arxiv" | "crossref",
    "count": int,
    "candidate_count": int,
    "truncated": bool,
    "ranking_method": "weighted_lexical_overlap_v2",
    "relevance_assessment": {
        "status": (
            "identifier_match"
            | "lexical_match"
            | "no_lexical_match"
            | "no_candidates"
        ),
        "best_score": int | None,
    },
    "papers": [
        {
            "source": str,
            "candidate_id": str,
            "arxiv_id": str | None,
            "doi": str | None,
            "title": str,
            "authors": list[str],
            "abstract": str | None,
            "published": str | None,
            "paper_url": str | None,
            "pdf_url": str | None,
            "local_relevance": {
                "score": int,
                "title_matches": list[str],
                "abstract_matches": list[str],
                "title_phrase_match": bool,
                "exact_title_match": bool,
                "source_rank": int,
            },
        }
    ],
}
```

Crossref 回退结果还包含 `fallback_reason`，用于区分 arXiv 无结果和不可用。
不可用时还包含 `arxiv_error`，用于在 Debug Trace 中保留 HTTP 429、超时等具体
原因。query 包含标准数字 arXiv ID 时，搜索层使用 `id_list` 定向查询；HTTP 429
会按 `Retry-After` 或默认 3 秒等待并重试一次。
`candidate_id` 优先使用去除版本号的 arXiv ID，其次使用 DOI；两者都没有时
使用标题和论文 URL 的稳定哈希。

普通关键词请求让搜索源最多返回 10 个候选；明确 arXiv ID 时仍只请求 1 个。
`search.py` 随后执行 `weighted_lexical_overlap_v2`：英文和数字按不区分大小写的
词项处理，连续中文拆为双字片段；标题命中每项 4 分、摘要命中每项 1 分，完整
query 词项序列连续出现在标题时再加 `词项数 × 2`；规范化标题与 query 完全
相等时再加 `词项数 × 2`。因此精确标题的最低得分也会高于只包含该短语的标题。
同分使用 `source_rank` 保持来源原始顺序。只有前 3 篇添加稳定 `candidate_id`
后进入 Tool Result；其余候选不会写入 State。

搜索结果写入 State 时保留完整摘要，供下一次 LLM 决策使用。`terminal.py` 显示
Debug Trace 时复制显示对象，把每篇 `abstract` 替换为最多 400 字符的
`abstract_preview`；这一显示层裁剪不改变 Tool Result 或模型 Context。

顶层相关性状态由确定性代码产生：查询中的 arXiv ID 与返回记录一致时为
`identifier_match`；否则最高分大于 0 为 `lexical_match`；候选存在但最高分为
0 时为 `no_lexical_match`；没有候选为 `no_candidates`。这只是最低词法信号，
不代替模型对标题、摘要和作者的判断。

### `save_paper`

LLM 可见的输入：

```python
{"candidate_id": "arxiv:1706.03762"}
```

Runtime 会在当前 State 的历史 `search_paper` Tool Results 中解析该 ID，再把
可信论文对象交给确认器。用户允许后，Runtime 才把论文交给内部的
`save_paper(paper)` 存储函数。输出会说明是否保存、论文 ID、标题、文献库路径
和当前总数。用户拒绝返回 `saved=false, reason=user_declined`；重复记录返回
`saved=false, reason=already_exists`，两者都不写入新记录。

### `list_library`

LLM 可见的输入：

```python
{"limit": 20}  # 可省略，范围 1..50
```

输出：

```python
{
    "count": int,
    "total": int,
    "truncated": bool,
    "papers": [
        {
            "id": str,
            "title": str,
            "authors": list[str],
            "source": str | None,
            "published": str | None,
            "paper_url": str | None,
            "pdf_url": str | None,
            "saved_at": str | None,
        }
    ],
    "library_path": str,
}
```

结果按最近保存优先排列，不返回 abstract。不存在的文献库返回空结果且不创建
文件；损坏的 JSON 会抛出异常，由 Runtime 转换为 `tool_execution_error`。

### `download_paper`

LLM 可见的输入：

```python
{"paper_id": "arxiv:1706.03762"}
```

Runtime 只在历史 `search_paper` 的 `candidate_id` 或 `list_library` 的 `id` 中
解析该值，再把可信论文对象交给内部下载函数。输出：

```python
{
    "downloaded": bool,
    "cached": bool,
    "paper_id": str,
    "title": str | None,
    "pdf_url": str,
    "local_path": str,
    "size_bytes": int,
}
```

下载函数只允许 HTTPS，拒绝带凭据、localhost 和直接使用私有 IP 的 URL；请求
超时 20 秒，最多读取 20 MiB，并校验 Content-Type 与 `%PDF-` 文件头。缓存采用
稳定文件名和临时文件替换。二进制内容不会进入 messages。

### `extract_paper_text`

LLM 可见的输入：

```python
{
    "paper_id": "arxiv:1706.03762",
    "max_pages": 5,     # 可省略，范围 1..10
    "max_chars": 12000, # 可省略，范围 500..20000
}
```

Runtime 只在当前 State 以前成功的 `download_paper` Tool Result 中解析
`paper_id`，然后把完整下载结果交给内部提取函数。模型不能传入 `local_path`。
提取函数还会将路径解析为真实绝对路径并确认它位于当前 `PAPER_CACHE_DIR` 内，
从而形成第二层路径边界。输出：

```python
{
    "paper_id": str,
    "title": str | None,
    "page_count": int,
    "pages_scanned": int,
    "pages_read": int,
    "page_numbers": list[int],
    "last_page_partial": bool,
    "char_count": int,
    "text_available": bool,
    "truncated": bool,
    "truncation_reasons": list["page_limit" | "character_limit"],
    "text": str,
    "warning": str,  # 仅空文本时存在
}
```

默认只读取开头 5 页、返回 12000 字符，硬上限为 10 页和 20000 字符；文本带有
`[Page N]` 标记。`pages_scanned` 表示解析器实际检查的页数，`pages_read` 和
`page_numbers` 只统计确实进入返回文本的页面；字符限制落在页面中间时
`last_page_partial=true`。一旦确认某段非空正文无法完整返回便停止；如果某页恰好
填满上限，则继续检查后续页来判断是否真的发生截断。正文中独立成行的同形
`[Page N]` 会被转义，避免被下游误认成结构标记。空文本是正常结果，加密或损坏
PDF 是结构化 Tool 错误。
`--debug` 只显示最多 400 个字符的正文预览。当前不做 OCR，也不表示已经读取
整篇论文。

### 内部 `load_or_build_paper_index`

这一函数不是模型可见 Tool。它只接受 Runtime 从可信下载历史恢复的
`download_result`，再次调用 `validate_cached_pdf_path()` 校验 PDF，然后计算
SHA-256 并检查本地 JSON 索引。索引结构为：

```python
{
    "version": 1,
    "paper_id": str,
    "title": str | None,
    "pdf_sha256": str,
    "pdf_size_bytes": int,
    "max_pages": 200,
    "max_chars": 2_000_000,
    "page_count": int,
    "pages_scanned": int,
    "pages_indexed": int,
    "page_numbers": list[int],
    "last_page_partial": bool,
    "char_count": int,
    "text_available": bool,
    "coverage_complete": bool,
    "truncated": bool,
    "truncation_reasons": list["page_limit" | "character_limit"],
    "pages": [{"page": int, "text": str}],
}
```

默认目录为 `data/indexes/`，可以用 `PAPER_INDEX_DIR` 覆盖。只有格式版本、提取
上限、论文 ID、PDF SHA-256 和文件大小全部匹配才复用；否则自动重建。索引最多
200 页、200 万字符、32 MiB。写入使用同目录唯一临时文件和原子替换。调用结果
额外带 `index_status=built|cached|rebuilt`，但绝对索引路径和完整 pages 不会返回
给模型。

### 内部 `chunk_paper_text` 与 `chunk_paper_index`

这两个函数都不是独立的模型可见 Tool。前者保留给有界临时提取，后者消费完整
索引的逐页结构；两者最终产生相同 chunk 契约：

```python
{
    "paper_id": str,
    "title": str | None,
    "source_char_count": int,
    "source_truncated": bool,
    "source_truncation_reasons": list[str],
    "page_numbers": list[int],
    "chunk_size": int,
    "overlap": int,
    "chunk_count": int,
    "chunks": [
        {
            "chunk_id": str,
            "paper_id": str,
            "page": int,
            "char_start": int,
            "char_end": int,
            "text": str,
        }
    ],
}
```

默认窗口 1200 字符、重叠 200 字符；每个窗口都限制在单页内。`char_start` 和
`char_end` 是页内偏移。全部 chunks 只在确定性 Python 逻辑中存在，不写入
messages。临时文本 Chunking 会验证结构页码与提取 metadata 一致；索引 Chunking
会先验证完整索引契约。两条路径都不允许正文伪造页码。

### 内部 `rank_paper_chunks`

这一函数也不是独立的模型可见 Tool。它接收 chunking 结果、查询和 `top_k`，输出：

```python
{
    "found": bool,
    "paper_id": str,
    "title": str | None,
    "query": str,
    "query_terms": list[str],
    "matched_query_terms": list[str],
    "query_term_coverage": float,
    "minimum_query_term_coverage": float,
    "rejected_low_query_coverage": bool,
    "scoring_method": "tfidf_cosine" | "bm25",
    "count": int,
    "total_matches": int,
    "total_chunks": int,
    "top_k": int,
    "truncated": bool,
    "source_truncated": bool,
    "source_truncation_reasons": list[str],
    "matches": [
        {
            # 原 chunk 字段，另加：
            "score": float,
            "matched_terms": list[str],
        }
    ],
}
```

英文和数字按不区分大小写的 token 处理，连续中文拆为双字片段。TF-IDF 的词频
使用 token 在当前文本中的比例，IDF 为 `ln((N+1)/(df+1))+1`，最后计算查询和
chunk 向量的余弦相似度。BM25 使用 `k1=1.5`、`b=0.75`，作为同数据集实验选项；
两种方法的绝对分数不直接比较。

排序之外还计算“整篇索引中出现的不同查询词数 / 不同查询词总数”。正式默认要求
覆盖率至少 0.5；不足时保留门槛前的 `total_matches` 供诊断，但清空 `matches` 并
返回 `found=false`。这个门槛解决弱通用词假阳性，不改变 chunk 间的排序。同分时
仍按页码、页内起点和 chunk ID 排序。默认 top 3、最多 top 5。

### `retrieve_paper_chunks`

LLM 可见的输入：

```python
{
    "paper_id": "arxiv:1706.03762",
    "query": "encoder decoder architecture",
    "top_k": 3,  # 可省略，范围 1..5
}
```

Runtime 只接受当前 State 以前成功下载过的 `paper_id`，从对应 Tool Result 恢复
完整 `download_result`。query 去除首尾空白后最多 500 字符；模型不能提交
`local_path`。Tool 内部固定执行：加载或懒建立持久化全文索引、页内切分、默认
TF-IDF 排序和 50% 查询词覆盖门槛。BM25 与门槛参数只留在内部评测接口，不暴露给
模型，避免模型在一次调用中改变检索质量策略。输出沿用上面的排序结果，并补充：

```python
{
    # rank_paper_chunks 的字段，其中 matches 最多 5 个
    "page_count": int,
    "pages_scanned": int,
    "pages_read": int,
    "page_numbers": list[int],
    "last_page_partial": bool,
    "coverage_complete": bool,
    "index_status": "built" | "cached" | "rebuilt",
    "index_version": int,
    "warning": str,  # 仅索引器产生提示时存在
}
```

完整索引、绝对索引路径和未入选 chunks 不会越过 Tool 边界。`--debug` 会把每个
入选 chunk 进一步缩短为最多 400 字符的预览，但实际 top-k 文本仍会进入模型
Context。

## 6. State、Context、Memory 与 RAG

- **State**：当前命令行进程、当前会话里的 `user_query`、完整 messages 和累计
  step。它让后续轮次能引用先前候选和 Tool Result，退出 CLI 后即清空。
- **Context**：某一次模型调用实际收到的 system instructions 加当前 messages。
  检索 Tool 限制的是新增到 Context 的结果大小，不会自动删除旧 State。
- **长期 Memory**：跨任务保存并按需召回的信息；当前未实现。本地文献库是显式
  JSON 数据，PDF 和全文索引是磁盘缓存；三者都不会自动注入模型，因此也不等同
  于 Agent Memory。
- **RAG**：先从外部知识中检索相关内容，再让模型据此生成答案的整体模式。当前
  Tool 已有持久化全文索引、有界词法检索和离线质量基线，但没有向量库、语义
  检索或 Context 压缩，所以仍不称为完整 RAG。

离线评测的数据流与在线 Agent 路径分离：

```text
固定逐页 JSON → 数据契约校验 → 内存索引
→ 正式 chunk_paper_index → 正式 rank_paper_chunks
→ 逐问题页码判断 → Recall@K / Hit Rate@K / MRR / 无答案准确率
```

有答案问题中，Recall@K 是召回的期望页数占全部期望页数的比例，Hit Rate@K 表示
至少命中一页，MRR 使用第一个正确 chunk 的倒数排名；无答案问题仅在没有返回任何
匹配 chunk 时算正确。两类问题分开聚合，避免大量无答案样例扭曲召回指标。

方法选择复用同一份数据和同一套正式函数：

```text
原始 TF-IDF ┐
原始 BM25   ├→ 相同 10 问、相同 top_k → 比较四项指标 → 选择默认配置
带门槛 TF-IDF ┤
带门槛 BM25 ┘
```

选择优先级为 Recall@K、无答案准确率、Hit Rate@K、MRR；指标完全相同时保留改动
和依赖更少的 TF-IDF。当前 BM25 与 TF-IDF 指标持平，50% 查询词覆盖门槛把无答案
准确率从 0.500 提升到 1.000，因此默认选择 `tfidf_guarded`。

资源测量的数据流是：

```text
临时合成 PDF → 首次 retrieve_paper_chunks → 建立并写入全文索引
             → 再次 retrieve_paper_chunks → 复用并校验索引
             → 汇总耗时、索引字节数和 Tool Result JSON 字节数
```

首次和缓存计时都包含 PDF SHA-256、chunking 和默认 TF-IDF 排序，因此衡量的是用户
实际调用一次检索 Tool 的时间，而不是孤立的 JSON 读写速度。Context 指标只计算
本次 Tool Result；完整模型 Context 还包含 instructions、Schemas 和历史消息。

## 7. 搜索和保存的信息流

仅搜索时：

```text
用户请求 → LLM 调用 search_paper → Runtime 执行搜索
→ 来源返回最多 10 个候选 → Python 基于标题/摘要计算词法分数
→ 前 3 个候选写回 messages → LLM 比较候选 → final
```

当前保存流程：

```text
用户明确请求搜索并保存 → search_paper → 候选写回 messages
→ LLM 把 candidate_id 交给 save_paper
→ Runtime 从历史搜索结果解析可信 metadata
→ terminal.py 展示论文并读取 y/N
├─ 拒绝 → 结构化拒绝结果写回 messages
└─ 同意 → Runtime 调用存储函数写入 JSON
→ 保存结果写回 messages → LLM 返回 final
```

来源校验和写入确认都由 Runtime 确定性执行。instructions 仍用于引导模型判断
何时提出保存动作，但不能绕过确认直接写入。

遇到候选歧义时：

```text
LLM 返回候选和澄清问题 → terminal.py 输出本轮回答并继续读取输入
→ 用户回复“第 1 篇” → 消息追加到原 State
→ LLM 复用历史 search_paper 结果中的 candidate_id → 继续保存流程
```

用户否定全部候选并提供新线索时：

```text
用户：“这些都不相关，我要场景重建”
→ LLM 生成更具体的单个 query
→ 再次调用同一个 search_paper
→ 新结果写回 State → LLM 再判断
```

同一用户轮次里模型再次提交相同 query 时：

```text
LLM 再次调用 search_paper("TASA scene")
→ Runtime 折叠 query 中的空白并命中本轮临时集合
→ 不执行搜索 Tool，不访问网络
→ duplicate_search_query 写回 State
→ LLM 改写 query 或返回澄清问题
```

临时集合属于一次 `run_agent()` 调用，而不是长期 State。用户发送下一条消息后，
新的 `run_agent()` 会建立空集合，因此用户主动要求重新执行相同搜索仍然允许。

如果模型在同一用户轮次提交第三个不同 query：

```text
LLM 第三次调用 search_paper("another query")
→ Runtime 发现本轮已经尝试 2 个不同 query
→ 不执行搜索 Tool，不访问网络
→ search_limit_reached 写回 State
→ LLM 停止搜索并请求用户澄清
```

Runtime 只计数和拒绝，不生成第二个 query。是否根据 `no_lexical_match` 或
`no_candidates` 精炼一次，仍是 LLM 的决策。

读取 PDF 时：

```text
用户明确请求搜索、下载并总结
→ search_paper 返回可信 candidate_id
→ download_paper 返回可信 paper_id + local_path
→ LLM 只把 paper_id 交给 extract_paper_text
→ Runtime 从历史下载结果恢复 local_path
→ extract.py 校验缓存边界并提取有界文本
→ 页码、截断信息和文本写回 State
→ LLM 基于这段有限内容回答
```

这里仍保持一个 Tool Call 一步。`extract_paper_text` 适合概览或总结开头内容；
针对方法、数据集、指标或页码等具体问题，走下面的检索链路：

```text
用户针对已下载论文提出具体问题
→ LLM 只提交 paper_id + query + 可选 top_k
→ Runtime 从历史 download_paper 结果恢复可信 local_path
→ Tool 校验或建立全文索引 → 页内 chunking → TF-IDF 排序 + 查询词覆盖门槛
→ 只有 top-k 片段、页码和截断信息写回 State
→ 下一次 LLM 调用把这份小结果放入 Context 并生成答案
```

这里的“重新搜索”不是新 Tool，而是 Agent Loop 对同一个 Tool 的再次调用。

查看本地文献库时：

```text
用户询问已保存论文 → LLM 调用 list_library
→ Runtime 执行只读查询 → 有界结果写回 messages → LLM 返回 final
```

这条链路不调用网络、不请求写入确认，也不修改文献库。

下载 PDF 时：

```text
用户明确要求下载 → LLM 先搜索或查询文献库
→ LLM 只提交稳定 paper_id → Runtime 解析可信 paper metadata
→ download_paper 校验 URL / 类型 / 大小 → 原子写入缓存
→ 路径和大小写回 messages → LLM 返回 final
```

## 8. 错误模型

- 没有搜索结果：正常 Tool Result，`found=false`。
- 同一用户轮次重复相同搜索：Runtime 返回 `duplicate_search_query`，不访问网络。
- 同一用户轮次提交第三个不同搜索：Runtime 返回 `search_limit_reached`，不访问
  网络。
- 未知工具、参数不匹配：Runtime 返回结构化错误。
- 保存 ID 不在当前任务搜索历史中：Runtime 返回 `unknown_candidate`，不写文件。
- 保存调用没有确认器：Runtime 返回 `approval_required`，不写文件。
- 确认器执行失败：Runtime 返回 `approval_error`，不写文件。
- 用户拒绝保存：正常 Tool Result，`saved=false, reason=user_declined`。
- 文献库不存在或为空：正常 Tool Result，`count=0, total=0`。
- 文献库 JSON 损坏：Runtime 返回 `tool_execution_error`。
- 下载 ID 不在可信历史中：Runtime 返回 `unknown_paper`，不发起网络请求。
- PDF URL 不安全、响应非 PDF、超时或超限：Runtime 返回
  `tool_execution_error`，不留下不完整缓存文件。
- 提取 ID 没有对应的可信下载结果：Runtime 返回 `unknown_download`。
- 检索 ID 没有对应的可信下载结果：Runtime 返回 `unknown_download`，不读取文件。
- 检索 query、`top_k` 越界或模型提交路径：Runtime 返回结构化 Tool 错误。
- 检索没有共同词：正常 Tool Result，`found=false`，不返回任意无关段落。
- 检索只命中不足 50% 的不同查询词：正常 Tool Result，
  `rejected_low_query_coverage=true`，终端和 Agent 将其描述为证据不足。
- 索引缺失：正常建立；PDF 变化或索引损坏/过期：自动原子重建。
- 索引超过 200 页、200 万字符或 32 MiB：有界停止或拒绝写入，并披露截断/错误。
- PDF 缓存路径越界、PDF 加密或损坏：Runtime 返回 `tool_execution_error`。
- PDF 没有可提取文本：正常 Tool Result，`text_available=false`，并提示尚无 OCR。
- 网络、解析或文件异常：Tool 抛出异常，Runtime 转成
  `tool_execution_error`。
- LLM 响应格式无效：`llm.py` 抛出 `RuntimeError`。
- Runtime Event 回调异常：被隔离并忽略，不改变 Agent 结果。
- LLM 或 Tool 执行期间按 Ctrl+C：Runtime 补写必要的 Assistant/Tool 消息，并以
  `cancelled` 结束当前轮；下一条用户输入仍可继续。
- 交互终端发生可恢复的 Runtime 错误：显示错误并继续等待；管道输入发生同类错误：
  以非零状态码退出。
- 达到单轮最大步数：Runtime 结束本轮；活动 Plan 保持 `running` 并返回可继续提示，
  非计划请求返回本轮停止信息。任务级预算仍独立限制累计 LLM 和 Tool 次数。

## 9. 数据和副作用边界

- arXiv 和 Crossref 请求是外部只读网络操作。
- DeepSeek 和 OpenRouter 调用会把当前 messages 发给所选 Provider。
- `save_paper` 会修改个人文献库，因此需要终端确认。
- `download_paper` 会写入可重新生成的本地 PDF 缓存。
- `extract_paper_text` 只读取可信 PDF 缓存，但其有界文本结果会进入 messages，
  并可能被发送给所选模型 Provider。
- `retrieve_paper_chunks` 同样只读取可信缓存；中间全文和全部 chunks 留在 Tool
  内部，只有 top-k 片段会进入 messages 并可能发送给模型 Provider。
- `.env` 和 `data/library.json` 都被 Git 忽略。
- `data/pdfs/` 被 Git 忽略。
- `data/indexes/` 被 Git 忽略；它是可重建缓存，不是长期 Agent Memory。
- Prompt Toolkit 历史只保存在当前进程内，不会默认把用户输入写入磁盘。
- 默认离线测试使用 Mock/Fake，不访问网络，也不写真实文献库。

## 10. V2 验收与 V3 入口

PDF 下载、缓存、有界文本提取、最小段落检索、候选本地重排、相关性状态和搜索
次数边界均已接入 Agent Loop。V2 已使用真实 arXiv、DeepSeek 和 PDF 完成搜索、
无结果澄清、保存、查询、下载、提取和检索的端到端验收。V2.0.1 进一步修正了
模型可见页数元数据，并通过 UI 无关事件把同一个 Runtime 接入增强终端和 Plain
终端；Agent 的决策和 Tool 执行边界没有改变。

V2.1 的全文覆盖、质量评测、资源测量和算法选择已经完成。固定 10 问默认 Top-3 为
Recall@3=0.875、Hit Rate@3=0.875、MRR=0.875、无答案准确率=1.000。BM25 与
TF-IDF 持平；50% 查询词覆盖门槛消除了固定集中的通用词无答案误命中，剩余失败是
中文查询无法直接召回英文正文。V3 不直接引入高级 Agent 框架，而是在当前结构上
增加可观察、可测试的计划控制层。

## 11. V3 计划架构（已实现）

V3 的目标信息流是：

```text
复合用户目标
  ↓
planner.py：生成或修订高层 Plan
  ↓
planning.py：校验契约与合法状态转换
  ↓
runtime.py：选择当前 step，调用现有 Executor/Tool Loop
  ↓
可信 Tool Result → evidence reference → Runtime 更新 step
  ├─ 成功：进入下一 step
  ├─ 前提失效：有限 Replan
  ├─ 需要用户选择：blocked
  └─ 完成：生成带证据最终回答
  ↓
checkpoint.py：每个重要状态转换后原子保存
```

### 11.1 计划模块职责

- `planning.py`：已定义纯数据 Plan 契约、常量、验证器和状态转换；不调用模型、不
  执行 Tool、不写文件，而且每次转换返回新的 Plan 值。
- `planner.py`：已定义内部 `submit_plan` Schema、首次规划 instructions 和响应
  归一化；它不直接修改 State。
- `executor.py`：已根据可信 Plan 组装任务目标、已完成步骤和当前 step 的有界控制
  Context，并复用现有 Tool Schemas 请求一次 `tool_call|final`；需要用户输入时还可
  返回内部 blocked 动作。只有 Runtime 授权时才临时加入动态收紧后的
  `submit_plan` Schema；Executor 不修改 State。
- `checkpoint.py`：已校验并原子保存/读取活动任务 Checkpoint；不调用 LLM 或执行
  Tool，但会复用稳定 ID 与 PDF 缓存校验规则。
- `state.py`：已增加可为空的任务 ID 和 Plan；预算计数器与证据引用保存在 Plan 中。
- `runtime.py`：已接入第一次 Planner 决策、可信 Plan 创建、串行 Executor、预算计数、
  evidence reference、Replan 条件、blocked 恢复、重复副作用保护和 Checkpoint 时机。
- `main.py` / `terminal.py`：已组合 `--resume`、有界恢复提示、`/plan`、`/cancel`
  和计划事件显示；显示层不修改 Plan 业务状态。

Planner 使用仅供结构化输出的 `submit_plan` Schema。它不是外部 Tool，不进入
`tools/` 包或 `TOOL_REGISTRY`，也不会被“执行”；`planner.py` 只读取归一化
`tool_call.arguments`，随后由 `planning.py` 校验。这样可以复用 DeepSeek 与
OpenRouter 已有 Tool Calling 协议，同时保持 Provider 解析仍只存在于 `llm.py`。
`planner.py` 在把 `submit_plan` 内部化为 `plan` 动作时同时保留已归一化的 reasoning
元数据；Runtime 生成 Plan 预览 Assistant 消息时再把它写入 State，避免下一次
DeepSeek 请求丢失创建计划那一轮的 `reasoning_content`。

新任务的第一次 LLM 决策可以在 `submit_plan`、现有外部 Tool Call 和 `final` 之间
选择：复合任务提交高层步骤，简单任务保持 V2 路径。这避免额外调用一次分类模型，
也不使用关键词硬编码路由。LLM 只提交步骤描述；Runtime 使用原始用户目标生成
`task_id`、状态、revision、step ID、计数器和 evidence reference。活动 Plan 已经
存在时，Runtime 只在满足 Replan 条件后才再次允许 `submit_plan`。

当前 Planner 的归一化输出有三种：

```python
{"type": "plan", "tool_call_id": str, "step_descriptions": list[str], ...}
{"type": "tool_call", ...}  # 简单任务沿用 V2
{"type": "final", ...}      # 信息不足或无需 Tool
```

其中 `plan` 只携带经过 `planning.normalize_step_descriptions` 校验的步骤描述。
Provider 返回的 `status`、`task_id`、预算、证据等额外参数会被拒绝；真正的 Plan
仍要等 Runtime 使用原始用户目标和可信 task ID 调用 `create_plan` 后才产生。

这条创建链路现已接入 Runtime：

```text
本轮最新 user message
  + Planner 的 step_descriptions
  + Runtime 生成的 UUID task_id
  → create_plan
  → record_plan_usage(llm_steps=1)
  → 同时写入 State.task_id / State.plan
```

Runtime 在完整 Plan 创建并通过校验前不会部分更新 State。内部 `submit_plan` 调用
不会作为 Assistant Tool Call 写进 messages，也不会触发 Tool Registry。创建后返回
有界 Plan 预览；下一条用户消息进入下面的最小执行链路：

```text
start_next_step
  → executor.py 注入 current step Context
  → tool_call → Runtime 执行并写回 Tool Result
  → Runtime 分配 tool-result-NNN evidence reference
  → 继续当前 step
  → step-level final → complete_current_step
  → 有 pending step：继续串行执行；全部完成：任务 completed
```

Tool Call ID 来自 Provider，不能直接作为可信 evidence reference。Runtime 使用任务内
单调递增的 `tool-result-NNN`，并把同一引用同时写入 Tool 消息和当前 step，因此后续
可以反向定位真实 Tool Result。LLM 与 Tool 预算都在调用前检查，达到上限后任务进入
`failed`，不会先产生第 33 次 LLM 调用或第 25 次 Tool 执行。

当前 Replan 链路为：

```text
Tool Result
  → Runtime 分类最新未处理观察
  → 无问题：Executor 不能看到 submit_plan
  → 失败/歧义/证据不足，且仍有预算和 Plan 容量
      → Executor 临时看到 submit_plan
      → 模型选择普通 Tool/final：继续当前步骤
      → 模型选择 submit_plan
          → fail_current_step
          → revise_plan：保留 completed/failed 历史和 evidence
          → 替换旧 pending 后缀并增加 revision/replans
          → 启动新的 pending step
```

硬失败包括结构化 Tool 错误、搜索无候选或无词法匹配、检索无有效片段、PDF 无可
提取文本和空文献库。搜索返回多个候选但没有精确标题匹配时属于软歧义：Runtime
允许 Replan，但如果当前 step 的目标只是展示候选，模型也可以正常结束该步骤。
`submit_plan` 的 `maxItems` 会根据保留历史动态缩小；Plan 总步数仍不能超过 8，
任务最多 Replan 2 次。

缺少候选选择等关键输入时，Executor 调用内部 `request_clarification`。它不会进入
Tool Registry；Runtime 把当前 step 和任务都标为 blocked，并把模型生成的一个有界
问题返回用户。只有 State 最后一条消息是新的 user 消息时，blocked step 才恢复为
running，attempts 增加后继续执行。硬失败后的普通 final 也不能误完成步骤，而会由
Runtime 收敛到 blocked。

为了防止新计划重复副作用，Runtime 在执行 `save_paper` 或 `download_paper` 前按
Tool 名和规范参数查找已成功的可信 Tool Result。命中后复制旧结果并增加
`runtime_reused` 标记，不再次调用 Tool，也不再次请求保存确认。这个保护只复用成功
或已经存在的结果；下载结果还必须重新通过本地缓存校验。用户拒绝、审批异常、
失效缓存和 Tool 错误不会被当作可复用成功结果。

### 11.2 计划状态

```python
{
    "version": 1,
    "task_id": str,
    "goal": str,
    "status": "running | blocked | completed | failed | cancelled",
    "revision": int,
    "current_step_id": str | None,
    "steps": [
        {
            "id": str,
            "description": str,
            "status": "pending | running | completed | blocked | failed",
            "attempts": int,
            "evidence_refs": list[str],
        }
    ],
    "budget": {
        "llm_steps": int,
        "tool_calls": int,
        "replans": int,
    },
}
```

LLM 可以提出步骤内容，但不能写任务状态、attempts、预算和可信 evidence reference。
这些字段由 Runtime 根据实际事件更新。证据引用指向 State 中已有的 Tool Result，
Checkpoint 不复制 PDF、全文索引或 Tool 内部数据。

步骤的正常转换是 `pending → running → completed|blocked|failed`。`blocked` 表示缺少
继续执行所需的用户信息，因此只允许在用户补充信息后恢复为 `running`；恢复会增加
一次 attempts。Replan 后 failed 步骤作为不可变历史保留；其数量与已使用 Replan
次数必须一致，最多只允许一个尚未解决的 failed 步骤。新后缀完成后，任务可以在
保留 failed 历史的同时进入 completed。终态步骤不能由模型自行回退。

### 11.3 Checkpoint 与 Memory 的边界

```text
Checkpoint = 恢复同一个未完成任务的精确运行状态
Long-term Memory = 在新任务中选择性召回过去信息
```

V3 只实现前者。默认只存在一个 `data/checkpoints/active.json`，并设格式版本和
4 MiB 上限。`--resume` 恢复前重新验证结构、可信 ID 与本地缓存；完成、不可恢复
失败后清除活动 Checkpoint；显式 `/cancel` 先调用同一清理函数，成功后才将 Plan
转为 cancelled。启动时存在活动文件就提示恢复，新任务不能静默覆盖它。任何
Checkpoint 内容都不会自动注入新的任务 Context。

Checkpoint 记录完整 State，包括 Plan 计数器、Tool Result、evidence reference 和
Provider 继续请求所需的不透明 reasoning 字段；不记录 API Key、PDF 二进制、完整
索引或 Tool 未返回的数据。写入只发生在消息协议平衡、Plan 已通过校验的边界。恢复
后的 Executor 从保存的 `current_step_id` 继续；已完成步骤不会重新执行，成功的保存
或下载仍由 Runtime 幂等保护复用。

### 11.4 实现顺序（全部完成）

```text
Plan 纯数据契约与状态机
→ Planner/Fake Planner
→ Runtime 单步执行与预算
→ 有限 Replanning / blocked
→ Checkpoint 原子保存与恢复
→ Terminal 命令和 Runtime Events
→ 离线评测
→ arXiv + DeepSeek + PDF 真实验收
```

开发和复盘方法见 `docs/agent-development-process.md`。

### 11.5 评测与真实验收

确定性 Planning 评测固定两篇论文数据和 6 个模型动作，但实际调用 Runtime、Plan
状态转换、Tool Registry 和原子 Checkpoint。当前快照为任务完成、0 次 Replan、
6 次 LLM 决策、2 次 Tool 执行、13 次 Checkpoint 保存、最大文件 4,572 B。它用于
发现控制流和持久化回归，不用于衡量模型答案质量。

真实系统测试固定四步高层 Plan，以减少 Planner 措辞和步数的随机性；后续步骤由
DeepSeek 通过正式 Executor 决策，并真实搜索 arXiv:1706.03762 与 arXiv:1810.04805、
下载两份 PDF、建立全文索引、取得两边带页码的片段并生成比较。测试还校验两篇的
可信 ID、下载结果、页码与文本证据以及最终回答的双论文页码引用。2026-09-13 的
全量联网结果为 202 项通过、0 跳过。
