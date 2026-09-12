# AgentReader Demo

一个用于学习 Agent 基础机制的文献搜索与管理 Demo。项目当前不依赖
LangGraph、Multi-Agent、MCP 或完整 RAG，而是显式保留下面的数据流：

```text
LLM 决策 -> Runtime 执行 Tool -> Tool Result 写回 State -> LLM 再决策
```

## 当前状态

- V1 最小 Agent Loop 已完成，并有离线测试覆盖。
- V2 已接入真实文献搜索：优先 arXiv，Crossref 作为备选。
- V2 已实现本地 JSON 文献保存原型，并能校验保存目标来自本次搜索结果。
- 本地写入前会在终端展示确切论文，并要求用户进行 `y/N` 确认。
- 命令行支持多轮对话，可以继续回答候选澄清问题。
- 用户否定旧候选并补充线索后，Agent 可以再次调用同一个搜索 Tool。
- Tool 已整理为 `tools/` 包，Schema、搜索和文献库按职责分开。
- 已支持通过只读 `list_library` Tool 查看本地保存的论文。
- 已支持从可信搜索或文献库结果下载 PDF，并进行有界本地缓存。
- 已支持从可信缓存 PDF 中提取有页数和字符数上限的文本。
- 已实现页码感知的 chunking，以及可比较的 TF-IDF/BM25 chunk 排序。
- 已将 `retrieve_paper_chunks` 接入 Agent Loop，只把有界 top-k 相关片段交给模型。
- V2.1 已实现懒加载的本地全文 JSON 索引；PDF 未变化时重复检索会复用索引。
- Runtime 已阻止同一用户轮次内重复执行相同的搜索 query。
- 搜索源最多取 10 个候选，本地词法重排后仍只向模型返回前 3 个。
- 搜索结果会显式标记相关性状态；每个用户轮次最多执行 2 次不同搜索。
- 已提供轻量交互式终端：实时显示 LLM/Tool 状态，支持输入历史、Slash Commands
  和无 ANSI 的 `--plain` 模式。
- V2 已通过真实 arXiv、DeepSeek 和 PDF 的端到端验收，V2.0.1 的提取正确性与
  交互式终端也已完成；V2.1 的全文覆盖、检索评测、资源测量和排序方法选择均已
  完成。下一版本开始前先讨论并定义 V3 的单一目标。

详细范围和信息流见：

- [需求说明](docs/requirements.md)
- [架构说明](docs/architecture.md)
- [Agent 开发流程与项目复盘](docs/agent-development-process.md)
- [开发路线](TODO.md)

## 创建开发环境

在项目根目录创建环境：

```bash
conda env create -f environment.yml
conda activate agent-reader-demo
```

激活成功后，终端提示符通常会出现 `(agent-reader-demo)`。使用下面的命令确认环境：

```bash
conda info --envs
which python
python -V
python -c "import openai; print(openai.__version__)"
```

以后修改了 `environment.yml`，可以同步环境：

```bash
conda env update -f environment.yml --prune
```

## 配置模型

如果项目中还没有 `.env`，从示例配置创建本地配置文件：

```bash
cp .env.example .env
```

`.env` 已被 Git 忽略，不应提交 API Key。默认的 `LLM_PROVIDER=fake`
不会请求模型 API，适合离线学习和测试。真实调用支持：

- `LLM_PROVIDER=deepseek`：配置 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_MODEL`
- `LLM_PROVIDER=openrouter`：配置 `OPENROUTER_API_KEY` 和 `OPENROUTER_MODEL`

`LLM_MODEL` 和 `LLM_BASE_URL` 可以临时覆盖当前 Provider 的默认配置。

## 运行

```bash
python main.py
```

程序会在每次 Agent 回答后继续等待输入，并把新消息追加到同一个 State。
例如 Agent 发现多个 TASA 候选后，可以直接回复 `第 1 篇`。交互终端使用
`You ›` 和 `Agent ›` 区分双方消息，真实 Tool 执行状态会在运行过程中显示；方向键
可以访问当前进程内的输入历史。空输入只会重新提示，不再结束会话。

终端命令：

```text
/help           显示帮助
/debug          切换 Debug 模式
/debug on|off   开启或关闭 Debug 模式
/clear          清空当前会话 State
/exit           退出 AgentReader
```

普通文本 `exit`、`quit`、`退出` 和 Ctrl+D 也可以退出。输入阶段按 Ctrl+C 会取消
当前输入，不会退出整个会话。

需要查看 LLM 决策、Tool Call、Tool Result 和步数时：

```bash
python main.py --debug
```

Debug 模式现在按实际发生顺序显示 LLM 和 Tool 事件、耗时、参数及有界结果，不再等
整轮结束后一次性打印。需要用于管道、CI 或不支持增强终端的环境时：

```bash
python main.py --plain
python main.py --plain --debug
```

非 TTY 输入输出会自动使用 Plain 模式。增强模式的命令历史只保存在当前进程内，
不会默认把用户查询写到磁盘。非交互管道发生运行错误时返回非零退出码。

## 测试

请在项目根目录使用 `python -m unittest ...` 启动测试，不要直接执行
`python tests/test_xxx.py`。使用 `-m` 时，Python 会从项目根目录解析
`tools/`、`runtime.py` 等模块。

运行默认离线测试（不会请求模型 API）：

```bash
python -m unittest discover -s tests -v
```

运行固定检索质量评测（同样不会访问网络或模型）：

```bash
python evaluate_retrieval.py
python evaluate_retrieval.py --json
```

显式运行真实搜索测试（会先请求 arXiv，失败或无结果时请求 Crossref）：

```bash
RUN_LIVE_ARXIV_TESTS=1 python -m unittest \
  tests.test_arxiv_integration -v
```

显式运行 DeepSeek 在线测试（会产生一次完整 Agent Loop 的 API 请求）：

```bash
RUN_LIVE_LLM_TESTS=1 python -m unittest \
  tests.test_deepseek_integration -v
```

在线测试会强制选择 DeepSeek，读取 `.env` 中的
`DEEPSEEK_API_KEY` 和 `DEEPSEEK_MODEL`，不会受
`LLM_PROVIDER` 当前值影响。

本项目的完整验收会同时启用全部联网测试：

```bash
RUN_LIVE_ARXIV_TESTS=1 RUN_LIVE_LLM_TESTS=1 \
  python -m unittest discover -s tests -v
```

环境变量开关仍然保留，使普通离线开发或没有 API Key 的 CI 不会意外访问网络和
产生模型费用；Codex 后续进行完整验收时使用上面的联网命令。

正常运行 `python main.py` 时，`search_paper` 会优先查询 arXiv；
只有 arXiv 超时、报错或无结果时，才会查询 Crossref。可选设置
`CROSSREF_MAILTO`，让 Crossref 请求进入官方推荐的 polite pool。
arXiv 返回 HTTP 429 时会等待后重试一次；如果 query 中包含 `1706.03762`
这样的 arXiv ID，则改用 API 的 `id_list` 定向查询。这样模型在补充明确 ID 后
能够取得权威 metadata 和 PDF URL，而不是继续依赖不含 PDF 的 Crossref 记录。
重试后仍需回退时，结果中的 `arxiv_error` 会保留简短原因供 Debug Trace 查看。
普通关键词搜索会从 arXiv 或 Crossref 取得最多 10 篇候选，再根据 query 与标题、
摘要的词项重叠做确定性本地重排，最后仍只返回前 3 篇给模型。标题每命中一个
query 词项计 4 分，摘要命中计 1 分；query 的完整词项序列出现在标题时，再增加
`query 词项数 × 2`；如果规范化后的标题与完整 query 完全相等，再增加
`query 词项数 × 2`，防止带前后缀的仿标题压过精确标题。同分保留搜索源顺序。
明确 arXiv ID 的定向查询仍只取 1 篇。

每篇结果包含 `local_relevance`，记录总分、标题/摘要命中词、标题短语命中、精确
标题命中状态和原始来源排名；顶层 `candidate_count` 和 `truncated` 说明候选池
是否被裁剪。这个分数只是可解释的词法证据，不等同于语义正确。Agent 仍需比较
标题、摘要和作者；如果信息不足，应展示候选，而不是假定第一篇一定正确。

`--debug` 只把搜索结果中的每篇摘要显示为最多 400 字符的
`abstract_preview`，避免长摘要把最终回答和下一轮输入提示挤出终端视野。完整摘要
仍保留在 State 中并交给模型判断，缩短只影响终端调试显示。

顶层 `relevance_assessment` 将结果归为四类：明确 arXiv ID 命中为
`identifier_match`；最高词法分数大于 0 为 `lexical_match`；来源返回了候选但
没有词项命中为 `no_lexical_match`；没有候选为 `no_candidates`。后两种情况下，
Agent 不应把候选说成相关；如果现有线索足够，可以生成一个不同的精炼 query
再搜一次，否则应请求用户补充信息。

如果用户认为候选不相关并补充新线索，Agent 会生成一个更具体的 query 再次
调用 `search_paper`。每个 LLM step 仍只执行一个 Tool Call；Provider 如果
同时建议多个调用，系统会先执行第一个并观察结果，再进行下一次决策。

`run_agent()` 会记录当前用户轮次里已经尝试过的有效搜索 query。比较前采用与
搜索 Tool 相同的空白规范化，因此 `TASA   scene` 和 `TASA scene` 会被视为重复；
第二次调用不会访问 arXiv 或 Crossref，而是产生
`duplicate_search_query` 结构化错误供模型观察。新的用户消息会启动新的
`run_agent()` 调用并清空这份临时集合，所以用户主动要求重新搜索相同 query
仍然允许。不同的精炼 query 不受重复检查影响，但每个用户轮次最多尝试 2 个
不同 query；第三个返回 `search_limit_reached`，不会访问网络。这里的“重搜”仍是
LLM 根据结果作出的下一步 Tool Call，不是 Runtime 自己生成搜索词。

## Tool 目录

Tool 相关代码按职责组织：

```text
tools/
├── __init__.py  # 稳定的公共导入入口
├── schemas.py   # 模型可见的 Tool Schema
├── search.py    # arXiv / Crossref 搜索、解析与候选重排
├── library.py   # 稳定 ID、本地 JSON 持久化与只读查询
├── download.py  # 受限 PDF 下载与缓存
├── extract.py   # 可信缓存 PDF 的有界文本提取
├── index.py     # 可失效的持久化全文 JSON 索引
└── retrieval.py # 索引页切分与后续最小检索
```

Agent 和 Runtime 仍使用 `from tools import ...`，不需要了解包内文件。以后增加
Tool 时，应先判断它属于模型契约、外部搜索还是本地文献库，避免再次堆回一个
大文件。

## 查看文献库

运行程序后可以直接输入：

```text
请列出我的文献库
```

Agent 会调用只读的 `list_library`，默认按最近保存顺序返回最多 20 篇论文。
也可以请求较小数量，例如“列出最近 5 篇”；单次最多返回 50 篇。结果包含
`total` 和 `truncated`，但不包含摘要，避免把整个文献库无界地放入 Context。
文献库不存在或为空时会返回正常空结果，不会为了查询而创建文件。

## 下载 PDF

可以搜索并下载：

```text
请搜索并下载 REST3D 的 PDF
```

也可以先列出文献库，再要求下载其中一篇。模型只能向 `download_paper` 提交
`search_paper` 的 `candidate_id` 或 `list_library` 的 `id`；Runtime 会从当前
State 解析对应的可信 `pdf_url`，模型不能直接传入 URL。

下载只允许 HTTPS，超时为 20 秒，单个 PDF 最大 20 MiB，并校验响应类型和
`%PDF-` 文件头。文件默认缓存到 `data/pdfs/`，使用临时文件写完后再原子替换；
有效缓存会直接复用。Tool Result 只进入本地路径和文件大小，不包含 PDF 二进制。

如需修改缓存位置：

```bash
PAPER_CACHE_DIR=/path/to/pdfs python main.py --debug
```

## 读取 PDF 文本

可以把搜索、下载和阅读放在同一个明确请求中：

```text
请搜索、下载并总结 Attention Is All You Need 的 PDF
```

Agent 会按顺序调用 `search_paper`、`download_paper` 和
`extract_paper_text`，每一步的 Tool Result 都先写回 State，再由模型决定下一步。
提取 Tool 的模型可见输入仍只有可信 `paper_id` 和可选上限；Runtime 从当前会话
之前的 `download_paper` 结果恢复真实路径，模型不能提交任意本地路径。

默认只解析开头 5 页并最多返回 12000 个字符；允许范围为 1–10 页、
500–20000 个字符。结果通过 `pages_scanned` 表示解析器检查过的页数，通过
`pages_read` 和 `page_numbers` 表示返回文本实际覆盖的页数，并用
`last_page_partial` 标记最后一页是否只返回了一部分；同时保留实际字符数和截断
原因。
空文本会返回结构化提示；加密或损坏 PDF 会返回结构化 Tool 错误。当前版本没有
OCR，因此扫描版 PDF 可能无法提取文本。针对具体问题，应使用下面的最小检索，
而不是把开头的连续文本全部交给模型。
使用 `--debug` 时只展示最多 400 个字符的正文预览，避免调试输出刷满终端。

## 全文索引与最小段落检索

`tools/retrieval.py` 已能把 `extract_paper_text` 的有界结果按页切分。默认每个
chunk 最多 1200 个字符，相邻 chunk 重叠 200 个字符；chunk 不跨页，并保留
`paper_id`、页码、页内字符起止位置和稳定 `chunk_id`。输入已经截断时，结果也会
保留 `source_truncated` 和原始截断原因。

`rank_paper_chunks` 支持 TF-IDF 余弦相似度和 BM25 两种确定性排序方法。英文按
不区分大小写的单词切分，连续中文使用双字片段；默认只返回 3 个匹配，最多返回
5 个。同分时按页码、页内位置和 `chunk_id` 排序，因此结果可重复。

正式 Tool 仍使用 TF-IDF，并增加独立的最低证据门槛：查询中至少 50% 的不同词项
必须在整篇索引中出现，才返回排名片段。门槛不足时返回 `found=false`，同时用
`query_term_coverage`、`matched_query_terms` 和
`rejected_low_query_coverage=true` 说明原因；`total_matches` 仍保留门槛前的弱
匹配数量，便于 Debug。这避免只因 `model` 等一个通用词碰巧出现就给出无依据答案。

模型可见的 `retrieve_paper_chunks` 会在第一次查询时自动建立本地全文索引：

```text
可信下载结果 → 校验 PDF SHA-256 与索引版本
→ 缺失/过期/损坏时建立索引 → 页内 chunking
→ TF-IDF 排序 + 查询词覆盖门槛 → 只返回 top-k 匹配片段
```

索引默认保存在 `data/indexes/`，可以通过 `PAPER_INDEX_DIR` 修改位置。索引最多
检查 200 页、保存 200 万字符，JSON 文件上限为 32 MiB；普通研究论文通常可以
完整覆盖，异常长文档仍会通过 `coverage_complete=false` 和
`source_truncation_reasons` 明确披露截断。索引包含格式版本和 PDF SHA-256；PDF
变化、索引版本变化或 JSON 损坏时会自动原子重建。索引正文和全部 chunks 不进入
State，模型只看到 top-k 片段、页码、覆盖状态和 `built/cached/rebuilt` 状态。

模型只能提交之前 `download_paper` 返回的 `paper_id`、最长 500 字符的检索 query
和可选 `top_k`；Runtime 恢复可信本地路径，模型不能传入路径。默认返回 3 段，
最多 5 段，全部 chunks 和连续提取文本都只存在于 Tool 内部，不写入 State。
没有匹配词或查询词证据不足时返回正常的 `found=false`，不会拿无关首段凑答案。

可以在一个请求中测试完整链路：

```text
请搜索、下载并回答 Attention Is All You Need 中 encoder-decoder architecture 在哪几页？
```

也可以先下载，再在下一轮针对同一篇论文提问。对于英文论文，query 应包含可能
出现在原文中的英文术语。当前检索可以覆盖索引范围内的完整 PDF，但仍是词法
匹配，不是全文语义检索；中文问题和英文原文之间不会自动建立跨语言语义对应。

这里的 State 是当前 CLI 进程、当前会话保存的完整消息历史，退出程序后就会清空；
Context 是每次调用模型时实际发送的 instructions、历史消息和本次 Tool Result。
磁盘上的文献库、PDF 和全文索引可以跨会话复用，但它们不会自动注入 State，因此
不等于 Agent Memory。检索 Tool 通过只产生少量 top-k 结果来限制新增 Context，
本项目还没有跨会话长期 Memory、向量索引或完整 RAG。`--debug` 对每个匹配片段
也只显示最多 400 字符预览。

## 离线检索质量评测

`evals/retrieval_cases.json` 保存一份版本化、确定性的教学评测集：6 页英文正文和
10 个问题，覆盖英文词法查询、多页证据、中文查询英文正文、零词项重叠和通用词
误命中。`evaluate_retrieval.py` 直接复用正式的 `chunk_paper_index` 与
`rank_paper_chunks`，但不经过 Agent、LLM、网络或磁盘索引缓存。

正式默认的 TF-IDF + 50% 查询词覆盖门槛，Top-3 结果为：

- Recall@3：0.875
- Hit Rate@3：0.875
- MRR：0.875
- 无答案准确率：1.000

7 个英文词法问题的 Recall@3、Hit Rate@3 和 MRR 均为 1.0。通用词 `model` 的
无答案误命中已被覆盖率门槛拦截；剩余失败是中文“位置编码”无法直接匹配英文
原文。测试会锁定这份确定性结果作为回归快照，任何有意的算法变化都要同步审查和
更新；它不是生产质量达标证明。

同一评测集的对比结果如下：

| 配置 | Recall@3 | Hit Rate@3 | MRR | 无答案准确率 |
|---|---:|---:|---:|---:|
| 原始 TF-IDF | 0.875 | 0.875 | 0.875 | 0.500 |
| 原始 BM25 | 0.875 | 0.875 | 0.875 | 0.500 |
| TF-IDF + 门槛 | 0.875 | 0.875 | 0.875 | 1.000 |
| BM25 + 门槛 | 0.875 | 0.875 | 0.875 | 1.000 |

BM25 没有带来指标提升，所以生产默认保持改动更小的 TF-IDF + 门槛。跨语言失败
需要查询翻译或语义匹配，而不是只更换词法打分公式；当前 Agent 会为英文论文生成
英文检索词，但 V2.1 不增加 embedding/hybrid 依赖。运行方式：

```bash
python evaluate_retrieval.py
python evaluate_retrieval.py --compare
python evaluate_retrieval.py --compare --json
python evaluate_retrieval.py --method bm25
```

## 检索资源测量

运行确定性的合成 PDF 基准：

```bash
python measure_retrieval_resources.py
python measure_retrieval_resources.py --json
```

默认在隔离临时目录中生成 12 页、每页约 3000 字符的 PDF，重复 5 次。首次检索
包含 PDF 哈希、全文索引建立与写盘、chunking 和 TF-IDF 排序；缓存检索包含哈希、
索引读取与校验、chunking 和排序。PDF 生成本身不计入时间，也不会写入个人缓存。

2026-09-12 在当前开发机上的参考结果：首次检索中位数约 8.9 ms，缓存检索中位数
约 1.4 ms；36,000 字符形成 37,196 B 索引和 36 个 chunks，top-3 Tool Result 为
5,131 B，即索引大小的 13.79%。该载荷是与 `llm.py` 相同 JSON 序列化方式得到的
“本次检索新增 Context”，不包含 system instructions、Tool Schemas、旧消息和
模型 tokenizer 开销。时间是机器相关观测值，不作为跨机器测试阈值。

PDF 提取功能依赖 `pypdf`。已有环境如尚未同步，可运行：

```bash
conda env update -f environment.yml --prune
```

## 保存文献

在请求中明确说明需要保存，例如：

```text
请搜索并保存 REST3D
```

Agent 会先调用 `search_paper`。每篇候选都有稳定的 `candidate_id`；Agent
选中候选后只把这个 ID 交给 `save_paper`，Runtime 再从当前任务的搜索历史中
取回真实 metadata。随后终端会显示标题、作者、来源、链接和候选 ID，只有输入
`y`、`yes`、`是` 或 `确认` 才会继续写入；直接回车和其他输入都视为拒绝。
不存在的 ID 或模型复制的整份论文对象都会被拒绝。
文献元数据默认保存在 `data/library.json`；重复的 arXiv ID 或 DOI
不会再次写入。该文件包含个人文献库数据，已被 Git 忽略。

至少成功保存一篇论文后，可以格式化查看文献库：

```bash
python -m json.tool data/library.json
```

当前保存功能仍是学习原型：Agent instructions 判断何时提出保存调用，Runtime
则强制执行两个确定性边界——只能保存本次会话搜索到的候选，而且没有显式确认
就不能写入。用户拒绝时，结构化结果会写回 State，由 LLM 正常说明已经取消。

如需修改保存位置，可设置：

```bash
PAPER_LIBRARY_PATH=/path/to/library.json python main.py --debug
```

完成后可退出虚拟环境：

```bash
conda deactivate
```
