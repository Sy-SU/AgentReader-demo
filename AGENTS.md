# Project instructions

## Required context

开始修改代码前，请阅读：

- README.md
- TODO.md
- docs/requirements.md
- docs/architecture.md
- docs/agent-development-process.md

## Development rules

- 当前目标是把最小 Agent 工作机制逐步演进为可复用的真实项目经验。
- V1 不使用 LangGraph、多 Agent、MCP 或完整 RAG。
- 每次只实现一个小步骤，并解释信息流和设计原因。
- 修改完成后运行相关测试。
- 完整验收时显式启用 arXiv 和 DeepSeek 联网测试，不把联网测试计为跳过；保留
  环境变量开关，避免普通离线测试意外产生网络请求或 API 费用。
- 开始新版本时，先在 `docs/agent-development-process.md` 中记录目标、非目标、
  信息流和验收标准。
- 完成里程碑、修改核心契约或职责边界、修复重要 Bug、完成真实验收后，同步更新
  `docs/agent-development-process.md` 的阶段记录和更新记录。
- 不要读取或提交 `.env`、API Key 等敏感信息。
