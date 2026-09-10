# AgentReader Demo

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

## 运行

```bash
python main.py
```

需要查看 LLM 决策、Tool Call、Tool Result 和步数时：

```bash
python main.py --debug
```

## 测试

运行默认离线测试（不会请求模型 API）：

```bash
python -m unittest discover -s tests -v
```

显式运行 DeepSeek 在线测试（会产生一次完整 Agent Loop 的 API 请求）：

```bash
RUN_LIVE_LLM_TESTS=1 python -m unittest \
  tests.test_deepseek_integration -v
```

在线测试会强制选择 DeepSeek，读取 `.env` 中的
`DEEPSEEK_API_KEY` 和 `DEEPSEEK_MODEL`，不会受
`LLM_PROVIDER` 当前值影响。

完成后可退出虚拟环境：

```bash
conda deactivate
```
