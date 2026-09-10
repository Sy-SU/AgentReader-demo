import json
import os

from dotenv import load_dotenv

try:
    from openai import APIError, OpenAI
except ModuleNotFoundError:
    APIError = Exception
    OpenAI = None


load_dotenv()


PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-v4-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": None,
    },
}


def call_llm(messages: list[dict], tools: list[dict]) -> dict:
    """Call the selected provider and return a normalized response."""
    provider = os.getenv("LLM_PROVIDER", "fake").lower()

    if provider == "fake":
        return _call_fake_llm(messages, tools)

    config = PROVIDERS.get(provider)
    if config is None:
        supported = ", ".join(["fake", *PROVIDERS])
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}'. Supported: {supported}."
        )

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Missing API key environment variable: {config['api_key_env']}"
        )

    model = (
        os.getenv("LLM_MODEL")
        or os.getenv(config["model_env"])
        or config["default_model"]
    )
    if not model:
        raise RuntimeError(
            "Set LLM_MODEL or OPENROUTER_MODEL for OpenRouter."
        )

    base_url = os.getenv("LLM_BASE_URL", config["base_url"])
    if OpenAI is None:
        raise RuntimeError(
            "当前 Python 环境没有安装 openai。请先创建并激活 "
            "Conda 环境：'conda env create -f environment.yml'，"
            "然后执行 'conda activate agent-reader-demo'。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=_to_api_messages(messages),
            tools=_to_api_tools(tools),
            tool_choice="auto",
        )
    except APIError as error:
        status_code = getattr(error, "status_code", None)
        status = f" HTTP {status_code}" if status_code else ""
        raise RuntimeError(
            f"{provider} API request failed with{status}: {error}"
        ) from error

    return _normalize_api_response(response.model_dump())


def _call_fake_llm(messages: list[dict], tools: list[dict]) -> dict:
    """Return deterministic responses for offline Agent Loop tests."""
    last_message = messages[-1]

    if last_message["role"] == "tool":
        result = last_message["content"]

        if "error" in result:
            answer = f"工具执行失败：{result['error']['message']}"
        elif result["found"]:
            answer = (
                f"找到论文：{result['title']}，"
                f"PDF 地址：{result['pdf_url']}"
            )
        else:
            answer = "没有找到相关论文。"

        return {
            "type": "final",
            "content": answer,
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }

    available_tool_names = {tool["name"] for tool in tools}

    if "search_paper" not in available_tool_names:
        return {
            "type": "final",
            "content": "当前没有可用的论文搜索工具。",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }

    return {
        "type": "tool_call",
        "content": None,
        "tool_call_id": "fake-call-1",
        "tool_name": "search_paper",
        "tool_arguments": {"query": "TASA"},
    }


def _to_api_tools(tools: list[dict]) -> list[dict]:
    """Translate internal tool schemas into Chat Completions format."""
    return [
        {"type": "function", "function": tool_schema}
        for tool_schema in tools
    ]


def _to_api_messages(messages: list[dict]) -> list[dict]:
    """Translate internal messages into Chat Completions messages."""
    api_messages = []

    for message in messages:
        if "tool_call" in message:
            tool_call = message["tool_call"]
            api_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(
                                    tool_call["arguments"],
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            )
            continue

        if message["role"] == "tool":
            api_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": json.dumps(
                        message["content"], ensure_ascii=False
                    ),
                }
            )
            continue

        api_messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    return api_messages


def _normalize_api_response(response: dict) -> dict:
    """Translate a provider response into the internal response contract."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("LLM API response has no assistant message.") from error

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        if len(tool_calls) != 1:
            raise RuntimeError("V1 supports exactly one tool call per LLM step.")

        tool_call = tool_calls[0]
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must decode to an object")
            return {
                "type": "tool_call",
                "content": None,
                "tool_call_id": tool_call["id"],
                "tool_name": tool_call["function"]["name"],
                "tool_arguments": arguments,
            }
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("LLM returned an invalid tool call.") from error

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("LLM returned neither text nor a tool call.")

    return {
        "type": "final",
        "content": content,
        "tool_call_id": None,
        "tool_name": None,
        "tool_arguments": None,
    }
