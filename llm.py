import json
import os
import re
from copy import deepcopy

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
        "thinking_env": "DEEPSEEK_THINKING",
        "reasoning_effort_env": "DEEPSEEK_REASONING_EFFORT",
        "default_thinking": "enabled",
        "default_reasoning_effort": "high",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": None,
        "thinking_env": "OPENROUTER_THINKING",
        "reasoning_effort_env": "OPENROUTER_REASONING_EFFORT",
        "default_thinking": None,
        "default_reasoning_effort": None,
    },
}
THINKING_MODES = frozenset({"enabled", "disabled"})
DEEPSEEK_REASONING_EFFORTS = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)
OPENROUTER_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


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
    request_options = _provider_reasoning_options(provider, config)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=_to_api_messages(messages),
            tools=_to_api_tools(tools),
            tool_choice="auto",
            **request_options,
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
    available_tool_names = {tool["name"] for tool in tools}

    if last_message["role"] == "tool":
        result = last_message["content"]

        if "error" in result:
            answer = f"工具执行失败：{result['error']['message']}"
        elif last_message["name"] == "save_paper":
            if result["saved"]:
                answer = f"论文已保存：{result['title']}"
            elif result.get("reason") == "user_declined":
                answer = f"已取消保存：{result['title']}"
            else:
                answer = f"论文已经存在：{result['title']}"
        elif last_message["name"] == "list_library":
            if (
                result["count"] > 0
                and _user_requested_download(messages)
                and "download_paper" in available_tool_names
            ):
                return {
                    "type": "tool_call",
                    "content": None,
                    "tool_call_id": "fake-download-call-2",
                    "tool_name": "download_paper",
                    "tool_arguments": {
                        "paper_id": result["papers"][0]["id"]
                    },
                }
            if result["count"] == 0:
                answer = "本地文献库为空。"
            else:
                titles = "；".join(
                    paper["title"] for paper in result["papers"]
                )
                answer = (
                    f"本地文献库共有 {result['total']} 篇论文：{titles}"
                )
        elif last_message["name"] == "download_paper":
            if (
                _user_requested_retrieval(messages)
                and "retrieve_paper_chunks" in available_tool_names
            ):
                return {
                    "type": "tool_call",
                    "content": None,
                    "tool_call_id": "fake-retrieve-call-3",
                    "tool_name": "retrieve_paper_chunks",
                    "tool_arguments": {
                        "paper_id": result["paper_id"],
                        "query": _fake_retrieval_query(messages),
                    },
                }
            if (
                _user_requested_read(messages)
                and "extract_paper_text" in available_tool_names
            ):
                return {
                    "type": "tool_call",
                    "content": None,
                    "tool_call_id": "fake-extract-call-3",
                    "tool_name": "extract_paper_text",
                    "tool_arguments": {
                        "paper_id": result["paper_id"]
                    },
                }
            if result["cached"]:
                answer = f"PDF 已在本地缓存：{result['local_path']}"
            else:
                answer = f"PDF 已下载：{result['local_path']}"
        elif last_message["name"] == "extract_paper_text":
            if result["text_available"]:
                answer = (
                    f"已读取 {result['pages_read']} 页、"
                    f"提取 {result['char_count']} 个字符。"
                )
            else:
                answer = "PDF 中没有可提取的文本，当前版本尚未实现 OCR。"
        elif last_message["name"] == "retrieve_paper_chunks":
            if result["found"]:
                pages = sorted(
                    {match["page"] for match in result["matches"]}
                )
                page_text = "、".join(str(page) for page in pages)
                answer = (
                    f"找到 {result['count']} 个相关片段，"
                    f"位于第 {page_text} 页。"
                )
            elif result.get("rejected_low_query_coverage"):
                answer = "查询词在论文中的覆盖不足，暂时没有足够的关键词证据。"
            else:
                answer = "有界关键词检索没有找到匹配段落。"
        elif (
            result["found"]
            and _user_requested_save(messages)
            and "save_paper" in available_tool_names
        ):
            candidate_id = result["papers"][0]["candidate_id"]
            return {
                "type": "tool_call",
                "content": None,
                "tool_call_id": "fake-call-2",
                "tool_name": "save_paper",
                "tool_arguments": {"candidate_id": candidate_id},
            }
        elif (
            result["found"]
            and _user_requested_download(messages)
            and "download_paper" in available_tool_names
        ):
            paper_id = result["papers"][0]["candidate_id"]
            return {
                "type": "tool_call",
                "content": None,
                "tool_call_id": "fake-download-call-2",
                "tool_name": "download_paper",
                "tool_arguments": {"paper_id": paper_id},
            }
        elif result["found"]:
            paper = result["papers"][0]
            answer = (
                f"找到论文：{paper['title']}，"
                f"PDF 地址：{paper['pdf_url']}"
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

    if (
        _user_requested_retrieval(messages)
        and "retrieve_paper_chunks" in available_tool_names
    ):
        download_result = _latest_successful_download(messages)
        if download_result is not None:
            return {
                "type": "tool_call",
                "content": None,
                "tool_call_id": "fake-retrieve-follow-up-call",
                "tool_name": "retrieve_paper_chunks",
                "tool_arguments": {
                    "paper_id": download_result["paper_id"],
                    "query": _fake_retrieval_query(messages),
                },
            }

    if (
        _user_requested_read(messages)
        and "extract_paper_text" in available_tool_names
    ):
        download_result = _latest_successful_download(messages)
        if download_result is not None:
            return {
                "type": "tool_call",
                "content": None,
                "tool_call_id": "fake-extract-follow-up-call",
                "tool_name": "extract_paper_text",
                "tool_arguments": {
                    "paper_id": download_result["paper_id"]
                },
            }

    if (
        _user_requested_library_list(messages)
        and "list_library" in available_tool_names
    ):
        return {
            "type": "tool_call",
            "content": None,
            "tool_call_id": "fake-list-call-1",
            "tool_name": "list_library",
            "tool_arguments": {},
        }

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


def _user_requested_save(messages: list[dict]) -> bool:
    user_text = " ".join(
        message.get("content", "")
        for message in messages
        if message["role"] == "user"
        and isinstance(message.get("content"), str)
    ).lower()
    return "保存" in user_text or "save" in user_text


def _user_requested_download(messages: list[dict]) -> bool:
    user_text = " ".join(
        message.get("content", "")
        for message in messages
        if message["role"] == "user"
        and isinstance(message.get("content"), str)
    ).lower()
    return "下载" in user_text or "download" in user_text


def _user_requested_read(messages: list[dict]) -> bool:
    user_text = _latest_user_text(messages).lower()
    return any(
        keyword in user_text
        for keyword in (
            "读取",
            "阅读",
            "分析",
            "总结",
            "read",
            "analyze",
            "summarize",
            "summary",
            "extract",
        )
    )


def _user_requested_retrieval(messages: list[dict]) -> bool:
    user_text = _latest_user_text(messages).lower()
    return any(
        keyword in user_text
        for keyword in (
            "哪一页",
            "在哪里",
            "在哪",
            "回答",
            "查找",
            "检索",
            "where",
            "which page",
            "answer",
            "find",
        )
    )


def _latest_user_text(messages: list[dict]) -> str:
    return next(
        (
            message["content"].strip()
            for message in reversed(messages)
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ),
        "",
    )


def _fake_retrieval_query(messages: list[dict]) -> str:
    """Extract a deterministic content query for Fake LLM flow tests."""
    user_text = _latest_user_text(messages)
    english_terms = re.findall(
        r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*",
        user_text,
    )
    if english_terms:
        return " ".join(english_terms)

    query = user_text
    for phrase in (
        "请搜索",
        "请下载",
        "并回答",
        "哪一页",
        "在哪里",
        "在哪",
        "查找",
        "检索",
        "回答",
        "搜索",
        "下载",
    ):
        query = query.replace(phrase, " ")
    return query.strip(" ，。！？?：:") or user_text


def _latest_successful_download(messages: list[dict]) -> dict | None:
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if message.get("name") != "download_paper":
            continue
        result = message.get("content")
        if (
            isinstance(result, dict)
            and "error" not in result
            and isinstance(result.get("paper_id"), str)
            and isinstance(result.get("local_path"), str)
        ):
            return result
    return None


def _user_requested_library_list(messages: list[dict]) -> bool:
    latest_user_text = next(
        (
            message["content"].lower()
            for message in reversed(messages)
            if message["role"] == "user"
            and isinstance(message.get("content"), str)
        ),
        "",
    )
    mentions_library = (
        "文献库" in latest_user_text or "library" in latest_user_text
    )
    requests_list = any(
        keyword in latest_user_text
        for keyword in ("查看", "列出", "哪些", "有什么", "list", "show")
    )
    return mentions_library and requests_list


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
            api_message = {
                "role": "assistant",
                "content": message.get("content"),
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
            _copy_reasoning_fields(message, api_message)
            api_messages.append(api_message)
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

        api_message = {
            "role": message["role"],
            "content": message["content"],
        }
        if message["role"] == "assistant":
            _copy_reasoning_fields(message, api_message)
        api_messages.append(api_message)

    return api_messages


def _normalize_api_response(response: dict) -> dict:
    """Translate a provider response into the internal response contract."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("LLM API response has no assistant message.") from error

    reasoning_fields = _normalize_reasoning_fields(message)
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        # Keep the Runtime sequential even if a provider proposes parallel
        # calls. The first result is observed before the next LLM decision.
        tool_call = tool_calls[0]
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must decode to an object")
            normalized = {
                "type": "tool_call",
                "content": (
                    message.get("content")
                    if isinstance(message.get("content"), str)
                    else None
                ),
                "tool_call_id": tool_call["id"],
                "tool_name": tool_call["function"]["name"],
                "tool_arguments": arguments,
            }
            normalized.update(reasoning_fields)
            return normalized
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("LLM returned an invalid tool call.") from error

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("LLM returned neither text nor a tool call.")

    normalized = {
        "type": "final",
        "content": content,
        "tool_call_id": None,
        "tool_name": None,
        "tool_arguments": None,
    }
    normalized.update(reasoning_fields)
    return normalized


def _provider_reasoning_options(provider: str, config: dict) -> dict:
    """Build explicit provider request options without exposing secrets."""
    thinking = _environment_setting(
        "LLM_THINKING",
        config["thinking_env"],
        config["default_thinking"],
    )
    effort = _environment_setting(
        "LLM_REASONING_EFFORT",
        config["reasoning_effort_env"],
        config["default_reasoning_effort"],
    )

    if thinking is not None and thinking not in THINKING_MODES:
        raise RuntimeError(
            "Thinking mode must be 'enabled' or 'disabled'."
        )

    if provider == "deepseek":
        if effort not in DEEPSEEK_REASONING_EFFORTS:
            allowed = ", ".join(sorted(DEEPSEEK_REASONING_EFFORTS))
            raise RuntimeError(
                f"DeepSeek reasoning effort must be one of: {allowed}."
            )
        options = {"extra_body": {"thinking": {"type": thinking}}}
        if thinking == "enabled":
            options["reasoning_effort"] = effort
        return options

    if effort is not None and effort not in OPENROUTER_REASONING_EFFORTS:
        allowed = ", ".join(sorted(OPENROUTER_REASONING_EFFORTS))
        raise RuntimeError(
            f"OpenRouter reasoning effort must be one of: {allowed}."
        )
    if thinking == "disabled" and effort not in {None, "none"}:
        raise RuntimeError(
            "OpenRouter disabled thinking requires reasoning effort 'none'."
        )
    if thinking == "enabled" and effort == "none":
        raise RuntimeError(
            "OpenRouter enabled thinking cannot use reasoning effort 'none'."
        )

    reasoning = None
    if thinking == "disabled" or effort == "none":
        reasoning = {"effort": "none"}
    elif effort is not None:
        reasoning = {"effort": effort, "exclude": False}
    elif thinking == "enabled":
        reasoning = {"enabled": True, "exclude": False}
    if reasoning is None:
        return {}
    return {"extra_body": {"reasoning": reasoning}}


def _environment_setting(
    generic_name: str,
    provider_name: str,
    default: str | None,
) -> str | None:
    value = os.getenv(generic_name) or os.getenv(provider_name) or default
    return value.strip().lower() if isinstance(value, str) else value


def _normalize_reasoning_fields(message: dict) -> dict:
    """Normalize provider reasoning metadata for opaque State replay."""
    reasoning_details = message.get("reasoning_details")
    if reasoning_details is not None:
        if not isinstance(reasoning_details, list):
            raise RuntimeError("LLM returned invalid reasoning_details.")
        return {"reasoning_details": deepcopy(reasoning_details)}

    for field in ("reasoning_content", "reasoning"):
        value = message.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise RuntimeError("LLM returned invalid reasoning content.")
        return {"reasoning_content": value}
    return {}


def _copy_reasoning_fields(source: dict, target: dict) -> None:
    """Copy one normalized reasoning representation without inspecting it."""
    if "reasoning_details" in source:
        target["reasoning_details"] = deepcopy(source["reasoning_details"])
    elif "reasoning_content" in source:
        target["reasoning_content"] = source["reasoning_content"]
