import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from unittest.mock import Mock, patch

from llm import (
    PROVIDERS,
    _normalize_api_response,
    _provider_reasoning_options,
    _to_api_messages,
    call_llm,
)
from terminal import print_debug_trace


FINAL_API_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "done",
                "tool_calls": None,
            }
        }
    ]
}


class LLMReasoningProtocolTests(unittest.TestCase):
    def test_deepseek_reasoning_content_is_normalized_for_tool_and_final(self):
        tool_response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "private-tool-reasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "search_paper",
                                    "arguments": '{"query": "BERT"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        final_response = deepcopy(FINAL_API_RESPONSE)
        final_response["choices"][0]["message"]["reasoning_content"] = (
            "private-final-reasoning"
        )

        normalized_tool = _normalize_api_response(tool_response)
        normalized_final = _normalize_api_response(final_response)

        self.assertEqual(
            normalized_tool["reasoning_content"],
            "private-tool-reasoning",
        )
        self.assertEqual(normalized_tool["content"], "")
        self.assertEqual(
            normalized_final["reasoning_content"],
            "private-final-reasoning",
        )

    def test_openrouter_reasoning_alias_and_details_are_normalized(self):
        string_response = deepcopy(FINAL_API_RESPONSE)
        string_response["choices"][0]["message"]["reasoning"] = (
            "openrouter-reasoning"
        )
        details = [
            {
                "id": "reasoning-1",
                "format": "unknown",
                "type": "reasoning.text",
                "text": "opaque",
            }
        ]
        details_response = deepcopy(FINAL_API_RESPONSE)
        details_response["choices"][0]["message"]["reasoning_details"] = (
            details
        )

        normalized_string = _normalize_api_response(string_response)
        normalized_details = _normalize_api_response(details_response)
        details[0]["text"] = "changed"

        self.assertEqual(
            normalized_string["reasoning_content"],
            "openrouter-reasoning",
        )
        self.assertEqual(
            normalized_details["reasoning_details"][0]["text"],
            "opaque",
        )
        self.assertNotIn("reasoning_content", normalized_details)

    def test_non_thinking_response_keeps_the_existing_contract(self):
        normalized = _normalize_api_response(FINAL_API_RESPONSE)

        self.assertEqual(
            normalized,
            {
                "type": "final",
                "content": "done",
                "tool_call_id": None,
                "tool_name": None,
                "tool_arguments": None,
            },
        )

    def test_invalid_reasoning_shapes_are_rejected(self):
        for field, value in (
            ("reasoning_content", {"not": "text"}),
            ("reasoning", ["not", "text"]),
            ("reasoning_details", "not-a-list"),
        ):
            response = deepcopy(FINAL_API_RESPONSE)
            response["choices"][0]["message"][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "reasoning"):
                    _normalize_api_response(response)

    def test_api_messages_replay_reasoning_without_modification(self):
        reasoning_details = [{"type": "reasoning.encrypted", "data": "abc"}]
        messages = [
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "tool reasoning",
                "tool_call": {
                    "id": "call-1",
                    "name": "search_paper",
                    "arguments": {"query": "BERT"},
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_paper",
                "content": {"found": False},
            },
            {
                "role": "assistant",
                "content": "step done",
                "reasoning_details": reasoning_details,
            },
        ]

        api_messages = _to_api_messages(messages)
        reasoning_details[0]["data"] = "changed"

        self.assertEqual(
            api_messages[1]["reasoning_content"],
            "tool reasoning",
        )
        self.assertEqual(api_messages[1]["content"], "")
        self.assertEqual(
            api_messages[3]["reasoning_details"],
            [{"type": "reasoning.encrypted", "data": "abc"}],
        )
        self.assertNotIn("reasoning_content", api_messages[3])

    @patch("llm.OpenAI")
    def test_deepseek_request_explicitly_enables_thinking(self, openai):
        response = Mock()
        response.model_dump.return_value = FINAL_API_RESPONSE
        client = openai.return_value
        client.chat.completions.create.return_value = response
        environment = {
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_MODEL": "deepseek-v4-flash",
            "DEEPSEEK_THINKING": "enabled",
            "DEEPSEEK_REASONING_EFFORT": "low",
        }

        with patch.dict("os.environ", environment, clear=True):
            result = call_llm([{"role": "user", "content": "hello"}], [])

        self.assertEqual(result["content"], "done")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(
            kwargs["extra_body"],
            {"thinking": {"type": "enabled"}},
        )

    def test_deepseek_can_explicitly_disable_thinking(self):
        environment = {
            "DEEPSEEK_THINKING": "disabled",
            "DEEPSEEK_REASONING_EFFORT": "high",
        }

        with patch.dict("os.environ", environment, clear=True):
            options = _provider_reasoning_options(
                "deepseek",
                PROVIDERS["deepseek"],
            )

        self.assertEqual(
            options,
            {"extra_body": {"thinking": {"type": "disabled"}}},
        )

    def test_openrouter_reasoning_is_optional_and_configurable(self):
        with patch.dict("os.environ", {}, clear=True):
            default = _provider_reasoning_options(
                "openrouter",
                PROVIDERS["openrouter"],
            )
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_THINKING": "enabled",
                "OPENROUTER_REASONING_EFFORT": "low",
            },
            clear=True,
        ):
            configured = _provider_reasoning_options(
                "openrouter",
                PROVIDERS["openrouter"],
            )

        self.assertEqual(default, {})
        self.assertEqual(
            configured,
            {
                "extra_body": {
                    "reasoning": {"effort": "low", "exclude": False}
                }
            },
        )

    def test_debug_trace_does_not_print_reasoning(self):
        state = {
            "user_query": "test",
            "messages": [
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "do-not-print-this",
                },
            ],
        }
        output = StringIO()

        with redirect_stdout(output):
            print_debug_trace(state)

        self.assertNotIn("do-not-print-this", output.getvalue())


if __name__ == "__main__":
    unittest.main()
