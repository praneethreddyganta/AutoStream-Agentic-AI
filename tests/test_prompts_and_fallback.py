import re

import pytest
from langchain_core.messages import AIMessage

from agent import prompts
from agent.intents import parse_json_reply
from agent.llm import GracefulLimitExceededFallback, extract_text_content, make_logging_wrapper


def test_prompts_contain_no_hardcoded_product_facts():
    """The knowledge base is the single source of truth; prices must never live in prompts."""
    for name in dir(prompts):
        value = getattr(prompts, name)
        if name.isupper() and isinstance(value, str):
            assert not re.search(r"\$\s?\d", value), f"{name} contains a hardcoded price"
            assert "7-day" not in value, f"{name} contains a hardcoded policy"


class Failing:
    def invoke(self, input, config=None, **kwargs):
        raise RuntimeError("429 rate limit")


class Working:
    def invoke(self, input, config=None, **kwargs):
        return AIMessage(content=[{"type": "text", "text": "hello from fallback"}])


def test_fallback_chain_recovers_from_primary_failure():
    chain = make_logging_wrapper(Failing(), "primary").with_fallbacks([make_logging_wrapper(Working(), "fallback")])
    assert chain.invoke("Hi").content == "hello from fallback"


def test_all_models_down_degrades_gracefully():
    chain = make_logging_wrapper(Failing(), "primary").with_fallbacks([
        make_logging_wrapper(Failing(), "fallback"),
        make_logging_wrapper(GracefulLimitExceededFallback(), "terminal"),
    ])
    assert "limit has been reached" in chain.invoke("Hi").content


def test_terminal_fallback_returns_flagged_json_for_structured_prompts():
    terminal = GracefulLimitExceededFallback()
    intent = parse_json_reply(terminal.invoke(prompts.INTENT_CLASSIFIER_SYSTEM_PROMPT).content)
    assert intent == {"intent": "Unknown", "limit_reached": True}
    slots = parse_json_reply(terminal.invoke(prompts.LEAD_EXTRACTION_SYSTEM_PROMPT).content)
    assert slots["limit_reached"] is True and slots["email"] is None


@pytest.mark.parametrize("content,expected", [
    ("plain", "plain"),
    ([{"type": "text", "text": "a"}, "b"], "ab"),
    ({"text": "c"}, "c"),
])
def test_extract_text_content(content, expected):
    assert extract_text_content(content) == expected
