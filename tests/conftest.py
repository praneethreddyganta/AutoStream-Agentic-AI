"""Offline test doubles: no API keys or network are needed to run this suite."""
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent.graph import detect_platform
from agent.intents import extract_email
from agent.service import AgentService

KB_CHUNK = "Basic Plan: $29/month. Pro Plan: $79/month."


class FakeLLM:
    """
    Stands in for the chat model. It recognises which node is calling from the
    system prompt and records every call, so tests can assert on routing.
    """

    def __init__(self, intent="Unknown", limit_reached=False):
        self.intent = intent
        self.limit_reached = limit_reached
        self.calls: list[str] = []
        self.prompts: list[str] = []

    def _invoke(self, prompt_value, config=None, **kwargs):
        messages = prompt_value.to_messages()
        system, last = messages[0].content, messages[-1].content
        self.prompts.append(system)

        if "routing assistant" in system:
            self.calls.append("intent")
            return AIMessage(content=json.dumps({"intent": self.intent, "limit_reached": self.limit_reached}))

        if "data extraction agent" in system:
            self.calls.append("extract")
            slots = {"name": None, "email": None, "platform": None}
            if self.limit_reached:
                return AIMessage(content=json.dumps({**slots, "limit_reached": True}))
            slots["email"] = extract_email(last)
            slots["platform"] = detect_platform(last)
            if not slots["email"] and not slots["platform"] and len(last.split()) <= 2:
                slots["name"] = last.strip()
            # Real models often wrap JSON in a code fence; the parser must cope
            return AIMessage(content="```json\n" + json.dumps(slots) + "\n```")

        self.calls.append("chat")
        return AIMessage(content="LLM_REPLY")

    def as_runnable(self):
        return RunnableLambda(self._invoke)


class FakeRAG:
    def __init__(self):
        self.queries: list[str] = []

    def retrieve_chunks(self, query):
        self.queries.append(query)
        return [KB_CHUNK]

    def retrieve_context(self, query):
        return "\n\n".join(self.retrieve_chunks(query))


class Harness:
    def __init__(self, **llm_kwargs):
        self.llm = FakeLLM(**llm_kwargs)
        self.rag = FakeRAG()
        self.captured: list[tuple] = []
        self.service = AgentService(self.llm.as_runnable(), self.rag, lead_capture=self._capture)

    def _capture(self, name, email, platform):
        self.captured.append((name, email, platform))
        return f"Lead captured successfully: {name}, {email}, {platform}"

    def say(self, message, session="s1"):
        return self.service.chat(session, message)


@pytest.fixture
def harness():
    return Harness()


@pytest.fixture
def make_harness():
    return Harness
