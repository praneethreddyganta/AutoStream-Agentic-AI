"""
service.py

The single entry point to the AutoStream agent. The CLI, the Streamlit dashboard
and any future HTTP backend all talk to AgentService instead of touching the
LangGraph graph, its checkpointer or its state keys directly.
"""
import os
from dataclasses import dataclass, field, asdict
from typing import Callable

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_agent_graph
from agent.llm import extract_text_content
from agent.tools import mock_lead_capture

DEFAULT_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge_base.md"
)
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"


@dataclass
class Lead:
    name: str | None = None
    email: str | None = None
    platform: str | None = None

    @property
    def filled_slots(self) -> int:
        return sum(1 for value in (self.name, self.email, self.platform) if value)


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class SessionSnapshot:
    """Everything a client may show about a conversation."""
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    intent: str = "Unknown"
    lead_stage: str = "none"
    lead: Lead = field(default_factory=Lead)
    retrieved_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChatResult(SessionSnapshot):
    """The outcome of one user turn."""
    reply: str = ""
    # True only on the turn in which the lead was completed and handed to lead_capture
    lead_captured: bool = False


class AgentService:
    def __init__(
        self,
        llm,
        rag,
        lead_capture: Callable[[str, str, str], str] = mock_lead_capture,
        checkpointer=None,
    ):
        self._checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self._graph = build_agent_graph(llm, rag, lead_capture=lead_capture, checkpointer=self._checkpointer)

    @classmethod
    def from_env(
        cls,
        google_api_key: str | None = None,
        embedding_model: str | None = None,
        kb_path: str | None = None,
        **kwargs,
    ) -> "AgentService":
        """Builds the production service: fallback LLM chain, Gemini embeddings and the FAISS index."""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from agent.llm import get_fallback_llm
        from agent.rag import AutoStreamRAG

        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            google_api_key=google_api_key or os.getenv("GOOGLE_API_KEY"),
        )
        rag = AutoStreamRAG(kb_path or DEFAULT_KB_PATH, embeddings)
        return cls(get_fallback_llm(), rag, **kwargs)

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def _snapshot_fields(self, session_id: str, values: dict) -> dict:
        messages = [
            ChatMessage(
                role="user" if isinstance(msg, HumanMessage) else "assistant",
                content=extract_text_content(msg.content),
            )
            for msg in values.get("messages", [])
        ]
        return dict(
            session_id=session_id,
            messages=messages,
            intent=values.get("current_intent") or "Unknown",
            lead_stage=values.get("lead_collection_stage") or "none",
            lead=Lead(values.get("lead_name"), values.get("lead_email"), values.get("lead_platform")),
            retrieved_chunks=list(values.get("retrieved_chunks") or []),
        )

    def get_session(self, session_id: str) -> SessionSnapshot:
        """Returns the current state of a conversation (empty for an unknown session)."""
        state = self._graph.get_state(self._config(session_id))
        return SessionSnapshot(**self._snapshot_fields(session_id, state.values if state else {}))

    def chat(self, session_id: str, message: str) -> ChatResult:
        """Sends one user message and returns the agent's reply with the updated session state."""
        stage_before = self.get_session(session_id).lead_stage
        values = self._graph.invoke({"messages": [HumanMessage(content=message)]}, self._config(session_id))
        fields = self._snapshot_fields(session_id, values)
        reply = fields["messages"][-1].content if fields["messages"] else ""
        return ChatResult(
            **fields,
            reply=reply,
            lead_captured=fields["lead_stage"] == "completed" and stage_before != "completed",
        )

    def reset(self, session_id: str) -> None:
        """Forgets one conversation. Other sessions are untouched."""
        self._checkpointer.delete_thread(session_id)
