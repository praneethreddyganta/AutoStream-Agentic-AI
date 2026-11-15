import re
from typing import Callable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.intents import (
    SLOT_STAGES,
    classify_intent,
    extract_email,
    is_valid_email,
    parse_json_reply,
    rule_based_intent,
)
from agent.tools import mock_lead_capture
from agent.rag import AutoStreamRAG
from agent.llm import DEFAULT_LIMIT_REACHED_MESSAGE, extract_text_content
from agent.prompts import (
    RAG_SYSTEM_PROMPT,
    LEAD_EXTRACTION_SYSTEM_PROMPT,
    GREETING_SYSTEM_PROMPT,
    OUT_OF_DOMAIN_SYSTEM_PROMPT
)

ASK_NAME = "Great! May I know your name?"
ASK_EMAIL = "Please share your email."
ASK_PLATFORM = "Which creator platform do you use?"
SLOT_QUESTIONS = {"ask_name": ASK_NAME, "ask_email": ASK_EMAIL, "ask_platform": ASK_PLATFORM}

KNOWN_PLATFORMS = {
    "youtube": "YouTube", "instagram": "Instagram", "tiktok": "TikTok", "twitch": "Twitch",
    "facebook": "Facebook", "linkedin": "LinkedIn", "twitter": "Twitter", "snapchat": "Snapchat",
}
NAME_RE = re.compile(r"[A-Za-z][A-Za-z .'-]{0,48}")


def detect_platform(text: str) -> str | None:
    """Finds a well-known creator platform in the text without calling the LLM."""
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in KNOWN_PLATFORMS:
            return KNOWN_PLATFORMS[word]
    return None


def _clean_slot(value) -> str | None:
    """Normalises an extracted slot value; LLMs sometimes return 'null' or 'Not set' as text."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() in {"null", "none", "not set", "n/a", "unknown"}:
        return None
    return value


def build_agent_graph(
    llm: BaseChatModel,
    rag_retriever: AutoStreamRAG,
    lead_capture: Callable[[str, str, str], str] = mock_lead_capture,
    checkpointer=None,
):
    """
    Creates and compiles the LangGraph StateGraph workflow with persistent memory.

    lead_capture: called once with (name, email, platform) when a lead is complete.
    checkpointer: conversation store; defaults to an in-process MemorySaver.
    """

    def _retrieve_chunks(query: str) -> list[str]:
        if hasattr(rag_retriever, "retrieve_chunks"):
            return list(rag_retriever.retrieve_chunks(query))
        context = rag_retriever.retrieve_context(query)
        return [context] if context else []

    # ------------------ Node Definitions ------------------

    def intent_detector_node(state: AgentState):
        """
        Classifies user intent based on the latest message and recent history context.
        Deterministic rules run first (no API quota); the LLM decides only when they are unsure.
        """
        messages = state.get("messages", [])
        if not messages:
            return {"current_intent": "Greeting", "retrieved_chunks": []}

        latest_message = extract_text_content(messages[-1].content)
        stage = state.get("lead_collection_stage", "none")

        detected_intent = rule_based_intent(latest_message, stage)

        if detected_intent is None:
            # Build context from last 5 messages for intent classification stability
            history_str = ""
            for msg in messages[:-1][-5:]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_str += f"{role}: {extract_text_content(msg.content)}\n"

            detected_intent = classify_intent(latest_message, history_str, llm)

            # While collecting details, anything that is not a product question is a slot answer
            if stage in SLOT_STAGES and detected_intent != "Product/Pricing Inquiry":
                detected_intent = "High-Intent Lead"

        # retrieved_chunks is reset every turn so it always describes the latest answer
        return {"current_intent": detected_intent, "retrieved_chunks": []}

    def rag_retriever_node(state: AgentState):
        """
        Answers product and pricing questions from the retrieved knowledge base chunks.
        """
        latest_message = extract_text_content(state["messages"][-1].content)
        chunks = _retrieve_chunks(latest_message)

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("placeholder", "{messages}")
        ])

        try:
            chain = prompt | llm
            response = chain.invoke({
                "context": "\n\n".join(chunks),
                "messages": state["messages"]
            })
            text = extract_text_content(response.content)
        except Exception as e:
            print(f"⚠️ [RAG Node Limit Fallback]: {e}")
            text = DEFAULT_LIMIT_REACHED_MESSAGE

        # If the user asked a question mid sign-up, answer it and then resume the pending question
        stage = state.get("lead_collection_stage", "none")
        if stage in SLOT_STAGES:
            text = f"{text}\n\nWhenever you're ready to continue signing up: {SLOT_QUESTIONS[stage]}"

        return {"messages": [AIMessage(content=text)], "retrieved_chunks": chunks}

    def lead_qualifier_node(state: AgentState):
        """
        Extracts lead information and manages the progressive slot-filling stage.
        """
        messages = state["messages"]
        latest_message = extract_text_content(messages[-1].content)
        stage = state.get("lead_collection_stage", "none")

        current_name = state.get("lead_name")
        current_email = state.get("lead_email")
        current_platform = state.get("lead_platform")

        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", LEAD_EXTRACTION_SYSTEM_PROMPT),
            ("human", "{message}")
        ])
        chain = extract_prompt | llm

        extracted = {}
        llm_unavailable = False
        try:
            response = chain.invoke({
                "current_name": current_name or "Not set",
                "current_email": current_email or "Not set",
                "current_platform": current_platform or "Not set",
                "message": latest_message
            })
            extracted = parse_json_reply(response.content)
            llm_unavailable = bool(extracted.get("limit_reached"))
        except Exception as e:
            print(f"[Lead Extraction Parsing Error]: {e}")
            llm_unavailable = True

        # Email: a regex match on the user's own text wins; an LLM value must be well-formed
        found_email = extract_email(latest_message)
        if not found_email:
            llm_email = _clean_slot(extracted.get("email"))
            found_email = llm_email if is_valid_email(llm_email) else None

        found_platform = _clean_slot(extracted.get("platform")) or detect_platform(latest_message)

        found_name = _clean_slot(extracted.get("name"))
        if not found_name and llm_unavailable and stage == "ask_name":
            # No model available: accept a short plain-text reply as the name we just asked for
            candidate = latest_message.strip()
            if NAME_RE.fullmatch(candidate) and len(candidate.split()) <= 4:
                found_name = candidate

        new_name = found_name or current_name
        new_email = found_email or current_email
        new_platform = found_platform or current_platform
        made_progress = (new_name, new_email, new_platform) != (current_name, current_email, current_platform)

        # Determine next missing slot and update the collection stage
        if not new_name:
            next_stage = "ask_name"
            reply = ASK_NAME if stage != "ask_name" else "Sorry, I didn't catch your name. Could you tell me your name?"
        elif not new_email:
            next_stage = "ask_email"
            if stage == "ask_email":
                reply = "That doesn't look like a valid email address. Could you share it again? (for example name@example.com)"
            else:
                reply = ASK_EMAIL
        elif not new_platform:
            next_stage = "ask_platform"
            if stage == "ask_platform":
                reply = "Sorry, I didn't catch that. Which creator platform do you use (for example YouTube, Instagram or TikTok)?"
            else:
                reply = ASK_PLATFORM
        else:
            next_stage = "completed"
            reply = None

        # The model is out of quota and we learned nothing: say so instead of silently re-asking
        if reply and llm_unavailable and not made_progress:
            reply = DEFAULT_LIMIT_REACHED_MESSAGE

        updates = {
            "lead_name": new_name,
            "lead_email": new_email,
            "lead_platform": new_platform,
            "lead_collection_stage": next_stage
        }

        if reply:
            updates["messages"] = [AIMessage(content=reply)]

        return updates

    def tool_execution_node(state: AgentState):
        """
        Triggers the lead capture tool once all variables are satisfied.
        """
        capture_result = lead_capture(
            state.get("lead_name"),
            state.get("lead_email"),
            state.get("lead_platform"),
        )

        return {
            "lead_collection_stage": "completed",
            "messages": [AIMessage(content=capture_result)]
        }

    def response_generator_node(state: AgentState):
        """
        Handles greetings, casual chat, and fallback responses.
        """
        intent = state.get("current_intent", "Unknown")
        chunks: list[str] = []

        if intent == "Greeting":
            system_prompt = GREETING_SYSTEM_PROMPT
            variables = {"messages": state["messages"]}
        else:
            # Unclassified messages may still touch product facts, so they get grounded context too
            chunks = _retrieve_chunks(extract_text_content(state["messages"][-1].content))
            system_prompt = OUT_OF_DOMAIN_SYSTEM_PROMPT
            variables = {"messages": state["messages"], "context": "\n\n".join(chunks)}

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])

        try:
            response = (prompt | llm).invoke(variables)
            text = extract_text_content(response.content)
        except Exception as e:
            print(f"⚠️ [Response Gen Limit Fallback]: {e}")
            text = DEFAULT_LIMIT_REACHED_MESSAGE
        return {"messages": [AIMessage(content=text)], "retrieved_chunks": chunks}

    # ------------------ Graph Edge Routing ------------------
    
    def route_after_intent(state: AgentState):
        """
        Determines the branch to follow based on classification and slot stages.
        """
        stage = state.get("lead_collection_stage", "none")
        intent = state.get("current_intent", "Unknown")
        
        # 1. If lead is already completed, do not re-trigger slot-filling or tools.
        # Answer follow-up questions (like payment) naturally using the conversational generator!
        if stage == "completed":
            if intent == "Product/Pricing Inquiry":
                return "rag_retriever"
            return "response_generator"
            
        # 2. If lead collection is active, prioritize qualification unless they pivot to pricing RAG questions
        if stage in SLOT_STAGES:
            if intent == "Product/Pricing Inquiry":
                return "rag_retriever"
            return "lead_qualifier"
            
        if intent == "Greeting":
            return "response_generator"
        elif intent == "Product/Pricing Inquiry":
            return "rag_retriever"
        elif intent == "High-Intent Lead":
            return "lead_qualifier"
        else:
            return "response_generator"

    def route_after_qualification(state: AgentState):
        """
        Routes either to tool execution node if slots are fully completed, or ends.
        """
        if state.get("lead_collection_stage") == "completed":
            return "tool_execution"
        return "end"

    # ------------------ Graph Compilation ------------------
    
    workflow = StateGraph(AgentState)
    
    # 1. Register Nodes
    workflow.add_node("intent_detector", intent_detector_node)
    workflow.add_node("rag_retriever", rag_retriever_node)
    workflow.add_node("lead_qualifier", lead_qualifier_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("response_generator", response_generator_node)
    
    # 2. Add Transitions
    workflow.add_edge(START, "intent_detector")
    
    # Intent conditional transitions
    workflow.add_conditional_edges(
        "intent_detector",
        route_after_intent,
        {
            "rag_retriever": "rag_retriever",
            "lead_qualifier": "lead_qualifier",
            "response_generator": "response_generator"
        }
    )
    
    # End branches
    workflow.add_edge("rag_retriever", END)
    workflow.add_edge("response_generator", END)
    
    # Lead qualifier split routing (either continue collecting or run tool)
    workflow.add_conditional_edges(
        "lead_qualifier",
        route_after_qualification,
        {
            "tool_execution": "tool_execution",
            "end": END
        }
    )
    
    workflow.add_edge("tool_execution", END)
    
    # 3. Compile with conversation memory (in-process by default; pass a persistent checkpointer in production)
    return workflow.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())
