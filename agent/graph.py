import json
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.intents import classify_intent
from agent.tools import mock_lead_capture
from agent.rag import AutoStreamRAG

def build_agent_graph(llm: BaseChatModel, rag_retriever: AutoStreamRAG):
    """
    Creates and compiles the LangGraph StateGraph workflow with persistent memory.
    """
    
    # ------------------ Node Definitions ------------------
    
    def intent_detector_node(state: AgentState):
        """
        Classifies user intent based on the latest message and recent history context.
        Uses local bypasses to save API quota on greetings and slot answers.
        """
        messages = state.get("messages", [])
        if not messages:
            return {"current_intent": "Greeting"}
        
        latest_message = messages[-1].content
        latest_clean = latest_message.strip().lower()
        
        # 1. Local Greeting Detection (Bypasses LLM call to save quota!)
        if latest_clean in ["hi", "hello", "hey", "hii", "hey there", "good morning", "good afternoon"]:
            return {"current_intent": "Greeting"}
            
        stage = state.get("lead_collection_stage", "none")
        
        # 2. Smart Slot-Filling Override (Bypasses intent LLM call when collecting details)
        if stage in ["ask_name", "ask_email", "ask_platform"] and stage != "completed":
            # Check if they are trying to ask an unrelated pricing question instead of answering
            inquiry_keywords = ["price", "pricing", "cost", "refund", "support", "features", "how much", "plan", "?"]
            has_inquiry = any(kw in latest_clean for kw in inquiry_keywords)
            if not has_inquiry:
                # Bypasses intent classifier entirely to save quota!
                return {"current_intent": "High-Intent Lead"}
        
        # 3. Standard intent classification using LLM
        # Build context from last 5 messages for intent classification stability
        history_messages = messages[:-1][-5:]
        history_str = ""
        for msg in history_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_str += f"{role}: {msg.content}\n"
            
        detected_intent = classify_intent(latest_message, history_str, llm)
        
        # Safety fallback
        if stage in ["ask_name", "ask_email", "ask_platform"] and stage != "completed":
            if detected_intent != "Product/Pricing Inquiry":
                detected_intent = "High-Intent Lead"
                
        return {"current_intent": detected_intent}

    def rag_retriever_node(state: AgentState):
        """
        Answers product and pricing questions by querying the local FAISS index.
        """
        latest_message = state["messages"][-1].content
        context = rag_retriever.retrieve_context(latest_message)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly helpful and professional AI sales and support representative for AutoStream.
AutoStream is a SaaS platform providing automated video editing tools for content creators.

Answer the user's question accurately using ONLY the provided local knowledge base context below.
If the answer is not present in the context, politely let the user know and offer to capture their lead details so an agent can follow up.

Knowledge Base Context:
{context}

Guidelines:
- Keep the response professional, clear, and concise.
- Direct users to the Pro plan if they ask about unlimited videos, 4K rendering, 24/7 priority live support, or AI captions.
"""),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "messages": state["messages"]
        })
        
        return {"messages": [response]}

    def lead_qualifier_node(state: AgentState):
        """
        Extracts lead information and manages the progressive slot-filling stage.
        """
        messages = state["messages"]
        latest_message = messages[-1].content
        
        # Extract fields using structured JSON extraction
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise data extraction agent for AutoStream.
Analyze the user's latest message and extract any of these customer slots if mentioned:
1. "name": The user's name (e.g. "Praneeth", "My name is John").
2. "email": The user's email address (e.g. "praneeth@gmail.com").
3. "platform": The content creator platform they publish to (e.g. "YouTube", "Instagram", "TikTok", "Twitch").

Current captured values:
- Name: {current_name}
- Email: {current_email}
- Platform: {current_platform}

Rules:
- Keep existing captured values unless the user is explicitly correcting or changing them.
- Respond with a valid JSON object ONLY, containing keys "name", "email", and "platform". Use null for fields not provided.
- Do not output backticks (```json) or markdown framing.

JSON format:
{{
    "name": "extracted_name" | null,
    "email": "extracted_email" | null,
    "platform": "extracted_platform" | null
}}
"""),
            ("human", "{message}")
        ])
        
        current_name = state.get("lead_name")
        current_email = state.get("lead_email")
        current_platform = state.get("lead_platform")
        
        chain = extract_prompt | llm
        
        try:
            response = chain.invoke({
                "current_name": current_name or "Not set",
                "current_email": current_email or "Not set",
                "current_platform": current_platform or "Not set",
                "message": latest_message
            })
            
            # Sanitise formatting
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
                
            extracted = json.loads(content)
            
            # Assign values
            new_name = extracted.get("name") or current_name
            new_email = extracted.get("email") or current_email
            new_platform = extracted.get("platform") or current_platform
        except Exception as e:
            print(f"[Lead Extraction Parsing Error]: {e}")
            new_name = current_name
            new_email = current_email
            new_platform = current_platform
            
        # Determine next missing slot and update the collection stage
        next_message = None
        next_stage = state.get("lead_collection_stage", "none")
        
        if not new_name:
            next_stage = "ask_name"
            next_message = AIMessage(content="Great! May I know your name?")
        elif not new_email:
            next_stage = "ask_email"
            next_message = AIMessage(content="Please share your email.")
        elif not new_platform:
            next_stage = "ask_platform"
            next_message = AIMessage(content="Which creator platform do you use?")
        else:
            next_stage = "completed"
            
        updates = {
            "lead_name": new_name,
            "lead_email": new_email,
            "lead_platform": new_platform,
            "lead_collection_stage": next_stage
        }
        
        if next_message:
            updates["messages"] = [next_message]
            
        return updates

    def tool_execution_node(state: AgentState):
        """
        Triggers the lead capture tool once all variables are satisfied.
        """
        name = state.get("lead_name")
        email = state.get("lead_email")
        platform = state.get("lead_platform")
        
        # Fire mock lead capture tool
        capture_result = mock_lead_capture(name, email, platform)
        
        # Produce message to show to user matching the required conversation output
        response = AIMessage(content=capture_result)
        
        return {
            "lead_collection_stage": "completed",
            "messages": [response]
        }

    def response_generator_node(state: AgentState):
        """
        Handles greetings, casual chat, and fallback responses.
        """
        intent = state.get("current_intent", "Unknown")
        
        if intent == "Greeting":
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are AutoStream's friendly sales assistant.
Greet the user enthusiastically, introduce yourself, and ask how you can help them with automated video editing solutions.
Keep it extremely short (1-2 sentences)."""),
                ("placeholder", "{messages}")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are AutoStream's sales assistant.
The user's message is outside normal domains. Politely respond and invite them to ask about AutoStream pricing, features, refund policy, or setting up a Pro account."""),
                ("placeholder", "{messages}")
            ])
            
        chain = prompt | llm
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

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
        if stage in ["ask_name", "ask_email", "ask_platform"]:
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
    
    # 3. Setup Persistent Checkpointing Memory
    memory = MemorySaver()
    
    # 4. Compile workflow runnable
    return workflow.compile(checkpointer=memory)
