import os
from dotenv import load_dotenv

if os.path.exists(".ignore/.env"):
    load_dotenv(".ignore/.env")
else:
    load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from agent.rag import AutoStreamRAG
from agent.graph import build_agent_graph
from agent.llm import get_fallback_llm

# Initialize LLM and Embeddings using fallback LLM chain
llm = get_fallback_llm()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

script_dir = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(script_dir, "data", "knowledge_base.md")
rag = AutoStreamRAG(kb_path, embeddings)

compiled_graph = build_agent_graph(llm, rag)
config = {"configurable": {"thread_id": "test-real-session"}}

state_input = {"messages": [HumanMessage(content="Hi")]}
try:
    print("Invoking graph...")
    state_output = compiled_graph.invoke(state_input, config)
    print("Graph execution successful!")
    print("Intent:", state_output.get("current_intent"))
    print("Messages:")
    for msg in state_output.get("messages", []):
        print(f"  {msg.__class__.__name__}: {msg.content}")
except Exception as e:
    print("Error executing graph:", e)
