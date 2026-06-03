import os
import sys
from dotenv import load_dotenv

# Load env
load_dotenv()

def run_simulation():
    print("=== STARTING SIMULATED CONVERSATION TEST ===")
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_core.messages import HumanMessage
        from agent.rag import AutoStreamRAG
        from agent.graph import build_agent_graph
        
        # Initialize
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(script_dir, "data", "knowledge_base.md")
        rag = AutoStreamRAG(kb_path, embeddings)
        
        compiled_graph = build_agent_graph(llm, rag)
        config = {"configurable": {"thread_id": "test-session-001"}}
        
        # Turns simulation
        turns = [
            "Hi",
            "Tell me about your pricing.",
            "That sounds good, I want to try the Pro plan for my YouTube channel.",
            "Praneeth",
            "praneeth@gmail.com"
        ]
        
        for i, user_message in enumerate(turns):
            print(f"\nTurn {i+1} - User: {user_message}")
            state_input = {"messages": [HumanMessage(content=user_message)]}
            
            state_output = compiled_graph.invoke(state_input, config)
            
            # Print state details
            intent = state_output.get("current_intent", "Unknown")
            stage = state_output.get("lead_collection_stage", "none")
            name = state_output.get("lead_name") or "None"
            email = state_output.get("lead_email") or "None"
            platform = state_output.get("lead_platform") or "None"
            
            print(f"[State] Intent: {intent} | Stage: {stage} | Slots: Name={name}, Email={email}, Platform={platform}")
            
            messages = state_output.get("messages", [])
            if messages:
                print(f"Bot: {messages[-1].content}")
                
        print("\n=== SIMULATED CONVERSATION TEST COMPLETE ===")
        
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("❌ Test cancelled: OPENAI_API_KEY is not configured in .env file.")
        sys.exit(0)
    run_simulation()
