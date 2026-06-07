import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def print_banner():
    """
    Prints a beautiful CLI banner for AutoStream Agent.
    """
    banner = """
======================================================================
     🚀  AUTOSTREAM CONVERSATIONAL AI SALES & SUPPORT AGENT  🚀
======================================================================
 Welcome to AutoStream's interactive terminal chatbot!
 
 Intern Assignment Project (ServiceHive - Inflx - Social-to-Lead)
 Powered by: LangGraph, LangChain, FAISS, and Gemini 2.5 Flash
 
 Type your messages below. Type 'exit' or 'quit' to end the chat.
======================================================================
"""
    print(banner)

def verify_google_key():
    """
    Verifies if Google Gemini API key is configured.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "" or "your_google_ai_studio" in api_key:
        print("\n❌ ERROR: GOOGLE_API_KEY is not configured in the environment or .env file.")
        print("👉 Action Required: Open the '.env' file in this directory and paste your valid Gemini API key.")
        print("   Get a FREE key instantly from Google AI Studio: https://aistudio.google.com/")
        print("   Format: GOOGLE_API_KEY=AIzaSy...\n")
        sys.exit(1)

def main():
    verify_google_key()
    print_banner()
    
    print("⚙️  Initializing AI models and local FAISS vector store. Please wait...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from langchain_core.messages import HumanMessage
        
        from agent.rag import AutoStreamRAG
        from agent.graph import build_agent_graph
        
        # 1. Initialize core LLM and Embeddings using Google Gemini (FREE tier)
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        print(f"  🔹 Step 1/3: Instantiating ChatGemini ({model_name}) and Embeddings...")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # 2. Build local RAG Knowledge Base
        print("  🔹 Step 2/3: Loading local markdown and generating FAISS vector index (via Google API)...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(script_dir, "data", "knowledge_base.md")
        rag = AutoStreamRAG(kb_path, embeddings)
        
        # 3. Assemble and compile the LangGraph workflow
        print("  🔹 Step 3/3: Assembling and compiling LangGraph StateGraph...")
        compiled_graph = build_agent_graph(llm, rag)
        
        print("✅ Initialization completed. The agent is ready!\n")
        
    except Exception as e:
        print(f"\n❌ FAILED TO INITIALIZE AGENT: {e}")
        print("Please ensure your Python dependencies are fully installed using:")
        print("   pip install -r requirements.txt\n")
        sys.exit(1)
        
    # Configure conversation checkpoint thread identifier
    config = {"configurable": {"thread_id": "autostream-cli-user-session"}}
    
    # Start conversational loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip() == "":
                continue
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                print("\n👋 Thank you for chatting with AutoStream! Have a great day!\n")
                break
                
            # Send message to LangGraph StateGraph workflow
            # Passing the user message as a list of HumanMessage
            state_input = {
                "messages": [HumanMessage(content=user_input)]
            }
            
            # Execute step of graph
            state_output = compiled_graph.invoke(state_input, config)
            
            # Print diagnostic internal state markers (Sleek agentic debugger)
            intent = state_output.get("current_intent", "Unknown")
            stage = state_output.get("lead_collection_stage", "none")
            name = state_output.get("lead_name") or "None"
            email = state_output.get("lead_email") or "None"
            platform = state_output.get("lead_platform") or "None"
            
            print(f"\033[90m[Agent Diagnostics | Intent: {intent} | Stage: {stage} | Lead: (Name={name}, Email={email}, Platform={platform})]\033[0m")
            
            # Print last AI Message
            messages = state_output.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(f"Bot: {last_msg.content}")
                
        except KeyboardInterrupt:
            print("\n👋 Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during conversation execution: {e}")

if __name__ == "__main__":
    main()
