import os
import sys
from dotenv import load_dotenv

from agent.llm import DEFAULT_LIMIT_REACHED_MESSAGE

# Load environment variables from .ignore/.env or .env file
if os.path.exists(".ignore/.env"):
    load_dotenv(".ignore/.env")
else:
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
 Powered by: LangGraph, LangChain, FAISS, Gemini 3.5 Flash-Lite & Groq GPT-OSS 20B
 
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
        from agent.service import AgentService

        # Builds the resilient LLM chain, embeddings, FAISS index and LangGraph workflow
        service = AgentService.from_env()
        print("✅ Initialization completed. The agent is ready!\n")

    except Exception as e:
        print(f"\n❌ FAILED TO INITIALIZE AGENT: {e}")
        print("Please ensure your Python dependencies are fully installed using:")
        print("   pip install -r requirements.txt\n")
        sys.exit(1)

    session_id = "autostream-cli-user-session"

    # Start conversational loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip() == "":
                continue
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                print("\n👋 Thank you for chatting with AutoStream! Have a great day!\n")
                break

            try:
                result = service.chat(session_id, user_input)
            except Exception as e:
                print(f"\033[90m[Chat Error: {e}]\033[0m")
                print(f"\nBot: {DEFAULT_LIMIT_REACHED_MESSAGE}\n")
                continue

            # Print diagnostic internal state markers (Sleek agentic debugger)
            lead = result.lead
            print(
                f"\033[90m[Agent Diagnostics | Intent: {result.intent} | Stage: {result.lead_stage} | "
                f"Lead: (Name={lead.name}, Email={lead.email}, Platform={lead.platform}) | "
                f"KB chunks used: {len(result.retrieved_chunks)}]\033[0m"
            )
            print(f"Bot: {result.reply}")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during conversation execution: {e}")

if __name__ == "__main__":
    main()
