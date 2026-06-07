import os
import sys
from dotenv import load_dotenv

if os.path.exists(".ignore/.env"):
    load_dotenv(".ignore/.env")
else:
    load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.llm import get_fallback_llm, make_logging_wrapper

class FailingModel:
    def __init__(self, name):
        self.name = name
    def invoke(self, input, config=None, **kwargs):
        print(f"💥 [Mock] Model '{self.name}' is simulated to fail!")
        raise RuntimeError(f"Rate limit exceeded (429) for {self.name}")

class SucceedingModel:
    def __init__(self, name):
        self.name = name
    def invoke(self, input, config=None, **kwargs):
        print(f"✅ [Mock] Model '{self.name}' invoked successfully!")
        return AIMessage(content=f"Hello from working model '{self.name}'!")

def test_fallback_flow():
    print("==================================================")
    print("🧪 RUNNING MULTI-LLM FALLBACK FLOW DIAGNOSTICS")
    print("==================================================")
    
    # 1. Create a chain where the first 2 models fail and the 3rd succeeds
    model1 = make_logging_wrapper(FailingModel("primary-gemini-2.5"), "primary-gemini-2.5")
    model2 = make_logging_wrapper(FailingModel("backup-groq"), "backup-groq")
    model3 = make_logging_wrapper(SucceedingModel("backup-gemini-2.0"), "backup-gemini-2.0")
    
    print("\n--- Compiling Mock Fallback Chain ---")
    fallback_chain = model1.with_fallbacks([model2, model3])
    
    print("\n--- Invoking Fallback Chain ---")
    try:
        response = fallback_chain.invoke("Hi")
        print("\n🎉 SUCCESS!")
        print("Final Response Content:", response.content)
        assert "backup-gemini-2.0" in response.content
        print("Test passed: Seamlessly recovered from 2 failures to get a reply!")
    except Exception as e:
        print("\n❌ FAILED:", e)
        sys.exit(1)
        
    print("\n--- Testing Live API Setup (without triggering fallbacks unless quota hit) ---")
    live_llm = get_fallback_llm()
    try:
        print("Invoking live configuration...")
        resp = live_llm.invoke("Hi")
        print("Live Response Success:", resp.content)
    except Exception as e:
        print("Live invocation failed (this is normal if no keys/quotas are active):", e)

if __name__ == "__main__":
    test_fallback_flow()
