import os
import sys
from dotenv import load_dotenv

if os.path.exists(".ignore/.env"):
    load_dotenv(".ignore/.env")
else:
    load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.llm import get_fallback_llm, make_logging_wrapper, GracefulLimitExceededFallback

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
    
    # 1. Create a chain where primary fails and fallback succeeds
    model1 = make_logging_wrapper(FailingModel("primary-gemini-3.5-flash-lite"), "primary-gemini-3.5-flash-lite")
    model2 = make_logging_wrapper(SucceedingModel("fallback-groq-gpt-oss-20b"), "fallback-groq-gpt-oss-20b")
    
    print("\n--- Compiling Mock Fallback Chain ---")
    fallback_chain = model1.with_fallbacks([model2])
    
    print("\n--- Invoking Fallback Chain ---")
    try:
        response = fallback_chain.invoke("Hi")
        print("\n🎉 SUCCESS!")
        print("Final Response Content:", response.content)
        assert "fallback-groq-gpt-oss-20b" in response.content
        print("Test passed: Seamlessly recovered from primary failure to get a reply from Groq fallback!")
    except Exception as e:
        print("\n❌ FAILED:", e)
        sys.exit(1)
        
    # 2. Test when BOTH models reach their limits (Gemini fails + Groq fails)
    print("\n--- Testing When BOTH Models Reach Limits ---")
    model_gemini_fail = make_logging_wrapper(FailingModel("primary-gemini-3.5-flash-lite"), "primary-gemini-3.5-flash-lite")
    model_groq_fail = make_logging_wrapper(FailingModel("fallback-groq-gpt-oss-20b"), "fallback-groq-gpt-oss-20b")
    model_terminal = make_logging_wrapper(GracefulLimitExceededFallback(), "graceful_limit_reached")
    
    both_failed_chain = model_gemini_fail.with_fallbacks([model_groq_fail, model_terminal])
    try:
        response_both = both_failed_chain.invoke("Hi")
        print("\n🎉 SUCCESSFUL GRACEFUL DEGRADATION!")
        print("Respectful Limit Message:", response_both.content)
        assert "limit has been reached" in response_both.content
        print("Test passed: Polite respectful message delivered without throwing an error!")
    except Exception as e:
        print("\n❌ FAILED TO DEGRADE GRACEFULLY:", e)
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
