import os
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

# Load environment variables from secret folder first, then local directory
if os.path.exists(".ignore/.env"):
    load_dotenv(".ignore/.env")
else:
    load_dotenv()

# Helper to format any prompt input into OpenAI-style role/content messages
def convert_to_api_messages(prompt_input):
    if hasattr(prompt_input, "to_messages"):
        messages = prompt_input.to_messages()
    elif isinstance(prompt_input, list):
        messages = prompt_input
    elif isinstance(prompt_input, str):
        messages = [HumanMessage(content=prompt_input)]
    else:
        messages = [HumanMessage(content=str(prompt_input))]
        
    api_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "user"
        api_messages.append({"role": role, "content": msg.content})
    return api_messages


# Custom implementation for Groq Cloud chat API using requests (No dependencies required)
class CustomGroqModel:
    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-20b"):
        self.api_key = api_key
        if model_name in ["gpt-oss-20b", "gpt-oss-120b"]:
            model_name = f"openai/{model_name}"
        self.model_name = model_name
        
    def invoke(self, input, config=None, **kwargs):
        api_messages = convert_to_api_messages(input)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": 0
        }
        # 8-second timeout so it fails fast if Groq is degraded and moves to next fallback
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=8.0
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return AIMessage(content=content)


DEFAULT_LIMIT_REACHED_MESSAGE = (
    "We are currently experiencing very high demand and our service limit has been reached. "
    "Please visit us again a little later, and we will be delighted to assist you. "
    "Thank you for your patience and understanding!"
)


class GracefulLimitExceededFallback:
    """
    Safety terminal fallback model invoked when both primary (Gemini) and fallback (Groq) models
    have reached their rate limits or quotas.
    Returns a polite, respectful message instead of throwing an error.
    """
    def __init__(self, message: str = None):
        self.message = message or os.getenv("LIMIT_REACHED_MESSAGE", DEFAULT_LIMIT_REACHED_MESSAGE)
        
    def invoke(self, input, config=None, **kwargs):
        input_str = str(input)
        # If the graph node is requesting intent classification or lead slot extraction JSON,
        # return safe JSON so graph flow continues without JSON decode error
        if "INTENT_CLASSIFIER" in input_str or '"intent"' in input_str:
            return AIMessage(content='{"intent": "Unknown"}')
        if "LEAD_EXTRACTION" in input_str or '"name":' in input_str:
            return AIMessage(content='{"name": null, "email": null, "platform": null}')
        return AIMessage(content=self.message)


def make_logging_wrapper(model_runnable, name):
    """Wraps any LangChain-compatible model in a logger that triggers on exception."""
    def _invoke(input, config=None, **kwargs):
        try:
            return model_runnable.invoke(input, config, **kwargs)
        except Exception as e:
            print(f"⚠️ [LLM Fallback Warning] Model '{name}' failed: {e}")
            raise e
    return RunnableLambda(_invoke)


def get_fallback_llm():
    """
    Constructs and returns the resilient LLM pipeline.
    Primary: Gemini 3.5 Flash-Lite (Google GenAI)
    Fallback: Groq GPT-OSS 20B (Groq Cloud API)
    Terminal: Graceful respectful message if both models reach limits
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Read API keys
    primary_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Read model names
    primary_gemini = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
    groq_model = os.getenv("GROQ_MODEL", "gpt-oss-20b")
    
    runnables = []
    
    # 1. Primary Model: Gemini 3.5 Flash-Lite
    try:
        gemini_model = ChatGoogleGenerativeAI(
            model=primary_gemini,
            google_api_key=primary_key,
            temperature=0,
            max_retries=1
        )
        runnables.append(make_logging_wrapper(gemini_model, f"gemini:{primary_gemini}"))
    except Exception as e:
        print(f"❌ [LLM Setup Error] Could not instantiate primary model '{primary_gemini}': {e}")
        
    # 2. Fallback Model: Groq GPT-OSS 20B
    if groq_api_key and groq_api_key.strip():
        groq_model_instance = CustomGroqModel(api_key=groq_api_key, model_name=groq_model)
        runnables.append(make_logging_wrapper(groq_model_instance, f"groq:{groq_model}"))
        
    # 3. Terminal Graceful Limit Reached Handler (always available)
    terminal_fallback = GracefulLimitExceededFallback()
    runnables.append(make_logging_wrapper(terminal_fallback, "limit_exceeded_fallback"))
    
    # Compile with fallbacks: Primary -> Groq -> Graceful message
    primary_runnable = runnables[0]
    if len(runnables) > 1:
        return primary_runnable.with_fallbacks(fallbacks=runnables[1:])
    return primary_runnable
