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
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
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


# Custom implementation for local Ollama chat API using requests (No dependencies required)
class CustomOllamaModel:
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        
    def invoke(self, input, config=None, **kwargs):
        api_messages = convert_to_api_messages(input)
        payload = {
            "model": self.model_name,
            "messages": api_messages,
            "stream": False,
            "options": {
                "temperature": 0
            }
        }
        # 5-second timeout because Ollama is local and should respond immediately
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=5.0
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return AIMessage(content=content)


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
    If GROQ_API_KEY is present, Groq/Llama is inserted as the 2nd model (after primary Gemini 2.5).
    Allows using different API keys for fallbacks to avoid single-key quota blocking.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Read keys
    primary_key = os.getenv("GOOGLE_API_KEY")
    fallback_key_1 = os.getenv("GOOGLE_API_KEY_FALLBACK_1") or primary_key
    fallback_key_2 = os.getenv("GOOGLE_API_KEY_FALLBACK_2") or primary_key
    
    # Read settings
    primary_gemini = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    fallback_gemini_str = os.getenv("FALLBACK_GEMINI_MODELS", "gemini-2.0-flash,gemini-1.5-flash-latest,gemini-1.5-flash")
    fallback_geminis = [m.strip() for m in fallback_gemini_str.split(",") if m.strip()]
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    ollama_model = os.getenv("OLLAMA_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    runnables = []
    
    # 1. Primary Gemini (Uses main API key)
    try:
        gemini_model = ChatGoogleGenerativeAI(model=primary_gemini, google_api_key=primary_key, temperature=0, max_retries=1)
        runnables.append(make_logging_wrapper(gemini_model, primary_gemini))
    except Exception as e:
        print(f"❌ [LLM Setup Error] Could not instantiate primary model '{primary_gemini}': {e}")
        
    # 2. Smart Dynamic Position: If Groq Key is available, place Groq 2nd!
    if groq_api_key and groq_api_key.strip():
        groq_model_instance = CustomGroqModel(api_key=groq_api_key, model_name=groq_model)
        runnables.append(make_logging_wrapper(groq_model_instance, f"groq:{groq_model}"))
        
    # 3. Fallback Gemini Models (Rotate fallback API keys)
    for i, model_name in enumerate(fallback_geminis):
        try:
            # Cycle keys: 1st fallback model gets fallback_key_1, subsequent models get fallback_key_2
            key_to_use = fallback_key_1 if i == 0 else fallback_key_2
            gemini_model = ChatGoogleGenerativeAI(model=model_name, google_api_key=key_to_use, temperature=0, max_retries=1)
            runnables.append(make_logging_wrapper(gemini_model, model_name))
        except Exception as e:
            print(f"❌ [LLM Setup Error] Could not instantiate fallback model '{model_name}': {e}")
            
    # 4. Ollama (if key is set)
    if ollama_model and ollama_model.strip():
        ollama_model_instance = CustomOllamaModel(base_url=ollama_base_url, model_name=ollama_model)
        runnables.append(make_logging_wrapper(ollama_model_instance, f"ollama:{ollama_model}"))
        
    if not runnables:
        # Emergency default fallback if somehow everything failed to instantiate
        fallback_default = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, max_retries=1)
        return make_logging_wrapper(fallback_default, "gemini-1.5-flash-latest")
        
    # Compile with fallbacks
    primary_runnable = runnables[0]
    if len(runnables) > 1:
        return primary_runnable.with_fallbacks(fallbacks=runnables[1:])
    return primary_runnable
