import os
import requests

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

# Load environment variables from secret folder first, then local directory.
if os.path.exists(".ignore/.env"):
    load_dotenv(".ignore/.env")
else:
    load_dotenv()


def convert_to_api_messages(prompt_input):
    """Convert LangChain messages/prompts into Groq chat-completions messages."""
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


class CustomGroqModel:
    """Minimal Groq chat-completions adapter used only as the Gemini fallback."""

    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-20b"):
        self.api_key = api_key
        self.model_name = model_name

    def invoke(self, input, config=None, **kwargs):
        api_messages = convert_to_api_messages(input)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": api_messages,
            # Low reasoning effort is preferred for this latency-sensitive chatbot.
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "max_completion_tokens": 1024,
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=8.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return AIMessage(content=content)


def make_logging_wrapper(model_runnable, name):
    """Log provider failures, then re-raise so LangChain can invoke the fallback."""
    def _invoke(input, config=None, **kwargs):
        try:
            return model_runnable.invoke(input, config=config, **kwargs)
        except Exception as exc:
            print(f"⚠️ [LLM Fallback] Model '{name}' failed: {exc}")
            raise

    return RunnableLambda(_invoke)


def get_fallback_llm(
    primary_api_key: str | None = None,
    groq_api_key: str | None = None,
    primary_model: str | None = None,
):
    """
    Build the two-provider LLM chain.

    Gemini Flash-Lite is always attempted first. Groq GPT-OSS 20B is called only
    when the Gemini invocation raises an exception. No legacy Gemini or Ollama
    models are added to the fallback chain.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    primary_key = primary_api_key or os.getenv("GOOGLE_API_KEY")
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
    model_name = primary_model or os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
    groq_model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not primary_key or not primary_key.strip():
        raise ValueError("GOOGLE_API_KEY is required for the primary Gemini model.")

    # Disable SDK-level retries so quota/rate-limit errors can move promptly to Groq.
    gemini_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=primary_key,
        temperature=0,
        max_retries=0,
    )
    primary_runnable = make_logging_wrapper(gemini_model, model_name)

    if not groq_key or not groq_key.strip():
        print("⚠️ [LLM Setup] GROQ_API_KEY is missing; Gemini has no configured fallback.")
        return primary_runnable

    groq_model = CustomGroqModel(api_key=groq_key, model_name=groq_model_name)
    groq_runnable = make_logging_wrapper(groq_model, f"groq:{groq_model_name}")
    return primary_runnable.with_fallbacks(fallbacks=[groq_runnable])
