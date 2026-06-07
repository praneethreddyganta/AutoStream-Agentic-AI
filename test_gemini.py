import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
print("Model name:", model_name)
print("API Key exists:", "GOOGLE_API_KEY" in os.environ)

try:
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    response = llm.invoke("Hi, please say hello!")
    print("Response successful:", response.content)
except Exception as e:
    print("Error invoking LLM:", e)
