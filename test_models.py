import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

candidate_models = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.5-flash"
]

print("API Key exists:", "GOOGLE_API_KEY" in os.environ)

for model in candidate_models:
    print(f"\n--- Testing model: {model} ---")
    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        response = llm.invoke("Hi")
        print(f"Success! Response: {response.content}")
    except Exception as e:
        print(f"Failed: {e}")
