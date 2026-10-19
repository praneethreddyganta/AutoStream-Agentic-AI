import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

candidate_models = [
    "gemini-3.5-flash-lite",
]

print("API Key exists:", "GOOGLE_API_KEY" in os.environ)

for model in candidate_models:
    print(f"\n--- Testing chat model: {model} ---")
    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        response = llm.invoke("Hi")
        print(f"Success! Response: {response.content}")
    except Exception as e:
        print(f"Failed: {e}")

embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
print(f"\n--- Testing embedding model: {embedding_model} ---")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vec = embeddings.embed_query("AutoStream pricing")
    print(f"Success! Embedding dimension: {len(vec)}")
except Exception as e:
    print(f"Embedding Failed: {e}")
