import os
import requests
from dotenv import load_dotenv

if os.path.exists(".ignore/.env"):
    load_dotenv(".ignore/.env")
else:
    load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
print("Groq Key exists:", bool(groq_key))
if groq_key:
    # Print length and sanitized representation for debugging without exposing the full key
    print("Groq Key Length:", len(groq_key))
    print("Groq Key Starts With:", groq_key[:8] if len(groq_key) > 8 else groq_key)

headers = {
    "Authorization": f"Bearer {groq_key}",
    "Content-Type": "application/json"
}
payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0
}

try:
    print("Sending POST request to Groq API...")
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=10.0)
    print("Response Status Code:", response.status_code)
    if response.status_code == 200:
        print("Success! Reply:", response.json()["choices"][0]["message"]["content"])
    else:
        print("Failure JSON:", response.json())
except Exception as e:
    print("Network/Request Exception:", e)
