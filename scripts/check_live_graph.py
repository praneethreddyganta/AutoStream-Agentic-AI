"""Live smoke test: runs a full conversation against the real Gemini/Groq APIs (uses quota)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

if os.path.exists(".ignore/.env"):
    load_dotenv(".ignore/.env")
else:
    load_dotenv()

from agent.service import AgentService

TURNS = [
    "Hi",
    "Tell me about your pricing.",
    "I want the Pro plan for my YouTube channel.",
    "Praneeth",
    "praneeth@example.com",
]

service = AgentService.from_env()
for turn in TURNS:
    result = service.chat("live-check-session", turn)
    print(f"\nUser: {turn}")
    print(f"[intent={result.intent} | stage={result.lead_stage} | kb_chunks={len(result.retrieved_chunks)}]")
    print(f"Bot: {result.reply}")

print("\nLead captured:", result.lead_stage == "completed")
