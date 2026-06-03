import json
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

def classify_intent(message: str, history_str: str, llm: BaseChatModel) -> str:
    """
    Uses the LLM to classify the user's intent based on the current message and history.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI sales and support routing assistant for AutoStream, a SaaS platform for automated video editing.
Your job is to classify the user's latest message into exactly one of these categories:
1. "Greeting": The user is saying hello, greeting you, or making casual chit-chat (e.g., "Hi", "Hello", "Good morning", "Hey").
2. "Product/Pricing Inquiry": The user is asking about pricing plans, features, limitations, company policies, refunds, or support options.
3. "High-Intent Lead": The user shows a clear intention to purchase, upgrade, subscribe, sign up, or try a premium plan (e.g., "I want to buy the Pro plan", "Sign me up for Pro", "I'd like to get the premium tier", "Let's purchase").
4. "Unknown": The message does not fit any of the above categories.

You MUST respond with a valid JSON object only. Do not output any markdown formatting, backticks (like ```json), or explanatory text.
JSON format:
{{
    "intent": "Greeting" | "Product/Pricing Inquiry" | "High-Intent Lead" | "Unknown",
    "rationale": "Brief 1-sentence explanation of classification"
}}
"""),
        ("human", """Conversation Context:
{history_str}

User's Latest Message: "{message}"
""")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({
            "message": message,
            "history_str": history_str
        })
        
        # Clean up any potential markdown formatting in response content
        content = response.content.strip()
        if content.startswith("```"):
            # Strip ```json ... ```
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        data = json.loads(content)
        intent = data.get("intent", "Unknown")
        if intent in ["Greeting", "Product/Pricing Inquiry", "High-Intent Lead", "Unknown"]:
            return intent
        return "Unknown"
    except Exception as e:
        print(f"[Error in Intent Detection]: {e}. Defaulting to 'Unknown'.")
        return "Unknown"
