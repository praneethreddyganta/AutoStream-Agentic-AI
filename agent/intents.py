import json
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from agent.prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT

def classify_intent(message: str, history_str: str, llm: BaseChatModel) -> str:
    """
    Uses the LLM to classify the user's intent based on the current message and history.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_CLASSIFIER_SYSTEM_PROMPT),
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
