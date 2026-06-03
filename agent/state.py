import operator
from typing import Annotated, Sequence, TypedDict, Literal

def add_messages(left: list, right: list) -> list:
    """
    Custom reducer for adding messages to conversation history.
    """
    # Simple list addition. Standard langgraph message reducer behaves similarly.
    return left + right

class AgentState(TypedDict):
    """
    Represents the state of our conversational agent.
    """
    # The history of chat messages in this conversation.
    # Annotated with add_messages so that returning new messages appends them to the history.
    messages: Annotated[list, add_messages]
    
    # The current classified intent of the user.
    # Can be: "Greeting", "Product/Pricing Inquiry", "High-Intent Lead", or "Unknown"
    current_intent: Literal["Greeting", "Product/Pricing Inquiry", "High-Intent Lead", "Unknown"]
    
    # Slot-filling data for capturing leads
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    
    # Tracks which field we are currently trying to collect.
    # "none": No active capture occurring.
    # "ask_name": Prompting/Waiting for the user's name.
    # "ask_email": Prompting/Waiting for the user's email.
    # "ask_platform": Prompting/Waiting for their content platform (e.g. YouTube, IG, TikTok).
    # "completed": All fields are captured and the mock capture tool is executed.
    lead_collection_stage: Literal["none", "ask_name", "ask_email", "ask_platform", "completed"]
