import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from agent.prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT
from agent.llm import extract_text_content

VALID_INTENTS = ["Greeting", "Product/Pricing Inquiry", "High-Intent Lead", "Unknown"]
SLOT_STAGES = ("ask_name", "ask_email", "ask_platform")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

GREETINGS = {"hi", "hello", "hey", "hii", "hey there", "good morning", "good afternoon", "good evening"}


def _words(*alternatives: str) -> re.Pattern:
    """Compiles a whole-word matcher, so 'pro' never matches 'problem'."""
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


# Explicit purchase verbs: a lead unless the user is really asking what it costs.
STRONG_LEAD_RE = _words(
    r"buy", r"buying", r"purchase", r"purchasing", r"sign\s+me\s+up", r"sign\s*up",
    r"signing\s+up", r"subscribe", r"subscribing", r"checkout", r"check\s+out",
    r"get\s+started", r"upgrade\s+me",
)

# "I want / I'll take / let's go with ... <plan>": a lead unless it reads as a question.
SOFT_LEAD_RE = re.compile(
    r"\b(?:want|need|take|get|try|choose|pick|go\s+with|going\s+with|upgrade\s+to|"
    r"switch\s+to|start\s+with|interested\s+in|ready\s+for)\b"
    r".{0,40}?\b(?:pro|basic|plan|plans|subscription|tier|premium|trial)\b",
    re.IGNORECASE,
)

# The user wants a number or a policy, not a checkout.
COST_QUESTION_RE = _words(
    r"how\s+much", r"price", r"prices", r"pricing", r"cost", r"costs", r"refund", r"refunds",
)

# The user is asking for information rather than stating a decision.
INFO_SEEKING_RE = _words(
    r"know", r"tell", r"explain", r"learn", r"compare", r"comparison", r"difference",
    r"differences", r"details", r"what", r"which", r"how", r"does", r"do\s+you", r"is\s+there",
)

PRICING_RE = _words(
    r"price", r"prices", r"pricing", r"cost", r"costs", r"plan", r"plans", r"tier", r"tiers",
    r"free", r"refund", r"refunds", r"support", r"feature", r"features", r"how\s+much",
    r"basic", r"pro", r"standard", r"money-back", r"guarantee", r"amount", r"rate", r"rates",
    r"charge", r"charges", r"fee", r"fees", r"pay", r"payment", r"payments", r"package",
    r"packages", r"quote", r"subscription", r"discount", r"discounts", r"4k", r"720p",
    r"captions", r"resolution", r"policy", r"policies", r"trial",
)


def extract_email(text: str) -> str | None:
    """Returns the first well-formed email address in the text, if any."""
    match = EMAIL_RE.search(text or "")
    return match.group(0).rstrip(".") if match else None


def is_valid_email(value) -> bool:
    return isinstance(value, str) and EMAIL_RE.fullmatch(value.strip()) is not None


def _looks_like_question(text: str) -> bool:
    return "?" in text or COST_QUESTION_RE.search(text) is not None or INFO_SEEKING_RE.search(text) is not None


def rule_based_intent(message: str, stage: str = "none") -> str | None:
    """
    Classifies the message with deterministic rules (no API quota used).
    Returns None when the rules are not confident and the LLM should decide.
    """
    text = (message or "").strip()
    clean = text.lower()

    if clean.rstrip("!.,") in GREETINGS:
        return "Greeting"

    if stage in SLOT_STAGES:
        # An email address is always an answer to our question, whatever words it contains.
        if extract_email(text):
            return "High-Intent Lead"
        # Anything that is not a question is treated as the answer to the pending slot.
        if not _looks_like_question(clean) and not PRICING_RE.search(clean):
            return "High-Intent Lead"

    # Purchase intent is checked BEFORE pricing words, so "I want the Pro plan" is a lead.
    if STRONG_LEAD_RE.search(clean) and not COST_QUESTION_RE.search(clean):
        return "High-Intent Lead"
    if SOFT_LEAD_RE.search(clean) and not _looks_like_question(clean):
        return "High-Intent Lead"

    if PRICING_RE.search(clean):
        return "Product/Pricing Inquiry"

    return None


def _strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def parse_json_reply(raw) -> dict:
    """Parses a JSON object from an LLM reply, tolerating markdown code fences."""
    data = json.loads(_strip_code_fence(extract_text_content(raw)))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


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
        intent = parse_json_reply(response.content).get("intent", "Unknown")
        return intent if intent in VALID_INTENTS else "Unknown"
    except Exception as e:
        print(f"[Error in Intent Detection]: {e}.")
        return rule_based_intent(message) or "Unknown"
