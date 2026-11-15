import pytest

from agent.intents import extract_email, is_valid_email, parse_json_reply, rule_based_intent

LEAD = "High-Intent Lead"
PRICING = "Product/Pricing Inquiry"


@pytest.mark.parametrize("message", [
    "That sounds good, I want to try the Pro plan for my YouTube channel.",
    "I want the Pro plan for my YouTube channel.",
    "Sign me up for Pro",
    "I'd like to buy the basic plan",
    "Can I purchase the Pro plan?",
    "let's go with the pro plan",
    "I want to subscribe",
])
def test_purchase_intent_beats_pricing_keywords(message):
    assert rule_based_intent(message) == LEAD


@pytest.mark.parametrize("message", [
    "Tell me about your pricing.",
    "How much does the Pro plan cost?",
    "What is your refund policy?",
    "I want to know the price of the pro plan",
    "How much is it to buy Pro?",
    "Which plan has 4K?",
    "Do you offer a free trial?",
])
def test_pricing_questions(message):
    assert rule_based_intent(message) == PRICING


@pytest.mark.parametrize("message", [
    "I have a problem with my account",   # contains "pro"
    "Can you generate subtitles",          # contains "rate"
    "I feel great today",                  # contains "fee"
    "My feedback is positive",             # contains "fee"
    "We are planning a launch",            # contains "plan"
    "What's the weather like",
])
def test_substrings_do_not_trigger_rules(message):
    assert rule_based_intent(message) is None


@pytest.mark.parametrize("message", ["hi", "Hello!", "hey there", "Good morning"])
def test_greetings(message):
    assert rule_based_intent(message) == "Greeting"


@pytest.mark.parametrize("message", ["support@acme.com", "costa@gmail.com", "my email is pro.plan@pricing.io"])
@pytest.mark.parametrize("stage", ["ask_name", "ask_email", "ask_platform"])
def test_email_is_always_a_slot_answer_while_collecting(message, stage):
    assert rule_based_intent(message, stage) == LEAD


def test_plain_answer_while_collecting_is_a_slot_answer():
    assert rule_based_intent("Praneeth", "ask_name") == LEAD
    assert rule_based_intent("YouTube", "ask_platform") == LEAD


def test_question_while_collecting_is_still_a_question():
    assert rule_based_intent("wait, what is the refund policy?", "ask_email") == PRICING


def test_extract_email():
    assert extract_email("reach me at a.b+c@mail.example.com.") == "a.b+c@mail.example.com"
    assert extract_email("no address here") is None


@pytest.mark.parametrize("value,expected", [
    ("a@b.co", True), ("praneeth@gmail", False), ("not an email", False), ("", False), (None, False),
])
def test_is_valid_email(value, expected):
    assert is_valid_email(value) is expected


def test_parse_json_reply_handles_code_fences():
    assert parse_json_reply('```json\n{"intent": "Greeting"}\n```') == {"intent": "Greeting"}
    with pytest.raises(Exception):
        parse_json_reply("sorry, I cannot do that")
