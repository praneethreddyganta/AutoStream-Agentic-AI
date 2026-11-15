"""End-to-end conversation tests through AgentService with a fake LLM and fake retriever."""
from agent.llm import DEFAULT_LIMIT_REACHED_MESSAGE
from tests.conftest import KB_CHUNK

LEAD = "High-Intent Lead"
PRICING = "Product/Pricing Inquiry"


def test_headline_demo_flow_captures_lead(harness):
    r = harness.say("Hi")
    assert r.intent == "Greeting" and r.lead_stage == "none"

    r = harness.say("Tell me about your pricing.")
    assert r.intent == PRICING and r.retrieved_chunks == [KB_CHUNK]

    r = harness.say("That sounds good, I want to try the Pro plan for my YouTube channel.")
    assert r.intent == LEAD
    assert r.lead_stage == "ask_name" and r.lead.platform == "YouTube"
    assert r.retrieved_chunks == []

    r = harness.say("Praneeth")
    assert r.lead_stage == "ask_email" and r.lead.name == "Praneeth"
    assert not r.lead_captured

    r = harness.say("praneeth@gmail.com")
    assert r.lead_stage == "completed" and r.lead_captured
    assert r.reply == "Lead captured successfully: Praneeth, praneeth@gmail.com, YouTube"
    assert harness.captured == [("Praneeth", "praneeth@gmail.com", "YouTube")]


def test_rules_skip_the_intent_llm_call(harness):
    harness.say("Hi")
    harness.say("Tell me about your pricing.")
    harness.say("sign me up")
    assert "intent" not in harness.llm.calls


def test_ambiguous_message_falls_back_to_llm_classifier(make_harness):
    h = make_harness(intent=LEAD)
    r = h.say("I have a problem, can someone take my details")
    assert h.llm.calls[0] == "intent"
    assert r.intent == LEAD and r.lead_stage == "ask_name"


def test_emails_containing_pricing_words_are_captured(harness):
    harness.say("sign me up")
    harness.say("Praneeth")
    r = harness.say("support@acme.com")
    assert r.intent == LEAD and r.lead.email == "support@acme.com"
    assert r.lead_stage == "ask_platform"


def test_invalid_email_is_rejected_with_clear_message(harness):
    harness.say("sign me up")
    harness.say("Praneeth")
    r = harness.say("praneeth at gmail")
    assert r.lead.email is None and r.lead_stage == "ask_email"
    assert "valid email" in r.reply
    assert harness.captured == []


def test_question_during_signup_is_answered_then_signup_resumes(harness):
    harness.say("sign me up")
    harness.say("Praneeth")
    r = harness.say("what is the refund policy?")
    assert r.intent == PRICING and r.lead_stage == "ask_email"
    assert r.reply.startswith("LLM_REPLY") and "Please share your email." in r.reply

    r = harness.say("p@example.com")
    r = harness.say("TikTok")
    assert r.lead_captured and harness.captured == [("Praneeth", "p@example.com", "TikTok")]


def test_lead_is_captured_only_once(harness):
    for message in ["sign me up", "Praneeth", "p@example.com", "YouTube"]:
        r = harness.say(message)
    assert r.lead_captured

    r = harness.say("I want to buy it again")
    assert not r.lead_captured and r.lead_stage == "completed"
    assert len(harness.captured) == 1


def test_all_details_in_one_message(harness):
    harness.say("sign me up")
    r = harness.say("p@example.com")
    assert r.lead_stage == "ask_name" and r.lead.email == "p@example.com"


def test_quota_exhausted_during_signup_is_not_silent(make_harness):
    h = make_harness(limit_reached=True)
    r = h.say("sign me up")
    assert r.reply == DEFAULT_LIMIT_REACHED_MESSAGE and r.lead_stage == "ask_name"


def test_signup_still_progresses_without_llm(make_harness):
    h = make_harness(limit_reached=True)
    h.say("sign me up")
    assert h.say("Praneeth Reddy").lead.name == "Praneeth Reddy"
    assert h.say("p@example.com").lead.email == "p@example.com"
    r = h.say("I post on Instagram")
    assert r.lead_captured and h.captured == [("Praneeth Reddy", "p@example.com", "Instagram")]


def test_unknown_messages_get_grounded_context(harness):
    r = harness.say("tell me a joke")
    assert r.intent == "Unknown"
    assert r.retrieved_chunks == [KB_CHUNK]
    assert KB_CHUNK in harness.llm.prompts[-1]  # the retrieved context reaches the model


def test_sessions_are_isolated_and_reset_affects_one_session(harness):
    harness.say("sign me up", session="alice")
    harness.say("Alice", session="alice")
    harness.say("Hi", session="bob")

    assert harness.service.get_session("alice").lead.name == "Alice"
    assert harness.service.get_session("bob").lead.name is None

    harness.service.reset("bob")
    assert harness.service.get_session("bob").messages == []
    assert harness.service.get_session("alice").lead.name == "Alice"
    assert len(harness.service.get_session("alice").messages) == 4


def test_snapshot_is_json_serialisable(harness):
    import json
    r = harness.say("Hi")
    data = json.loads(json.dumps(r.to_dict()))
    assert data["messages"][0] == {"role": "user", "content": "Hi"}
    assert data["lead"] == {"name": None, "email": None, "platform": None}
