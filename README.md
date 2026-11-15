# AutoStream Agentic AI: Social-to-Lead Conversational Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://autostream-agentic-ai.streamlit.app/)

🚀 **Live App Link:** [https://autostream-agentic-ai.streamlit.app/](https://autostream-agentic-ai.streamlit.app/)

Welcome to the **AutoStream Agentic AI** repository. This project is a production-grade, conversational sales and support agent built for the fictional SaaS company **AutoStream** (operating under the ServiceHive/Inflx platform). 


The agent utilizes **LangGraph** state machines to orchestrate multi-turn slot-filling lead capture workflows, retrieves real-time pricing and features from a local **FAISS Vector Database** using Retrieval-Augmented Generation (RAG), and features a premium, responsive **Streamlit** user interface designed to replicate modern commercial AI products (like ChatGPT and Claude).

---

## 📖 Overview

### What is Agentic AI?
Unlike simple prompt-based chatbots, **Agentic AI** systems are autonomous entities that make decisions, determine execution routing, and call external APIs (tools) based on conversation state. This project models the conversational sales funnel as a state machine. It uses LangGraph to manage complex state transitions deterministically, ensuring that intent is classified accurately, information is collected progressively (slot-filling), and backend capture APIs are triggered only when all lead variables are satisfied.

### Retrieval-Augmented Generation (RAG)
To handle product inquiries, the agent employs **RAG**. When users ask about pricing, features, or refund policies, the agent queries a local vector store containing AutoStream's knowledge database. This retrieved context is injected into the LLM prompt, ensuring the chatbot provides highly accurate, factually grounded answers and completely avoids hallucinating custom plans or pricing details.

### Lead Qualification Funnel
Once the agent detects high user intent (e.g., a desire to upgrade or purchase a subscription), it transitions into a structured **lead qualification workflow**. It progressively prompts for, validates, and stores:
1. **Lead Name**
2. **Lead Email**
3. **Creator Platform** (e.g. YouTube, Instagram, TikTok)

Once all slots are successfully filled, the agent automatically executes the mock CRM lead capture tool to write the customer details to the database.

---

## 🏛️ Architecture

The conversational state graph is structured as follows:

```mermaid
graph TD
    User([User Input]) --> Intent[Intent Detection Node]
    Intent --> Router{Conditional Edge}
    
    Router -- "Greeting" --> ResponseGen[Response Generator Node]
    Router -- "Product/Pricing Inquiry" --> RAG[RAG Retrieval Node]
    Router -- "High-Intent Lead" --> Qualifier[Lead Slot Extraction Node]
    
    Qualifier --> Complete{All Slots Filled?}
    Complete -- "No" --> User
    Complete -- "Yes" --> Tool[Tool Execution Node]
    
    Tool --> END([End Turn / Wait for Input])
    RAG --> END
    ResponseGen --> END
    
    END -.->|Restores Memory via thread_id| User
```

---

## ✨ Features

- **LangGraph State Graph**: Models the entire conversational flow as a state machine with persistent session checkpointer memory.
- **Gemini 3.5 Flash-Lite Integration**: Powered by Google Gemini 3.5 Flash-Lite for ultra-fast, cost-effective intent classification and response generation.
- **Groq GPT-OSS 20B Fallback Engine**: High-throughput fallback model via Groq Cloud ensuring zero-downtime availability.
- **Graceful Degradation Safeguard**: When both providers reach their usage limits, the system politely notifies users instead of throwing exceptions.
- **FAISS Vector Store RAG**: Performs semantic search over local markdown documents using Google Gemini Embedding 2 (`models/gemini-embedding-2`) for pricing and policy queries.
- **Structured Lead Slot-Filling**: Dynamically tracks and extracts parameters progressively over multiple turns.
- **Mock CRM Integration**: Automatically executes backend mock lead capture tools upon slot completion.
- **Premium Streamlit UI**: Modern dark theme with custom glassmorphic cards, linear progress bars, active intent badges, and visual RAG retrieval status indicators.

---

## 🛡️ Resilient Multi-LLM Architecture

AutoStream is built with a 3-tier fallback pipeline:
1. **Primary**: **Google Gemini 3.5 Flash-Lite** handles primary intent routing and conversation flow.
2. **Secondary**: **Groq GPT-OSS 20B** automatically takes over within milliseconds if Google reaches rate limits or encounters downtime.
3. **Terminal**: **Graceful Rate-Limit Notification** returns a courteous notice (*"We are currently experiencing very high demand and our service limit has been reached. Please visit us again a little later..."*) without throwing application errors.

---

## 🛠️ Tech Stack

- **Core Logic**: Python 3.11 / 3.12 / 3.13
- **Orchestration**: LangGraph, LangChain Core
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Primary LLM**: Google Gemini 3.5 Flash-Lite
- **Fallback LLM**: Groq GPT-OSS 20B (`openai/gpt-oss-20b`)
- **Embedding Model**: Google Gemini Embedding 2 (`models/gemini-embedding-2`)
- **Dashboard Interface**: Streamlit

---

## 📂 Project Structure

```text
AutoStream-Agentic-AI/
│
├── app.py                     # CLI Interactive Chatbot Runner
├── streamlit_app.py           # Premium Streamlit GUI Web Dashboard
├── requirements.txt           # Python Package Dependencies
├── requirements-dev.txt       # Adds pytest for the offline test suite
├── pytest.ini                 # Test configuration
├── .gitignore                 # Files excluded from Git
├── .env.example               # Environment Variables template
│
├── agent/                     # Core Agent Module
│   ├── __init__.py
│   ├── service.py             # AgentService: the one entry point used by CLI, UI and backends
│   ├── graph.py               # LangGraph StateGraph state machine structure
│   ├── intents.py             # Rule-based + LLM intent classification, email validation
│   ├── prompts.py             # System prompts (behaviour only, no product facts)
│   ├── llm.py                 # Resilient LLM setup (Gemini 3.5 Flash-Lite + Groq GPT-OSS 20B)
│   ├── rag.py                 # FAISS Loader and semantic retrieval helper
│   ├── state.py               # TypedDict schema definition for Agent State
│   └── tools.py               # Mock CRM API Lead Capture function
│
├── data/                      # Local Knowledge Base (single source of product facts)
│   └── knowledge_base.md      # AutoStream pricing tiers, features, and policies
│
├── tests/                     # Offline pytest suite (no API keys or network needed)
│
├── scripts/                   # Live API smoke checks (use real quota)
│   ├── check_models.py        # Verifies the Gemini chat + embedding models respond
│   ├── check_groq.py          # Verifies the Groq fallback responds
│   └── check_live_graph.py    # Runs a full conversation against the live APIs
│
└── README.md                  # Comprehensive documentation and setup guide

```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11 or 3.12 installed
- A **Google Gemini API Key** (Get a free developer key from [Google AI Studio](https://aistudio.google.com/))
- *(Optional but recommended)* A **Groq API Key** for GPT-OSS 20B fallback (Get from [Groq Console](https://console.groq.com/))

### 1. Clone & Navigate
```bash
git clone https://github.com/praneethreddyganta/AutoStream-Agentic-AI.git
cd AutoStream-Agentic-AI
```

### 2. Set Up Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Configure Secrets
Copy the environment template:
```bash
cp .env.example .env
```
Open `.env` and configure your keys:
```env
GOOGLE_API_KEY=AIzaSyYourActualGoogleGeminiKeyHere
MODEL_NAME=gemini-3.5-flash-lite
GROQ_API_KEY=gsk_YourActualGroqKeyHere
GROQ_MODEL=gpt-oss-20b
EMBEDDING_MODEL=models/gemini-embedding-2
```

### 5. Running the Application
- **Run the CLI Interactive Chatbot**:
  ```bash
  python app.py
  ```
- **Run the Premium Streamlit Dashboard**:
  ```bash
  streamlit run streamlit_app.py
  ```

### 6. Running the Tests
The test suite is fully offline: it uses a fake LLM and retriever, so no API keys are required.
```bash
pip install -r requirements-dev.txt
python -m pytest
```
To smoke-test against the live APIs (uses quota):
```bash
python scripts/check_live_graph.py
```

---

## 🧩 Using the Agent from Your Own Code

Every client goes through `AgentService`, which hides the graph, its memory and its state keys:
```python
from agent.service import AgentService

service = AgentService.from_env()
result = service.chat("session-123", "I want the Pro plan for my YouTube channel.")
print(result.reply, result.intent, result.lead_stage, result.lead, result.retrieved_chunks)

service.get_session("session-123")   # full conversation snapshot (JSON-serialisable via .to_dict())
service.reset("session-123")         # forget one conversation only
```
Pass `lead_capture=your_function` to send completed leads to a real CRM, and `checkpointer=...` to persist conversations outside the process.

**Single source of truth:** all plans, prices and policies live in `data/knowledge_base.md`. The prompts contain no product facts, so editing the knowledge base is all that is needed to change what the agent says (a test enforces this).

---

## ☁️ Streamlit Community Cloud Deployment

To deploy this application to **Streamlit Community Cloud** for production access:

1. Push your repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
3. Click **New app**, select your repository, set the branch to `main`, and set the file path to:
   ```text
   streamlit_app.py
   ```
4. Click **Advanced settings** before deploying. Under the **Secrets** section, configure your environment variables:
   ```toml
   GOOGLE_API_KEY = "AIzaSyYourActualGoogleGeminiKeyHere"
   MODEL_NAME = "gemini-3.5-flash-lite"
   GROQ_API_KEY = "gsk_YourActualGroqKeyHere"
   GROQ_MODEL = "gpt-oss-20b"
   EMBEDDING_MODEL = "models/gemini-embedding-2"
   ```
5. Click **Save** and then **Deploy**. Your app will be live on the web in seconds!

---

## 💬 Example Conversation

```text
User: Hi
Agent: Hello! Welcome to AutoStream, your automated video editing partner. How can I assist you with your content creation journey today?

User: How much does the Pro plan cost?
Agent: The AutoStream Pro Plan costs $79/month. It includes unlimited videos, 4K rendering, AI captions, and 24/7 priority live support.

User: I want the Pro plan for my YouTube channel.
Agent: Great! May I know your name?

User: Praneeth
Agent: Please share your email.

User: praneeth@gmail.com
Agent: Lead captured successfully: Praneeth, praneeth@gmail.com, YouTube
```

---

## 🔮 Future Improvements

- **Omnichannel Webhooks**: Deploy FastAPI endpoints to integrate the agent directly with WhatsApp Business and Telegram.
- **CRM Integration**: Connect the lead capture tool to write details directly to Salesforce, Hubspot, or Notion databases.
- **Multi-Agent Collaboration**: Split the graph into specialized subagents (e.g., an Billing Specialist agent, a Technical Support agent, and a Sales agent) routing queries dynamically.
