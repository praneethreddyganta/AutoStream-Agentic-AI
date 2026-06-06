import os
import sys
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables
load_dotenv()

# Set up Streamlit Page configuration
st.set_page_config(
    page_title="AutoStream Agentic AI - Sales & Support Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom Glassmorphic Styling ------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #0d1117, #07090e);
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Headers & Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Custom Glassmorphic Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    
    /* Brand styling */
    .brand-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Dynamic Intent Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 1px solid transparent;
    }
    .badge-greeting {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.1);
    }
    .badge-pricing {
        background-color: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.1);
    }
    .badge-lead {
        background-color: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.3);
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.1);
    }
    .badge-unknown {
        background-color: rgba(156, 163, 175, 0.1);
        color: #9ca3af;
        border-color: rgba(156, 163, 175, 0.2);
    }
    
    /* Progress bar design */
    .progress-bar-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 9999px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #3b82f6, #f59e0b);
        height: 100%;
        transition: width 0.4s ease;
    }
    
    /* Styled checklist items */
    .checklist-item {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    
    /* Chat message overrides */
    [data-testid="stChatMessage"] {
        background-color: rgba(22, 27, 34, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Scrollbars customization */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Initialization & Config ------------------

def verify_env_key():
    """Checks if GOOGLE_API_KEY is available."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "" or "your_google_ai_studio" in api_key:
        return False
    return True

@st.cache_resource
def load_agent_graph():
    """Initialises ChatGemini, Embeddings, RAG context, and compiles LangGraph StateGraph once."""
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from agent.rag import AutoStreamRAG
    from agent.graph import build_agent_graph
    
    # Instantiate LLM and Embeddings using Google Gemini API
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Assemble local FAISS RAG index
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, "data", "knowledge_base.md")
    rag = AutoStreamRAG(kb_path, embeddings)
    
    # Compile StateGraph state machine
    return build_agent_graph(llm, rag)

# Initialize Session thread_id for state preservation
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Set up graph config
graph_config = {"configurable": {"thread_id": f"autostream-web-{st.session_state.thread_id}"}}

# ------------------ UI Structure ------------------

# Check if environment is configured properly
if not verify_env_key():
    st.error("🔑 Google Gemini API Key Missing")
    st.info(
        "AutoStream Agent requires a valid `GOOGLE_API_KEY` to be set in your `.env` file or environment secrets.\n\n"
        "1. Create a `.env` file in the root folder.\n"
        "2. Add your Gemini key: `GOOGLE_API_KEY=AIzaSy...`\n"
        "3. Restart the Streamlit application."
    )
    st.stop()

# Load the compiled LangGraph agent
try:
    compiled_graph = load_agent_graph()
except Exception as e:
    st.error(f"Failed to compile LangGraph state machine: {e}")
    st.stop()

# Retrieve latest LangGraph state values for this thread
current_state = compiled_graph.get_state(graph_config)
state_values = current_state.values if current_state else {}

# Extract active variables
messages = state_values.get("messages", [])
current_intent = state_values.get("current_intent", "Unknown")
lead_collection_stage = state_values.get("lead_collection_stage", "none")
lead_name = state_values.get("lead_name")
lead_email = state_values.get("lead_email")
lead_platform = state_values.get("lead_platform")

# Calculate lead completion progress metrics
completed_slots = sum([1 for val in [lead_name, lead_email, lead_platform] if val])
progress_percentage = int((completed_slots / 3.0) * 100)

# Sidebar layout
with st.sidebar:
    # Branding header
    st.markdown('<div class="brand-title">⚡ AutoStream Agentic AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.8rem; color: #6b7280; margin-top: -5px; margin-bottom: 25px;">Conversational Sales & Support System</p>', unsafe_allow_html=True)
    
    # Glass card for Lead Qualification Progress
    st.markdown('### 📊 Lead Capture Progress')
    with st.container(border=True):
        st.markdown(f"**Stage:** `{lead_collection_stage.upper()}`")
        
        # Linear progress bar
        st.markdown(f"""
        <div class="progress-bar-container">
            <div class="progress-bar-fill" style="width: {progress_percentage}%;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Checklist items
        name_icon = "🟢 ✓" if lead_name else "⚪"
        name_text = f"**Name:** {lead_name}" if lead_name else "Name: *Pending*"
        st.markdown(f"<div class='checklist-item'>{name_icon} {name_text}</div>", unsafe_allow_html=True)
        
        email_icon = "🟢 ✓" if lead_email else "⚪"
        email_text = f"**Email:** {lead_email}" if lead_email else "Email: *Pending*"
        st.markdown(f"<div class='checklist-item'>{email_icon} {email_text}</div>", unsafe_allow_html=True)
        
        platform_icon = "🟢 ✓" if lead_platform else "⚪"
        platform_text = f"**Platform:** {lead_platform}" if lead_platform else "Platform: *Pending*"
        st.markdown(f"<div class='checklist-item'>{platform_icon} {platform_text}</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # RAG Retrieval Status Indicator Panel
    st.markdown('### 🔍 Retrieval status')
    with st.container(border=True):
        if current_intent == "Product/Pricing Inquiry":
            st.markdown("🌐 **FAISS Vector DB:** `🟢 RETRIEVED`")
            st.markdown("📄 **Index Chunks:** `2 loaded`")
            st.markdown("<p style='font-size: 0.8rem; color: #10b981;'>Last turn retrieved pricing context successfully.</p>", unsafe_allow_html=True)
        else:
            st.markdown("🌐 **FAISS Vector DB:** `⚪ STANDBY`")
            st.markdown("📄 **Index Chunks:** `4 loaded`")
            st.markdown("<p style='font-size: 0.8rem; color: #6b7280;'>Retriever will trigger on pricing inquiries.</p>", unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Session Details card
    st.markdown('### ⚙️ Session details')
    with st.container(border=True):
        st.markdown(f"**Session ID:** `{st.session_state.thread_id[:8]}...`")
        st.markdown(f"**Model:** `gemini-2.5-flash`")
        
        # Clear Conversation / Reset Session button
        if st.button("Reset Conversation", use_container_width=True, type="secondary"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

# Main Interface Area
# Top row metadata banner
col_header, col_badge = st.columns([3, 1])

with col_header:
    st.title("⚡ AutoStream Chat Assistant")
    st.markdown("<p style='color: #9ca3af; margin-top: -15px;'>Ask about our pricing plans, refund policies, or sign up for a Pro plan account.</p>", unsafe_allow_html=True)

with col_badge:
    # Build intent display badges dynamically
    badge_class = "badge-unknown"
    display_intent = "Unknown"
    
    if current_intent == "Greeting":
        badge_class = "badge-greeting"
        display_intent = "Greeting"
    elif current_intent == "Product/Pricing Inquiry":
        badge_class = "badge-pricing"
        display_intent = "Pricing Inquiry"
    elif current_intent == "High-Intent Lead":
        badge_class = "badge-lead"
        display_intent = "Lead Capture"
        
    st.markdown(f"""
    <div style="text-align: right; padding-top: 15px;">
        <span style="font-size: 0.75rem; color: #6b7280; display: block; margin-bottom: 4px;">LAST DETECTED INTENT</span>
        <span class="badge {badge_class}">{display_intent}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Display a success toast and balloons if lead just completed
if lead_collection_stage == "completed" and completed_slots == 3:
    st.success(f"🎉 Lead Captured Successfully! Details: **{lead_name}**, **{lead_email}**, **{lead_platform}**")
    st.balloons()

# Container for holding the chat thread history
chat_container = st.container()

# Render chat messages inside the container
with chat_container:
    if not messages:
        # Pre-populate greeting statically to avoid empty chat screens
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! Welcome to AutoStream, your automated video editing partner. "
                "How can I assist you with your content creation journey today?"
            )
            
        # Quick actions buttons helper for starting
        st.markdown("<p style='font-size: 0.85rem; color: #4b5563; margin-top: 20px;'>Quick Queries:</p>", unsafe_allow_html=True)
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("🏷️ Show Pricing & Features", use_container_width=True):
                st.session_state.quick_query = "Tell me about your pricing."
                st.rerun()
        with col_q2:
            if st.button("💼 Sign Up for Pro Plan", use_container_width=True):
                st.session_state.quick_query = "I want the Pro plan for my YouTube channel."
                st.rerun()
        with col_q3:
            if st.button("💸 Check Refund Policy", use_container_width=True):
                st.session_state.quick_query = "What is your refund policy?"
                st.rerun()
    else:
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

# Logic to handle user chat submissions (from either chat input or quick queries)
user_query = st.chat_input("Ask a question or request a plan...")

# Check if a quick query button was clicked
if "quick_query" in st.session_state and st.session_state.quick_query:
    user_query = st.session_state.quick_query
    # Clean it up immediately to avoid repeating
    st.session_state.quick_query = None

if user_query:
    # Render user query in chat log immediately
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_query)
            
    # Send user message payload to LangGraph state machine
    state_input = {
        "messages": [HumanMessage(content=user_query)]
    }
    
    with st.spinner("Processing..."):
        try:
            compiled_graph.invoke(state_input, graph_config)
        except Exception as e:
            st.error(f"Conversation execution error: {e}")
            
    # Refresh to render latest assistant state and response
    st.rerun()

# Dynamic bottom footer information
st.markdown("""
<div style="text-align: center; font-size: 0.75rem; color: #4b5563; margin-top: 50px; padding-bottom: 20px;">
    AutoStream CRM & Interactive Assistant • Powered by LangGraph + FAISS + Gemini 1.5 Flash
</div>
""", unsafe_allow_html=True)
