"""
app.py

Streamlit entry point for the Multi-Turn Conversational Data Assistant.

Run with:
    streamlit run app.py

UI layout:
- Left sidebar: dataset info, session stats, controls
- Main area: chat bubbles with expandable code sections
- Every answer shows response time and the code/SQL that produced it
"""

# ── Silence noisy warnings before any imports ─────────────────────────────────
import os
import warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*model_fields.*")
warnings.filterwarnings("ignore", message=".*asyncio_default_fixture_loop_scope.*")

import json
import time
import logging
import sys

import streamlit as st

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Data Assistant",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.assistant import DataAssistant, AssistantResponse
from utils.data_loader import get_dataframe, get_memory_usage

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3748;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #1a1f2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px !important;
    }

    /* User message bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #1e3a5f;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 8px;
    }

    /* Assistant message bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #1a2332;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 8px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 8px 12px;
    }

    /* Response time text */
    .response-time {
        color: #718096;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* Code blocks */
    .stCodeBlock {
        background-color: #0d1117 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #161b27;
        border: 1px solid #2d3748;
        border-radius: 8px;
    }

    /* Sidebar headers */
    .sidebar-section {
        color: #a0aec0;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 16px 0 8px 0;
    }

    /* Divider */
    hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def init_session_state() -> None:
    """Initialise all session state variables on first run."""
    if "assistant" not in st.session_state:
        st.session_state.assistant = DataAssistant()

    if "messages" not in st.session_state:
        # Each message: {role, content, code_used, tool_used, response_time_s}
        st.session_state.messages = []

    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    if "response_times" not in st.session_state:
        st.session_state.response_times = []


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the left sidebar with dataset info, stats, and controls."""
    with st.sidebar:
        st.markdown("## 🚕 Data Assistant")
        st.markdown("*NYC Yellow Taxi · Jan 2023*")
        st.divider()

        # ── Dataset info ──────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">Dataset</p>', unsafe_allow_html=True)
        try:
            mem_info = get_memory_usage()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", mem_info["row_count"])
            with col2:
                st.metric("Columns", mem_info["column_count"])
            st.metric("Memory", mem_info["total_mb"])
        except Exception:
            st.warning("Dataset not loaded yet")

        st.divider()

        # ── Session stats ─────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">Session</p>', unsafe_allow_html=True)

        assistant: DataAssistant = st.session_state.assistant
        stats = assistant.get_session_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Turns", stats["turn_count"])
        with col2:
            st.metric("Queries", st.session_state.total_queries)

        avg_time = (
            round(sum(st.session_state.response_times) /
                  len(st.session_state.response_times), 2)
            if st.session_state.response_times else 0.0
        )
        st.metric("Avg Response", f"{avg_time}s")

        # Context window progress bar
        st.markdown("**Context Window**")
        ctx_pct = stats["context_window_pct"]
        st.progress(
            ctx_pct / 100,
            text=f"{stats['turn_count']}/{stats['window_k']} turns ({ctx_pct}%)"
        )

        st.divider()

        # ── Ollama status ─────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">Model</p>', unsafe_allow_html=True)
        ollama_ok = assistant.is_ollama_reachable()
        if ollama_ok:
            st.success("✓ Ollama connected", icon="🟢")
        else:
            st.error("✗ Ollama offline", icon="🔴")
            st.code("ollama serve", language="bash")

        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        st.caption(f"Model: `{model_name}`")
        st.caption(f"Temp: `{os.getenv('OLLAMA_TEMPERATURE', '0.1')}`")
        st.caption(f"CTX: `{os.getenv('OLLAMA_NUM_CTX', '8192')}` tokens")

        st.divider()

        # ── Controls ──────────────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">Controls</p>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.response_times = []
            assistant.clear_session()
            st.rerun()

        if st.button("📥 Export Chat as JSON", use_container_width=True):
            json_str = assistant.export_chat_json()
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"chat_{stats['session_id']}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()

        # ── Quick questions ───────────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">Quick Questions</p>', unsafe_allow_html=True)
        quick_questions = [
            "How many rows are in this dataset?",
            "What is the average fare amount?",
            "Which payment type is most common?",
            "Which hour has the most pickups?",
            "What was my first question?",
        ]
        for q in quick_questions:
            if st.button(q, use_container_width=True, key=f"quick_{q[:20]}"):
                st.session_state["pending_question"] = q
                st.rerun()


# ── Chat area ─────────────────────────────────────────────────────────────────

def render_chat_history() -> None:
    """Render all previous messages in the chat area."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Assistant messages get extra metadata
            if msg["role"] == "assistant":
                # Expandable code section
                code = msg.get("code_used", "")
                tool = msg.get("tool_used", "none")

                if code and tool not in ("memory", "schema", "none"):
                    lang = "sql" if tool == "sql" else "python"
                    label = "🔍 Show SQL" if tool == "sql" else "🔍 Show Code"
                    with st.expander(label):
                        st.code(code, language=lang)

                # Response time
                resp_time = msg.get("response_time_s", 0)
                if resp_time:
                    st.markdown(
                        f'<p class="response-time">⏱ {resp_time}s · {tool}</p>',
                        unsafe_allow_html=True,
                    )


def render_welcome() -> None:
    """Show welcome message on first load."""
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #718096;">
            <h2 style="color: #e2e8f0;">🚕 NYC Taxi Data Assistant</h2>
            <p style="font-size: 1.1rem;">
                Ask me anything about 97,723 Yellow Taxi trips from January 2023.
            </p>
            <p>Try: <em>"What is the average fare amount?"</em><br>
            or: <em>"Which hour of the day has the most pickups?"</em><br>
            or: <em>"Show me the top 5 pickup locations"</em></p>
            <hr style="border-color: #2d3748; margin: 24px 40px;">
            <p style="font-size: 0.85rem;">
                Powered by <strong>qwen2.5:7b</strong> via Ollama · 
                LangChain · ChromaDB · pandas
            </p>
        </div>
        """, unsafe_allow_html=True)


def process_question(question: str) -> None:
    """
    Process a user question and add both messages to session state.

    Parameters
    ----------
    question : str
        The user's question string.
    """
    assistant: DataAssistant = st.session_state.assistant

    # Add user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })
    st.session_state.total_queries += 1

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Show assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response: AssistantResponse = assistant.chat(question)

        # Render the answer
        st.markdown(response.answer)

        # Render numeric results or DataFrames if present
        # (the answer text handles formatting; this handles any structured data)

        # Expandable code section
        if response.code_used and response.tool_used not in ("memory", "schema", "none"):
            lang = "sql" if response.tool_used == "sql" else "python"
            label = "🔍 Show SQL" if response.tool_used == "sql" else "🔍 Show Code"
            with st.expander(label):
                st.code(response.code_used, language=lang)

        # Response time
        st.markdown(
            f'<p class="response-time">⏱ {response.response_time_s}s · {response.tool_used}</p>',
            unsafe_allow_html=True,
        )

        # Error banner if something went wrong
        if response.error:
            st.warning(f"⚠️ {response.error}", icon="⚠️")

    # Track response time for sidebar
    st.session_state.response_times.append(response.response_time_s)

    # Save to session state for history rendering
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "code_used": response.code_used,
        "tool_used": response.tool_used,
        "response_time_s": response.response_time_s,
    })


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Main Streamlit app entry point."""
    init_session_state()
    render_sidebar()

    # Main header
    st.markdown("### 🚕 Multi-Turn Conversational Data Assistant")
    st.caption(
        "NYC Yellow Taxi · January 2023 · "
        "Powered by qwen2.5:7b · Remembers up to 15 turns"
    )
    st.divider()

    # Ollama warning banner (shown prominently if offline)
    assistant: DataAssistant = st.session_state.assistant
    if not assistant.is_ollama_reachable():
        st.error(
            "⚠️ **Ollama is not running.** Start it with `ollama serve` "
            "in a terminal, then refresh this page.",
            icon="🔴",
        )
        st.stop()

    # Welcome screen or chat history
    render_welcome()
    render_chat_history()

    # Handle quick question button clicks from sidebar
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        process_question(question)
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask anything about the NYC Taxi data..."):
        process_question(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
