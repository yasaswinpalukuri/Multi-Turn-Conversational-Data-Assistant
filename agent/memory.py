"""
agent/memory.py

Two-layer conversation memory for the data assistant:

Layer 1 — In-session rolling window
    LangChain ConversationBufferWindowMemory with k=15 turns.
    This is what the LLM sees in its context window on every call.
    Fast, in-RAM, resets when the process ends.

Layer 2 — ChromaDB persistence
    Every turn is written to a local ChromaDB collection so:
    - Sessions can be resumed after restart
    - The agent can answer "what was my first question?" accurately
    - Turn metadata (timestamp, turn number, code used) is preserved

Design decisions
----------------
- ChromaDB uses its built-in DefaultEmbeddingFunction (onnxruntime/CPU)
  instead of sentence-transformers to avoid CUDA conflicts with Ollama.
- Each turn is stored as a single ChromaDB document with the format:
  "User: <question>\nAssistant: <answer>" plus metadata.
- Session IDs are timestamp-based so multiple sessions never collide.
- The class is Streamlit-agnostic — no st.* calls anywhere.

Migration note
--------------
ConversationMemory plugs directly into LangGraph as a state manager.
Pass get_langchain_memory() to any LangGraph node that needs history.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PERSIST_DIR = ".chroma_store"
DEFAULT_COLLECTION  = "conversation_history"
DEFAULT_WINDOW_K    = 15
DEFAULT_SESSION_PFX = "session"


# ── ConversationMemory ────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Two-layer memory: LangChain rolling window + ChromaDB persistence.

    Usage
    -----
    memory = ConversationMemory()
    memory.add_turn("How many rows?", "There are 97,723 rows.", code="result = len(df)")
    history = memory.get_recent_turns(5)
    full    = memory.search_history("first question")

    Parameters
    ----------
    session_id : str | None
        Unique ID for this session. Auto-generated if not provided.
        Pass an existing ID to resume a previous session.
    persist_dir : str | None
        Directory for ChromaDB storage. Defaults to .env CHROMA_PERSIST_DIR.
    collection_name : str | None
        ChromaDB collection name. Defaults to .env CHROMA_COLLECTION.
    window_k : int | None
        Number of turns to keep in the rolling LangChain window.
        Defaults to .env MEMORY_WINDOW_K.
    """

    def __init__(
        self,
        session_id: str | None = None,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        window_k: int | None = None,
    ) -> None:
        # Session identity
        prefix = os.getenv("SESSION_PREFIX", DEFAULT_SESSION_PFX)
        self.session_id: str = session_id or (
            f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Config
        self.persist_dir: str = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR
        )
        self.collection_name: str = collection_name or os.getenv(
            "CHROMA_COLLECTION", DEFAULT_COLLECTION
        )
        self.window_k: int = window_k or int(
            os.getenv("MEMORY_WINDOW_K", DEFAULT_WINDOW_K)
        )

        # Turn counter
        self._turn_count: int = 0

        # Layer 1: LangChain in-session window
        # Using ChatMessageHistory directly (ConversationBufferWindowMemory
        # is deprecated in langchain 0.3.x — migrating early avoids breakage)
        self._chat_history = ChatMessageHistory()

        # Layer 2: ChromaDB persistence
        self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self._embedding_fn  = embedding_functions.DefaultEmbeddingFunction()
        self._collection    = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Load existing turn count for this session from ChromaDB
        self._turn_count = self._load_session_turn_count()

        logger.info(
            f"ConversationMemory initialised | session={self.session_id} "
            f"| window_k={self.window_k} | existing_turns={self._turn_count}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(
        self,
        user_question: str,
        assistant_answer: str,
        code_used: str = "",
        response_time_s: float = 0.0,
    ) -> int:
        """
        Record one conversation turn to both memory layers.

        Parameters
        ----------
        user_question : str
            The user's raw question.
        assistant_answer : str
            The assistant's full response.
        code_used : str
            The pandas/SQL code that produced the answer (empty if N/A).
        response_time_s : float
            Time in seconds the response took (for sidebar metrics).

        Returns
        -------
        int
            The turn number just recorded (1-indexed).
        """
        self._turn_count += 1
        turn_number = self._turn_count
        timestamp   = datetime.now().isoformat()

        # Layer 1: add to ChatMessageHistory
        self._chat_history.add_user_message(user_question)
        self._chat_history.add_ai_message(assistant_answer)

        # Layer 2: persist to ChromaDB
        doc_id   = f"{self.session_id}_turn_{turn_number:04d}"
        document = f"User: {user_question}\nAssistant: {assistant_answer}"
        metadata = {
            "session_id":       self.session_id,
            "turn_number":      turn_number,
            "timestamp":        timestamp,
            "user_question":    user_question[:500],   # ChromaDB metadata limit
            "assistant_answer": assistant_answer[:500],
            "code_used":        code_used[:500] if code_used else "",
            "response_time_s":  round(response_time_s, 3),
        }

        self._collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata],
        )

        logger.debug(
            f"Turn {turn_number} saved | session={self.session_id} "
            f"| q='{user_question[:60]}'"
        )
        return turn_number

    def get_langchain_memory(self) -> ChatMessageHistory:
        """
        Return the ChatMessageHistory object for passing to the agent chain.

        The window (last k turns) is enforced here — only the most recent
        window_k * 2 messages (k user + k assistant) are kept in context.

        Returns
        -------
        ChatMessageHistory
            Rolling window message history.
        """
        # Enforce window: keep only last k turns (2 messages per turn)
        all_messages = self._chat_history.messages
        max_messages = self.window_k * 2
        if len(all_messages) > max_messages:
            windowed = ChatMessageHistory()
            for msg in all_messages[-max_messages:]:
                windowed.messages.append(msg)
            return windowed
        return self._chat_history

    def get_chat_history_str(self) -> str:
        """
        Return the current rolling window as a plain string for injection
        into prompts that don't use the LangChain memory object directly.

        Returns
        -------
        str
            Formatted chat history, newest last.
        """
        messages = self._chat_history.messages
        if not messages:
            return "No conversation history yet."

        lines: list[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")
        return "\n".join(lines)

    def get_recent_turns(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Return the n most recent turns from ChromaDB with full metadata.

        Parameters
        ----------
        n : int
            Number of recent turns to return.

        Returns
        -------
        list[dict]
            List of turn dicts with keys: turn_number, timestamp,
            user_question, assistant_answer, code_used, response_time_s.
        """
        if self._turn_count == 0:
            return []

        # Query ChromaDB for turns from this session, sorted by turn number
        results = self._collection.get(
            where={"session_id": self.session_id},
            include=["metadatas"],
        )

        if not results["metadatas"]:
            return []

        turns = sorted(
            results["metadatas"],
            key=lambda x: x.get("turn_number", 0),
            reverse=True,
        )
        return turns[:n]

    def search_history(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """
        Semantic search across all stored turns in this session.

        Used by the agent to answer questions like:
        "What was my first question?" or "What did I ask about payments?"

        Parameters
        ----------
        query : str
            Natural language search query.
        n_results : int
            Max number of matching turns to return.

        Returns
        -------
        list[dict]
            Matching turns sorted by relevance, each with metadata.
        """
        if self._turn_count == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._turn_count),
                where={"session_id": self.session_id},
                include=["metadatas", "distances"],
            )

            turns = []
            for meta, dist in zip(
                results["metadatas"][0],
                results["distances"][0],
            ):
                turn = dict(meta)
                turn["relevance_score"] = round(1 - dist, 4)  # cosine → similarity
                turns.append(turn)

            return sorted(turns, key=lambda x: x.get("turn_number", 0))

        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
            return []

    def get_first_turn(self) -> dict[str, Any] | None:
        """
        Return the very first turn of this session.

        Used to answer "What was my first question?".

        Returns
        -------
        dict | None
            First turn metadata, or None if no turns yet.
        """
        if self._turn_count == 0:
            return None

        results = self._collection.get(
            where={
                "$and": [
                    {"session_id": self.session_id},
                    {"turn_number": 1},
                ]
            },
            include=["metadatas"],
        )

        if results["metadatas"]:
            return results["metadatas"][0]
        return None

    def get_turn_count(self) -> int:
        """
        Return the total number of turns in the current session.

        Returns
        -------
        int
            Turn count (0 if no turns yet).
        """
        return self._turn_count

    def get_session_stats(self) -> dict[str, Any]:
        """
        Return session-level statistics for the Streamlit sidebar.

        Returns
        -------
        dict
            Keys: session_id, turn_count, window_k, context_window_pct,
            avg_response_time_s, session_start.
        """
        recent = self.get_recent_turns(n=self._turn_count or 1)

        avg_response = 0.0
        if recent:
            times = [t.get("response_time_s", 0.0) for t in recent]
            avg_response = round(sum(times) / len(times), 2) if times else 0.0

        session_start = ""
        if recent:
            oldest = min(recent, key=lambda x: x.get("turn_number", 999))
            session_start = oldest.get("timestamp", "")[:19]

        context_pct = min(100, round((self._turn_count / self.window_k) * 100))

        return {
            "session_id":          self.session_id,
            "turn_count":          self._turn_count,
            "window_k":            self.window_k,
            "context_window_pct":  context_pct,
            "avg_response_time_s": avg_response,
            "session_start":       session_start,
        }

    def clear_session(self) -> None:
        """
        Clear the in-session LangChain window memory.

        Does NOT delete ChromaDB records — those are permanent audit history.
        Call this when the user clicks "Clear conversation" in the UI.
        """
        self._chat_history.clear()
        self._turn_count = 0
        logger.info(f"Session window cleared | session={self.session_id}")

    def format_history_for_context_test(self) -> str:
        """
        Format the last 5 questions as a numbered list.

        Used by the agent to answer eval question 14:
        "How many questions have I asked so far?"

        Returns
        -------
        str
            Numbered list of recent questions.
        """
        recent = self.get_recent_turns(n=self._turn_count or 1)
        if not recent:
            return "No questions asked yet."

        sorted_turns = sorted(recent, key=lambda x: x.get("turn_number", 0))
        lines = [f"Total questions asked: {self._turn_count}", ""]
        for turn in sorted_turns:
            n   = turn.get("turn_number", "?")
            q   = turn.get("user_question", "")
            lines.append(f"  Turn {n}: {q}")
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_session_turn_count(self) -> int:
        """
        On init, check ChromaDB for existing turns in this session.
        Allows resuming a session by passing the same session_id.

        Returns
        -------
        int
            Number of existing turns found, or 0 for a new session.
        """
        try:
            results = self._collection.get(
                where={"session_id": self.session_id},
                include=["metadatas"],
            )
            if results["metadatas"]:
                max_turn = max(
                    m.get("turn_number", 0) for m in results["metadatas"]
                )
                logger.info(
                    f"Resumed session '{self.session_id}' "
                    f"with {max_turn} existing turns"
                )
                return max_turn
        except Exception as e:
            logger.warning(f"Could not load session turn count: {e}")
        return 0
