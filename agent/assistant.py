"""
agent/assistant.py

Core orchestrator for the Multi-Turn Conversational Data Assistant.

Wires together:
- ChatOllama (qwen2.5:7b via langchain-ollama)
- Four tools from agent/tools.py
- Two-layer memory from agent/memory.py
- System prompt + few-shot examples from agent/prompts.py
- Retry logic for failed code generation
- Response time tracking for sidebar metrics

Usage
-----
    from agent.assistant import DataAssistant
    assistant = DataAssistant()
    response = assistant.chat("How many rows are in this dataset?")
    print(response.answer)
    print(response.code_used)
    print(response.response_time_s)

Migration note
--------------
DataAssistant has no Streamlit dependency. The same class plugs into
LangGraph as a node by calling assistant.chat() inside a node function.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from agent.memory import ConversationMemory
from agent.prompts import build_system_prompt, build_retry_prompt
from agent.tools import (
    execute_pandas_code,
    sql_query_tool,
    schema_tool,
    stats_tool,
    _strip_code_fences,
    ALL_TOOLS,
)
from utils.data_loader import get_dataframe, get_schema_summary

load_dotenv()
logger = logging.getLogger(__name__)


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class AssistantResponse:
    """
    Structured response from a single chat() call.

    Attributes
    ----------
    answer : str
        The final answer shown to the user.
    code_used : str
        The pandas/SQL code that produced the answer (empty if N/A).
    tool_used : str
        Which tool was called: 'pandas', 'sql', 'schema', 'stats', 'memory', 'none'.
    response_time_s : float
        Wall-clock seconds from question received to answer ready.
    turn_number : int
        Which turn in the session this was.
    error : str
        Non-empty if something went wrong (shown to user in friendly form).
    raw_llm_output : str
        Full raw LLM output for debugging (not shown in UI).
    """
    answer: str          = ""
    code_used: str       = ""
    tool_used: str       = "none"
    response_time_s: float = 0.0
    turn_number: int     = 0
    error: str           = ""
    raw_llm_output: str  = ""


# ── DataAssistant ─────────────────────────────────────────────────────────────

class DataAssistant:
    """
    Multi-turn conversational data assistant backed by a local Ollama LLM.

    Parameters
    ----------
    session_id : str | None
        Pass an existing session ID to resume a previous conversation.
        Leave None to start a fresh session.
    memory : ConversationMemory | None
        Inject an existing memory instance (useful for testing).
    """

    def __init__(
        self,
        session_id: str | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:

        # Config from .env
        self._base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._model      = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self._temp       = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
        self._num_ctx    = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

        # Memory
        self.memory = memory or ConversationMemory(session_id=session_id)

        # LLM — initialised lazily to give a clean error if Ollama is down
        self._llm: ChatOllama | None = None

        # System prompt (built once, schema injected at construction)
        self._system_prompt = build_system_prompt()

        # Preload the dataframe so first query isn't slow
        try:
            get_dataframe()
            logger.info("DataAssistant ready — dataframe preloaded")
        except FileNotFoundError as e:
            logger.error(f"Dataset not found: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_question: str) -> AssistantResponse:
        """
        Process one user question and return a structured response.

        This is the single entry point called by Streamlit (and LangGraph).
        Handles tool selection, code execution, retry on failure, and memory.

        Parameters
        ----------
        user_question : str
            Raw question from the user.

        Returns
        -------
        AssistantResponse
            Structured response with answer, code, timing, and metadata.
        """
        start_time = time.time()

        # Ensure LLM is reachable
        llm, error = self._get_llm()
        if error:
            return AssistantResponse(
                answer=error,
                error=error,
                response_time_s=time.time() - start_time,
                turn_number=self.memory.get_turn_count() + 1,
            )

        # Route the question
        response = self._route_and_answer(user_question, llm)
        response.response_time_s = round(time.time() - start_time, 2)

        # Save to memory
        response.turn_number = self.memory.add_turn(
            user_question=user_question,
            assistant_answer=response.answer,
            code_used=response.code_used,
            response_time_s=response.response_time_s,
        )

        logger.info(
            f"Turn {response.turn_number} | tool={response.tool_used} "
            f"| time={response.response_time_s}s | q='{user_question[:60]}'"
        )

        return response

    def is_ollama_reachable(self) -> bool:
        """
        Quick health check — returns True if Ollama is running.

        Used by Streamlit to show a warning banner if the LLM is down.
        """
        _, error = self._get_llm()
        return error == ""

    def get_session_stats(self) -> dict[str, Any]:
        """Return session statistics for the Streamlit sidebar."""
        return self.memory.get_session_stats()

    def clear_session(self) -> None:
        """Clear the in-session memory window (keeps ChromaDB history)."""
        self.memory.clear_session()

    def export_chat_json(self) -> str:
        """
        Export the full session history as a JSON string.

        Returns
        -------
        str
            JSON string of all turns in this session.
        """
        import json
        turns = self.memory.get_recent_turns(n=999)
        turns_sorted = sorted(turns, key=lambda x: x.get("turn_number", 0))
        export = {
            "session_id": self.memory.session_id,
            "turns": turns_sorted,
        }
        return json.dumps(export, indent=2, default=str)

    # ── Routing logic ──────────────────────────────────────────────────────────

    def _route_and_answer(
        self,
        user_question: str,
        llm: ChatOllama,
    ) -> AssistantResponse:
        """
        Decide which tool to use and execute the full answer pipeline.

        Routing priority:
        1. Memory questions → answered from chat history, no tool call
        2. Schema/stats questions → direct tool call
        3. Everything else → LLM generates pandas or SQL, we execute it

        Parameters
        ----------
        user_question : str
            The user's raw question.
        llm : ChatOllama
            Initialised LLM instance.

        Returns
        -------
        AssistantResponse
            Populated response (without response_time_s, set by caller).
        """
        q_lower = user_question.lower().strip()

        # ── Route 1: Memory questions ─────────────────────────────────────────
        if self._is_memory_question(q_lower):
            return self._answer_from_memory(user_question)

        # ── Route 2: Schema question ──────────────────────────────────────────
        if self._is_schema_question(q_lower):
            result = schema_tool.invoke("full")
            answer = self._llm_format_answer(user_question, result, llm)
            return AssistantResponse(answer=answer, tool_used="schema")

        # ── Route 3: Stats question ───────────────────────────────────────────
        col = self._extract_stats_column(q_lower)
        if col:
            result = stats_tool.invoke(col)
            answer = self._llm_format_answer(user_question, result, llm)
            return AssistantResponse(answer=answer, tool_used="stats")

        # ── Route 4: SQL question ─────────────────────────────────────────────
        if self._is_sql_question(q_lower):
            return self._answer_with_sql(user_question, llm)

        # ── Route 5: Default — pandas code generation ─────────────────────────
        return self._answer_with_pandas(user_question, llm)

    # ── Memory routing ─────────────────────────────────────────────────────────

    def _is_memory_question(self, q_lower: str) -> bool:
        """Return True if the question is about conversation history."""
        memory_signals = [
            "first question", "what did i ask", "what have i asked",
            "how many questions", "what was my", "previous question",
            "summarize everything", "summarise everything",
            "what did we discuss", "what have we discussed",
            "what did i say", "recall", "remember when",
            "earlier i asked", "how many turns",
        ]
        return any(signal in q_lower for signal in memory_signals)

    def _answer_from_memory(self, user_question: str) -> AssistantResponse:
        """
        Answer a memory/history question from ChromaDB + chat history.

        Handles three cases:
        1. "What was my first question?" → get_first_turn()
        2. "How many questions have I asked?" → get_turn_count()
        3. "What did I ask about X?" → search_history(X)
        4. "Summarize everything" → get_recent_turns(all)
        """
        q_lower = user_question.lower()

        # Case 1: First question
        if "first question" in q_lower or "first ask" in q_lower:
            first = self.memory.get_first_turn()
            if first:
                answer = (
                    f"Your first question was:\n\n"
                    f"**\"{first['user_question']}\"**\n\n"
                    f"_(Asked at turn 1, {first.get('timestamp', '')[:19]})_"
                )
            else:
                answer = "You haven't asked any questions yet in this session."
            return AssistantResponse(answer=answer, tool_used="memory")

        # Case 2: Turn count
        if any(p in q_lower for p in ["how many questions", "how many turns", "how many have i"]):
            count = self.memory.get_turn_count()
            summary = self.memory.format_history_for_context_test()
            answer = (
                f"You have asked **{count} question{'s' if count != 1 else ''}** "
                f"so far in this session.\n\n{summary}"
            )
            return AssistantResponse(answer=answer, tool_used="memory")

        # Case 3: Summarize everything
        if any(p in q_lower for p in ["summarize", "summarise", "what have we discussed", "everything we"]):
            turns = self.memory.get_recent_turns(n=999)
            turns_sorted = sorted(turns, key=lambda x: x.get("turn_number", 0))
            if not turns_sorted:
                answer = "We haven't discussed anything yet."
            else:
                lines = ["Here's a summary of everything we've discussed:\n"]
                for t in turns_sorted:
                    lines.append(
                        f"**Turn {t['turn_number']}**: {t['user_question']}\n"
                        f"→ {t['assistant_answer'][:200]}{'...' if len(t.get('assistant_answer','')) > 200 else ''}\n"
                    )
                answer = "\n".join(lines)
            return AssistantResponse(answer=answer, tool_used="memory")

        # Case 4: What did I ask about X? — semantic search
        search_results = self.memory.search_history(user_question, n_results=3)
        if search_results:
            lines = [f"Here's what I found in our conversation history:\n"]
            for r in search_results:
                lines.append(
                    f"**Turn {r['turn_number']}**: You asked: \"{r['user_question']}\"\n"
                    f"→ {r['assistant_answer'][:200]}{'...' if len(r.get('assistant_answer','')) > 200 else ''}\n"
                )
            answer = "\n".join(lines)
        else:
            answer = "I couldn't find anything relevant in our conversation history."

        return AssistantResponse(answer=answer, tool_used="memory")

    # ── Pandas routing ─────────────────────────────────────────────────────────

    def _answer_with_pandas(
        self,
        user_question: str,
        llm: ChatOllama,
    ) -> AssistantResponse:
        """
        Ask the LLM to generate pandas code, execute it, retry once on failure.

        Returns
        -------
        AssistantResponse
            With answer, code_used, tool_used='pandas', and error if both attempts failed.
        """
        schema = get_schema_summary()

        # Step 1: Ask LLM to generate pandas code
        code = self._generate_pandas_code(user_question, schema, llm)
        if not code:
            return AssistantResponse(
                answer="I wasn't able to generate code for that question. Could you rephrase it?",
                error="LLM returned no code",
                tool_used="pandas",
            )

        # Step 2: Execute the code
        result_str = execute_pandas_code.invoke(code)

        # Step 3: Check for errors and retry once
        if result_str.startswith("CODE_ERROR") or result_str.startswith("SECURITY"):
            logger.warning(f"First attempt failed: {result_str[:100]}")
            error_msg = result_str.split("\n")[0]

            retry_prompt = build_retry_prompt(
                original_question=user_question,
                failed_code=code,
                error_message=error_msg,
            )
            fixed_code = self._ask_llm_for_code(retry_prompt, llm)

            if fixed_code:
                result_str = execute_pandas_code.invoke(fixed_code)
                code = fixed_code  # Update code to the fixed version

            if result_str.startswith("CODE_ERROR") or result_str.startswith("SECURITY"):
                return AssistantResponse(
                    answer=(
                        "I tried twice but couldn't compute that answer. "
                        f"The error was: {error_msg}\n\n"
                        "Try rephrasing your question, or ask me to show the schema first."
                    ),
                    code_used=code,
                    tool_used="pandas",
                    error=error_msg,
                )

        # Step 4: Format the result into a natural language answer
        answer = self._llm_format_answer(user_question, result_str, llm)

        # Extract the code from the result string
        code_used = self._extract_code_from_result(result_str, code)

        return AssistantResponse(
            answer=answer,
            code_used=code_used,
            tool_used="pandas",
        )

    def _generate_pandas_code(
        self,
        question: str,
        schema: dict,
        llm: ChatOllama,
    ) -> str:
        """Ask the LLM to write pandas code for a question."""
        prompt = (
            f"Write Python/pandas code to answer this question about the NYC Taxi dataset:\n"
            f"Question: {question}\n\n"
            f"Available columns: {schema['columns']}\n"
            f"Derived columns available: {schema['derived_columns']}\n\n"
            f"Rules:\n"
            f"- DataFrame is 'df', assign answer to 'result'\n"
            f"- Do not import anything\n"
            f"- Use payment_type_label for readable payment names\n"
            f"- For time: use pickup_hour, pickup_day_of_week columns\n\n"
            f"Return ONLY the Python code, nothing else. No explanation."
        )
        return self._ask_llm_for_code(prompt, llm)

    # ── SQL routing ────────────────────────────────────────────────────────────

    def _answer_with_sql(
        self,
        user_question: str,
        llm: ChatOllama,
    ) -> AssistantResponse:
        """Generate and execute a SQL query."""
        schema = get_schema_summary()

        sql_prompt = (
            f"Write a SQL SELECT query to answer: {user_question}\n\n"
            f"Table name: trips\n"
            f"Available columns: {schema['columns']}\n\n"
            f"Rules:\n"
            f"- Only SELECT statements\n"
            f"- No subqueries deeper than 2 levels\n"
            f"- Return ONLY the SQL, no explanation\n"
        )

        sql = self._ask_llm_for_code(sql_prompt, llm)
        if not sql:
            return self._answer_with_pandas(user_question, llm)

        result_str = sql_query_tool.invoke(sql)

        if result_str.startswith("SQL_ERROR"):
            # Fall back to pandas on SQL failure
            logger.warning(f"SQL failed, falling back to pandas: {result_str[:100]}")
            return self._answer_with_pandas(user_question, llm)

        answer = self._llm_format_answer(user_question, result_str, llm)
        return AssistantResponse(
            answer=answer,
            code_used=sql,
            tool_used="sql",
        )

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _get_llm(self) -> tuple[ChatOllama | None, str]:
        """
        Initialise and return the ChatOllama instance.

        Returns (llm, "") on success, (None, error_message) if Ollama is down.
        Uses a cached instance after first successful connection.
        """
        if self._llm is not None:
            return self._llm, ""

        try:
            llm = ChatOllama(
                base_url=self._base_url,
                model=self._model,
                temperature=self._temp,
                num_ctx=self._num_ctx,
            )
            # Ping with a tiny request to confirm it's alive
            llm.invoke([HumanMessage(content="hi")])
            self._llm = llm
            logger.info(f"Ollama connected: {self._model} at {self._base_url}")
            return self._llm, ""

        except Exception as e:
            error = (
                f"⚠️ Cannot connect to Ollama at {self._base_url}.\n\n"
                f"Please ensure Ollama is running:\n"
                f"```\nollama serve\n```\n"
                f"And that the model is pulled:\n"
                f"```\nollama pull {self._model}\n```\n\n"
                f"Technical detail: {str(e)}"
            )
            logger.error(f"Ollama connection failed: {e}")
            return None, error

    def _ask_llm_for_code(self, prompt: str, llm: ChatOllama) -> str:
        """
        Send a code-generation prompt to the LLM and return clean code.

        Parameters
        ----------
        prompt : str
            The code generation prompt.
        llm : ChatOllama
            Connected LLM instance.

        Returns
        -------
        str
            Cleaned code string (markdown fences stripped), or "" on failure.
        """
        try:
            messages = [
                SystemMessage(content=(
                    "You are a Python/pandas code generator. "
                    "Return ONLY executable Python code, nothing else. "
                    "No markdown, no explanation, no comments."
                )),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            return _strip_code_fences(raw)
        except Exception as e:
            logger.error(f"LLM code generation failed: {e}")
            return ""

    def _llm_format_answer(
        self,
        question: str,
        raw_result: str,
        llm: ChatOllama,
    ) -> str:
        """
        Ask the LLM to turn a raw tool result into a natural language answer.

        Parameters
        ----------
        question : str
            The original user question.
        raw_result : str
            The raw output from execute_pandas_code or sql_query_tool.
        llm : ChatOllama
            Connected LLM instance.

        Returns
        -------
        str
            Clean natural language answer for the user.
        """
        history = self.memory.get_chat_history_str()

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=(
                f"Conversation so far:\n{history}\n\n"
                f"The user asked: {question}\n\n"
                f"The data tool returned this result:\n{raw_result}\n\n"
                f"Write a clear, concise answer in 1-3 sentences. "
                f"Include the key number or finding. "
                f"Format numbers with commas and appropriate units ($, %, mph). "
                f"Do NOT repeat the raw result verbatim — interpret it."
            )),
        ]

        try:
            response = llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"LLM formatting failed: {e}")
            # Fall back to returning the raw result if LLM formatting fails
            return raw_result

    # ── Question classifiers ───────────────────────────────────────────────────

    def _is_schema_question(self, q_lower: str) -> bool:
        """Return True if the question is about dataset structure."""
        schema_signals = [
            "what columns", "column names", "what fields",
            "show schema", "describe the dataset", "what data",
            "available columns", "list columns", "what is available",
        ]
        return any(s in q_lower for s in schema_signals)

    def _is_sql_question(self, q_lower: str) -> bool:
        """Return True if SQL is a better fit than pandas for this question."""
        sql_signals = [
            "top 5", "top 10", "rank", "group by",
            "per location", "by location", "by day",
            "join", "having", "distinct",
        ]
        return any(s in q_lower for s in sql_signals)

    def _extract_stats_column(self, q_lower: str) -> str:
        """
        If the question is about stats for a specific column, return that column name.
        Returns "" if not a stats question.
        """
        schema = get_schema_summary()
        stats_triggers = ["distribution", "statistics", "stats for", "describe", "percentile"]

        if not any(t in q_lower for t in stats_triggers):
            return ""

        # Check if a known column name appears in the question
        for col in schema["columns"]:
            if col.lower() in q_lower:
                return col
        return ""

    def _extract_code_from_result(self, result_str: str, fallback_code: str) -> str:
        """
        Extract the CODE_USED section from execute_pandas_code output.

        Parameters
        ----------
        result_str : str
            Raw output from execute_pandas_code tool.
        fallback_code : str
            Use this if CODE_USED section is not found.

        Returns
        -------
        str
            The code string.
        """
        if "CODE_USED:" in result_str:
            parts = result_str.split("CODE_USED:")
            if len(parts) > 1:
                return parts[1].strip()
        return fallback_code
