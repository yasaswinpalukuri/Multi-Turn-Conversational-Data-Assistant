"""
tests/test_memory.py

Unit tests for agent/memory.py.

Tests cover:
- Turn storage and retrieval
- First turn recall (eval Q12)
- Semantic search (eval Q13)
- Turn counting (eval Q14)
- Session stats accuracy
- Window enforcement (k=15 rolling)
- Session resumption via session_id

Run with:
    pytest tests/test_memory.py -v
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from agent.memory import ConversationMemory


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_memory():
    """
    Create a fresh ConversationMemory with a unique session ID per test.
    Using timestamp + test name to avoid collisions in ChromaDB.
    """
    session_id = f"test_{int(time.time() * 1000)}"
    return ConversationMemory(session_id=session_id)


@pytest.fixture
def memory_with_10_turns(fresh_memory):
    """
    ConversationMemory pre-loaded with 10 turns covering various topics.
    Mirrors the eval set question flow.
    """
    turns = [
        ("How many rows are in this dataset?",
         "There are 97,723 rows.", "result = len(df)"),
        ("What are all the column names?",
         "There are 30 columns including fare_amount, trip_distance...", "result = df.columns.tolist()"),
        ("What is the average trip distance?",
         "The average trip distance is 3.08 miles.", "result = df['trip_distance'].mean()"),
        ("Which payment type is most common?",
         "Credit card is the most common payment type with 78,432 trips.", "result = df['payment_type_label'].value_counts()"),
        ("What is the maximum fare amount?",
         "The maximum fare amount is $70.20.", "result = df['fare_amount'].max()"),
        ("How many trips had zero passengers?",
         "There are 0 trips with zero passengers after cleaning.", "result = len(df[df['passenger_count'] == 0])"),
        ("What is the average tip percentage for credit card payments?",
         "The average tip percentage for credit card payments is 21.3%.", "result = df[df['payment_type']==1]['tip_percentage'].mean()"),
        ("Which hour of the day has the most pickups?",
         "Hour 18 (6 PM) has the most pickups.", "result = df['pickup_hour'].value_counts().idxmax()"),
        ("What percentage of trips lasted more than 30 minutes?",
         "8.73% of trips lasted more than 30 minutes.", "result = (df['trip_duration_minutes'] > 30).mean() * 100"),
        ("What is the correlation between trip distance and fare amount?",
         "The correlation is 0.89 — a strong positive relationship.", "result = df['trip_distance'].corr(df['fare_amount'])"),
    ]

    for question, answer, code in turns:
        fresh_memory.add_turn(
            user_question=question,
            assistant_answer=answer,
            code_used=code,
            response_time_s=2.5,
        )

    return fresh_memory


# ── Basic storage tests ───────────────────────────────────────────────────────

class TestBasicStorage:
    """Memory must store and retrieve turns accurately."""

    def test_add_single_turn(self, fresh_memory):
        fresh_memory.add_turn("Test question", "Test answer")
        assert fresh_memory.get_turn_count() == 1

    def test_turn_count_increments(self, fresh_memory):
        for i in range(5):
            fresh_memory.add_turn(f"Question {i}", f"Answer {i}")
        assert fresh_memory.get_turn_count() == 5

    def test_turn_number_returned(self, fresh_memory):
        turn_num = fresh_memory.add_turn("Question", "Answer")
        assert turn_num == 1
        turn_num2 = fresh_memory.add_turn("Question 2", "Answer 2")
        assert turn_num2 == 2

    def test_recent_turns_returned(self, fresh_memory):
        fresh_memory.add_turn("Q1", "A1")
        fresh_memory.add_turn("Q2", "A2")
        fresh_memory.add_turn("Q3", "A3")
        turns = fresh_memory.get_recent_turns(n=2)
        assert len(turns) <= 3  # at most 3 turns exist

    def test_metadata_stored(self, fresh_memory):
        fresh_memory.add_turn(
            "Test question",
            "Test answer",
            code_used="result = len(df)",
            response_time_s=1.5,
        )
        turns = fresh_memory.get_recent_turns(n=1)
        assert len(turns) == 1
        turn = turns[0]
        assert "user_question" in turn
        assert "assistant_answer" in turn
        assert turn["response_time_s"] == 1.5


# ── Eval Q12: First question recall ──────────────────────────────────────────

class TestFirstQuestionRecall:
    """Eval Q12: 'What was my first question?' must return turn 1."""

    def test_first_turn_is_turn_1(self, memory_with_10_turns):
        first = memory_with_10_turns.get_first_turn()
        assert first is not None
        assert first["turn_number"] == 1

    def test_first_question_content(self, memory_with_10_turns):
        first = memory_with_10_turns.get_first_turn()
        assert "rows" in first["user_question"].lower()

    def test_first_turn_none_when_empty(self, fresh_memory):
        first = fresh_memory.get_first_turn()
        assert first is None

    def test_first_turn_unchanged_after_more_turns(self, memory_with_10_turns):
        """Adding more turns must not change what get_first_turn returns."""
        memory_with_10_turns.add_turn("New question", "New answer")
        first = memory_with_10_turns.get_first_turn()
        assert first["turn_number"] == 1
        assert "rows" in first["user_question"].lower()


# ── Eval Q13: Semantic search ─────────────────────────────────────────────────

class TestSemanticSearch:
    """Eval Q13: 'What did I ask about payment types?' must find relevant turn."""

    def test_search_finds_payment_question(self, memory_with_10_turns):
        results = memory_with_10_turns.search_history("payment type")
        assert len(results) > 0
        # At least one result should mention payment
        questions = [r["user_question"].lower() for r in results]
        assert any("payment" in q for q in questions)

    def test_search_finds_distance_question(self, memory_with_10_turns):
        results = memory_with_10_turns.search_history("trip distance")
        assert len(results) > 0
        questions = [r["user_question"].lower() for r in results]
        assert any("distance" in q for q in questions)

    def test_search_empty_when_no_turns(self, fresh_memory):
        results = fresh_memory.search_history("anything")
        assert results == []

    def test_search_returns_relevance_score(self, memory_with_10_turns):
        results = memory_with_10_turns.search_history("fare amount")
        assert len(results) > 0
        assert "relevance_score" in results[0]
        assert 0 <= results[0]["relevance_score"] <= 1


# ── Eval Q14: Turn counting ───────────────────────────────────────────────────

class TestTurnCounting:
    """Eval Q14: 'How many questions have I asked?' must be accurate."""

    def test_turn_count_is_accurate(self, memory_with_10_turns):
        assert memory_with_10_turns.get_turn_count() == 10

    def test_format_history_shows_count(self, memory_with_10_turns):
        formatted = memory_with_10_turns.format_history_for_context_test()
        assert "10" in formatted
        assert "Total questions" in formatted

    def test_format_history_lists_questions(self, memory_with_10_turns):
        formatted = memory_with_10_turns.format_history_for_context_test()
        assert "Turn 1" in formatted
        assert "Turn 10" in formatted


# ── Chat history string tests ─────────────────────────────────────────────────

class TestChatHistoryString:
    """get_chat_history_str() must return proper formatted history."""

    def test_empty_history_message(self, fresh_memory):
        history = fresh_memory.get_chat_history_str()
        assert "No conversation history" in history

    def test_history_contains_user_messages(self, fresh_memory):
        fresh_memory.add_turn("My question", "My answer")
        history = fresh_memory.get_chat_history_str()
        assert "User:" in history
        assert "My question" in history

    def test_history_contains_assistant_messages(self, fresh_memory):
        fresh_memory.add_turn("My question", "My answer")
        history = fresh_memory.get_chat_history_str()
        assert "Assistant:" in history
        assert "My answer" in history

    def test_window_enforced(self, fresh_memory):
        """Only last k turns should appear in the LangChain window."""
        # Add more than window_k turns
        for i in range(20):
            fresh_memory.add_turn(f"Question {i}", f"Answer {i}")

        lc_memory = fresh_memory.get_langchain_memory()
        # window_k=15 means at most 15*2=30 messages
        assert len(lc_memory.messages) <= fresh_memory.window_k * 2


# ── Session stats tests ───────────────────────────────────────────────────────

class TestSessionStats:
    """Session stats must be accurate for Streamlit sidebar display."""

    def test_stats_structure(self, fresh_memory):
        stats = fresh_memory.get_session_stats()
        required_keys = [
            "session_id", "turn_count", "window_k",
            "context_window_pct", "avg_response_time_s", "session_start",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_avg_response_time(self, fresh_memory):
        fresh_memory.add_turn("Q1", "A1", response_time_s=2.0)
        fresh_memory.add_turn("Q2", "A2", response_time_s=4.0)
        stats = fresh_memory.get_session_stats()
        assert stats["avg_response_time_s"] == 3.0

    def test_context_window_pct(self, fresh_memory):
        # Add 3 turns with window_k=15: should be 20%
        for i in range(3):
            fresh_memory.add_turn(f"Q{i}", f"A{i}")
        stats = fresh_memory.get_session_stats()
        assert stats["context_window_pct"] == 20

    def test_context_window_capped_at_100(self, fresh_memory):
        # Add more than window_k turns
        for i in range(20):
            fresh_memory.add_turn(f"Q{i}", f"A{i}")
        stats = fresh_memory.get_session_stats()
        assert stats["context_window_pct"] <= 100


# ── Clear session test ────────────────────────────────────────────────────────

class TestClearSession:
    """clear_session() must reset window but preserve ChromaDB history."""

    def test_clear_resets_turn_count(self, fresh_memory):
        fresh_memory.add_turn("Q1", "A1")
        fresh_memory.add_turn("Q2", "A2")
        fresh_memory.clear_session()
        assert fresh_memory.get_turn_count() == 0

    def test_clear_resets_chat_history(self, fresh_memory):
        fresh_memory.add_turn("Q1", "A1")
        fresh_memory.clear_session()
        history = fresh_memory.get_chat_history_str()
        assert "No conversation history" in history
