"""
tests/test_tools.py

Unit tests for agent/tools.py.

Tests cover:
- All four tools return expected output format
- Security sandbox blocks all dangerous patterns
- Security sandbox allows all safe patterns
- Code fence stripping works correctly
- SQL tool rejects non-SELECT statements
- Error messages are clean (no raw tracebacks)

Run with:
    pytest tests/test_tools.py -v
"""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from agent.tools import (
    execute_pandas_code,
    sql_query_tool,
    schema_tool,
    stats_tool,
    _is_code_safe,
    _strip_code_fences,
    _execute_pandas_code,
)
from utils.data_loader import get_dataframe


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    """Load dataframe once for all tests in this module."""
    return get_dataframe()


# ── Security sandbox tests ────────────────────────────────────────────────────

class TestSecuritySandbox:
    """Gate 4: Security — malicious code must be blocked."""

    BLOCKED_CASES = [
        ("import os", "os module import"),
        ("import sys", "sys module import"),
        ("import subprocess", "subprocess import"),
        ("from os import path", "from-os import"),
        ("__import__('os')", "dunder import"),
        ("open('/etc/passwd').read()", "file open"),
        ("exec('import os')", "nested exec"),
        ("eval('1+1')", "eval call"),
        ("__class__", "dunder class access"),
        ("import pathlib", "pathlib import"),
        ("import socket", "socket import"),
    ]

    SAFE_CASES = [
        ("result = df['fare_amount'].mean()", "mean aggregation"),
        ("result = df.groupby('payment_type_label').size()", "groupby"),
        ("result = df['trip_distance'].describe()", "describe"),
        ("result = len(df)", "len of df"),
        ("result = df['fare_amount'].max()", "max"),
        ("result = df[df['trip_distance'] > 5]['speed_mph'].mean()", "filtered mean"),
        ("result = df['pickup_hour'].value_counts()", "value counts"),
        ("result = df['trip_distance'].corr(df['fare_amount'])", "correlation"),
    ]

    @pytest.mark.parametrize("code,description", BLOCKED_CASES)
    def test_blocked_patterns(self, code, description):
        """Dangerous code patterns must be blocked before exec."""
        safe, reason = _is_code_safe(code)
        assert not safe, (
            f"SECURITY FAILURE: '{description}' was not blocked.\n"
            f"Code: {code}\n"
            f"This is a critical security vulnerability."
        )

    @pytest.mark.parametrize("code,description", SAFE_CASES)
    def test_safe_patterns_allowed(self, code, description):
        """Safe pandas code must not be blocked by the sandbox."""
        safe, reason = _is_code_safe(code)
        assert safe, (
            f"Safe code was incorrectly blocked: '{description}'\n"
            f"Code: {code}\n"
            f"Reason: {reason}"
        )

    def test_os_system_blocked(self, df):
        """The original Gate 4 test: os.system must be blocked at execution."""
        result, error = _execute_pandas_code(
            "import os; result = os.system('echo hacked')", df
        )
        assert result is None
        assert "Blocked" in error or "Security" in error or "blocked" in error.lower()

    def test_open_file_blocked(self, df):
        """File open must be blocked."""
        result, error = _execute_pandas_code(
            "result = open('/etc/passwd').read()", df
        )
        assert result is None
        assert error != ""


# ── Code fence stripper tests ─────────────────────────────────────────────────

class TestCodeFenceStripper:
    """_strip_code_fences must handle all LLM output formats."""

    def test_python_fence(self):
        code = "```python\nresult = df['fare_amount'].mean()\n```"
        assert _strip_code_fences(code) == "result = df['fare_amount'].mean()"

    def test_plain_fence(self):
        code = "```\nresult = len(df)\n```"
        assert _strip_code_fences(code) == "result = len(df)"

    def test_no_fence(self):
        code = "result = df['trip_distance'].max()"
        assert _strip_code_fences(code) == code

    def test_sql_fence(self):
        sql = "```sql\nSELECT COUNT(*) FROM trips\n```"
        assert _strip_code_fences(sql) == "SELECT COUNT(*) FROM trips"


# ── execute_pandas_code tool tests ────────────────────────────────────────────

class TestPandasTool:
    """Tool 1: execute_pandas_code must return ANSWER and CODE_USED."""

    def test_row_count(self):
        result = execute_pandas_code.invoke("result = len(df)")
        assert "ANSWER" in result
        assert "97" in result  # 97,723 rows

    def test_mean_fare(self):
        result = execute_pandas_code.invoke("result = df['fare_amount'].mean()")
        assert "ANSWER" in result
        # Should be a number between 10 and 30
        import re
        numbers = re.findall(r'\d+\.\d+', result)
        assert numbers, "No numeric result found"
        fare = float(numbers[0])
        assert 10 < fare < 30, f"Average fare {fare} outside expected range"

    def test_max_fare(self):
        result = execute_pandas_code.invoke("result = df['fare_amount'].max()")
        assert "ANSWER" in result

    def test_payment_type_value_counts(self):
        result = execute_pandas_code.invoke(
            "result = df['payment_type_label'].value_counts()"
        )
        assert "ANSWER" in result
        assert "Credit card" in result

    def test_code_used_in_output(self):
        code = "result = df['trip_distance'].mean()"
        result = execute_pandas_code.invoke(code)
        assert "CODE_USED" in result

    def test_invalid_column_returns_error(self):
        result = execute_pandas_code.invoke("result = df['nonexistent_column'].mean()")
        assert "CODE_ERROR" in result or "ERROR" in result.upper()
        # Must NOT show a raw Python traceback — only clean message
        assert "Traceback" not in result

    def test_result_not_assigned_returns_error(self):
        result = execute_pandas_code.invoke("x = df['fare_amount'].mean()")
        # Should report that result wasn't set
        assert "CODE_ERROR" in result or "result" in result.lower()


# ── sql_query_tool tests ──────────────────────────────────────────────────────

class TestSQLTool:
    """Tool 2: sql_query_tool must execute SELECT and block everything else."""

    def test_basic_count(self):
        result = sql_query_tool.invoke("SELECT COUNT(*) as total FROM trips")
        assert "ANSWER" in result
        assert "97" in result  # 97,723 rows

    def test_avg_fare(self):
        result = sql_query_tool.invoke("SELECT AVG(fare_amount) FROM trips")
        assert "ANSWER" in result

    def test_top_5_locations(self):
        result = sql_query_tool.invoke(
            "SELECT PULocationID, COUNT(*) as cnt FROM trips "
            "GROUP BY PULocationID ORDER BY cnt DESC LIMIT 5"
        )
        assert "ANSWER" in result
        assert "PULocationID" in result

    def test_sql_used_in_output(self):
        sql = "SELECT MAX(fare_amount) FROM trips"
        result = sql_query_tool.invoke(sql)
        assert "SQL_USED" in result

    def test_non_select_blocked(self):
        result = sql_query_tool.invoke("DROP TABLE trips")
        assert "SQL_ERROR" in result

    def test_delete_blocked(self):
        result = sql_query_tool.invoke("DELETE FROM trips WHERE 1=1")
        assert "SQL_ERROR" in result

    def test_insert_blocked(self):
        result = sql_query_tool.invoke("INSERT INTO trips VALUES (1,2,3)")
        assert "SQL_ERROR" in result


# ── schema_tool tests ─────────────────────────────────────────────────────────

class TestSchemaTool:
    """Tool 3: schema_tool must return accurate schema information."""

    def test_full_schema(self):
        result = schema_tool.invoke("full")
        assert "DATASET" in result
        assert "97,723" in result
        assert "fare_amount" in result
        assert "trip_duration_minutes" in result  # derived column

    def test_columns_only(self):
        result = schema_tool.invoke("columns")
        assert "COLUMNS" in result
        assert "tpep_pickup_datetime" in result
        assert "payment_type_label" in result

    def test_sample_rows(self):
        result = schema_tool.invoke("sample")
        assert "SAMPLE" in result

    def test_numeric_columns(self):
        result = schema_tool.invoke("numeric")
        assert "NUMERIC" in result
        assert "fare_amount" in result
        assert "min=" in result

    def test_payment_map_shown(self):
        result = schema_tool.invoke("full")
        assert "Credit card" in result
        assert "Cash" in result


# ── stats_tool tests ──────────────────────────────────────────────────────────

class TestStatsTool:
    """Tool 4: stats_tool must return descriptive stats for valid columns."""

    def test_fare_amount_stats(self):
        result = stats_tool.invoke("fare_amount")
        assert "STATISTICS FOR: fare_amount" in result
        assert "mean" in result
        assert "median" in result
        assert "min" in result
        assert "max" in result

    def test_payment_type_label_stats(self):
        result = stats_tool.invoke("payment_type_label")
        assert "STATISTICS FOR: payment_type_label" in result
        assert "Credit card" in result

    def test_invalid_column_error(self):
        result = stats_tool.invoke("nonexistent_column")
        assert "STATS_ERROR" in result
        assert "not found" in result.lower() or "available" in result.lower()

    def test_trip_duration_stats(self):
        result = stats_tool.invoke("trip_duration_minutes")
        assert "mean" in result
        # Average trip ~15 min
        import re
        numbers = re.findall(r'mean:\s*([\d.]+)', result)
        if numbers:
            mean_duration = float(numbers[0])
            assert 5 < mean_duration < 60, f"Mean duration {mean_duration} outside expected range"
