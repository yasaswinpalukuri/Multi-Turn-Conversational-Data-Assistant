"""
agent/tools.py

Four LangChain tools the agent uses to answer questions about the NYC Taxi dataset:

1. pandas_query_tool  — generates + executes pandas code, returns result + code
2. sql_query_tool     — generates + executes SQLite query, returns result + SQL
3. schema_tool        — returns column names, dtypes, ranges, sample rows
4. stats_tool         — descriptive statistics for any column

Security model
--------------
All LLM-generated code runs inside exec() with a RESTRICTED globals dict.
The sandbox physically prevents:
  - import of os, sys, subprocess, pathlib, shutil, socket, and others
  - open() file access
  - __import__ calls
  - access to __builtins__ beyond a safe whitelist

Every exec() attempt (allowed or blocked) is logged to logs/executions.log.

Migration note
--------------
These tools are plain Python classes with zero Streamlit dependency.
They plug directly into LangGraph nodes without modification.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain.tools import tool
from dotenv import load_dotenv

from utils.data_loader import get_dataframe, get_schema_summary, get_column_stats

load_dotenv()
logger = logging.getLogger(__name__)

# ── Execution logger setup ────────────────────────────────────────────────────

def _get_exec_logger() -> logging.Logger:
    """
    Return a dedicated logger that writes every exec() attempt to
    logs/executions.log regardless of the root log level.
    """
    exec_logger = logging.getLogger("exec_audit")
    if exec_logger.handlers:
        return exec_logger  # already configured

    log_path = Path(os.getenv("EXEC_LOG_PATH", "logs/executions.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    exec_logger.setLevel(logging.DEBUG)
    exec_logger.addHandler(handler)
    exec_logger.propagate = False
    return exec_logger


EXEC_LOGGER = _get_exec_logger()


# ── Security sandbox ──────────────────────────────────────────────────────────

# Modules that must never be importable inside the sandbox
BLOCKED_MODULES: set[str] = {
    "os", "sys", "subprocess", "pathlib", "shutil", "socket",
    "http", "urllib", "requests", "ftplib", "smtplib", "builtins",
    "importlib", "pkgutil", "ctypes", "multiprocessing", "threading",
    "tempfile", "glob", "fnmatch", "pickle", "shelve", "dbm",
    "pty", "tty", "termios", "signal", "resource", "gc",
}

# Safe builtins the sandbox exposes — nothing that touches the filesystem,
# network, or process control
SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset,
    "getattr": getattr, "hasattr": hasattr, "hash": hash,
    "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "object": object,
    "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}

# Dangerous code patterns — regex checked before exec()
BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bimport\s+(" + "|".join(BLOCKED_MODULES) + r")\b"),
    re.compile(r"\bfrom\s+(" + "|".join(BLOCKED_MODULES) + r")\b"),
    re.compile(r"\b__import__\s*\("),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bcompile\s*\("),
    re.compile(r"\bgetattr\s*\(.*__"),   # getattr(x, '__class__') style escapes
    re.compile(r"__[a-zA-Z]+__"),         # dunder access (covers __class__, __globals__, etc.)
    re.compile(r"\bbreakpoint\s*\("),
    re.compile(r"\binput\s*\("),
]


def _is_code_safe(code: str) -> tuple[bool, str]:
    """
    Check generated code against the blocked patterns list.

    Parameters
    ----------
    code : str
        The code string to validate.

    Returns
    -------
    tuple[bool, str]
        (True, "") if safe, or (False, reason) if blocked.
    """
    for pattern in BLOCKED_PATTERNS:
        match = pattern.search(code)
        if match:
            return False, f"Blocked pattern detected: '{match.group()}'"

    # Extra check: code must reference 'df' (the dataframe variable)
    # Pure arithmetic or string manipulation without df is suspicious
    if "df" not in code and "result" not in code:
        return False, "Code does not reference the dataframe variable 'df'"

    return True, ""


def _execute_pandas_code(code: str, df: pd.DataFrame) -> tuple[Any, str]:
    """
    Execute LLM-generated pandas code in a restricted sandbox.

    The code must assign its final answer to a variable named `result`.
    Example valid code:
        result = df['fare_amount'].mean()

    Parameters
    ----------
    code : str
        Python code string generated by the LLM.
    df : pd.DataFrame
        The loaded taxi dataframe, passed as a read reference.

    Returns
    -------
    tuple[Any, str]
        (result_value, error_message). error_message is "" on success.

    Side effects
    ------------
    Logs every attempt to logs/executions.log.
    """
    timestamp = datetime.now().isoformat()

    # Security check first
    safe, reason = _is_code_safe(code)
    if not safe:
        EXEC_LOGGER.warning(f"BLOCKED | {timestamp} | Reason: {reason} | Code: {repr(code)}")
        return None, f"Security violation: {reason}"

    EXEC_LOGGER.info(f"EXEC | {timestamp} | Code: {repr(code)}")

    # Restricted globals — df and numpy/pandas are the only things exposed
    sandbox_globals: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "df": df,
        "pd": pd,
        "np": np,
        "result": None,
    }

    try:
        exec(code, sandbox_globals)  # noqa: S102
        result = sandbox_globals.get("result")

        if result is None:
            # LLM sometimes forgets to assign to result — try to be helpful
            EXEC_LOGGER.warning(f"NO_RESULT | {timestamp} | Code did not set 'result'")
            return None, (
                "The generated code ran but did not assign anything to 'result'. "
                "The code should end with: result = <your answer>"
            )

        EXEC_LOGGER.info(f"SUCCESS | {timestamp} | Result type: {type(result).__name__}")
        return result, ""

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        EXEC_LOGGER.error(f"ERROR | {timestamp} | {error_msg} | Code: {repr(code)}")
        return None, error_msg


def _format_result(result: Any) -> str:
    """
    Convert a pandas/Python result to a clean string for the agent response.

    Handles: DataFrames, Series, scalars, lists, dicts.
    Truncates large DataFrames to 20 rows to keep context window manageable.
    """
    if isinstance(result, pd.DataFrame):
        if len(result) > 20:
            preview = result.head(20)
            return (
                f"DataFrame ({len(result)} rows × {len(result.columns)} cols) "
                f"— showing first 20:\n{preview.to_string(index=True)}"
            )
        return result.to_string(index=True)

    if isinstance(result, pd.Series):
        if len(result) > 20:
            preview = result.head(20)
            return f"Series ({len(result)} items) — showing first 20:\n{preview.to_string()}"
        return result.to_string()

    if isinstance(result, float):
        return f"{result:,.4f}"

    if isinstance(result, (int, np.integer)):
        return f"{int(result):,}"

    if isinstance(result, (list, tuple)):
        if len(result) > 20:
            return str(result[:20]) + f"... ({len(result)} items total)"
        return str(result)

    return str(result)


# ── Tool 1: pandas_query_tool ─────────────────────────────────────────────────

@tool
def pandas_query_tool(question: str) -> str:
    """
    Answer a question about the NYC Taxi dataset by generating and executing
    pandas code. Use this for most analytical questions about the data.

    The tool generates Python/pandas code, runs it safely, and returns
    both the answer and the exact code used to produce it.

    Format of return value:
        ANSWER: <the result>
        CODE_USED: <the pandas code>

    Parameters
    ----------
    question : str
        A natural language question about the taxi dataset.

    Returns
    -------
    str
        The answer with the code that produced it, or an error explanation.
    """
    df = get_dataframe()
    schema = get_schema_summary()

    # Build a minimal schema hint for the code-generation prompt
    numeric_cols = [c for c, d in schema["dtypes"].items()
                    if "float" in d or "int" in d]
    datetime_cols = schema["datetime_columns"]
    all_cols = schema["columns"]

    schema_hint = (
        f"Available columns: {all_cols}\n"
        f"Numeric columns: {numeric_cols}\n"
        f"Datetime columns: {datetime_cols}\n"
        f"Derived columns: {schema['derived_columns']}\n"
        f"payment_type codes: {schema['payment_type_map']}\n"
        f"Note: {schema['notes']}\n"
        f"Total rows: {schema['row_count']:,}\n"
    )

    # This prompt is sent back to the LLM to generate code
    # The agent/assistant.py will handle the actual LLM call;
    # here we expose the tool interface that returns structured output
    code_prompt = (
        f"Write Python/pandas code to answer: {question}\n\n"
        f"Schema:\n{schema_hint}\n"
        f"Rules:\n"
        f"- The dataframe is available as 'df'\n"
        f"- Assign the final answer to a variable named 'result'\n"
        f"- Do NOT import anything\n"
        f"- Do NOT use open(), exec(), eval(), or any file I/O\n"
        f"- For datetime operations, use df['tpep_pickup_datetime'].dt.hour etc.\n"
        f"- For payment type, use df['payment_type_label'] for readable names\n"
        f"- Keep it simple — one expression assigned to result\n"
        f"Example: result = df['fare_amount'].mean()\n"
    )

    return code_prompt  # The agent will call the LLM, get code, then call execute_pandas


@tool
def execute_pandas_code(code: str) -> str:
    """
    Execute a pandas code string generated by the LLM and return the result.

    This is called AFTER pandas_query_tool generates the code prompt and
    the LLM produces actual code. The code must assign its answer to 'result'.

    Parameters
    ----------
    code : str
        Python pandas code string. Must assign final answer to 'result'.
        Example: result = df['fare_amount'].mean()

    Returns
    -------
    str
        Formatted result string, or error message with retry guidance.
    """
    df = get_dataframe()

    # Clean the code — LLMs often wrap in markdown fences
    code = _strip_code_fences(code)

    result, error = _execute_pandas_code(code, df)

    if error:
        return (
            f"CODE_ERROR: {error}\n"
            f"FAILED_CODE: {code}\n"
            f"SUGGESTION: Check column names exist in df. "
            f"Use df.columns to verify. Assign answer to 'result'."
        )

    formatted = _format_result(result)
    return f"ANSWER: {formatted}\nCODE_USED:\n{textwrap.indent(code, '  ')}"


# ── Tool 2: sql_query_tool ────────────────────────────────────────────────────

@tool
def sql_query_tool(sql: str) -> str:
    """
    Execute a SQL SELECT query against the NYC Taxi dataset using SQLite.

    The entire DataFrame is registered as a table named 'trips'.
    Only SELECT statements are permitted — no INSERT, UPDATE, DROP, etc.

    Parameters
    ----------
    sql : str
        A SQL SELECT query. The table name is 'trips'.
        Example: SELECT AVG(fare_amount) FROM trips

    Returns
    -------
    str
        Query results as a formatted string, or an error message.
    """
    df = get_dataframe()

    # Clean markdown fences if present
    sql = _strip_code_fences(sql).strip()

    # Security: only SELECT statements
    sql_upper = sql.upper().lstrip()
    if not sql_upper.startswith("SELECT"):
        EXEC_LOGGER.warning(f"SQL_BLOCKED | Non-SELECT statement attempted: {repr(sql)}")
        return (
            "SQL_ERROR: Only SELECT statements are permitted.\n"
            f"Attempted: {sql[:100]}"
        )

    # Block SQL injection patterns
    dangerous_sql = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE",
                     "ALTER", "ATTACH", "DETACH", "PRAGMA"]
    for keyword in dangerous_sql:
        if keyword in sql_upper:
            EXEC_LOGGER.warning(f"SQL_BLOCKED | Dangerous keyword '{keyword}': {repr(sql)}")
            return f"SQL_ERROR: Keyword '{keyword}' is not permitted in queries."

    timestamp = datetime.now().isoformat()
    EXEC_LOGGER.info(f"SQL_EXEC | {timestamp} | Query: {repr(sql)}")

    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql("trips", conn, index=False, if_exists="replace")
        result_df = pd.read_sql_query(sql, conn)
        conn.close()

        EXEC_LOGGER.info(f"SQL_SUCCESS | {timestamp} | Rows returned: {len(result_df)}")

        if result_df.empty:
            return f"ANSWER: Query returned no rows.\nSQL_USED:\n  {sql}"

        formatted = _format_result(result_df)
        return f"ANSWER:\n{formatted}\nSQL_USED:\n  {sql}"

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        EXEC_LOGGER.error(f"SQL_ERROR | {timestamp} | {error_msg} | Query: {repr(sql)}")
        return (
            f"SQL_ERROR: {error_msg}\n"
            f"FAILED_SQL: {sql}\n"
            f"HINT: Table name is 'trips'. "
            f"Check column names — use schema_tool to see available columns."
        )


# ── Tool 3: schema_tool ───────────────────────────────────────────────────────

@tool
def schema_tool(query: str = "full") -> str:
    """
    Return schema information about the NYC Taxi dataset.

    Use this tool to orient yourself before writing pandas or SQL code.
    Always call this first if you are unsure what columns exist.

    Parameters
    ----------
    query : str
        What schema info to return. Options:
        - "full"    : all columns, dtypes, null counts, derived columns
        - "columns" : just the column list
        - "sample"  : 3 sample rows
        - "numeric" : numeric columns with min/max/mean

    Returns
    -------
    str
        Schema information as a formatted string.
    """
    df = get_dataframe()
    schema = get_schema_summary()
    query = query.lower().strip()

    if query == "columns":
        cols = "\n".join(f"  - {c} ({schema['dtypes'][c]})" for c in schema["columns"])
        return f"COLUMNS ({schema['column_count']} total):\n{cols}"

    if query == "sample":
        sample = df.sample(3, random_state=42)
        return f"SAMPLE ROWS (3 random):\n{sample.to_string(index=False)}"

    if query == "numeric":
        lines = ["NUMERIC COLUMNS (with ranges):"]
        for col, ranges in schema["numeric_ranges"].items():
            lines.append(
                f"  {col}: min={ranges['min']}, max={ranges['max']}, mean={ranges['mean']}"
            )
        return "\n".join(lines)

    # Default: full schema
    lines = [
        f"DATASET: NYC Yellow Taxi Trips",
        f"ROWS: {schema['row_count']:,}",
        f"COLUMNS: {schema['column_count']}",
        "",
        "ALL COLUMNS (name | dtype | null_count):",
    ]
    for col in schema["columns"]:
        dtype = schema["dtypes"][col]
        nulls = schema["null_counts"].get(col, 0)
        lines.append(f"  {col:<30} {dtype:<12} nulls={nulls}")

    lines += [
        "",
        "DERIVED COLUMNS (added at load time):",
    ]
    for col in schema["derived_columns"]:
        lines.append(f"  {col}")

    lines += [
        "",
        "DATETIME COLUMNS:",
        f"  {schema['datetime_columns']}",
        "",
        "PAYMENT TYPE CODES:",
    ]
    for code, label in schema["payment_type_map"].items():
        lines.append(f"  {code} = {label}")

    lines += [
        "",
        "IMPORTANT NOTES:",
        f"  {schema['notes']}",
        "",
        "SQL TABLE NAME: trips",
        "PANDAS VARIABLE: df",
    ]

    return "\n".join(lines)


# ── Tool 4: stats_tool ────────────────────────────────────────────────────────

@tool
def stats_tool(column_name: str) -> str:
    """
    Return descriptive statistics for a specific column in the dataset.

    Use this when the user asks about distributions, ranges, averages,
    or percentiles for a specific column.

    Parameters
    ----------
    column_name : str
        Exact name of the column to profile.
        Use schema_tool first if you are unsure of the column name.

    Returns
    -------
    str
        Formatted statistics for the column, or an error if not found.
    """
    try:
        stats = get_column_stats(column_name)
    except ValueError as e:
        return f"STATS_ERROR: {str(e)}"

    lines = [f"STATISTICS FOR: {column_name}"]
    for key, value in stats.items():
        if key == "top_5_values" and isinstance(value, dict):
            lines.append("  top_5_values:")
            for val, count in value.items():
                lines.append(f"    {val}: {count:,}")
        elif isinstance(value, float):
            lines.append(f"  {key}: {value:,.4f}")
        elif isinstance(value, int):
            lines.append(f"  {key}: {value:,}")
        else:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


# ── Utility: strip markdown code fences ──────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences that LLMs often wrap code in.

    Handles:
        ```python
        result = df['fare_amount'].mean()
        ```
    and:
        ```
        result = df['fare_amount'].mean()
        ```

    Parameters
    ----------
    text : str
        Raw LLM output that may contain markdown fences.

    Returns
    -------
    str
        Clean code string without fences.
    """
    text = text.strip()
    # Remove ```python ... ``` or ``` ... ```
    text = re.sub(r"^```(?:python|sql|py)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ── Public exports ────────────────────────────────────────────────────────────

# All tools in one list for easy registration with the LangChain agent
ALL_TOOLS = [
    execute_pandas_code,
    sql_query_tool,
    schema_tool,
    stats_tool,
]

# Tool name → function map for direct lookup
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
