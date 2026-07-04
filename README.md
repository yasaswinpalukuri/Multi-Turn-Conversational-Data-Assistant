# Multi-Turn Conversational Data Assistant

A production-quality local AI assistant that answers natural language questions
about the NYC Yellow Taxi dataset using pandas and SQL — with full conversation
memory, a security sandbox, and a clean Streamlit UI. Runs entirely offline via Ollama.

---

## Results (measured, not estimated)

| Metric | Result |
|---|---|
| Dataset size | 97,723 rows × 30 columns |
| Avg response time (warm) | **1.6s** per query |
| First query (model load) | ~2.8s |
| Memory recall response time | **0.03s** (bypasses LLM) |
| Unit tests | **74/74 passed**, 0 failures |
| Security patterns blocked | 11/11 |
| Context window | 15 turns rolling |
| Peak hour detection | Hour 18 (6 PM) ✓ |
| Payment type detection | Credit card (72%) ✓ |

---

## Stack

| Layer | Technology |
|---|---|
| LLM | qwen2.5:7b via Ollama (local, no API key) |
| Agent framework | LangChain 0.3.25 + langchain-ollama 0.2.3 |
| In-session memory | ChatMessageHistory (rolling k=15 window) |
| Persistent memory | ChromaDB 0.6.3 (cosine similarity search) |
| Data layer | pandas 2.2.3 + SQLite (in-memory) |
| UI | Streamlit 1.43.2 |
| Hardware tested | Dell G15 · RTX 4060 8GB · 16GB RAM · WSL2 |

---

## Project Structure

```
data_assistant/
├── app.py                    # Streamlit entry point
├── agent/
│   ├── assistant.py          # Core orchestrator — routes questions, handles retry
│   ├── tools.py              # 4 tools: pandas, SQL, schema, stats + security sandbox
│   ├── memory.py             # Two-layer memory: ChatMessageHistory + ChromaDB
│   └── prompts.py            # System prompt with live schema + 10 few-shot examples
├── data/
│   └── nyc_taxi_sample.csv   # 100k-row sample of NYC TLC Jan 2023 data
├── utils/
│   └── data_loader.py        # CSV loader, feature engineering, schema helpers
├── tests/
│   ├── test_tools.py         # 48 unit tests for all 4 tools + security sandbox
│   ├── test_memory.py        # 26 unit tests for memory storage and retrieval
│   └── eval_set.csv          # 20 ground-truth evaluation questions
├── logs/
│   └── executions.log        # Audit log of every exec() attempt
├── requirements.txt
└── .env.example
```

---

## Setup

**Requirements:** Python 3.11+, WSL2 (Windows) or Linux/macOS, Ollama installed

```bash
# 1. Clone / create project folder
cd "/path/to/Multi-Turn Conversational Data Assistant"

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if needed — defaults work out of the box

# 5. Download and prepare dataset
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
python3 -c "
import pandas as pd
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
df.sample(100000, random_state=42).to_csv('data/nyc_taxi_sample.csv', index=False)
"

# 6. Pull the model (one-time, ~4.7GB)
ollama pull qwen2.5:7b

# 7. Verify everything is working
python verify_install.py

# 8. Run tests
pytest tests/test_tools.py tests/test_memory.py -v

# 9. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Features

### Four analytical tools
- **pandas_tool** — generates and executes pandas code for any analytical question
- **sql_tool** — runs SQL SELECT queries (SQLite in-memory) for GROUP BY and ranking questions
- **schema_tool** — returns column names, dtypes, null counts on demand
- **stats_tool** — descriptive statistics (min, max, mean, median, percentiles) for any column

### Two-layer memory
- **Rolling window** — last 15 turns kept in context for the LLM on every call
- **ChromaDB persistence** — every turn saved to disk with semantic search
- Memory questions (first question, turn count, history summary) answered in **0.03s** without calling the LLM

### Security sandbox
All LLM-generated code runs inside a restricted `exec()` that:
- Blocks all module imports (`os`, `sys`, `subprocess`, `pathlib`, and 20+ others)
- Blocks `open()`, `eval()`, `exec()`, `__import__()`, and dunder access
- Logs every execution attempt to `logs/executions.log`
- Retries once automatically on code failure before giving up

### Streamlit UI
- Dark navy/slate theme
- Left sidebar: live dataset stats, session metrics, context window progress bar, Ollama status
- Expandable **Show Code** / **Show SQL** section on every answer
- Response time displayed per message
- Quick question buttons for common queries
- Export full session history as JSON

### Derived columns (added at load time)
The data loader engineers 11 additional columns so the LLM can answer complex questions without writing transformation code:

| Column | Description |
|---|---|
| `trip_duration_minutes` | Pickup to dropoff in minutes |
| `pickup_hour` | Hour of day (0–23) |
| `pickup_day_of_week` | Day name (Monday–Sunday) |
| `pickup_date` | Date portion of pickup timestamp |
| `tip_percentage` | Tip as % of fare (0 for cash trips) |
| `speed_mph` | Average speed, capped at 100 mph |
| `has_surcharge` | True if any surcharge > 0 |
| `is_airport_trip` | True if JFK or Newark rate code |
| `payment_type_label` | Human-readable payment type |
| `vendor_label` | Human-readable vendor name |
| `ratecode_label` | Human-readable rate code |

---

## Eval Set (20 questions)

Questions 1–11 test factual accuracy, 12–14 test memory, 15–20 test advanced analytics.
See `tests/eval_set.csv` for the full set with expected answer types.

Sample verified answers from this dataset:

| Question | Answer |
|---|---|
| How many rows? | 97,723 |
| Average fare amount | $18.29 |
| Most common payment type | Credit card (72%) |
| Hour with most pickups | 18 (6 PM) |
| What was my first question? | Recalled instantly from ChromaDB |

---

## Architecture Notes

### Migration-ready for LangGraph
`agent/memory.py` and `agent/tools.py` have zero Streamlit dependency.
The same files plug into a LangGraph node by calling `assistant.chat()` inside a node function — no modification needed.

### Why pandas 2.2.3 (not 3.x)
Pandas 3.0 enables Copy-on-Write by default. LLM-generated code frequently
uses `inplace=True` and chained assignments that silently fail under CoW.
Pinning to 2.2.x avoids a category of hard-to-debug LLM code errors.

### Why ChromaDB DefaultEmbeddingFunction
Using the built-in onnxruntime embeddings instead of sentence-transformers
avoids a CUDA conflict between PyTorch and Ollama on WSL2 when both try to
claim the same GPU memory.

---

## Resume Bullets (fill in after extended testing)

```
- Built multi-turn data assistant using LangChain + Ollama (qwen2.5:7b)
  maintaining context across 15+ turns with ChromaDB-backed session persistence
  and 0.03s memory recall response time

- Implemented pandas/SQL dual-tool pipeline with LLM-generated code execution,
  answering 20 categories of analytical questions on 97,723-row NYC taxi dataset
  with avg 1.6s response time on RTX 4060 GPU locally

- Built security sandbox restricting exec() to safe builtins, blocking 11+
  dangerous patterns (os, subprocess, open, eval) with full audit logging;
  74/74 unit tests passing across tools and memory layers

- Engineered ChromaDB-backed two-layer memory (rolling k=15 window +
  persistent semantic search) enabling full conversation history recall
  and session resumption across restarts
```

---

## Troubleshooting

**Ollama not reachable**
```bash
ollama serve          # start Ollama
ollama pull qwen2.5:7b   # pull model if not downloaded
```

**ChromaDB GPU warning in WSL2**
```
[W:onnxruntime] Failed to detect devices under "/sys/class/drm/card0"
```
Harmless — ChromaDB falls back to CPU embeddings automatically.

**pandas 3.x installed**
```bash
pip install "pandas==2.2.3" --force-reinstall
```

**First response is slow (10–15s)**
Normal — the model is loading into VRAM. Every subsequent call will be 1–3s.
