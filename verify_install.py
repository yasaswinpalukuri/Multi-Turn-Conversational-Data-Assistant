"""
verify_install.py — Run this after `pip install -r requirements.txt`

Usage:
    python verify_install.py

Checks every critical import and prints version numbers so you can
confirm nothing silently resolved to a wrong version.
Also pings Ollama to confirm it's reachable.
"""

import sys
import importlib

REQUIRED_PYTHON = (3, 11)

def check(label: str, import_path: str, version_attr: str = "__version__") -> None:
    """Import a module and print its version."""
    try:
        mod = importlib.import_module(import_path)
        version = getattr(mod, version_attr, "version unknown")
        print(f"  ✓  {label:<30} {version}")
    except ImportError as e:
        print(f"  ✗  {label:<30} MISSING — {e}")
        sys.exit(1)


def check_ollama_reachable(base_url: str = "http://localhost:11434") -> None:
    """Ping Ollama HTTP endpoint."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
            print(f"  ✓  Ollama reachable at {base_url}")
    except urllib.error.URLError:
        print(f"  ⚠  Ollama NOT reachable at {base_url}")
        print("     Start it with: ollama serve")
        print("     Then pull the model: ollama pull qwen2.5:7b")


def main() -> None:
    print("\n── Python version ─────────────────────────────────")
    major, minor = sys.version_info[:2]
    if (major, minor) < REQUIRED_PYTHON:
        print(f"  ✗  Python {major}.{minor} found; {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required")
        sys.exit(1)
    print(f"  ✓  Python {major}.{minor}.{sys.version_info.micro}")

    print("\n── LangChain ecosystem ────────────────────────────")
    check("langchain",           "langchain")
    check("langchain-core",      "langchain_core")
    check("langchain-community", "langchain_community")
    check("langchain-ollama",    "langchain_ollama")

    print("\n── Data layer ─────────────────────────────────────")
    check("pandas",  "pandas")
    check("numpy",   "numpy")
    check("pyarrow", "pyarrow")

    print("\n── Vector store ───────────────────────────────────")
    check("chromadb", "chromadb")

    print("\n── UI ─────────────────────────────────────────────")
    check("streamlit", "streamlit")

    print("\n── Utilities ──────────────────────────────────────")
    check("python-dotenv", "dotenv", "VERSION")
    check("ollama SDK",    "ollama")

    print("\n── Testing ────────────────────────────────────────")
    check("pytest",        "pytest")
    check("pytest-asyncio","pytest_asyncio")

    print("\n── Ollama connectivity ────────────────────────────")
    check_ollama_reachable()

    print("\n── Version sanity checks ──────────────────────────")
    import pandas as pd
    import langchain
    pd_major = int(pd.__version__.split(".")[0])
    lc_major = int(langchain.__version__.split(".")[0])

    if pd_major >= 3:
        print(f"  ⚠  pandas {pd.__version__} detected — 3.x has Copy-on-Write breaking changes.")
        print("     Run: pip install 'pandas==2.2.3'")
    else:
        print(f"  ✓  pandas version OK ({pd.__version__})")

    if lc_major >= 1:
        print(f"  ⚠  langchain {langchain.__version__} detected — 1.x has API changes.")
        print("     Run: pip install 'langchain==0.3.25'")
    else:
        print(f"  ✓  langchain version OK ({langchain.__version__})")

    print("\n✅  All checks passed. Proceed to Step 2.\n")


if __name__ == "__main__":
    main()
