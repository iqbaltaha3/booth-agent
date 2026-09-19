
#!/usr/bin/env python3
"""
Launch the Gyanpur Booth Agent (pure LLM ReAct on Groq).

Setup:
  export GROQ_API_KEY=gsk_...
  export GROQ_MODEL=llama-3.3-70b-versatile   # optional

Layout:
  test_apps/
    orchestrator.py
    gyanpur_booth/
    gyanpur_voters.db
    gyanpur_history.db

Run:
  python orchestrator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gyanpur_booth.booth_agent import main

if __name__ == "__main__":
    main()