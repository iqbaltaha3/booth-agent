#!/usr/bin/env python3
"""
Product identity + live schema discovery.

Paths are resolved relative to the project. All region/party/station
vocabularies are read from the databases at runtime — nothing is
hardcoded as query logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

PACKAGE_DIR = Path(__file__).resolve().parent


def find_project_root() -> Path:
    """Walk up from package / cwd until DBs or markers are found."""
    candidates = [PACKAGE_DIR.parent, PACKAGE_DIR, Path.cwd()]
    for base in candidates:
        if (base / "gyanpur_history.db").exists() or (base / "gyanpur_voters.db").exists():
            return base
    return Path.cwd()


PROJECT_ROOT = find_project_root()
VOTERS_DB = PROJECT_ROOT / "gyanpur_voters.db"
HISTORY_DB = PROJECT_ROOT / "gyanpur_history.db"

CONSTITUENCY = {
    "name": "Gyanpur",
    "type": "Vidhan Sabha (Assembly)",
    "district": "Bhadohi",
    "state": "Uttar Pradesh",
    "product_name": "Gyanpur Booth Agent",
    "product_tagline": "Booth-level electoral intelligence for Gyanpur Vidhan Sabha",
}

SYSTEM_ARCHITECTURE = f"""
You are part of a three-agent product for political candidates in
{CONSTITUENCY['name']} Vidhan Sabha ({CONSTITUENCY['district']},
{CONSTITUENCY['state']}, India).

STACK
-----
BOOTH AGENT  (this conversation face / orchestrator)
   ├── VOTER AGENT      → gyanpur_voters.db          (live electoral roll SQL)
   ├── HISTORY AGENT    → gyanpur_history.db         (Form-20 results SQL)
   └── PORTFOLIO AGENT  → booth_analysis/*.json      (precomputed booth cards)
                          + gyanpur-boothlist.csv

VOTER AGENT: counts, surnames, age, gender, caste, religion, EPIC,
house, polling station, region composition from the live roll.

HISTORY AGENT: candidate/party vote totals, winners, margins, year
comparisons. Form-20 is long-format — totals need SUM(votes) GROUP BY.

PORTFOLIO AGENT: rich precomputed booth portfolios (demographics,
caste-gender cross-tabs, household structure, age bands) keyed by
part number or booth name. Prefer this when the user asks for a
"booth profile", "portfolio", "part number N", or detailed booth stats
that live in the JSON analysis files.

BOOTH AGENT routes to the right specialist(s) and synthesises.
Never invent numbers — only use tool evidence.

DATA CAUTIONS (reason about these; do not invent filters)
- Surnames usually appear in the name column of the voter roll, not caste.
- Caste labels may be Hindi script.
- Party strings differ by election year; discover DISTINCT party values
  from the relevant table before filtering.
- Do not invent area/region names the user did not mention. If you need
  exact spellings, query DISTINCT values from the database.
- META / TOTAL / NOTA style rows are not real candidates — exclude them
  when ranking winners unless the user asks about NOTA specifically.
""".strip()


def live_voters_context(db) -> str:
    """Build prompt context from the live voters DB."""
    if not db.path.exists():
        return f"VOTERS DB missing: {db.path}"
    lines = [f"DATABASE: {db.path.name}", db.schema_text()]
    try:
        if "voters" in db.tables():
            for col in ("region", "caste", "religion", "gender", "social_category"):
                if col in db.columns("voters"):
                    sample = db.sample_values("voters", col, limit=30)
                    lines.append(f"Sample DISTINCT {col} ({len(sample)} shown): {sample}")
    except Exception as e:
        lines.append(f"(sample error: {e})")
    return "\n".join(lines)


def live_history_context(db) -> str:
    """Build prompt context from the live history DB."""
    if not db.path.exists():
        return f"HISTORY DB missing: {db.path}"
    lines = [f"DATABASE: {db.path.name}", db.schema_text()]
    try:
        for table in db.tables():
            cols = db.columns(table)
            lines.append(f"\n--- {table} ---")
            if "party" in cols:
                parties = db.sample_values(table, "party", limit=25)
                lines.append(f"DISTINCT party: {parties}")
            if "area" in cols:
                areas = db.sample_values(table, "area", limit=25)
                lines.append(f"Sample DISTINCT area: {areas}")
    except Exception as e:
        lines.append(f"(sample error: {e})")
    return "\n".join(lines)
