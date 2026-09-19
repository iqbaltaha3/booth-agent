#!/usr/bin/env python3
"""
History Agent — pure ReAct over gyanpur_history.db Form-20 tables.
All planning is done by the LLM. No deterministic query builders.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import domain
from .common import (
    GroqClient,
    ReactState,
    ReactStep,
    SQLiteDB,
    ToolFailure,
    compact_json,
    extract_sql,
    safe_json,
    validate_select,
)

MAX_TURNS = 6


class HistoryAgent:
    def __init__(self, db_path=None, llm: Optional[GroqClient] = None, max_turns: int = MAX_TURNS):
        self.db_path = db_path or domain.HISTORY_DB
        self.db = SQLiteDB(self.db_path)
        self.llm = llm or GroqClient()
        self.max_turns = max_turns

    def _system(self) -> str:
        live = domain.live_history_context(self.db)
        return f"""
You are the HISTORY AGENT in the Gyanpur Booth Intelligence product.

{domain.SYSTEM_ARCHITECTURE}

LIVE DATABASE CONTEXT
---------------------
{live}

YOUR JOB
--------
Answer election-result questions from Form-20 tables using SQLite SELECT.
Table names look like form20_2017_vidhan, form20_2022_vidhan, etc.
There is no table named gyanpur_history.

Respond with ONLY a JSON object each turn:
{{
  "thought": "reasoning about what is still needed",
  "action": "sql" | "finish",
  "sql": "SELECT ...",
  "reason": "why this query"
}}

GUIDANCE (principles — reason, do not invent data)
- Long-format rows: totals need SUM(votes) GROUP BY candidate or party.
- Use exact party strings from DISTINCT samples (or query DISTINCT party).
  Party labels change by year (e.g. "BJP" vs full names). Prefer
  party LIKE '%BJP%' or check DISTINCT party first when filtering.
- Do not invent area names. User-named places (e.g. KHOKHAR) may match
  the `area` column — verify with DISTINCT area if unsure.
- Seat-level "Gyanpur" / "how many times has X won Gyanpur" means the
  whole assembly segment across form20_*_vidhan tables — not only area
  GYANPUR, and not Lok Sabha tables unless the user asked about LS.
- Exclude META / TOTAL / NOTA / TENDERED rows when ranking winners
  unless the user asked about them.
- Multi-year: prefer ONE query with UNION ALL of the relevant tables.
  SQLite does NOT allow ORDER BY / LIMIT directly on the parts of a UNION.
  Wrap EVERY part in a subquery, like this (top 3 per year):
    SELECT * FROM (
      SELECT '2019' AS yr, candidate, party, SUM(votes) AS total
      FROM form20_2019_lok WHERE COALESCE(party,'') <> 'META'
      GROUP BY candidate, party ORDER BY total DESC LIMIT 3)
    UNION ALL
    SELECT * FROM (
      SELECT '2024' AS yr, candidate, party, SUM(votes) AS total
      FROM form20_2024_lok WHERE COALESCE(party,'') <> 'META'
      GROUP BY candidate, party ORDER BY total DESC LIMIT 3)
- BLANK PARTY: some real candidates have party = '' (the party is written
  inside the candidate name, e.g. "Name (AITC (TMC))"). Never filter with
  party NOT IN ('META','') — that silently drops them. Exclude only the
  summary rows where party = 'META' (TOTAL VOTES / TOTAL VALID VOTES / NOTA).
- If an observation starts with ERROR, read the message, correct the SQL and
  try again. Do not finish until you have a successful result.
- Only SELECT. When done, action=finish.
""".strip()

    def plan(self, state: ReactState) -> Dict[str, Any]:
        user = f"""
User question:
{state.question}

Research so far:
{state.memory() or "(none yet)"}

Return the next JSON plan only.
"""
        text = self.llm.chat(
            [
                {"role": "system", "content": self._system()},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        if text.startswith("<<LLM_ERROR"):
            return {"thought": text, "action": "finish", "sql": ""}

        parsed = safe_json(text)
        if parsed:
            action = str(parsed.get("action", "")).lower().strip()
            if action == "finish":
                return {"thought": str(parsed.get("thought", "")), "action": "finish", "sql": ""}
            sql = str(parsed.get("sql", "")).strip()
            if not sql:
                sql = extract_sql(text) or ""
            ok, cleaned = validate_select(sql) if sql else (False, "")
            if ok:
                return {
                    "thought": str(parsed.get("thought", "")),
                    "action": "sql",
                    "sql": cleaned,
                    "reason": str(parsed.get("reason", "")),
                }

        sql = extract_sql(text)
        if sql:
            return {"thought": "Extracted SQL from model text", "action": "sql", "sql": sql}
        return {"thought": "Could not parse plan", "action": "finish", "sql": ""}

    def _synthesize(self, state: ReactState) -> str:
        sql_steps = [s for s in state.steps if s.action == "sql"]
        good = [s for s in sql_steps if not s.observation.startswith("ERROR")]
        if not good:
            if sql_steps:
                raise ToolFailure(
                    "the election-history query kept failing and could not be corrected",
                    detail=sql_steps[-1].observation[:300],
                )
            raise ToolFailure(
                "I could not work out a database query for the election-history question"
            )
        blocks = [
            f"SQL:\n{s.action_input}\nResult:\n{s.observation[:3500]}" for s in good
        ]

        text = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the History Agent for Gyanpur Vidhan Sabha. "
                        "Answer using ONLY the SQL evidence. Do not invent votes. "
                        "If assessing a party's 'scope', use multi-year totals, "
                        "presence of candidates, and comparison to winners when available. "
                        "Be honest when the party is weak."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {state.question}\n\nEvidence:\n"
                        + "\n\n".join(blocks)
                        + "\n\nWrite the final answer."
                    ),
                },
            ],
            temperature=0.1,
        )
        if text.startswith("<<LLM_ERROR"):
            # Data was retrieved but could not be summarised: give rows, never SQL.
            print(f"[history] summary LLM failed: {text[:200]}")
            return "Results (summary unavailable):\n" + "\n".join(
                s.observation[:3500] for s in good
            )
        return text

    def answer(self, question: str) -> str:
        if not self.db_path.exists():
            raise ToolFailure("the election-history database is missing", detail=str(self.db_path))

        state = ReactState(question=question.strip())
        print(f"[history] {question}")

        for turn in range(1, self.max_turns + 1):
            print(f"[history] turn {turn}/{self.max_turns}")
            plan = self.plan(state)
            action = plan.get("action", "finish")
            print(f"  thought: {plan.get('thought', '')[:220]}")
            print(f"  action:  {action}")

            if action != "sql":
                state.steps.append(ReactStep(turn, plan.get("thought", ""), "finish", "", "(finish)"))
                break

            sql = plan.get("sql", "")
            print(f"  sql: {sql[:240]}")
            try:
                rows = self.db.execute(sql)
                obs = f"{len(rows)} row(s): {compact_json(rows, 5000)}"
            except Exception as e:
                obs = f"ERROR: {e}. Fix the SQL and try again."
            print(f"  obs: {obs[:220]}")
            state.steps.append(ReactStep(turn, plan.get("thought", ""), "sql", sql, obs))

        return self._synthesize(state)