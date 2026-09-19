#!/usr/bin/env python3
"""
Voter Agent — pure ReAct over gyanpur_voters.db.
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

MAX_TURNS = 5


class VoterAgent:
    def __init__(self, db_path=None, llm: Optional[GroqClient] = None, max_turns: int = MAX_TURNS):
        self.db_path = db_path or domain.VOTERS_DB
        self.db = SQLiteDB(self.db_path)
        self.llm = llm or GroqClient()
        self.max_turns = max_turns

    def _system(self) -> str:
        live = domain.live_voters_context(self.db)
        return f"""
You are the VOTER AGENT in the Gyanpur Booth Intelligence product.

{domain.SYSTEM_ARCHITECTURE}

LIVE DATABASE CONTEXT
---------------------
{live}

YOUR JOB
--------
Answer questions about the electoral ROLL only, using SQLite SELECT on
the single table `voters`.

HARD BOUNDARIES
- Table name is exactly: voters  (never gyanpur_voters, never gyanpur_history)
- Columns are only those in the schema above. There is NO year, candidate,
  party, or votes column. You cannot answer "who won" or "how they voted".
- If the question is only about election results / winners / vote totals,
  respond with action=finish and explain in thought that History Agent owns
  Form-20 results — do not invent SQL against missing columns.

Respond with ONLY a JSON object each turn:
{{
  "thought": "reasoning about what is still needed",
  "action": "sql" | "finish",
  "sql": "SELECT ...",
  "reason": "why this query"
}}

GUIDANCE
- Prefer COUNT / GROUP BY. LIMIT large dumps.
- Surnames usually sit in name / relative_name, not caste.
- Caste may be Hindi; use sample values above or DISTINCT.
- Booth ranking: GROUP BY polling_station, region ORDER BY count DESC.
- Never invent region labels the user did not imply.
- When done, action=finish.
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
                    "the voter-roll query kept failing and could not be corrected",
                    detail=sql_steps[-1].observation[:300],
                )
            raise ToolFailure("I could not work out a database query for the voter-roll question")
        blocks = [
            f"SQL:\n{s.action_input}\nResult:\n{s.observation[:3000]}" for s in good
        ]

        text = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the Voter Agent. Answer the campaign user using ONLY "
                        "the SQL evidence. Do not invent numbers. Be clear and concise."
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
            print(f"[voter] summary LLM failed: {text[:200]}")
            return "Results (summary unavailable):\n" + "\n".join(
                s.observation[:3000] for s in good
            )
        return text

    def answer(self, question: str) -> str:
        if not self.db_path.exists():
            raise ToolFailure("the voter-roll database is missing", detail=str(self.db_path))

        state = ReactState(question=question.strip())
        print(f"[voter] {question}")

        for turn in range(1, self.max_turns + 1):
            print(f"[voter] turn {turn}/{self.max_turns}")
            plan = self.plan(state)
            action = plan.get("action", "finish")
            print(f"  thought: {plan.get('thought', '')[:200]}")
            print(f"  action:  {action}")

            if action != "sql":
                state.steps.append(ReactStep(turn, plan.get("thought", ""), "finish", "", "(finish)"))
                break

            sql = plan.get("sql", "")
            print(f"  sql: {sql[:220]}")
            try:
                rows = self.db.execute(sql)
                obs = f"{len(rows)} row(s): {compact_json(rows, 5000)}"
            except Exception as e:
                obs = f"ERROR: {e}. Fix the SQL and try again."
            print(f"  obs: {obs[:220]}")
            state.steps.append(ReactStep(turn, plan.get("thought", ""), "sql", sql, obs))

        return self._synthesize(state)