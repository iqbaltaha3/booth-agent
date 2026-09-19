#!/usr/bin/env python3
"""
Booth Agent — product face. Pure ReAct orchestrator.

The LLM decides when to call voter / history / finish.
No keyword routing. No deterministic plans.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import domain
from .common import GroqClient, ReactStep, ToolFailure, safe_json
from .history_agent import HistoryAgent
from .portfolio_agent import PortfolioAgent
from .voter_agent import VoterAgent

MAX_STEPS = 5

CAPABILITIES_TEXT = f"""
I am the {domain.CONSTITUENCY['product_name']} for
{domain.CONSTITUENCY['name']} Vidhan Sabha
({domain.CONSTITUENCY['district']}, {domain.CONSTITUENCY['state']}).

I can help with three kinds of questions:

1) Voter roll (live database)
   - Counts by region, caste, religion, gender, age
   - Surname searches, EPIC lookup, booth composition

2) Election history (Form-20)
   - Candidate / party vote totals by year (2017, 2019, 2022, 2024)
   - Winners and margins for the seat or a region (e.g. Khokhar)

3) Booth portfolios (precomputed analysis)
   - Full demographic cards for a part number (Part_1, …)
   - Named booth / school profiles from booth_analysis

Ask a concrete question, for example:
  • "How many Yadav voters in Khokhar?"
  • "Who won in Khokhar in 2022?"
  • "Demographic breakdown for part number 1"
  • "How many times has BJP won Gyanpur?"
""".strip()



@dataclass
class ToolResult:
    tool: str
    question: str
    answer: str
    ok: bool
    error: Optional[str] = None   # plain-language reason, safe to show the user
    detail: str = ""              # technical info: logs / planner only


@dataclass
class BoothState:
    question: str
    steps: List[ReactStep] = field(default_factory=list)
    results: List[ToolResult] = field(default_factory=list)

    def memory(self) -> str:
        parts = []
        for s in self.steps:
            parts.append(
                f"Step {s.step} | action={s.action}\n"
                f"Thought: {s.thought}\n"
                f"Input: {s.action_input}\n"
                f"Observation: {s.observation[:3500]}"
            )
        return "\n\n".join(parts)


class BoothAgent:
    def __init__(self, llm: Optional[GroqClient] = None, max_steps: int = MAX_STEPS):
        self.llm = llm or GroqClient()
        self.max_steps = max_steps
        self.voter = VoterAgent(llm=self.llm)
        self.history = HistoryAgent(llm=self.llm)
        self.portfolio = PortfolioAgent(llm=self.llm)

    def _system(self) -> str:
        return f"""
You are the BOOTH AGENT — the product interface for political candidates
in {domain.CONSTITUENCY['name']} Vidhan Sabha
({domain.CONSTITUENCY['district']}, {domain.CONSTITUENCY['state']}).

{domain.SYSTEM_ARCHITECTURE}

TOOLS
-----
1) voter
   Live electoral roll SQL (table `voters`). Counts, surnames, caste,
   religion, gender, age, EPIC, region composition.
   No votes/winners/years.

2) history
   Form-20 election results SQL. Votes, winners, party performance,
   how a region voted in a given year.

3) portfolio
   Precomputed booth portfolio JSON (from booth_analysis/).
   Use for: part number queries (Part_1), named booth profiles,
   detailed demographic cards, caste-gender cross-tabs, household
   structure already analysed per booth.
   Prefer portfolio over voter when the user asks for a booth
   "profile", "portfolio", "stats for part N", or a specific school/booth
   card that exists in the analysis folder.

4) finish
   Stop and answer from observations.dont return json when aswering. 
   Only return json when using tools. If you have enough evidence to answer the question, use finish.

Respond with ONLY JSON each turn while using tools:
{{
  "thought": "what the candidate still needs",
  "action": "voter" | "history" | "portfolio" | "finish",
  "action_input": "self-contained sub-question for the tool"
}}

ROUTING EXAMPLES
- "demographics in part number 1" → portfolio
- "who won in Khokhar in 2022" → history
- "how many Yadav voters in Khokhar" → voter (or portfolio if booth card asked)
- "how did Khokhar vote in 2022" → history
- Do not send election results to voter or portfolio.
- Do not repeat the same failed tool call with the same input.
- action_input must stand alone. Never invent numbers.
- When observations answer the question, action=finish.
""".strip()

    def plan(self, state: BoothState) -> Dict[str, Any]:
        user = f"""
Candidate question:
{state.question}

Research so far:
{state.memory() or "(none yet)"}

Next step as JSON only.
"""
        text = self.llm.chat(
            [
                {"role": "system", "content": self._system()},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        if text.startswith("<<LLM_ERROR"):
            return {
                "thought": text,
                "action": "finish",
                "action_input": "",
            }

        parsed = safe_json(text)
        if not parsed:
            return {
                "thought": "Unparseable plan; finishing with available evidence.",
                "action": "finish",
                "action_input": "",
            }

        action = str(parsed.get("action", "")).lower().strip()
        if action not in {"voter", "history", "portfolio", "finish"}:
            action = "finish"
        return {
            "thought": str(parsed.get("thought", "")).strip() or "Planning.",
            "action": action,
            "action_input": str(parsed.get("action_input", state.question)).strip()
            or state.question,
        }

    def call_tool(self, name: str, question: str) -> ToolResult:
        try:
            if name == "voter":
                ans = self.voter.answer(question)
            elif name == "history":
                ans = self.history.answer(question)
            elif name == "portfolio":
                ans = self.portfolio.answer(question)
            else:
                return ToolResult(name, question, "", False, f"Unknown tool: {name}")
            return ToolResult(name, question, str(ans).strip(), True)
        except ToolFailure as e:
            print(f"  [{name}] failed: {e.message} | {e.detail[:200]}")
            return ToolResult(name, question, "", False, e.message, detail=e.detail)
        except Exception as e:
            traceback.print_exc()
            return ToolResult(
                name, question, "", False,
                f"the {name} lookup hit an unexpected error", detail=f"{type(e).__name__}: {e}",
            )

    _TOOL_LABELS = {
        "voter": "voter-roll",
        "history": "election-history",
        "portfolio": "booth-portfolio",
    }

    @staticmethod
    def _failure_message(failed: List[ToolResult]) -> str:
        reasons: List[str] = []
        for r in failed:
            if r.error and r.error not in reasons:
                reasons.append(r.error)
        return (
            "Sorry, I couldn't answer that this time: "
            + "; ".join(reasons)
            + ".\n\nThat's a problem on my side, not with your question. "
            "Please try again in a moment, or ask it in a slightly different way "
            "(for example one election at a time)."
        )

    def synthesise(self, state: BoothState) -> str:
        ok = [r for r in state.results if r.ok and r.answer]
        ok_tools = {r.tool for r in ok}
        # Report a failure only if that tool never succeeded later on a retry.
        failed = [r for r in state.results if not r.ok and r.tool not in ok_tools]

        if not ok:
            if failed:
                return self._failure_message(failed)
            # No tool was used: meta / capability question, or LLM/API failure
            q = state.question.lower()
            if any(
                x in q
                for x in (
                    "what can you",
                    "what do you do",
                    "help",
                    "capabilities",
                    "who are you",
                )
            ):
                return CAPABILITIES_TEXT
            err_bits = [
                s.thought for s in state.steps if s.thought.startswith("<<LLM_ERROR")
            ]
            if err_bits:
                return (
                    "The language model API failed before I could research that.\n"
                    f"Detail: {err_bits[0]}\n\n"
                    "Check GROQ_API_KEY and GROQ_MODEL, then try again.\n\n"
                    + CAPABILITIES_TEXT
                )
            return (
                "I could not gather enough evidence.\n\n" + CAPABILITIES_TEXT
            )

        evidence = "\n\n".join(
            f"[{r.tool.upper()}]\nQ: {r.question}\nA: {r.answer}" for r in ok
        )
        if failed:
            evidence += "\n\n[COULD NOT RETRIEVE]\n" + "\n".join(
                f"- {self._TOOL_LABELS.get(r.tool, r.tool)} data: {r.error}"
                for r in failed
            )
        text = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"You are the Booth Agent speaking to a campaign team in "
                        f"{domain.CONSTITUENCY['name']} Vidhan Sabha. "
                        "Use ONLY the tool evidence. Do not invent numbers. "
                        "Be practical, honest, and concise. "
                        "NEVER show SQL, database errors, stack traces or internal "
                        "tool names to the user. If the evidence contains a "
                        "[COULD NOT RETRIEVE] section, say plainly in one sentence "
                        "which part could not be found, and answer only the rest."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Candidate question:\n{state.question}\n\n"
                        f"Evidence:\n{evidence}\n\n"
                        "Write the final answer for the candidate."
                    ),
                },
            ],
            temperature=0.1,
        )
        if text.startswith("<<LLM_ERROR"):
            print(f"[booth] synthesis LLM failed: {text[:200]}")
            body = "\n\n".join(r.answer for r in ok)
            note = (
                "\n\n(I couldn't write a full summary just now, so these are the "
                "raw results.)"
            )
            if failed:
                note += " " + self._failure_message(failed).split("\n\n")[0]
            return body + note
        return text

    def run(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "Please ask a question about voters or election results."

        state = BoothState(question=question)
        print(f"\n[booth] question: {question}")

        for step in range(1, self.max_steps + 1):
            print(f"\n[booth] step {step}/{self.max_steps}")
            plan = self.plan(state)
            thought = plan["thought"]
            action = plan["action"]
            inp = plan["action_input"]
            print(f"  thought: {thought[:220]}")
            print(f"  action:  {action}")
            print(f"  input:   {inp[:160]}")

            if action == "finish":
                state.steps.append(ReactStep(step, thought, "finish", inp, "(finish)"))
                break

            result = self.call_tool(action, inp)
            state.results.append(result)
            obs = (
                result.answer
                if result.ok
                else f"ERROR: {result.error}. {result.detail}".strip()
            )
            print(f"  tool_ok: {result.ok} ({len(obs)} chars)")
            state.steps.append(ReactStep(step, thought, action, inp, obs))

        print("[booth] synthesising…")
        return self.synthesise(state)


def main() -> None:
    print("=" * 72)
    print(f"  {domain.CONSTITUENCY['product_name']}")
    print(f"  {domain.CONSTITUENCY['product_tagline']}")
    print("=" * 72)
    print(f"  Constituency : {domain.CONSTITUENCY['name']} Vidhan Sabha")
    print(f"  District     : {domain.CONSTITUENCY['district']}, {domain.CONSTITUENCY['state']}")
    print(f"  Voters DB    : {domain.VOTERS_DB}")
    print(f"  History DB   : {domain.HISTORY_DB}")
    client = GroqClient()
    print(f"  LLM          : Groq / {client.model}")
    print(f"  API key set  : {'yes' if client.api_key else 'NO — export GROQ_API_KEY'}")
    print("=" * 72)
    print("Type 'exit' to quit.\n")

    agent = BoothAgent(llm=client)

    if len(sys.argv) > 1:
        print(agent.run(" ".join(sys.argv[1:])))
        return

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        try:
            print("\n" + agent.run(q) + "\n")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()