#!/usr/bin/env python3
"""
Visualization Agent — turns a Booth Agent answer into chart specs.

Input : the final, user-facing text produced by BoothAgent.run()
Output: a validated, library-agnostic JSON-style dict:

    {
      "has_charts": true,
      "charts": [
        {
          "chart_type": "bar",              # see SUPPORTED_CHART_TYPES
          "title": "Votes by party, 2022",
          "x_label": "Party",               # label of the category axis
          "y_label": "Votes",               # label of the value axis
          "categories": ["BJP", "SP", "BSP"],
          "series": [{"name": "Votes", "values": [90000, 85000, 20000]}]
        }
      ],
      "reason": ""                          # why no charts / what was dropped
    }

The LLM proposes the spec; plain Python then validates it. Every plotted
value must literally appear in the answer text (STRICT_NUMBERS) so the
agent can never chart a number the Booth Agent did not say.

Rendering lives in chart_renderer.py (Plotly and Matplotlib).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .common import GroqClient, safe_json

SUPPORTED_CHART_TYPES = (
    "bar",
    "horizontal_bar",
    "grouped_bar",
    "stacked_bar",
    "line",
    "pie",
)
MAX_CHARTS = 3
MAX_CATEGORIES = 30
MAX_PIE_SLICES = 12
MAX_SERIES = 8

_ALIASES = {
    "column": "bar",
    "barh": "horizontal_bar",
    "hbar": "horizontal_bar",
    "grouped": "grouped_bar",
    "clustered_bar": "grouped_bar",
    "stacked": "stacked_bar",
    "donut": "pie",
    "doughnut": "pie",
}

# "1,23,456" / "1,234.5" / "45.2" — a trailing "." or "," is not part of the number.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# "9.7 lakh", "2 crore", "45 thousand", "12k"  (also लाख / करोड़)
_NUMBER_UNIT_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*((?:lakhs?|lacs?|crores?|thousand|million|mn|k)\b|लाख|करोड़)",
    re.I,
)
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _unit_scale(unit: str) -> float:
    u = unit.lower()
    if u.startswith(("lakh", "lac")) or u == "लाख":
        return 1e5
    if u.startswith("crore") or u == "करोड़":
        return 1e7
    if u in ("thousand", "k"):
        return 1e3
    return 1e6  # million / mn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> List[float]:
    """
    All numbers in `text`: thousands separators removed, Devanagari digits
    converted, and "9.7 lakh"-style figures added in their expanded form too.
    """
    text = (text or "").translate(_DEVANAGARI_DIGITS)
    out: List[float] = []
    for m in _NUMBER_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    for num, unit in _NUMBER_UNIT_RE.findall(text):
        try:
            out.append(round(float(num.replace(",", "")) * _unit_scale(unit), 6))
        except ValueError:
            continue
    return out


def _strip_think(text: str) -> str:
    """
    Remove reasoning-model <think> blocks. An UNCLOSED <think> means the model
    ran out of tokens while still reasoning, so nothing usable is left.
    """
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if re.search(r"<think>", text, flags=re.I):
        return ""
    return text.strip()


def _tidy(x: float) -> Any:
    return int(x) if float(x).is_integer() else x


def _empty(reason: str = "") -> Dict[str, Any]:
    return {"has_charts": False, "charts": [], "reason": reason}


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------

class VisualizationAgent:
    def __init__(
        self,
        llm: Optional[GroqClient] = None,
        max_charts: int = MAX_CHARTS,
        strict_numbers: bool = True,
    ):
        """
        strict_numbers=True  -> a chart is dropped if any plotted value does
                                not appear in the answer text (no invented or
                                LLM-computed numbers).
        strict_numbers=False -> values are only checked for being numeric.
        """
        self.llm = llm or GroqClient()
        self.max_charts = max_charts
        self.strict_numbers = strict_numbers

    # -- prompt -------------------------------------------------------------

    def _system(self) -> str:
        types = ", ".join(SUPPORTED_CHART_TYPES)
        return f"""
You are the VISUALIZATION AGENT in the Gyanpur Booth Intelligence product
(election analytics for a candidate's campaign team).

You receive the FINAL ANSWER that another agent wrote for the user. Decide
whether the numbers in it can be shown as a chart. If yes, describe the
chart(s) as JSON. You never draw anything yourself; code will render your JSON.

OUTPUT: start your reply with the JSON immediately. Do NOT write out your
reasoning first. ONLY one JSON object, no prose, no markdown fences:
{{
  "has_charts": true | false,
  "reason": "one short sentence",
  "charts": [
    {{
      "chart_type": one of [{types}],
      "title": "short descriptive title",
      "x_label": "label of the category axis",
      "y_label": "label of the value axis (include the unit)",
      "categories": ["label 1", "label 2", "..."],
      "series": [ {{"name": "series name", "values": [number, number, "..."]}} ]
    }}
  ]
}}
If nothing is chartable return {{"has_charts": false, "reason": "...", "charts": []}}.

HARD RULES
- Use ONLY numbers that appear verbatim in the answer. Never compute,
  estimate, round, sum, or convert (no deriving percentages, totals or
  differences). If a value is not written in the answer, do not chart it.
- Every series needs exactly one value per category, as a plain number
  (no "1,234", no "45%", no null). Drop a category rather than guess.
- Do not mix units in one chart (e.g. voter counts with percentages).
  Make separate charts instead.
- A chart needs at least 2 categories. A single number is not chartable.
- At most {self.max_charts} charts. Prefer the few that best answer the
  user's question; do not chart the same data twice.
- Ignore ids and codes that are not measurements (part numbers, EPIC ids,
  house numbers, rank numbers, list numbering like "1)").
- Keep category labels in the same language/script as the answer.

CHOOSING chart_type
- bar             : compare one measure across categories (votes per party).
- horizontal_bar  : same, but many categories or long labels.
- grouped_bar     : several series side by side (party votes in 2017 vs 2022).
- stacked_bar     : parts that add up to a whole per category.
- line            : values over time (years) — categories must be in order.
- pie             : shares of ONE whole (gender split, caste share), with
                    at most {MAX_PIE_SLICES} slices; series must be a single series.
""".strip()

    # -- public API ---------------------------------------------------------

    def _ask(self, user: str, max_tokens: int) -> Tuple[Optional[Dict[str, Any]], str]:
        """One LLM round-trip -> (parsed_json_or_None, failure_reason)."""
        raw = self.llm.chat(
            [
                {"role": "system", "content": self._system()},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if raw.startswith("<<LLM_ERROR"):
            return None, f"LLM call failed: {raw[:160]}"
        cleaned = _strip_think(raw)
        if not cleaned:
            return None, (
                "model used all its tokens reasoning before writing JSON"
                if raw.strip()
                else "model returned an empty reply"
            )
        parsed = safe_json(cleaned)
        if not parsed:
            return None, "model reply was not valid JSON"
        return parsed, ""

    def analyse(self, answer: str, question: str = "") -> Dict[str, Any]:
        """Return the validated chart spec for a Booth Agent answer."""
        answer = (answer or "").strip()
        if len(extract_numbers(answer)) < 2:
            return _empty("Too few numbers in the answer to chart.")

        user = (
            (f"User question (for context only):\n{question}\n\n" if question else "")
            + f"FINAL ANSWER TO ANALYSE:\n{answer}\n\nReturn the JSON object now."
        )
        parsed, why = self._ask(user, max_tokens=4000)
        if parsed is None and not why.startswith("LLM call failed"):
            print(f"[viz] first attempt unusable ({why}); retrying once")
            parsed, why = self._ask(
                user + "\n\nREPLY WITH THE JSON OBJECT ONLY. NO THINKING, NO EXPLANATION.",
                max_tokens=6000,
            )
        if parsed is None:
            print(f"[viz] no spec: {why}")
            return _empty(why)

        result = self.validate(parsed, answer)
        print(
            f"[viz] has_charts={result['has_charts']} "
            f"charts={len(result['charts'])} note={result['reason'][:160]!r}"
        )
        return result

    def validate(self, spec: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        """Sanitise an LLM spec. Bad charts are dropped, never repaired by guessing."""
        if not isinstance(spec, dict) or not spec.get("has_charts"):
            reason = str(spec.get("reason", "")) if isinstance(spec, dict) else ""
            return _empty(reason or "No chartable data.")

        raw_charts = spec.get("charts")
        if not isinstance(raw_charts, list):
            return _empty("Spec had no charts list.")

        numbers = extract_numbers(source_text)
        charts: List[Dict[str, Any]] = []
        dropped: List[str] = []
        seen_titles = set()

        for i, raw in enumerate(raw_charts):
            chart, why = self._clean_chart(raw, numbers)
            if chart is None:
                dropped.append(f"chart {i + 1}: {why}")
                continue
            key = chart["title"].strip().lower()
            if key in seen_titles:
                dropped.append(f"chart {i + 1}: duplicate title")
                continue
            seen_titles.add(key)
            charts.append(chart)
            if len(charts) >= self.max_charts:
                break

        if not charts:
            return _empty("; ".join(dropped) or "No valid charts.")
        return {"has_charts": True, "charts": charts, "reason": "; ".join(dropped)}

    # -- validation ---------------------------------------------------------

    def _value_in_text(self, v: float, numbers: List[float]) -> bool:
        return any(abs(v - n) < 0.006 for n in numbers)

    def _clean_chart(
        self, raw: Any, numbers: List[float]
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not isinstance(raw, dict):
            return None, "not an object"

        ctype = str(raw.get("chart_type", "")).strip().lower()
        ctype = ctype.replace("-", "_").replace(" ", "_")
        ctype = _ALIASES.get(ctype, ctype)
        if ctype not in SUPPORTED_CHART_TYPES:
            return None, f"unsupported chart_type {ctype!r}"

        cats = raw.get("categories")
        if not isinstance(cats, list) or not 2 <= len(cats) <= MAX_CATEGORIES:
            return None, f"need 2-{MAX_CATEGORIES} categories"
        cats = [str(c).strip() for c in cats]

        series_raw = raw.get("series")
        if not isinstance(series_raw, list) or not 1 <= len(series_raw) <= MAX_SERIES:
            return None, f"need 1-{MAX_SERIES} series"

        series: List[Dict[str, Any]] = []
        for i, s in enumerate(series_raw):
            vals = s.get("values") if isinstance(s, dict) else None
            if not isinstance(vals, list) or len(vals) != len(cats):
                return None, "series length does not match categories"
            clean: List[float] = []
            for v in vals:
                if isinstance(v, bool) or v is None:
                    return None, "non-numeric value"
                if isinstance(v, str):
                    v = v.replace(",", "").replace("%", "").strip()
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return None, "non-numeric value"
                if not math.isfinite(f):
                    return None, "non-finite value"
                if self.strict_numbers and not self._value_in_text(f, numbers):
                    return None, f"value {f:g} not present in the answer"
                clean.append(_tidy(f))
            series.append({"name": str(s.get("name") or f"Series {i + 1}").strip(), "values": clean})

        # Make chart_type consistent with the data shape.
        if ctype == "pie":
            if len(series) != 1:
                return None, "pie needs exactly one series"
            vals = series[0]["values"]
            if len(vals) > MAX_PIE_SLICES or any(v < 0 for v in vals) or sum(vals) <= 0:
                return None, "pie needs 2-12 non-negative slices"
        elif len(series) == 1 and ctype in ("grouped_bar", "stacked_bar"):
            ctype = "bar"
        elif len(series) > 1 and ctype in ("bar", "horizontal_bar"):
            ctype = "grouped_bar"

        title = str(raw.get("title") or "").strip() or "Chart"
        return (
            {
                "chart_type": ctype,
                "title": title,
                "x_label": str(raw.get("x_label") or "").strip(),
                "y_label": str(raw.get("y_label") or "").strip(),
                "categories": cats,
                "series": series,
            },
            "",
        )