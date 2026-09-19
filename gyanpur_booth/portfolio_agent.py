#!/usr/bin/env python3
"""
Booth Portfolio Agent

Data source (different from voter roll / Form-20):
  - gyanpur-boothlist.csv   → part_no ↔ pooling_station mapping
  - booth_analysis/*.json   → precomputed booth demographic portfolios

Answers booth-level portfolio questions (part number, booth name, locality).
"""

from __future__ import annotations

import difflib
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import domain
from .common import GroqClient, ToolFailure, safe_json

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


class PortfolioAgent:
    def __init__(
        self,
        csv_path: Optional[Path] = None,
        analysis_dir: Optional[Path] = None,
        llm: Optional[GroqClient] = None,
    ):
        root = domain.PROJECT_ROOT
        self.csv_path = Path(csv_path) if csv_path else root / "gyanpur-boothlist.csv"
        self.analysis_dir = Path(analysis_dir) if analysis_dir else root / "booth_analysis"
        self.llm = llm or GroqClient()

        self.booth_df = None
        self.json_lookup: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if pd is not None and self.csv_path.exists():
            self.booth_df = pd.read_csv(self.csv_path)
        else:
            self.booth_df = None

        if self.analysis_dir.exists():
            files = glob.glob(str(self.analysis_dir / "*.json"))
            self.json_lookup = {os.path.basename(f).lower(): f for f in files}
        else:
            self.json_lookup = {}

    def _system_extract(self) -> str:
        return """
You extract booth targets from a user question about Gyanpur polling booths.

Return ONLY JSON:
{
  "part_numbers": ["Part_1", "Part_2"],
  "booth_names": ["PRIMARY SCHOOL KHOKHAR R.NO.-1"]
}

Rules:
- Normalize any part mention (part 1, part number 1, part1, PART-1) to Part_N.
- booth_names: polling station names, school names, or locality/region phrases.
- Empty lists if nothing is mentioned.
""".strip()

    def _extract_targets(self, question: str) -> Dict[str, List[str]]:
        text = self.llm.chat(
            [
                {"role": "system", "content": self._system_extract()},
                {"role": "user", "content": f"User query: {question}"},
            ],
            temperature=0.0,
        )
        parsed = safe_json(text) or {}
        parts = parsed.get("part_numbers") or []
        names = parsed.get("booth_names") or []
        # Soft normalize Part_N
        norm_parts = []
        for p in parts:
            s = str(p).strip()
            import re
            m = re.search(r"(\d+)", s)
            if m:
                norm_parts.append(f"Part_{int(m.group(1))}")
            elif s:
                norm_parts.append(s)
        return {
            "part_numbers": norm_parts,
            "booth_names": [str(n).strip() for n in names if str(n).strip()],
        }

    def _match_files(self, extraction: Dict[str, List[str]]) -> List[str]:
        matched_stations = set()

        if self.booth_df is not None:
            # Part numbers
            if "part_no" in self.booth_df.columns:
                for part in extraction.get("part_numbers") or []:
                    hits = self.booth_df[
                        self.booth_df["part_no"].astype(str).str.lower() == part.lower()
                    ]
                    for _, row in hits.iterrows():
                        matched_stations.add(str(row.get("pooling_station", "")).strip())

            # Booth / region name fuzzy match against pooling_station
            stations = (
                self.booth_df["pooling_station"].dropna().astype(str).tolist()
                if "pooling_station" in self.booth_df.columns
                else []
            )
            for name in extraction.get("booth_names") or []:
                closest = difflib.get_close_matches(name, stations, n=3, cutoff=0.45)
                if closest:
                    matched_stations.update(closest)
                else:
                    nl = name.lower()
                    for st in stations:
                        if nl in st.lower() or st.lower() in nl:
                            matched_stations.add(st)

        # Resolve to JSON paths
        target_files: set = set()
        for station in matched_stations:
            if not station:
                continue
            expected = station.replace(" ", "_") + "_stats.json"
            if expected.lower() in self.json_lookup:
                target_files.add(self.json_lookup[expected.lower()])
                continue
            clean = station.replace(" ", "").lower()
            for key, path in self.json_lookup.items():
                if clean in key.replace("_", ""):
                    target_files.add(path)

        # If extraction only has booth names and CSV missing, try JSON keys directly
        if not target_files:
            for name in extraction.get("booth_names") or []:
                closest = difflib.get_close_matches(
                    name.replace(" ", "_").lower() + "_stats.json",
                    list(self.json_lookup.keys()),
                    n=2,
                    cutoff=0.4,
                )
                for c in closest:
                    target_files.add(self.json_lookup[c])
                nl = name.replace(" ", "").lower()
                for key, path in self.json_lookup.items():
                    if nl in key.replace("_", ""):
                        target_files.add(path)

        return list(target_files)

    def answer(self, question: str) -> str:
        if not self.json_lookup:
            raise ToolFailure(
                "the booth portfolio files are missing",
                detail=f"Expected JSON files under {self.analysis_dir}",
            )
        if self.booth_df is None and not self.csv_path.exists():
            # Can still try name→json matching without CSV
            pass

        print(f"[portfolio] {question}")
        extraction = self._extract_targets(question)
        print(f"[portfolio] extraction: {extraction}")

        files = self._match_files(extraction)
        if not files:
            raise ToolFailure(
                "no booth portfolio matched that part number or booth name "
                "(check the spelling, e.g. Part_1)"
            )

        print(
            f"[portfolio] matched {len(files)} file(s): "
            f"{[os.path.basename(f) for f in files[:8]]}"
        )

        contexts: Dict[str, Any] = {}
        for fp in files[:5]:
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    contexts[os.path.basename(fp)] = json.load(fh)
            except Exception as e:
                contexts[os.path.basename(fp)] = {"error": str(e)}

        context_str = json.dumps(contexts, ensure_ascii=False, indent=2)
        if len(context_str) > 25000:
            context_str = context_str[:25000] + "\n...[truncated]"

        text = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an electoral booth-portfolio analyst for Gyanpur "
                        "Vidhan Sabha (Bhadohi, UP). Answer ONLY from the provided "
                        "JSON booth portfolios. Be precise with numbers. "
                        "Do not invent statistics."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Booth portfolios (JSON):\n{context_str}\n\n"
                        f"User question: {question}\n\n"
                        "Write a clear, data-backed answer."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        if text.startswith("<<LLM_ERROR"):
            print(f"[portfolio] LLM failed: {text[:200]}")
            raise ToolFailure(
                "the language model failed while reading the booth portfolio",
                detail=text[:300],
            )
        return text