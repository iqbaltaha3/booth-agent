#!/usr/bin/env python3
"""
Gyanpur Booth Agent — Streamlit UI

  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import streamlit as st

# =============================================================================
# HARDCODED Groq credentials (change here if needed)
# =============================================================================
import os
# Replace the hardcoded string with this:
HARDCODED_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HARDCODED_GROQ_MODEL = os.environ.get("GROQ_MODEL", "")

os.environ["GROQ_API_KEY"] = HARDCODED_GROQ_API_KEY
os.environ["GROQ_MODEL"] = HARDCODED_GROQ_MODEL
# =============================================================================

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Gyanpur Booth Agent",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 920px;
  }
  .hero {
    background: linear-gradient(135deg, #0f2744 0%, #1a4a7a 55%, #2d6a9f 100%);
    color: #fff;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(15, 39, 68, 0.25);
  }
  .hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.65rem;
    font-weight: 700;
  }
  .hero p { margin: 0; opacity: 0.92; font-size: 0.98rem; }
  .pill-row {
    display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.85rem;
  }
  .pill {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
  }
  div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 10px;
    border: 1px solid #d0d7e2;
    background: #f7f9fc;
    text-align: left;
    height: auto;
    white-space: normal;
    padding: 0.55rem 0.75rem;
  }
  .ok { color: #1b7f3a; font-weight: 600; }
  .bad { color: #b42318; font-weight: 600; }
  .footnote { color: #667085; font-size: 0.82rem; margin-top: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

api_key = HARDCODED_GROQ_API_KEY
model = HARDCODED_GROQ_MODEL

with st.sidebar:
    st.markdown("### ⚙️ System")
    st.caption(f"Model: `{model}`")
    st.caption("API key: hardcoded")

    st.divider()
    st.markdown("### 📁 Data sources")
    try:
        from gyanpur_booth import domain

        voters_ok = domain.VOTERS_DB.exists()
        history_ok = domain.HISTORY_DB.exists()
        csv_path = domain.PROJECT_ROOT / "gyanpur-boothlist.csv"
        analysis_dir = domain.PROJECT_ROOT / "booth_analysis"
        csv_ok = csv_path.exists()
        analysis_ok = analysis_dir.exists() and any(analysis_dir.glob("*.json"))

        def _status(ok: bool, label: str, path: Path):
            mark = (
                '<span class="ok">● ready</span>'
                if ok
                else '<span class="bad">● missing</span>'
            )
            st.markdown(f"**{label}**  {mark}", unsafe_allow_html=True)
            st.caption(str(path))

        _status(voters_ok, "Voters DB", domain.VOTERS_DB)
        _status(history_ok, "History DB", domain.HISTORY_DB)
        _status(csv_ok, "Booth list CSV", csv_path)
        _status(analysis_ok, "Booth portfolios", analysis_dir)
        st.caption(f"Project root: `{domain.PROJECT_ROOT}`")
    except Exception as e:
        st.error(f"Could not load package: {e}")

    st.divider()
    show_charts = st.toggle(
        "📊 Auto-generate charts",
        value=True,
        help="Runs the Visualization Agent on each answer (one extra LLM call).",
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<p class="footnote">Answers use only your local databases '
        "and booth portfolios. Numbers are not invented.</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="hero">
  <h1>🗳️ Gyanpur Booth Agent</h1>
  <p>Booth-level electoral intelligence for Gyanpur Vidhan Sabha · Bhadohi, UP</p>
  <div class="pill-row">
    <span class="pill">Voter roll</span>
    <span class="pill">Election history</span>
    <span class="pill">Booth portfolios</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_agent(api_key_value: str, model_name: str):
    from gyanpur_booth.common import GroqClient
    from gyanpur_booth.booth_agent import BoothAgent

    client = GroqClient(api_key=api_key_value, model=model_name)
    return BoothAgent(llm=client)


@st.cache_resource(show_spinner=False)
def get_viz_agent(api_key_value: str, model_name: str):
    from gyanpur_booth.common import GroqClient
    from gyanpur_booth.visualization_agent import VisualizationAgent

    client = GroqClient(api_key=api_key_value, model=model_name)
    return VisualizationAgent(llm=client)


def render_charts(charts, key_prefix: str) -> None:
    """Draw chart specs from the Visualization Agent (Plotly, else Matplotlib)."""
    from gyanpur_booth import chart_renderer

    for i, chart in enumerate(charts or []):
        key = f"{key_prefix}_{i}"
        try:
            try:
                fig = chart_renderer.to_plotly(chart)
                try:
                    st.plotly_chart(fig, width="stretch", key=key)  # new Streamlit
                except TypeError:
                    st.plotly_chart(fig, use_container_width=True, key=key)  # old
            except ImportError:  # plotly not installed -> static fallback
                fig = chart_renderer.to_matplotlib(chart)
                try:
                    st.pyplot(fig, width="stretch")
                except TypeError:
                    st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"⚠️ Could not draw chart “{chart.get('title', '')}”: {e}")


EXAMPLES = [
    "How many times has BJP won Gyanpur?",
    "Who won in Khokhar in 2022?",
    "How many Yadav voters are in Khokhar?",
    "Demographic breakdown for part number 1",
    "Does Congress have any scope in Gyanpur?",
    "What can you do for me?",
]

st.markdown("**Try an example**")
cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    with cols[i % 2]:
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_prompt = example

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("charts"):
            render_charts(msg["charts"], key_prefix=f"chart_{idx}")

prompt = st.chat_input("Ask about voters, results, or a booth portfolio…")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Researching with specialist agents…", expanded=True)
        try:
            status.write("Routing (voter · history · portfolio)…")
            agent = get_agent(api_key, model)
            with st.spinner("Working…"):
                answer = agent.run(prompt)
            status.update(label="Done", state="complete", expanded=False)
            st.markdown(answer)

            # Visualization Agent: never allowed to break the text answer.
            charts, chart_note = [], ""
            if show_charts:
                try:
                    with st.spinner("Checking for charts…"):
                        viz = get_viz_agent(api_key, model).analyse(answer, prompt)
                    charts = viz.get("charts", []) if viz.get("has_charts") else []
                    chart_note = viz.get("reason", "")
                except Exception as e:
                    charts, chart_note = [], f"visualization agent crashed: {e}"
                    traceback.print_exc()
            msg_idx = len(st.session_state.messages)
            if charts:
                render_charts(charts, key_prefix=f"chart_{msg_idx}")
            if show_charts and chart_note:
                st.caption(f"📊 {'Some charts skipped' if charts else 'No chart'}: {chart_note}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "charts": charts}
            )
        except Exception as e:
            status.update(label="Failed", state="error")
            err = f"**Error:** {e}\n\n```\n{traceback.format_exc()[-1500:]}\n```"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})