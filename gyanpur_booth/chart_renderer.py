#!/usr/bin/env python3
"""
Chart renderer — turns a validated chart spec (see visualization_agent.py)
into a Plotly or Matplotlib figure. No LLM, no I/O; pure functions.

    fig = to_plotly(chart)          # interactive, preferred in Streamlit
    fig = to_matplotlib(chart)      # static fallback

Both libraries are imported lazily, so the app still runs if one is missing.
"""

from __future__ import annotations

from typing import Any, Dict, List

Chart = Dict[str, Any]


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def _labels(values: List[float]) -> List[str]:
    return [_fmt(v) for v in values]


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

def to_plotly(chart: Chart):
    import plotly.graph_objects as go

    ctype = chart["chart_type"]
    cats = chart["categories"]
    series = chart["series"]
    x_label, y_label = chart.get("x_label", ""), chart.get("y_label", "")

    fig = go.Figure()

    if ctype == "pie":
        fig.add_trace(
            go.Pie(
                labels=cats,
                values=series[0]["values"],
                hole=0.3,
                sort=False,
                textinfo="label+percent",
            )
        )
    elif ctype == "line":
        for s in series:
            fig.add_trace(
                go.Scatter(
                    x=cats,
                    y=s["values"],
                    name=s["name"],
                    mode="lines+markers+text",
                    text=_labels(s["values"]),
                    textposition="top center",
                )
            )
    elif ctype == "horizontal_bar":
        for s in series:
            fig.add_trace(
                go.Bar(
                    y=cats,
                    x=s["values"],
                    name=s["name"],
                    orientation="h",
                    text=_labels(s["values"]),
                    textposition="auto",
                )
            )
        fig.update_yaxes(autorange="reversed")  # first category on top
    else:  # bar / grouped_bar / stacked_bar
        for s in series:
            fig.add_trace(
                go.Bar(
                    x=cats,
                    y=s["values"],
                    name=s["name"],
                    text=_labels(s["values"]),
                    textposition="auto",
                )
            )

    if ctype == "stacked_bar":
        fig.update_layout(barmode="stack")
    elif ctype in ("grouped_bar", "horizontal_bar"):
        fig.update_layout(barmode="group")

    horizontal = ctype == "horizontal_bar"
    if ctype != "pie":
        fig.update_layout(
            xaxis_title=y_label if horizontal else x_label,
            yaxis_title=x_label if horizontal else y_label,
        )
        if not horizontal:
            fig.update_xaxes(type="category")
        else:
            fig.update_yaxes(type="category")

    fig.update_layout(
        title=chart.get("title", ""),
        template="plotly_white",
        height=430,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=(len(series) > 1 or ctype == "pie"),
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------

def to_matplotlib(chart: Chart):
    import numpy as np
    from matplotlib.figure import Figure  # no pyplot: safe across Streamlit threads

    ctype = chart["chart_type"]
    cats = chart["categories"]
    series = chart["series"]
    x_label, y_label = chart.get("x_label", ""), chart.get("y_label", "")

    fig = Figure(figsize=(8, 4.6), dpi=110, layout="constrained")
    ax = fig.subplots()
    ax.set_title(chart.get("title", ""), fontsize=12, fontweight="bold")

    if ctype == "pie":
        ax.pie(
            series[0]["values"],
            labels=cats,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        ax.axis("equal")
        return fig

    n = len(cats)
    idx = np.arange(n)

    if ctype == "line":
        for s in series:
            ax.plot(cats, s["values"], marker="o", label=s["name"])
            for x, y in zip(cats, s["values"]):
                ax.annotate(_fmt(y), (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.3)

    elif ctype == "horizontal_bar":
        k = len(series)
        h = 0.8 / k
        for i, s in enumerate(series):
            bars = ax.barh(idx + (i - (k - 1) / 2) * h, s["values"], h, label=s["name"])
            ax.bar_label(bars, labels=_labels(s["values"]), fontsize=8, padding=2)
        ax.set_yticks(idx, cats)
        ax.invert_yaxis()
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)
        ax.grid(axis="x", alpha=0.3)

    elif ctype == "stacked_bar":
        bottom = np.zeros(n)
        for s in series:
            vals = np.array(s["values"], dtype=float)
            ax.bar(idx, vals, 0.6, bottom=bottom, label=s["name"])
            bottom += vals
        ax.set_xticks(idx, cats, rotation=30 if max(map(len, cats)) > 8 else 0,
                      ha="right" if max(map(len, cats)) > 8 else "center")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.3)

    else:  # bar / grouped_bar
        k = len(series)
        w = 0.8 / k
        for i, s in enumerate(series):
            bars = ax.bar(idx + (i - (k - 1) / 2) * w, s["values"], w, label=s["name"])
            ax.bar_label(bars, labels=_labels(s["values"]), fontsize=8, padding=2)
        long_labels = max(map(len, cats)) > 8
        ax.set_xticks(idx, cats, rotation=30 if long_labels else 0,
                      ha="right" if long_labels else "center")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.3)

    ax.set_axisbelow(True)
    ax.margins(y=0.12) if ctype != "horizontal_bar" else ax.margins(x=0.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False)
    return fig