# app.py
# Mini Terminal — Fundamentals & Valuation (Yahoo/yfinance)
# Works with your existing modules:
# - data_engine.py (fetch_true_ttm)
# - growth_margin.py (compute_growth_and_margins)
# - annual_normalized.py (compute_annual_normalized)
# - capital_efficiency.py (compute_capital_efficiency)
# - capital_allocation.py (compute_capital_allocation)
# - allocation_scorecard.py (compute_allocation_scorecard)
# - value_creation.py (compute_value_creation)
# - valuation_sanity.py (load_valuation_sanity)  <-- Step-4 (Option A tuple output)

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import plotly.express as px

from data_engine import fetch_true_ttm
from growth_margin import compute_growth_and_margins
from annual_normalized import compute_annual_normalized
from capital_efficiency import compute_capital_efficiency
from capital_allocation import compute_capital_allocation
from allocation_scorecard import compute_allocation_scorecard
from value_creation import compute_value_creation

# Step-4 module (your file)
# IMPORTANT: load_valuation_sanity should return a tuple if you're using Option A:
# (mispricing_df, decision_df, warnings)
from valuation_sanity import compute_valuation_sanity, load_valuation_sanity



# -----------------------------
# Defaults / Manual Overrides
# -----------------------------
DEFAULT_TICKERS = ["MDLN", "BDX", "SYK", "ZBH", "MDT", "CAH", "MCK", "HSIC", "TFX", "OMI"]

# Optional: Put any known TTM Net Income overrides here (absolute USD, not millions)
MANUAL_NET_INCOME_TTM_USD: Dict[str, float] = {
    # "MDLN": 1266000000.0,
}


# -----------------------------
# UI Helpers
# -----------------------------
def tab_intro(
    title: str,
    what: str,
    why: str,
    how: str,
    interpret: List[str],
    caveats: List[str],
) -> None:
    st.subheader(title)
    st.markdown(f"**What it is:** {what}")
    st.markdown(f"**Why it matters:** {why}")
    st.markdown(f"**How it's computed:** {how}")

    with st.expander("How to read this (Interpretation)"):
        for b in interpret:
            st.markdown(f"- {b}")

    with st.expander("Caveats / Data quality notes"):
        for c in caveats:
            st.markdown(f"- {c}")


def _safe_upper_list(text: str) -> List[str]:
    return [t.strip().upper() for t in text.split(",") if t.strip()]


def _as_num(x):
    try:
        return float(x)
    except Exception:
        return None


def _pct(x: Optional[float]) -> Optional[float]:
    if x is None or pd.isna(x):
        return None
    return float(x) * 100.0


# -----------------------------
# Cached Loaders (fast UI)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_true_ttm(tickers: List[str], sleep_s: float) -> Tuple[pd.DataFrame, List[str]]:
    res = fetch_true_ttm(tickers, manual_net_income_ttm=MANUAL_NET_INCOME_TTM_USD, sleep_s=sleep_s)
    return res.df, list(res.warnings or [])


@st.cache_data(show_spinner=False)
def load_growth_margins(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    gm = compute_growth_and_margins(tickers, manual_net_income_ttm_usd=MANUAL_NET_INCOME_TTM_USD)
    return gm.df, list(gm.warnings or [])


@st.cache_data(show_spinner=False)
def load_annual_normalized(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df, warnings = compute_annual_normalized(tickers)
    return df, list(warnings or [])


@st.cache_data(show_spinner=False)
def load_capital_efficiency(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    ce = compute_capital_efficiency(tickers)
    return ce.df, list(ce.warnings or [])


@st.cache_data(show_spinner=False)
def load_capital_allocation(tickers: List[str], politeness_delay_s: float) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df, warnings = compute_capital_allocation(tickers, politeness_delay_s=politeness_delay_s)
    scorecard = compute_allocation_scorecard(df) if df is not None and not df.empty else pd.DataFrame()
    return df, scorecard, list(warnings or [])


@st.cache_data(show_spinner=False)
def load_value_creation(annual_norm_df: pd.DataFrame, cap_eff_df: pd.DataFrame) -> pd.DataFrame:
    if annual_norm_df is None or annual_norm_df.empty or cap_eff_df is None or cap_eff_df.empty:
        return pd.DataFrame()
    return compute_value_creation(annual_norm_df, cap_eff_df)


@st.cache_data(show_spinner=False)
def load_valuation_layer(base_ttm_df: pd.DataFrame, vc_df: pd.DataFrame, cap_eff_df: pd.DataFrame):
    # Option A expected return:
    # mispricing_df, decision_df, warnings
    return load_valuation_sanity(base_ttm_df, vc_df, cap_eff_df)


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Mini Terminal", layout="wide")
st.title("Mini Terminal — Fundamentals & Valuation (Yahoo/yfinance)")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Watchlist")

    tickers_text = st.text_area("Tickers (comma separated)", value=",".join(DEFAULT_TICKERS))
    tickers = _safe_upper_list(tickers_text)

    refresh = st.button("Refresh Now", use_container_width=True)

    st.caption("Politeness delay (seconds per ticker)")
    politeness_delay = st.slider("", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    st.divider()
    st.subheader("Chart options")
    log_scale = st.checkbox("Log scale for Price (recommended)", value=True)

    st.divider()
    st.subheader("Exports")
    export_csv = st.button("Export CSV", use_container_width=True)

    with st.expander("Glossary (quick)"):
        st.markdown(
            """
- **TTM**: Trailing twelve months  
- **EV**: Enterprise value  
- **ROIC**: Return on invested capital  
- **FCF**: Free cash flow  
- **P/E**: Price / earnings  
- **EV/EBITDA**: Enterprise value / EBITDA  
"""
        )

if refresh:
    # Clear all cached data and re-run
    st.cache_data.clear()
    st.rerun()


# -----------------------------
# Load Base Data (TTM table)
# -----------------------------
with st.spinner("Fetching TTM fundamentals & valuation..."):
    base_df, base_warn = load_true_ttm(tickers, sleep_s=politeness_delay)

# A stable ordering helps UX
if base_df is None:
    base_df = pd.DataFrame()
if "Ticker" in base_df.columns:
    base_df = base_df.sort_values("Ticker")


# -----------------------------
# Tabs
# -----------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Dashboard",
        "Table",
        "Growth & Margins (TTM)",
        "Annual Normalized",
        "Capital Efficiency",
        "Capital Allocation",
        "Value Creation",
        "Valuation Sanity",
        "Diagnostics",
    ]
)

# -----------------------------
# Dashboard
# -----------------------------
with tab0:
    tab_intro(
        "Dashboard",
        what="High-level snapshot of each ticker across scale, profitability, and valuation.",
        why="Scan quickly for outliers and then drill down in other tabs.",
        how="Uses the base TTM table (price, EV, multiples) and your computed modules (ROIC, growth, valuation sanity).",
        interpret=[
            "Start with economics (ROIC + Growth) before valuation.",
            "If a stock looks 'cheap', check profitability and leverage first.",
            "Use Diagnostics for missing inputs when something looks odd.",
        ],
        caveats=[
            "Yahoo/yfinance coverage varies by ticker; some fields may be missing.",
            "Negative EBITDA/Net Income makes EV/EBITDA or P/E not meaningful.",
        ],
    )

    if base_df.empty:
        st.warning("No base data returned. Try fewer tickers or increase politeness delay.")
    else:
        # KPI row
        cols = st.columns(4)
        mcap_col = "MarketCap_USD_B" if "MarketCap_USD_B" in base_df.columns else None
        ev_col = "EV_USD_B" if "EV_USD_B" in base_df.columns else None

        with cols[0]:
            st.metric("Tickers", len(tickers))
        with cols[1]:
            if mcap_col:
                st.metric("Total Market Cap (B)", f"{base_df[mcap_col].fillna(0).sum():,.2f}")
            else:
                st.metric("Total Market Cap (B)", "N/A")
        with cols[2]:
            if ev_col:
                st.metric("Total EV (B)", f"{base_df[ev_col].fillna(0).sum():,.2f}")
            else:
                st.metric("Total EV (B)", "N/A")
        with cols[3]:
            st.metric("Data warnings", len(base_warn))

        st.divider()

        # Scatter: EV/EBITDA vs Market Cap (if available)
        if "EV_EBITDA" in base_df.columns and mcap_col:
            tmp = base_df.copy()
            tmp["EV_EBITDA_num"] = pd.to_numeric(tmp["EV_EBITDA"], errors="coerce")
            tmp[mcap_col] = pd.to_numeric(tmp[mcap_col], errors="coerce")

            fig = px.scatter(
                tmp.dropna(subset=["EV_EBITDA_num", mcap_col]),
                x=mcap_col,
                y="EV_EBITDA_num",
                text="Ticker" if "Ticker" in tmp.columns else None,
                hover_data=[c for c in ["Ticker", "Price_Last_USD", "EV_USD_B", "MarketCap_USD_B", "PE_Proxy"] if c in tmp.columns],
                title="Valuation snapshot: EV/EBITDA vs Market Cap",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dashboard scatter needs EV_EBITDA and MarketCap_USD_B columns.")


# -----------------------------
# Base Table
# -----------------------------
with tab1:
    tab_intro(
        "Base Table (TTM fundamentals + valuation)",
        what="Price, market cap, EV, and simple valuation multiples.",
        why="This is the 'raw terminal' view. Everything else builds on top.",
        how="Pulled from Yahoo/yfinance: price & market cap; EV computed using net debt where available; multiples from EV and TTM items.",
        interpret=[
            "Use as a screening sheet to sort by EV/EBITDA or P/E proxy.",
            "Treat negative multiples as invalid (usually negative EBITDA / earnings).",
        ],
        caveats=[
            "EV and multiples depend on balance-sheet completeness in Yahoo.",
            "Some tickers return stale or missing fields; check Diagnostics.",
        ],
    )

    st.dataframe(base_df, use_container_width=True)
    if base_warn:
        st.warning("Base data warnings:")
        st.write(base_warn)


# -----------------------------
# Growth & Margins (TTM)
# -----------------------------
with tab2:
    tab_intro(
        "Growth & Margins (TTM)",
        what="TTM revenue/EBITDA/net income with growth and margin sanity checks.",
        why="Separates 'growing' from 'profitable' and catches margin compression early.",
        how="Computes TTM levels, YoY changes, margins, and approximate 3Y CAGR where data exists.",
        interpret=[
            "Higher and stable EBITDA margin usually indicates pricing power or operating leverage.",
            "Watch for revenue growth with collapsing margins (quality problem).",
        ],
        caveats=[
            "TTM can be distorted by one-time events.",
            "Some tickers lack EBITDA or reliable quarterly statements via yfinance.",
        ],
    )

    with st.spinner("Computing Growth & Margins (TTM)..."):
        gm_df, gm_warn = load_growth_margins(tickers)

    st.dataframe(gm_df, use_container_width=True)
    if gm_warn:
        st.warning("Growth/Margins warnings:")
        st.write(gm_warn)


# -----------------------------
# Annual Normalized
# -----------------------------
with tab3:
    tab_intro(
        "Annual Normalized (Bloomberg-style)",
        what="Annual income statement view with normalized growth and average margins.",
        why="Annual series reduces noise vs TTM and is better for longer-term economics.",
        how="Uses annual financials from Yahoo; computes normalized revenue/EBITDA/net income, CAGR where possible, and average margins.",
        interpret=[
            "Prefer steady multi-year margin stability over a single strong year.",
            "CAGR is more meaningful when the business isn't highly cyclical.",
        ],
        caveats=[
            "Yahoo annual statements can be missing for some tickers.",
            "Accounting changes can create jumps (especially in NI).",
        ],
    )

    with st.spinner("Computing Annual Normalized..."):
        ann_df, ann_warn = load_annual_normalized(tickers)

    st.dataframe(ann_df, use_container_width=True)
    if ann_warn:
        st.warning("Annual normalized warnings:")
        st.write(ann_warn)


# -----------------------------
# Capital Efficiency
# -----------------------------
with tab4:
    tab_intro(
        "Capital Efficiency",
        what="ROIC and balance-sheet risk ratios (leverage + coverage).",
        why="ROIC is a strong long-run value creation signal; leverage/coverage show fragility.",
        how="ROIC = NOPAT / Invested Capital; risk ratios from income + balance sheet where available.",
        interpret=[
            "ROIC comfortably above ~8–10% is often a good sign (rough WACC range).",
            "Net Debt / EBITDA above ~3 can reduce flexibility.",
            "Higher interest coverage means resilience when rates rise.",
        ],
        caveats=[
            "Invested capital fields can be missing depending on Yahoo coverage.",
            "Negative EBITDA breaks some leverage ratios; treat those as N/A.",
        ],
    )

    with st.spinner("Computing Capital Efficiency..."):
        cap_df, cap_warn = load_capital_efficiency(tickers)

    st.dataframe(cap_df, use_container_width=True)
    if cap_warn:
        st.warning("Capital efficiency warnings:")
        st.write(cap_warn)


# -----------------------------
# Capital Allocation
# -----------------------------
with tab5:
    tab_intro(
        "Capital Allocation",
        what="FCF margin, capex intensity, debt change, dividends + a flags-based scorecard.",
        why="Good capital allocation separates great companies from value traps.",
        how="Uses cash flow + balance sheet: FCF, capex, total debt changes, dividends where available; scorecard applies rule-based scoring and flags.",
        interpret=[
            "Sustained negative FCF with rising debt is a red flag.",
            "High capex can be good if ROIC remains high (productive reinvestment).",
            "Dividend yield is meaningful only if economics remain strong.",
        ],
        caveats=[
            "Cash flow labels vary in Yahoo; some fields may be missing.",
            "Debt changes can be distorted by acquisitions or reclassifications.",
        ],
    )

    with st.spinner("Computing Capital Allocation..."):
        ca_df, score_df, ca_warn = load_capital_allocation(tickers, politeness_delay_s=politeness_delay)

    st.caption("Capital allocation metrics")
    st.dataframe(ca_df, use_container_width=True)

    st.caption("Allocation scorecard (rules + flags)")
    st.dataframe(score_df, use_container_width=True)

    if ca_warn:
        st.warning("Capital allocation warnings:")
        st.write(ca_warn)


# -----------------------------
# Value Creation
# -----------------------------
with tab6:
    tab_intro(
        "Value Creation",
        what="Combines economics (growth + ROIC) into a Value Creation Score and Economic Quadrant.",
        why="This is your primary 'business quality' lens — valuation comes after.",
        how="Merges Annual Normalized output with Capital Efficiency output; computes score and assigns quadrant (Compounder, Melting Ice Cube, etc.).",
        interpret=[
            "Compounder: high ROIC + positive growth.",
            "Melting Ice Cube: weak/declining economics — cheap may be a trap.",
            "Turnaround: improving growth but weak ROIC — needs confirmation.",
        ],
        caveats=[
            "If ROIC or growth is missing, the quadrant may be 'Insufficient data'.",
            "Short-term growth spikes can overstate long-run quality.",
        ],
    )

    with st.spinner("Building Value Creation table..."):
        ann_df, _ = load_annual_normalized(tickers)
        cap_df, _ = load_capital_efficiency(tickers)
        vc_df = load_value_creation(ann_df, cap_df)

    st.dataframe(vc_df, use_container_width=True)


# -----------------------------
# Valuation Sanity (Step-4)
# -----------------------------
with tab7:
    tab_intro(
        "Valuation Sanity (Implied Fair Multiples)",
        what="Maps ROIC + Growth into implied fair EV/EBITDA and P/E; labels each ticker Cheap/Fair/Expensive.",
        why="Adds a pressure-test layer so you don’t overpay for quality or misread value traps.",
        how="Builds a quality score from ROIC and growth, converts that into fair multiples, then compares to current market multiples.",
        interpret=[
            "Cheap + Compounder → best hunting ground (verify data quality).",
            "Expensive + Compounder → wait for pullback unless fundamentals accelerate.",
            "Cheap + Melting Ice Cube → deep-value only; confirm no permanent impairment.",
        ],
        caveats=[
            "Negative EBITDA/earnings makes multiples invalid — should be treated as N/A.",
            "This is a sanity screen, not intrinsic value.",
            "Sector differences can distort comparisons (healthcare vs software vs distribution).",
        ],
    )

    with st.spinner("Computing Valuation Sanity..."):
        # Ensure we have VC + capital efficiency
        ann_df, _ = load_annual_normalized(tickers)
        cap_df, _ = load_capital_efficiency(tickers)
        vc_df = load_value_creation(ann_df, cap_df)

        mispricing_df, decision_df, warnings = load_valuation_sanity(base_df, vc_df, cap_df)
        st.dataframe(mispricing_df)
        st.dataframe(decision_df)

    st.caption("Mispricing table (Cheap / Fair / Expensive)")
    st.dataframe(mispricing_df, use_container_width=True)

    st.caption("Final decision quadrant (Economics × Valuation)")
    st.dataframe(decision_df, use_container_width=True)

    if warnings:
        st.warning("Valuation sanity warnings:")
        st.write(warnings)


# -----------------------------
# Diagnostics
# -----------------------------
with tab8:
    tab_intro(
        "Diagnostics",
        what="Shows missing fields and module warnings to keep the model honest.",
        why="Yahoo coverage gaps can silently distort ratios; diagnostics helps you trust the output.",
        how="Aggregates warnings from each module and highlights missing key columns.",
        interpret=[
            "Use this tab whenever something looks 'too good' or 'too weird'.",
            "If key inputs are missing, rely less on derived ratios.",
        ],
        caveats=[
            "Missing fields can be a Yahoo/yfinance limitation, not the company.",
        ],
    )

    # Collect warnings
    diag_blocks = []

    if base_warn:
        diag_blocks.append(("Base (TTM) warnings", base_warn))

    gm_df, gm_warn = load_growth_margins(tickers)
    if gm_warn:
        diag_blocks.append(("Growth & Margins warnings", gm_warn))

    ann_df, ann_warn = load_annual_normalized(tickers)
    if ann_warn:
        diag_blocks.append(("Annual Normalized warnings", ann_warn))

    cap_df, cap_warn = load_capital_efficiency(tickers)
    if cap_warn:
        diag_blocks.append(("Capital Efficiency warnings", cap_warn))

    ca_df, _, ca_warn = load_capital_allocation(tickers, politeness_delay_s=politeness_delay)
    if ca_warn:
        diag_blocks.append(("Capital Allocation warnings", ca_warn))

    # Show warnings
    if not diag_blocks:
        st.success("No warnings detected.")
    else:
        for title, items in diag_blocks:
            with st.expander(title, expanded=False):
                for w in items:
                    st.markdown(f"- {w}")

    st.divider()

    # Missing key columns audit (base table)
    st.caption("Missing key columns audit (Base TTM table)")
    key_cols = ["Ticker", "Price_Last_USD", "MarketCap_USD_B", "EV_USD_B", "EV_EBITDA", "PE_Proxy"]
    missing = [c for c in key_cols if c not in base_df.columns]
    if missing:
        st.warning(f"Missing columns in base_df: {missing}")
    else:
        st.success("All key base columns present.")


# -----------------------------
# Export
# -----------------------------
if export_csv:
    if base_df.empty:
        st.error("Nothing to export — base table is empty.")
    else:
        out = base_df.copy()
        out.to_csv("terminal_export.csv", index=False)
        st.success("Saved: terminal_export.csv (in your working directory)")
        st.caption("Tip: if you want browser download, switch to st.download_button.")
