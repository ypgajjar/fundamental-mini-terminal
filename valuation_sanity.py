# valuation_sanity.py
# Step-4: Valuation sanity layer (Implied Fair Multiples)
# - Builds ROIC_Score and Growth_Score (0–100) → Quality_0_100
# - Maps Quality_0_100 → Fair EV/EBITDA and Fair P/E (with light leverage adjustment)
# - Labels mispricing: Cheap / Fair / Expensive (with tolerance band)
# - Produces final "Decision" by combining Economics (Economic Quadrant) × Valuation

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Result container
# -----------------------------
@dataclass
class ValuationSanityResult:
    mispricing_df: pd.DataFrame
    decision_df: pd.DataFrame
    warnings: List[str]


# -----------------------------
# Helpers
# -----------------------------
def _to_num(x) -> float:
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _clip(x: float, lo: float, hi: float) -> float:
    if np.isnan(x):
        return np.nan
    return float(min(max(x, lo), hi))


def roic_score_0_100(roic_pct: float, roic_lo: float = 0.0, roic_hi: float = 30.0) -> float:
    """
    Convert ROIC (%) into a 0–100 score.
    - ROIC below 0% → treated as 0 score
    - ROIC above roic_hi → saturated at 100
    """
    r = _to_num(roic_pct)
    if np.isnan(r):
        return np.nan
    r = _clip(r, roic_lo, roic_hi)
    return (r - roic_lo) / (roic_hi - roic_lo) * 100.0 if roic_hi > roic_lo else np.nan


def growth_score_0_100(growth_pct: float, g_lo: float = 0.0, g_hi: float = 25.0) -> float:
    """
    Convert Growth (%) into a 0–100 score.
    - Growth below 0% → treated as 0 score (no credit for shrinkage)
    - Growth above g_hi → saturated at 100
    """
    g = _to_num(growth_pct)
    if np.isnan(g):
        return np.nan
    g = _clip(g, g_lo, g_hi)
    return (g - g_lo) / (g_hi - g_lo) * 100.0 if g_hi > g_lo else np.nan


def quality_0_100(
    roic_score: float,
    growth_score: float,
    w_roic: float = 0.6,
    w_growth: float = 0.4,
) -> float:
    """
    Quality score derived from ROIC and Growth scores.
    If one score is missing, the missing part contributes 0 (conservative).
    """
    rs = _to_num(roic_score)
    gs = _to_num(growth_score)
    rs = 0.0 if np.isnan(rs) else rs
    gs = 0.0 if np.isnan(gs) else gs
    return float(w_roic * rs + w_growth * gs)


def _fair_multiple_from_quality(
    q_0_100: float,
    lo: float,
    hi: float,
) -> float:
    """
    Linearly maps Quality_0_100 into [lo, hi].
    """
    q = _to_num(q_0_100)
    if np.isnan(q):
        return np.nan
    q = _clip(q, 0.0, 100.0)
    return lo + (q / 100.0) * (hi - lo)


def _leverage_adjust_multiple(fair_mult: float, net_debt_ebitda: float) -> float:
    """
    Light leverage sanity adjustment:
    - Net cash (<=0): +5%
    - Moderate leverage (2–3): -5%
    - Higher leverage (3–4): -10%
    - Very high (>=4): -20%
    """
    fm = _to_num(fair_mult)
    nd = _to_num(net_debt_ebitda)
    if np.isnan(fm):
        return np.nan
    if np.isnan(nd):
        return fm

    if nd <= 0:
        return fm * 1.05
    if 2.0 <= nd < 3.0:
        return fm * 0.95
    if 3.0 <= nd < 4.0:
        return fm * 0.90
    if nd >= 4.0:
        return fm * 0.80
    return fm


def _label_vs_fair(current: float, fair: float, band: float = 0.15) -> str:
    """
    Labels valuation vs fair multiple:
    - Cheap if current <= fair*(1-band)
    - Expensive if current >= fair*(1+band)
    - Fair otherwise
    Returns "N/A" if current or fair is invalid.
    """
    c = _to_num(current)
    f = _to_num(fair)
    if np.isnan(c) or np.isnan(f) or f <= 0:
        return "N/A"
    if c <= f * (1.0 - band):
        return "Cheap"
    if c >= f * (1.0 + band):
        return "Expensive"
    return "Fair"


def _overall_valuation(ev_label: str, pe_label: str) -> str:
    """
    Combines EV and PE valuations into one label.
    - If both available and match → that label
    - If both available and disagree → Mixed
    - If one available → that one
    - If neither → N/A
    """
    ev_ok = ev_label in {"Cheap", "Fair", "Expensive"}
    pe_ok = pe_label in {"Cheap", "Fair", "Expensive"}

    if ev_ok and pe_ok:
        return ev_label if ev_label == pe_label else "Mixed"
    if ev_ok:
        return ev_label
    if pe_ok:
        return pe_label
    return "N/A"


def _decision(econ: str, val: str) -> str:
    """
    Final decision logic (Economics × Valuation).
    Keep it simple and conservative.
    """
    econ = (econ or "").strip()
    val = (val or "").strip()

    if econ in {"Insufficient data", "", "N/A"} or val == "N/A":
        return "Incomplete"

    if econ == "Compounder":
        if val == "Cheap":
            return "Buy / Consider"
        if val == "Fair":
            return "Watch / Hold"
        if val == "Mixed":
            return "Watch"
        if val == "Expensive":
            return "Wait for pullback"
        return "Watch"

    if econ == "Melting Ice Cube":
        if val == "Cheap":
            return "Deep value only"
        return "Avoid"

    if econ == "Turnaround":
        if val == "Cheap":
            return "Speculative / Confirm"
        if val in {"Fair", "Mixed"}:
            return "Watch"
        return "Avoid"

    # Fallback
    if val == "Cheap":
        return "Watch"
    if val == "Fair":
        return "Watch"
    return "Avoid"


# -----------------------------
# Main compute function
# -----------------------------
def compute_valuation_sanity(
    base_df: pd.DataFrame,
    vc_df: pd.DataFrame,
    cap_eff_df: Optional[pd.DataFrame] = None,
    *,
    # Normalization ranges
    roic_hi: float = 30.0,
    growth_hi: float = 25.0,
    # Weighting
    w_roic: float = 0.6,
    w_growth: float = 0.4,
    # Fair multiple bands
    ev_lo: float = 6.0,
    ev_hi: float = 20.0,
    pe_lo: float = 8.0,
    pe_hi: float = 30.0,
    # Valuation band for Cheap/Fair/Expensive
    mispricing_band: float = 0.15,
) -> ValuationSanityResult:
    warnings: List[str] = []

    if base_df is None or base_df.empty:
        return ValuationSanityResult(pd.DataFrame(), pd.DataFrame(), ["Base table is empty (base_df)."])

    # Defensive copies
    b = base_df.copy()
    v = vc_df.copy() if vc_df is not None else pd.DataFrame()
    c = cap_eff_df.copy() if cap_eff_df is not None else pd.DataFrame()

    # Normalize column names expected
    # Required keys: Ticker
    if "Ticker" not in b.columns:
        return ValuationSanityResult(pd.DataFrame(), pd.DataFrame(), ["base_df missing 'Ticker' column."])
    if "Ticker" not in v.columns:
        warnings.append("vc_df missing 'Ticker' column; economics will be incomplete.")

    # Bring ROIC_% and Growth_% from vc_df
    # (vc_df already has ROIC_% and Growth_% in your current pipeline)
    keep_v_cols = [col for col in ["Ticker", "ROIC_%", "Growth_%", "Value_Creation_Score", "Economic_Quadrant"] if col in v.columns]
    v_small = v[keep_v_cols].drop_duplicates(subset=["Ticker"]) if keep_v_cols else pd.DataFrame({"Ticker": b["Ticker"]})

    # Bring NetDebt_EBITDA from capital efficiency if present
    netdebt_col = None
    if c is not None and not c.empty:
        if "NetDebt_EBITDA" in c.columns:
            netdebt_col = "NetDebt_EBITDA"
        elif "NetDebt/EBITDA" in c.columns:
            netdebt_col = "NetDebt/EBITDA"  # just in case
    if netdebt_col:
        c_small = c[["Ticker", netdebt_col]].drop_duplicates(subset=["Ticker"]).rename(columns={netdebt_col: "NetDebt_EBITDA"})
    else:
        c_small = pd.DataFrame({"Ticker": b["Ticker"], "NetDebt_EBITDA": np.nan})
        warnings.append("NetDebt_EBITDA not found in capital efficiency output; leverage adjustment skipped.")

    # Merge base + economics + leverage
    m = (
        b.merge(v_small, on="Ticker", how="left")
         .merge(c_small, on="Ticker", how="left")
    )

    # Ensure key multiples exist in base_df
    if "EV_EBITDA" not in m.columns:
        warnings.append("base_df missing EV_EBITDA; EV valuation will be N/A.")
        m["EV_EBITDA"] = np.nan
    if "PE_Proxy" not in m.columns:
        warnings.append("base_df missing PE_Proxy; PE valuation will be N/A.")
        m["PE_Proxy"] = np.nan

    # Convert key inputs to numeric
    for col in ["ROIC_%", "Growth_%", "EV_EBITDA", "PE_Proxy", "NetDebt_EBITDA"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    # Scores
    m["ROIC_Score_0_100"] = m["ROIC_%"].apply(lambda x: roic_score_0_100(x, 0.0, roic_hi))
    m["Growth_Score_0_100"] = m["Growth_%"].apply(lambda x: growth_score_0_100(x, 0.0, growth_hi))
    m["Quality_0_100"] = m.apply(
        lambda r: quality_0_100(r["ROIC_Score_0_100"], r["Growth_Score_0_100"], w_roic=w_roic, w_growth=w_growth),
        axis=1,
    )

    # Fair multiples from quality
    m["Fair_EV_EBITDA"] = m["Quality_0_100"].apply(lambda q: _fair_multiple_from_quality(q, ev_lo, ev_hi))
    m["Fair_PE"] = m["Quality_0_100"].apply(lambda q: _fair_multiple_from_quality(q, pe_lo, pe_hi))

    # Leverage adjustment (only affects EV multiple; P/E often already bakes in leverage indirectly)
    m["Fair_EV_EBITDA"] = m.apply(lambda r: _leverage_adjust_multiple(r["Fair_EV_EBITDA"], r["NetDebt_EBITDA"]), axis=1)

    # Invalidate multiples if denominators are negative / nonsensical
    # (EV/EBITDA <= 0 often indicates negative EBITDA; P/E <= 0 indicates negative earnings)
    m.loc[(m["EV_EBITDA"].notna()) & (m["EV_EBITDA"] <= 0), "EV_EBITDA"] = np.nan
    m.loc[(m["PE_Proxy"].notna()) & (m["PE_Proxy"] <= 0), "PE_Proxy"] = np.nan

    # Labels
    m["EV_Valuation"] = m.apply(lambda r: _label_vs_fair(r["EV_EBITDA"], r["Fair_EV_EBITDA"], band=mispricing_band), axis=1)
    m["PE_Valuation"] = m.apply(lambda r: _label_vs_fair(r["PE_Proxy"], r["Fair_PE"], band=mispricing_band), axis=1)
    m["Valuation_Overall"] = m.apply(lambda r: _overall_valuation(r["EV_Valuation"], r["PE_Valuation"]), axis=1)

    # Useful diagnostic gaps (%)
    def _gap_pct(curr, fair):
        c0 = _to_num(curr)
        f0 = _to_num(fair)
        if np.isnan(c0) or np.isnan(f0) or f0 <= 0:
            return np.nan
        return (c0 / f0) - 1.0

    m["EV_Gap_%"] = m.apply(lambda r: _gap_pct(r["EV_EBITDA"], r["Fair_EV_EBITDA"]), axis=1) * 100.0
    m["PE_Gap_%"] = m.apply(lambda r: _gap_pct(r["PE_Proxy"], r["Fair_PE"]), axis=1) * 100.0

    # Mispricing table (top table in your UI)
    cols_mispricing = [
        "Ticker",
        "ROIC_%", "Growth_%", "ROIC_Score_0_100", "Growth_Score_0_100", "Quality_0_100",
        "EV_EBITDA", "Fair_EV_EBITDA", "EV_Valuation", "EV_Gap_%",
        "PE_Proxy", "Fair_PE", "PE_Valuation", "PE_Gap_%",
        "Valuation_Overall",
        "NetDebt_EBITDA",
    ]
    mispricing_df = m[[c for c in cols_mispricing if c in m.columns]].copy()

    # Final decision table (Economics × Valuation)
    # Ensure economic quadrant exists
    if "Economic_Quadrant" not in m.columns:
        m["Economic_Quadrant"] = "Insufficient data"

    decision = m[[
        "Ticker", "Economic_Quadrant", "Valuation_Overall",
        "Value_Creation_Score", "ROIC_%", "Growth_%", "EV_EBITDA", "PE_Proxy"
    ]].copy()

    decision["Decision"] = decision.apply(lambda r: _decision(r["Economic_Quadrant"], r["Valuation_Overall"]), axis=1)

    # Reorder to match your UI style
    decision_df = decision[[
        "Ticker", "Economic_Quadrant", "Valuation_Overall", "Decision",
        "Value_Creation_Score", "ROIC_%", "Growth_%", "EV_EBITDA", "PE_Proxy"
    ]].copy()

    # Sorting (optional: most actionable first)
    # Put Compounders first, then by cheapness
    econ_rank = {"Compounder": 0, "Turnaround": 1, "Melting Ice Cube": 2, "Insufficient data": 3}
    val_rank = {"Cheap": 0, "Fair": 1, "Mixed": 2, "Expensive": 3, "N/A": 4}
    decision_df["_econ_rank"] = decision_df["Economic_Quadrant"].map(econ_rank).fillna(9)
    decision_df["_val_rank"] = decision_df["Valuation_Overall"].map(val_rank).fillna(9)
    decision_df = decision_df.sort_values(["_econ_rank", "_val_rank", "Ticker"]).drop(columns=["_econ_rank", "_val_rank"])

    return ValuationSanityResult(
        mispricing_df=mispricing_df,
        decision_df=decision_df,
        warnings=warnings,
    )


# Optional wrapper (handy if some app code expects tuple unpacking)
def load_valuation_sanity(base_df: pd.DataFrame, vc_df: pd.DataFrame, cap_eff_df: Optional[pd.DataFrame] = None):
    """
    Tuple-return wrapper:
    (mispricing_df, decision_df, warnings)
    """
    vs = compute_valuation_sanity(base_df, vc_df, cap_eff_df)
    return vs.mispricing_df, vs.decision_df, vs.warnings
