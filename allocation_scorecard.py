from __future__ import annotations
import pandas as pd
import numpy as np

def _score_fcf_margin(x):
    if x is None or pd.isna(x): return None
    if x < 0: return 0
    if x >= 15: return 5
    if x >= 10: return 4
    if x >= 5:  return 3
    if x >= 0:  return 2
    return 0

def _score_capex_intensity(x):
    if x is None or pd.isna(x): return None
    if x <= 2:  return 5
    if x <= 4:  return 4
    if x <= 6:  return 3
    if x <= 10: return 2
    return 1

def _score_debt_change(x):
    if x is None or pd.isna(x): return None
    if x <= -5: return 5
    if x <= 5:  return 4
    if x <= 15: return 3
    if x <= 30: return 2
    return 1

def _score_div_yield(x):
    # Treat 0.00 from yahoo as missing unless you explicitly know it pays no dividend
    if x is None or pd.isna(x): return None
    if x == 0: return None
    if x >= 2: return 5
    if x >= 1: return 4
    if x > 0:  return 2
    return None

def compute_allocation_scorecard(cap_alloc_df: pd.DataFrame) -> pd.DataFrame:
    df = cap_alloc_df.copy()

    df["Score_CashGen"] = df["FCF_Margin_%"].apply(_score_fcf_margin)
    df["Score_Reinvest"] = df["Capex_%Rev"].apply(_score_capex_intensity)
    df["Score_Leverage"] = df["Debt_Change_%"].apply(_score_debt_change)
    df["Score_ShareholderReturn"] = df["Dividend_Yield_%"].apply(_score_div_yield)

    # Total: average of available pillar scores (so missing dividend doesn’t punish)
    score_cols = ["Score_CashGen","Score_Reinvest","Score_Leverage","Score_ShareholderReturn"]
    df["Pillars_Available"] = df[score_cols].notna().sum(axis=1)

    df["Score_Total_20"] = df[score_cols].mean(axis=1) * 4  # mean(0-5) -> scale to 0-20
    df["Score_Total_100"] = (df["Score_Total_20"] / 20 * 100).round(1)

    # Flags
    flags = []
    for _, r in df.iterrows():
        f = []
        if pd.notna(r.get("FCF_Margin_%")) and r["FCF_Margin_%"] < 0:
            f.append("NEG_FCF")
        if pd.notna(r.get("Debt_Change_%")) and r["Debt_Change_%"] > 30:
            f.append("DEBT_SPIKE")
        if pd.notna(r.get("Capex_%Rev")) and r["Capex_%Rev"] > 8:
            f.append("HIGH_CAPEX")
        flags.append(", ".join(f) if f else "")
    df["Flags"] = flags

    # Clean view
    out = df[[
        "Ticker",
        "FCF_Margin_%",
        "Capex_%Rev",
        "Debt_Change_%",
        "Dividend_Yield_%",
        "Score_Total_100",
        "Pillars_Available",
        "Flags"
    ]].sort_values("Score_Total_100", ascending=False, na_position="last")

    return out
