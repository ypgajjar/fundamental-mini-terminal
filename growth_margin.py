from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class GrowthMarginResult:
    df: pd.DataFrame
    warnings: List[str]


def _get_quarterly_row(qfin: pd.DataFrame, row: str) -> Optional[pd.Series]:
    """Return numeric quarterly series sorted newest->oldest."""
    if qfin is None or qfin.empty or row not in qfin.index:
        return None
    s = pd.to_numeric(qfin.loc[row], errors="coerce").dropna()
    if s.empty:
        return None
    try:
        s.index = pd.to_datetime(s.index, errors="ignore")
        s = s.sort_index(ascending=False)
    except Exception:
        pass
    return s


def _ttm_sum(s: Optional[pd.Series], offset: int = 0) -> Optional[float]:
    """
    Sum 4 quarters to create a TTM number.
    offset=0 => latest 4Q, offset=4 => prior 4Q (YoY baseline).
    """
    if s is None or len(s) < offset + 4:
        return None
    return float(s.iloc[offset:offset + 4].sum())


def _yoy_growth(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    """Return YoY growth as a decimal (0.10 = 10%)."""
    if curr is None or prev is None or prev == 0:
        return None
    return (curr / prev) - 1.0


def _cagr(start: Optional[float], end: Optional[float], years: float) -> Optional[float]:
    if start is None or end is None or start <= 0 or years <= 0:
        return None
    return (end / start) ** (1.0 / years) - 1.0


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def compute_growth_and_margins(
    tickers: List[str],
    manual_net_income_ttm_usd: Optional[Dict[str, float]] = None,  # absolute USD (not millions)
    manual_revenue_ttm_usd: Optional[Dict[str, float]] = None,     # optional
    manual_ebitda_ttm_usd: Optional[Dict[str, float]] = None,      # optional
) -> GrowthMarginResult:
    """
    Computes:
      - TTM Revenue / EBITDA / EBIT / Net Income
      - Prior TTM Revenue / EBITDA / EBIT / Net Income (4Q before last 4Q)
      - YoY growth for the above
      - Margins on TTM: EBITDA%, EBIT%, Net%
      - 3Y CAGR (approx) using 12 quarters back if available

    Outputs are in USD millions for levels; % for margins/growth.
    """
    manual_net_income_ttm_usd = manual_net_income_ttm_usd or {}
    manual_revenue_ttm_usd = manual_revenue_ttm_usd or {}
    manual_ebitda_ttm_usd = manual_ebitda_ttm_usd or {}

    warnings: List[str] = []
    rows = []

    for sym in tickers:
        t = yf.Ticker(sym)

        try:
            qfin = t.quarterly_financials
        except Exception:
            qfin = None

        rev_q = _get_quarterly_row(qfin, "Total Revenue")
        ebitda_q = _get_quarterly_row(qfin, "EBITDA")
        ebit_q = _get_quarterly_row(qfin, "EBIT")
        ni_q = _get_quarterly_row(qfin, "Net Income")

        rev_ttm = _ttm_sum(rev_q, 0)
        ebitda_ttm = _ttm_sum(ebitda_q, 0)
        ebit_ttm = _ttm_sum(ebit_q, 0)
        ni_ttm = _ttm_sum(ni_q, 0)

        # Manual overrides (useful for MDLN if Yahoo/yfinance is incomplete)
        src_note = []
        if rev_ttm is None and sym in manual_revenue_ttm_usd:
            rev_ttm = float(manual_revenue_ttm_usd[sym])
            src_note.append("Revenue manual")
        if ebitda_ttm is None and sym in manual_ebitda_ttm_usd:
            ebitda_ttm = float(manual_ebitda_ttm_usd[sym])
            src_note.append("EBITDA manual")
        if ni_ttm is None and sym in manual_net_income_ttm_usd:
            ni_ttm = float(manual_net_income_ttm_usd[sym])
            src_note.append("NI manual")

        # Prior TTM (YoY baseline)
        rev_prev = _ttm_sum(rev_q, 4)
        ebitda_prev = _ttm_sum(ebitda_q, 4)
        ebit_prev = _ttm_sum(ebit_q, 4)
        ni_prev = _ttm_sum(ni_q, 4)

        # YoY growth
        rev_yoy = _yoy_growth(rev_ttm, rev_prev)
        ebitda_yoy = _yoy_growth(ebitda_ttm, ebitda_prev)
        ni_yoy = _yoy_growth(ni_ttm, ni_prev)

        # Margins (TTM)
        ebitda_margin = _safe_div(ebitda_ttm, rev_ttm)
        ebit_margin = _safe_div(ebit_ttm, rev_ttm)
        net_margin = _safe_div(ni_ttm, rev_ttm)

        # 3Y CAGR (approx): compare latest TTM vs TTM 12 quarters back (3 years)
        # Need at least 16 quarters to form two TTMs separated by 12 quarters.
        rev_ttm_3y_ago = _ttm_sum(rev_q, 12)
        ebitda_ttm_3y_ago = _ttm_sum(ebitda_q, 12)
        ni_ttm_3y_ago = _ttm_sum(ni_q, 12)

        rev_cagr_3y = _cagr(rev_ttm_3y_ago, rev_ttm, 3.0)
        ebitda_cagr_3y = _cagr(ebitda_ttm_3y_ago, ebitda_ttm, 3.0)
        ni_cagr_3y = _cagr(ni_ttm_3y_ago, ni_ttm, 3.0)

        # Warnings
        if rev_ttm is None:
            warnings.append(f"{sym}: Missing TTM Revenue (quarterly).")
        if ebitda_ttm is None:
            warnings.append(f"{sym}: Missing TTM EBITDA (quarterly).")
        if ni_ttm is None:
            warnings.append(f"{sym}: Missing TTM Net Income (quarterly).")

        rows.append({
            "Ticker": sym,
            "TTM_Revenue_USD_M": round(rev_ttm / 1e6, 2) if rev_ttm is not None else None,
            "TTM_EBITDA_USD_M": round(ebitda_ttm / 1e6, 2) if ebitda_ttm is not None else None,
            "TTM_NetIncome_USD_M": round(ni_ttm / 1e6, 2) if ni_ttm is not None else None,

            "Rev_YoY_%": round(rev_yoy * 100, 2) if rev_yoy is not None else None,
            "EBITDA_YoY_%": round(ebitda_yoy * 100, 2) if ebitda_yoy is not None else None,
            "NI_YoY_%": round(ni_yoy * 100, 2) if ni_yoy is not None else None,

            "EBITDA_Margin_%": round(ebitda_margin * 100, 2) if ebitda_margin is not None else None,
            "EBIT_Margin_%": round(ebit_margin * 100, 2) if ebit_margin is not None else None,
            "Net_Margin_%": round(net_margin * 100, 2) if net_margin is not None else None,

            "Rev_CAGR_3Y_%": round(rev_cagr_3y * 100, 2) if rev_cagr_3y is not None else None,
            "EBITDA_CAGR_3Y_%": round(ebitda_cagr_3y * 100, 2) if ebitda_cagr_3y is not None else None,
            "NI_CAGR_3Y_%": round(ni_cagr_3y * 100, 2) if ni_cagr_3y is not None else None,

            "Source_Notes": ", ".join(src_note) if src_note else "",
        })

    df = pd.DataFrame(rows)
    return GrowthMarginResult(df=df, warnings=warnings)
