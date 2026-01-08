from __future__ import annotations

import time
import numpy as np
import pandas as pd
import yfinance as yf


def _to_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _first_non_null(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return None


def _get_latest(df: pd.DataFrame | None, row_names: list[str]):
    """Return most recent numeric value from the FIRST matching row name."""
    if df is None or df.empty:
        return None, None
    for r in row_names:
        if r in df.index:
            s = pd.to_numeric(df.loc[r], errors="coerce").dropna()
            if not s.empty:
                return _to_float(s.iloc[0]), r
    return None, None


def _get_prev(df: pd.DataFrame | None, row_names: list[str]):
    """Return second most recent numeric value from the FIRST matching row name."""
    if df is None or df.empty:
        return None, None
    for r in row_names:
        if r in df.index:
            s = pd.to_numeric(df.loc[r], errors="coerce").dropna()
            if len(s) >= 2:
                return _to_float(s.iloc[1]), r
    return None, None


def _infer_dividend_yield_pct(info: dict) -> float | None:
    """
    Yahoo/yfinance fields vary. We try multiple fields and sanity-check.
    Acceptable dividend yield typically 0–15% for most stocks (higher is rare/special).
    """
    # candidates (some are ratios, some are absolute)
    cand = _first_non_null(
        _to_float(info.get("dividendYield")),        # usually 0.0–0.1
        _to_float(info.get("trailingAnnualDividendYield")),  # also ratio
    )
    if cand is None:
        return None

    # If cand looks like already percent (e.g., 4.5 not 0.045), detect
    # Heuristic: if > 1, treat as percent already
    if cand > 1:
        pct = cand
    else:
        pct = cand * 100

    # Sanity cap: if > 30%, likely bad/misreported -> return None
    if pct < 0 or pct > 30:
        return None
    return round(pct, 2)


def _revenue_fallback(info: dict, fin: pd.DataFrame | None, warnings: list[str], sym: str) -> float | None:
    """
    Try income statement first; if missing (MDLN), fall back to info['totalRevenue'].
    """
    REV_ROWS = ["Total Revenue"]
    rev, _ = _get_latest(fin, REV_ROWS)
    if rev is not None:
        return rev

    info_rev = _to_float(info.get("totalRevenue"))
    if info_rev is not None:
        warnings.append(f"{sym}: revenue taken from info.totalRevenue (income statement missing).")
        return info_rev

    return None


def compute_capital_allocation(
    tickers: list[str],
    politeness_delay_s: float = 0.15,
):
    """
    Returns:
      df (USD Millions/Billions standardized)
      warnings (list[str])
    """
    rows = []
    warnings: list[str] = []

    OCF_ROWS = ["Total Cash From Operating Activities", "Operating Cash Flow"]
    CAPEX_ROWS = ["Capital Expenditures", "Capital Expenditure"]
    FCF_ROWS = ["Free Cash Flow"]

    TOTAL_DEBT_ROWS = ["Total Debt", "Total Debt & Capital Lease Obligation"]
    SHORT_DEBT_ROWS = ["Short Long Term Debt", "Short Term Debt", "Current Debt", "Current Debt And Capital Lease Obligation"]
    LONG_DEBT_ROWS = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]

    for sym in tickers:
        t = yf.Ticker(sym)

        try:
            fin = t.financials
            cf = t.cashflow
            bs = t.balance_sheet
            info = getattr(t, "info", {}) or {}
        except Exception as e:
            warnings.append(f"{sym}: fetch failed: {e}")
            continue

        # --- Revenue (USD) with fallback ---
        rev = _revenue_fallback(info, fin, warnings, sym)

        # --- Cash flow ---
        ocf, ocf_row = _get_latest(cf, OCF_ROWS)
        capex, capex_row = _get_latest(cf, CAPEX_ROWS)
        fcf, fcf_row = _get_latest(cf, FCF_ROWS)

        # derive FCF if missing but OCF/Capex available
        if fcf is None and ocf is not None and capex is not None:
            fcf = ocf + capex  # capex typically negative
            fcf_row = f"derived({ocf_row}+{capex_row})"

        # --- Debt ---
        debt_now, debt_row = _get_latest(bs, TOTAL_DEBT_ROWS)
        if debt_now is None:
            sd, _ = _get_latest(bs, SHORT_DEBT_ROWS)
            ld, _ = _get_latest(bs, LONG_DEBT_ROWS)
            if sd is not None or ld is not None:
                debt_now = (sd or 0.0) + (ld or 0.0)
                debt_row = "derived(short+long)"

        debt_prev, _ = _get_prev(bs, TOTAL_DEBT_ROWS)
        if debt_prev is None:
            sd_prev, _ = _get_prev(bs, SHORT_DEBT_ROWS)
            ld_prev, _ = _get_prev(bs, LONG_DEBT_ROWS)
            if sd_prev is not None or ld_prev is not None:
                debt_prev = (sd_prev or 0.0) + (ld_prev or 0.0)

        debt_change_pct = None
        if debt_now is not None and debt_prev not in (None, 0):
            debt_change_pct = (debt_now - debt_prev) / abs(debt_prev)

        # --- Dividend yield (fixed + sanity checked) ---
        div_yield_pct = _infer_dividend_yield_pct(info)

        # --- Standardized metrics ---
        rev_m = (rev / 1e6) if rev is not None else None
        fcf_m = (fcf / 1e6) if fcf is not None else None
        capex_m = (capex / 1e6) if capex is not None else None

        debt_b = (debt_now / 1e9) if debt_now is not None else None

        fcf_margin_pct = None
        if fcf is not None and rev not in (None, 0):
            fcf_margin_pct = (fcf / rev) * 100

        capex_pct_rev = None
        if capex is not None and rev not in (None, 0):
            capex_pct_rev = (abs(capex) / rev) * 100

        rows.append({
            "Ticker": sym,

            "Revenue_USD_M": round(rev_m, 2) if rev_m is not None else None,
            "FCF_USD_M": round(fcf_m, 2) if fcf_m is not None else None,
            "FCF_Margin_%": round(fcf_margin_pct, 2) if fcf_margin_pct is not None else None,

            "Capex_USD_M": round(capex_m, 2) if capex_m is not None else None,
            "Capex_%Rev": round(capex_pct_rev, 2) if capex_pct_rev is not None else None,

            "Debt_USD_B": round(debt_b, 2) if debt_b is not None else None,
            "Debt_Change_%": round(debt_change_pct * 100, 2) if debt_change_pct is not None else None,

            "Dividend_Yield_%": div_yield_pct,

            "FCF_Source": fcf_row,
            "Debt_Source": debt_row,
        })

        # --- warnings for missing critical fields ---
        if rev is None:
            warnings.append(f"{sym}: missing revenue (needed for margins).")
        if capex is None:
            warnings.append(f"{sym}: missing capex (reinvestment proxy limited).")
        if ocf is None and fcf is None:
            warnings.append(f"{sym}: missing cash flow (FCF unavailable).")

        time.sleep(politeness_delay_s)

    df = pd.DataFrame(rows)

    # nicer sort default: highest FCF margin
    if "FCF_Margin_%" in df.columns:
        df = df.sort_values("FCF_Margin_%", ascending=False, na_position="last")

    return df, warnings
