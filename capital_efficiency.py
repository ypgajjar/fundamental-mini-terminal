# capital_efficiency.py
# Bloomberg-style Capital Efficiency & Risk metrics with robust Yahoo/yfinance field fallbacks.
#
# Outputs (latest annual where possible):
# - ROIC_% (NOPAT / Invested Capital)
# - NetDebt_EBITDA (Net Debt / EBITDA)
# - Interest_Coverage (EBIT / |Interest Expense|)
# - FCF_to_NI (Free Cash Flow / Net Income)
# - Data_Quality flags (which inputs were used / estimated)
#
# Notes:
# - Yahoo labels vary a lot; this module tries multiple row names for each concept.
# - ROIC is only computed when the math is meaningful (positive invested capital, non-null EBIT).
# - Effective tax rate is derived from Tax Provision / Pretax Income when possible,
#   otherwise defaults to a normalized tax rate (default 21%).
#
# Install:
#   pip install yfinance pandas numpy

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class CapitalEfficiencyResult:
    df: pd.DataFrame
    warnings: List[str]


# ---------- helpers ----------

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    a = _to_float(a)
    b = _to_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _get_latest_value(df: Optional[pd.DataFrame], row_candidates: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    """
    Return latest value from df for the first matching row name in row_candidates.
    Yahoo 'yfinance' frames are typically indexed by line item, columns by period.
    """
    if df is None or df.empty:
        return None, None
    for r in row_candidates:
        if r in df.index:
            s = pd.to_numeric(df.loc[r], errors="coerce").dropna()
            if not s.empty:
                # yfinance usually has most recent first; if not, this still picks first non-null.
                return _to_float(s.iloc[0]), r
    return None, None


def _abs_interest(x: Optional[float]) -> Optional[float]:
    v = _to_float(x)
    if v is None:
        return None
    return abs(v)


# ---------- core ----------

def compute_capital_efficiency(
    tickers: List[str],
    normalized_tax_rate: float = 0.21,
    use_ttm_for_income_if_available: bool = False
) -> CapitalEfficiencyResult:
    """
    Computes ROIC and key risk ratios.
    - Uses annual statements by default.
    - If use_ttm_for_income_if_available=True, will prefer TTM income items when present
      (Yahoo sometimes provides "TTM" columns via a different endpoint; yfinance often does not).
    """

    warnings: List[str] = []
    rows: List[Dict] = []

    # Candidate row names across Yahoo/yfinance variants
    EBIT_ROWS = [
    "EBIT",
    "Operating Income",
    "Operating Income or Loss",
    "Operating Income (Loss)",
    "Total Operating Income as Reported",   # seen on Yahoo Finance tables
    "Operating Profit"
    ]
    EBITDA_ROWS = ["EBITDA"]
    NET_INCOME_ROWS = ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"]
    PRETAX_ROWS = ["Pretax Income", "Earnings Before Tax", "Income Before Tax"]
    TAX_ROWS = ["Tax Provision", "Income Tax Expense", "Provision for Income Taxes"]
    INTEREST_EXP_ROWS = ["Interest Expense", "Interest Expense Non Operating", "Interest Expense, Net"]
    # Balance sheet rows
    TOTAL_ASSETS_ROWS = ["Total Assets"]
    CASH_ROWS = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"]
    CURR_LIAB_ROWS = ["Total Current Liabilities", "Current Liabilities"]
    # Debt: Yahoo varies; try totals first, then components
    TOTAL_DEBT_ROWS = ["Total Debt", "Total Debt & Capital Lease Obligation"]
    SHORT_DEBT_ROWS = ["Short Long Term Debt", "Short Term Debt", "Current Debt", "Current Debt And Capital Lease Obligation"]
    LONG_DEBT_ROWS = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
    EQUITY_ROWS = ["Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"]
    MINORITY_ROWS = ["Minority Interest", "Minority Interests"]

    # Cashflow rows
    FCF_ROWS = ["Free Cash Flow"]  # yfinance sometimes has this; otherwise we derive from OCF + CapEx
    OCF_ROWS = ["Total Cash From Operating Activities", "Operating Cash Flow"]
    CAPEX_ROWS = ["Capital Expenditures", "Capital Expenditure"]

    for sym in tickers:
        t = yf.Ticker(sym)

        try:
            fin_a = t.financials          # annual income statement
            bal_a = t.balance_sheet       # annual balance sheet
            cf_a = t.cashflow             # annual cash flow
        except Exception as e:
            warnings.append(f"{sym}: Yahoo fetch failed: {e}")
            continue

        # ---------- Income ----------
        ebit, ebit_row = _get_latest_value(fin_a, EBIT_ROWS)
        ebitda, ebitda_row = _get_latest_value(fin_a, EBITDA_ROWS)
        ni, ni_row = _get_latest_value(fin_a, NET_INCOME_ROWS)
        pretax, pretax_row = _get_latest_value(fin_a, PRETAX_ROWS)
        tax, tax_row = _get_latest_value(fin_a, TAX_ROWS)
        interest_exp, interest_row = _get_latest_value(fin_a, INTEREST_EXP_ROWS)

        # Fallback: derive EBIT if missing (EBIT ≈ Pretax Income + |Interest Expense|)
        if ebit is None and pretax is not None and interest_exp is not None:
            ebit = pretax + abs(interest_exp)
            ebit_row = f"derived({pretax_row}+abs({interest_row}))"

        # Effective tax rate
        tax_rate = None
        tax_rate_source = None
        if pretax is not None and pretax > 0 and tax is not None:
            tr = tax / pretax
            # clamp to a sensible band [0, 0.5] to avoid one-off weirdness
            tr = max(0.0, min(float(tr), 0.5))
            tax_rate = tr
            tax_rate_source = f"effective({tax_row}/{pretax_row})"
        else:
            tax_rate = float(normalized_tax_rate)
            tax_rate_source = f"normalized({normalized_tax_rate:.2%})"

        # NOPAT
        nopat = None
        if ebit is not None:
            nopat = ebit * (1.0 - tax_rate)

        # ---------- Balance Sheet ----------
        total_assets, ta_row = _get_latest_value(bal_a, TOTAL_ASSETS_ROWS)
        cash, cash_row = _get_latest_value(bal_a, CASH_ROWS)
        curr_liab, cl_row = _get_latest_value(bal_a, CURR_LIAB_ROWS)

        total_debt, td_row = _get_latest_value(bal_a, TOTAL_DEBT_ROWS)
        short_debt, sd_row = _get_latest_value(bal_a, SHORT_DEBT_ROWS)
        long_debt, ld_row = _get_latest_value(bal_a, LONG_DEBT_ROWS)

        equity, eq_row = _get_latest_value(bal_a, EQUITY_ROWS)
        minority, mi_row = _get_latest_value(bal_a, MINORITY_ROWS)

        # Debt fallback: if Total Debt missing, use short+long
        if total_debt is None:
            if short_debt is not None or long_debt is not None:
                total_debt = (short_debt or 0.0) + (long_debt or 0.0)
                td_row = f"derived({sd_row}+{ld_row})"

        # Invested Capital (Bloomberg-ish):
        # IC = (Total Assets - Cash) - (Current Liabilities - Short-term debt)
        invested_capital = None
        ic_source = None
        if total_assets is not None and curr_liab is not None:
            c = cash or 0.0
            sd = short_debt or 0.0
            invested_capital = (total_assets - c) - (curr_liab - sd)
            ic_source = f"IC=({ta_row}-{cash_row or 'cash0'})-({cl_row}-{sd_row or 'sd0'})"
        # Alternative fallback if key pieces are missing:
        # IC ≈ Total Debt + Equity (+ Minority) - Cash
        if invested_capital is None and total_debt is not None and equity is not None:
            c = cash or 0.0
            m = minority or 0.0
            invested_capital = (total_debt + equity + m) - c
            ic_source = f"IC=({td_row}+{eq_row}+{mi_row or 'mi0'})-{cash_row or 'cash0'}"

        # ROIC
        roic = None
        roic_note = ""
        if nopat is not None and invested_capital is not None and invested_capital > 0:
            roic = nopat / invested_capital
        else:
            if ebit is None:
                roic_note = "missing EBIT"
            elif invested_capital is None:
                roic_note = "missing IC inputs"
            elif invested_capital is not None and invested_capital <= 0:
                roic_note = "IC<=0"

        # ---------- Leverage ----------
        net_debt = None
        if total_debt is not None:
            net_debt = total_debt - (cash or 0.0)

        net_debt_ebitda = None
        if net_debt is not None and ebitda is not None and ebitda != 0:
            net_debt_ebitda = net_debt / ebitda

        # ---------- Interest coverage ----------
        interest_cov = None
        if ebit is not None and interest_exp is not None:
            denom = _abs_interest(interest_exp)
            interest_cov = _safe_div(ebit, denom)

        # ---------- Free cash flow ----------
        fcf, fcf_row = _get_latest_value(cf_a, FCF_ROWS)
        if fcf is None:
            ocf, ocf_row = _get_latest_value(cf_a, OCF_ROWS)
            capex, capex_row = _get_latest_value(cf_a, CAPEX_ROWS)
            # Yahoo capex typically negative. FCF = OCF + CapEx
            if ocf is not None and capex is not None:
                fcf = ocf + capex
                fcf_row = f"derived({ocf_row}+{capex_row})"

        fcf_to_ni = None
        if fcf is not None and ni is not None and ni != 0:
            fcf_to_ni = fcf / ni

        # ---------- Collect row ----------
        rows.append({
            "Ticker": sym,

            "ROIC_%": round(roic * 100, 2) if roic is not None else None,
            "NetDebt_EBITDA": round(net_debt_ebitda, 2) if net_debt_ebitda is not None else None,
            "Interest_Coverage": round(interest_cov, 2) if interest_cov is not None else None,
            "FCF_to_NI": round(fcf_to_ni, 2) if fcf_to_ni is not None else None,

            # Debug/quality columns (helpful for terminal diagnostics)
            "TaxRate_Source": tax_rate_source,
            "IC_Source": ic_source,
            "EBIT_Row": ebit_row,
            "EBITDA_Row": ebitda_row,
            "NI_Row": ni_row,
            "Interest_Row": interest_row,
            "FCF_Row": fcf_row,
            "ROIC_Note": roic_note,
        })

        # Warnings for missing major inputs
        if sym and roic is None and roic_note:
            warnings.append(f"{sym}: ROIC not computed ({roic_note}).")

    df = pd.DataFrame(rows)
    return CapitalEfficiencyResult(df=df, warnings=warnings)


# Convenience: keep backward-compatible signature if your app expects (df, warnings)
def compute_capital_efficiency_legacy(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    res = compute_capital_efficiency(tickers)
    return res.df, res.warnings
