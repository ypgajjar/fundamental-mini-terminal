from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class FetchResult:
    df: pd.DataFrame
    warnings: List[str]


def _last_valid_close(tkr: yf.Ticker, days: int = 10) -> Optional[float]:
    """Last available close (EOD) over recent sessions (handles weekends/holidays)."""
    try:
        hist = tkr.history(period=f"{days}d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        s = hist["Close"].dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:
        return None


def _last_trade_price(tkr: yf.Ticker) -> Optional[float]:
    """
    Last trade / last price from fast_info when available.
    This is often near-real-time but depends on Yahoo feed/limits.
    """
    try:
        fi = getattr(tkr, "fast_info", None)
        if fi:
            lp = fi.get("last_price", None)
            if lp is not None:
                return float(lp)
    except Exception:
        pass

    # fallback
    try:
        info = tkr.info
        lp = info.get("regularMarketPrice")
        return float(lp) if lp is not None else None
    except Exception:
        return None


def _market_cap(tkr: yf.Ticker) -> Optional[float]:
    """Market cap in USD."""
    try:
        fi = getattr(tkr, "fast_info", None)
        if fi:
            mc = fi.get("market_cap", None)
            if mc is not None:
                return float(mc)
    except Exception:
        pass

    try:
        info = tkr.info
        mc = info.get("marketCap")
        return float(mc) if mc is not None else None
    except Exception:
        return None


def _get_quarterly_series(qfin: pd.DataFrame, row_name: str) -> Optional[pd.Series]:
    """Return quarterly row as a Series with newest-first ordering."""
    if qfin is None or qfin.empty:
        return None
    if row_name not in qfin.index:
        return None
    s = qfin.loc[row_name].copy()
    # yfinance columns are typically timestamps; keep numeric and drop NaN
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    # Usually newest quarter is first column already, but enforce sort desc by date
    try:
        s = s.sort_index(ascending=False)
    except Exception:
        pass
    return s


def _ttm_from_quarters(s: Optional[pd.Series]) -> Optional[float]:
    """Sum last 4 quarters."""
    if s is None or len(s) < 4:
        return None
    return float(s.iloc[:4].sum())


def _latest_quarter_value(s: Optional[pd.Series]) -> Optional[float]:
    if s is None or len(s) < 1:
        return None
    return float(s.iloc[0])


def _balance_latest(tkr: yf.Ticker) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (cash_and_equivalents, total_debt, net_debt) using latest quarter balance sheet.
    Many tickers have inconsistent naming; we try common Yahoo labels.
    Values in USD.
    """
    try:
        qb = tkr.quarterly_balance_sheet
    except Exception:
        qb = None

    if qb is None or qb.empty:
        return (None, None, None)

    # Work with the latest column (most recent quarter)
    try:
        qb = qb.copy()
        qb.columns = pd.to_datetime(qb.columns, errors="ignore")
        latest_col = sorted(qb.columns, reverse=True)[0]
    except Exception:
        latest_col = qb.columns[0]

    def pick(labels: List[str]) -> Optional[float]:
        for lab in labels:
            if lab in qb.index:
                v = qb.loc[lab, latest_col]
                v = pd.to_numeric(v, errors="coerce")
                if pd.notna(v):
                    return float(v)
        return None

    cash = pick([
        "Cash And Cash Equivalents",
        "Cash And Short Term Investments",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash",
    ])

    # Debt labels vary a lot
    total_debt = pick([
        "Total Debt",
        "Short Long Term Debt Total",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
    ])

    # If “Total Debt” missing, try approximate: long term + short term
    if total_debt is None:
        ltd = pick(["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        std = pick(["Short Term Debt", "Current Debt", "Short Long Term Debt"])
        if ltd is not None or std is not None:
            total_debt = (ltd or 0.0) + (std or 0.0)

    net_debt = None
    if total_debt is not None and cash is not None:
        net_debt = total_debt - cash

    return cash, total_debt, net_debt


def fetch_true_ttm(
    tickers: List[str],
    manual_net_income_ttm: Optional[Dict[str, float]] = None,  # USD (absolute)
    sleep_s: float = 0.25,
) -> FetchResult:
    """
    Fetch:
      - EOD close
      - last trade price
      - market cap
      - True TTM from quarterly_financials: revenue, EBITDA, EBIT, net income
      - EV and EV/EBITDA using latest quarter balance sheet
    All values returned in:
      - Prices in USD
      - MarketCap, EV in USD Billions
      - TTM metrics in USD Millions
    """
    manual_net_income_ttm = manual_net_income_ttm or {}
    warnings: List[str] = []
    rows = []

    for sym in tickers:
        tkr = yf.Ticker(sym)

        eod = _last_valid_close(tkr)
        last_px = _last_trade_price(tkr)
        mcap = _market_cap(tkr)

        # Quarterly income statement (true TTM = sum last 4 quarters)
        try:
            qfin = tkr.quarterly_financials
        except Exception:
            qfin = None

        rev_q = _get_quarterly_series(qfin, "Total Revenue")
        ebitda_q = _get_quarterly_series(qfin, "EBITDA")
        ebit_q = _get_quarterly_series(qfin, "EBIT")
        ni_q = _get_quarterly_series(qfin, "Net Income")

        rev_ttm = _ttm_from_quarters(rev_q)
        ebitda_ttm = _ttm_from_quarters(ebitda_q)
        ebit_ttm = _ttm_from_quarters(ebit_q)
        ni_ttm = _ttm_from_quarters(ni_q)

        ni_src = "Quarterly sum(4Q)"
        if ni_ttm is None and sym in manual_net_income_ttm:
            ni_ttm = float(manual_net_income_ttm[sym])
            ni_src = "Manual override"

        # Balance sheet for EV
        cash, total_debt, net_debt = _balance_latest(tkr)

        ev = None
        ev_src = "mcap + debt - cash"
        if mcap is not None and total_debt is not None and cash is not None:
            ev = mcap + total_debt - cash
        else:
            ev_src = "Missing (balance sheet or mcap)"

        # Multiples
        ev_ebitda = None
        if ev is not None and ebitda_ttm is not None and ebitda_ttm != 0:
            ev_ebitda = ev / ebitda_ttm  # both USD
        # Market cap / net income proxy P/E (not perfect, but useful)
        pe_proxy = None
        if mcap is not None and ni_ttm is not None and ni_ttm > 0:
            pe_proxy = mcap / ni_ttm

        # Units
        row = {
            "Ticker": sym,
            "Price_EOD_USD": round(eod, 2) if eod is not None else None,
            "Price_Last_USD": round(last_px, 2) if last_px is not None else None,
            "MarketCap_USD_B": round(mcap / 1e9, 2) if mcap is not None else None,
            "Revenue_TTM_USD_M": round(rev_ttm / 1e6, 2) if rev_ttm is not None else None,
            "EBITDA_TTM_USD_M": round(ebitda_ttm / 1e6, 2) if ebitda_ttm is not None else None,
            "EBIT_TTM_USD_M": round(ebit_ttm / 1e6, 2) if ebit_ttm is not None else None,
            "NetIncome_TTM_USD_M": round(ni_ttm / 1e6, 2) if ni_ttm is not None else None,
            "NI_Source": ni_src,
            "Cash_USD_B": round(cash / 1e9, 2) if cash is not None else None,
            "TotalDebt_USD_B": round(total_debt / 1e9, 2) if total_debt is not None else None,
            "EV_USD_B": round(ev / 1e9, 2) if ev is not None else None,
            "EV_Source": ev_src,
            "EV_EBITDA": round(ev_ebitda, 2) if ev_ebitda is not None else None,
            "PE_Proxy": round(pe_proxy, 2) if pe_proxy is not None else None,
        }

        # Warnings for visibility
        if row["NetIncome_TTM_USD_M"] is None:
            warnings.append(f"{sym}: Missing quarterly net income on Yahoo/yfinance.")
        if row["EBITDA_TTM_USD_M"] is None:
            warnings.append(f"{sym}: Missing quarterly EBITDA on Yahoo/yfinance.")
        if row["EV_USD_B"] is None:
            warnings.append(f"{sym}: Cannot compute EV (need cash & debt).")

        rows.append(row)
        time.sleep(sleep_s)

    df = pd.DataFrame(rows)
    return FetchResult(df=df, warnings=warnings)
