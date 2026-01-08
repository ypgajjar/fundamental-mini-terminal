import yfinance as yf
import pandas as pd
import numpy as np


def safe_round(x, ndigits=2):
    try:
        if x is None:
            return None
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return round(x, ndigits)
    except Exception:
        return None


def cagr(start, end, years):
    """
    CAGR only makes sense for positive start/end.
    Returns decimal (0.10 = 10%) or None.
    """
    if start is None or end is None or years is None:
        return None
    if years <= 0:
        return None
    try:
        start = float(start)
        end = float(end)
    except Exception:
        return None
    # Prevent complex results / nonsense
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def compute_annual_normalized(tickers):
    rows = []
    warnings = []

    for sym in tickers:
        t = yf.Ticker(sym)

        try:
            fin = t.financials  # annual income statement
        except Exception:
            fin = None

        if fin is None or fin.empty:
            warnings.append(f"{sym}: No annual financials from Yahoo/yfinance")
            continue

        def get_row(name):
            if name not in fin.index:
                return None
            s = pd.to_numeric(fin.loc[name], errors="coerce").dropna()
            if s.empty:
                return None
            # Sort newest -> oldest if columns are dates
            try:
                s.index = pd.to_datetime(s.index)
                s = s.sort_index(ascending=False)
            except Exception:
                pass
            return s

        rev = get_row("Total Revenue")
        ebitda = get_row("EBITDA")
        ni = get_row("Net Income")

        # Latest values
        rev_latest = float(rev.iloc[0]) if rev is not None else None
        ebitda_latest = float(ebitda.iloc[0]) if ebitda is not None else None
        ni_latest = float(ni.iloc[0]) if ni is not None else None

        # Determine available history length (we’ll use up to 4 years of points)
        rev_len = len(rev) if rev is not None else 0
        ebitda_len = len(ebitda) if ebitda is not None else 0
        ni_len = len(ni) if ni is not None else 0

        max_len = max(rev_len, ebitda_len, ni_len)
        years_available = min(max_len, 4)  # cap at 4 points
        span_years = years_available - 1

        if rev is None:
            warnings.append(f"{sym}: Missing annual Total Revenue")
        if ebitda is None:
            warnings.append(f"{sym}: Missing annual EBITDA")
        if ni is None:
            warnings.append(f"{sym}: Missing annual Net Income")

        # CAGR using available span (needs at least 3 points => span_years>=2)
        rev_cagr = None
        ebitda_cagr = None
        ni_cagr = None

        if years_available >= 3:
            # Use the oldest point within our chosen window
            idx = span_years
            if rev is not None and rev_len > idx:
                rev_cagr = cagr(rev.iloc[idx], rev.iloc[0], span_years)
            if ebitda is not None and ebitda_len > idx:
                ebitda_cagr = cagr(ebitda.iloc[idx], ebitda.iloc[0], span_years)
            if ni is not None and ni_len > idx:
                ni_cagr = cagr(ni.iloc[idx], ni.iloc[0], span_years)

        # Average margins over last up to 3 years (where available)
        margins = []
        net_margins = []
        if rev is not None:
            n = min(len(rev), 3)
            for i in range(n):
                r = float(rev.iloc[i])
                if r == 0:
                    continue
                if ebitda is not None and i < len(ebitda):
                    margins.append(float(ebitda.iloc[i]) / r)
                if ni is not None and i < len(ni):
                    net_margins.append(float(ni.iloc[i]) / r)

        rows.append({
            "Ticker": sym,

            "Revenue_Annual_USD_M": safe_round(rev_latest / 1e6) if rev_latest is not None else None,
            "EBITDA_Annual_USD_M": safe_round(ebitda_latest / 1e6) if ebitda_latest is not None else None,
            "NetIncome_Annual_USD_M": safe_round(ni_latest / 1e6) if ni_latest is not None else None,

            "Revenue_CAGR_%": safe_round(rev_cagr * 100) if rev_cagr is not None else None,
            "EBITDA_CAGR_%": safe_round(ebitda_cagr * 100) if ebitda_cagr is not None else None,
            "NetIncome_CAGR_%": safe_round(ni_cagr * 100) if ni_cagr is not None else None,

            "Avg_EBITDA_Margin_%": safe_round(np.mean(margins) * 100) if margins else None,
            "Avg_Net_Margin_%": safe_round(np.mean(net_margins) * 100) if net_margins else None,

            "CAGR_Span_Years": span_years if years_available >= 3 else None,
        })

    return pd.DataFrame(rows), warnings
