# Mini Terminal — Fundamentals & Valuation (Yahoo Finance)

A modular, Bloomberg-style mini research terminal for equities using Yahoo Finance (`yfinance`) data.  
It computes True TTM fundamentals, annual normalized growth, capital efficiency, capital allocation quality, value creation, valuation sanity, fair multiples, and final decision logic.

---

## 📈 Features

- True TTM (sum of quarterly financials)
- Annual normalized growth + margins (Bloomberg-style)
- ROIC + leverage + coverage (capital efficiency)
- Capital allocation (FCF, capex intensity, debt change, dividends)
- Allocation scorecard (rule-based + flags)
- Value creation quadrant:
  - Compounder  
  - Cash Cow  
  - Value Destroyer  
  - Melting Ice Cube
- Valuation sanity:
  - Fair EV/EBITDA band
  - Fair P/E band
  - Cheap / Fair / Expensive classification
  - Final decision logic (Economics × Valuation)
- Streamlit UI for interactive screening & diagnostics

---

## 🧩 Pipeline Architecture

```
TTM → Annual Normalized → Capital Efficiency → Allocation → Value Creation → Valuation Sanity
```

---

## 🗂 Module Overview

| Module | Description |
|---|---|
| `data_engine.py` | Fetches True TTM + EV from balance sheet |
| `growth_margin.py` | Computes YoY growth & margins |
| `annual_normalized.py` | Computes normalized annual CAGR + margins |
| `capital_efficiency.py` | Computes ROIC, leverage, coverage |
| `capital_allocation.py` | Computes FCF, capex intensity, debt change, dividends |
| `allocation_scorecard.py` | Rule-based allocation & flagging |
| `value_creation.py` | ROIC × Growth quadrant + conclusion |
| `valuation_sanity.py` | Fair multiples + mispricing + decision |
| `app.py` | Streamlit interface tying modules together |

---

## 🛠 Tech Stack

- Python
- Streamlit
- yfinance
- pandas / numpy
- plotly (visualization)

---

## ⬇ Installation

Clone the repo:

```
git clone https://github.com/ypgajjar/fundamental-mini-terminal.git
cd fundamental-mini-terminal
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶ Run

```
streamlit run app.py
```

Add/modify default tickers inside `app.py`.

---

## 📊 Data Source

Yahoo Finance via `yfinance`  
(No paid APIs required)

---

## ⚠ Disclaimer

For research & educational purposes only.  
Not investment advice.  
Data quality depends on Yahoo Finance coverage.

---

## 📄 License

MIT

---

## 🤝 Contributions

Modular architecture enables easy extensions:
- additional factors
- valuation models
- dashboard views
- sector / ETF support

Pull requests welcome.
