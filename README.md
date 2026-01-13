# FINSTREET-Submission-Signal-Cartel-
Submission for Round 2 of the Finstreet Hackathon

# 📈 ML-Driven Algo Trading Strategy (Kshitij 2026 - FinStreet)

**Team Name:** Signal Cartel  
**Stock Selected:** `NSE:SONATSOFTW-EQ` (Sonata Software Ltd)

## 📖 Overview
This project is an end-to-end algorithmic trading system developed for the **Round 2 Submission of FinStreet (Kshitij 2026)**. 

The system leverages a **Random Forest Classifier** to predict short-term price movements and executes trades automatically via the **Fyers API**. It features a robust data pipeline, advanced feature engineering, and strict risk management rules (Volatility Targeting).

---

## 🚀 Key Features
* **Automated Data Pipeline**: Fetches historical daily OHLCV data directly from the Fyers API.
* **Machine Learning Core**: Uses a Random Forest model trained on technical indicators (RSI, MACD, Bollinger Bands) to predict directional moves.
* **Walk-Forward Validation**: Retrains the model daily to prevent look-ahead bias and adapt to changing market regimes.
* **Risk Management**: Implements **Volatility Targeting** to dynamically adjust position sizes based on market risk.
* **Live Execution Ready**: Generates compliant API payloads for the Fyers trading ecosystem.

---

## 📂 Repository Structure

```text
Kshitij2026_AlgoStrategy/
│
├── Data/                        # Stores fetched market data & access tokens
│   └── SONATSOFTW_daily_...csv  # Historical data used for backtesting
│
├── src/                         # Source code for strategy logic
│   ├── config.py                # API Credentials & Strategy Settings
│   ├── features.py              # Technical Indicator Calculation
│   ├── strategy.py              # Random Forest Model & Backtest Logic
│   └── execution.py             # Fyers API Order Placement Logic
│
├── main.py                      # MAIN SCRIPT: Orchestrates the entire workflow
├── FYERS_API_Integration.ipynb  # Notebook to Authenticate & Fetch Data
├── predict_january.py           # Output logic for Jan 1 - Jan 8 Predictions
└── requirements.txt             # List of dependencies
