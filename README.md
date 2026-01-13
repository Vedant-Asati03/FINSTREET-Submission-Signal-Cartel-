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
├── Data/                        # Data Generation Zone
│   ├── FYERS_API_Integration.ipynb  # [STEP 1] Run this to fetch data
│   ├── SONATSOFTW_daily_...csv      # The output CSV (Historical Data)
│   └── access_token.txt             # The generated token
│
├── src/                         # Strategy Logic Zone
│   ├── __init__.py              # (Empty file)
│   ├── config.py                # API Credentials & Settings
│   ├── features.py              # Feature Engineering
│   ├── strategy.py              # Random Forest Model
│   └── execution.py             # Trade Execution Logic
│
├── main.py                      # [STEP 2] Main Strategy Script
├── predict_january.py           # [STEP 3] Prediction Rules for Judges
├── requirements.txt
└── README.md
