import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier

# --- IMPORT SETTINGS FROM SRC ---
try:
    from src.config import TICKER, DATA_FILE_PATH, CONFIDENCE_THRESHOLD
except ImportError:
    # Fallback if running standalone without setting pythonpath
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    from config import TICKER, DATA_FILE_PATH, CONFIDENCE_THRESHOLD

def generate_prediction_report():
    print(f"🔮 PREDICTION REPORT: {TICKER}")
    print("=" * 60)
    print(f"Target Period: Jan 1, 2026 - Jan 8, 2026")
    print("=" * 60)

    # 1. LOAD & PREPARE DATA (Nov-Dec 2025)
    if not os.path.exists(DATA_FILE_PATH):
        print(f"❌ Error: Data file not found at {DATA_FILE_PATH}")
        return

    df = pd.read_csv(DATA_FILE_PATH, index_col=0, parse_dates=True)
    
    # Re-create features for training
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    
    # Indicators
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
    
    data['Rolling_Vol'] = data['Returns'].rolling(20).std() * np.sqrt(252)

    # Lag features (Yesterday's data predicts Today)
    features = ['Returns', 'Range', 'RSI', 'MACD_Hist']
    for feat in features:
        data[f'{feat}_Lag1'] = data[feat].shift(1)

    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data.dropna(inplace=True)

    # 2. TRAIN MODEL (Full Nov-Dec Data)
    feature_cols = [f'{col}_Lag1' for col in features]
    X = data[feature_cols]
    y = data['Target']

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)

    # 3. OUTPUT PREDICTION LOGIC
    print("\n[MODEL LOGIC FOR JANUARY]")
    print(f"The model has been trained on {len(data)} days of historical data.")
    print("It will generate signals based on the following feature importance:")
    
    importances = model.feature_importances_
    for name, importance in zip(feature_cols, importances):
        print(f"  - {name:<15}: {importance:.4f}")

    print("\n[DEPLOYMENT SCHEDULE JAN 1 - JAN 8]")
    print("Since future market data is unavailable, the strategy follows these rules:")
    
    dates = [
        "2026-01-01 (Thu)", "2026-01-02 (Fri)", 
        "2026-01-05 (Mon)", "2026-01-06 (Tue)", 
        "2026-01-07 (Wed)", "2026-01-08 (Thu)"
    ]

    print(f"\n{'DATE':<20} | {'STATUS':<15} | {'ACTION RULE'}")
    print("-" * 65)
    
    for d in dates:
        print(f"{d:<20} | PENDING         | IF Prob(Up) > {CONFIDENCE_THRESHOLD} -> BUY")
        print(f"{'':<20} |                 | IF Prob(Up) < {1-CONFIDENCE_THRESHOLD} -> SHORT")
        print(f"{'':<20} |                 | ELSE -> CASH")
        print("-" * 65)

    print("\n✅ Verification: This logic is automated in 'main.py' using Fyers API.")

if __name__ == "__main__":
    generate_prediction_report()
