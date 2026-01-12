# src/strategy.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.config import CONFIDENCE_THRESHOLD, TARGET_VOL, MAX_LEVERAGE

def run_backtest(data):
    """
    Runs a Walk-Forward Backtest using Random Forest.
    """
    trade_log = []
    # Use only the Lagged features for prediction (No Look-ahead bias)
    feature_cols = [c for c in data.columns if 'Lag1' in c]
    
    # Start loop after enough data is available (e.g., 50 days)
    for i in range(50, len(data)):
        current_date = data.index[i]
        
        # 1. Train on PAST data only
        train_df = data.iloc[:i]
        X_train = train_df[feature_cols]
        y_train = train_df['Target']
        
        # 2. Predict for TODAY
        X_today = data.iloc[[i]][feature_cols]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)
        
        prob_up = model.predict_proba(X_today)[0][1]
        
        # 3. Generate Signal
        action = "CASH"
        if prob_up > CONFIDENCE_THRESHOLD:
            action = "LONG"
        elif prob_up < (1 - CONFIDENCE_THRESHOLD):
            action = "SHORT"
            
        # 4. Position Sizing (Volatility Targeting)
        # Use YESTERDAY'S volatility to determine size
        vol = data.iloc[i-1]['Rolling_Vol']
        size = min(TARGET_VOL / vol, MAX_LEVERAGE) if vol > 0 else 0
        
        # 5. Calculate PnL
        actual_ret = data.iloc[i]['Returns']
        strat_ret = 0
        if action == "LONG": strat_ret = size * actual_ret
        elif action == "SHORT": strat_ret = size * -actual_ret
        
        trade_log.append({
            "Date": current_date,
            "Action": action,
            "Prob": round(prob_up, 2),
            "Size": round(size, 2),
            "Strat_Ret": strat_ret
        })
        
    return pd.DataFrame(trade_log).set_index("Date")