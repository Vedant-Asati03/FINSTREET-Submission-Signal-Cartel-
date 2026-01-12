# src/features.py
import pandas as pd
import numpy as np

def generate_features(df):
    """
    Generates technical indicators (RSI, MACD, Bollinger Bands) for the strategy.
    """
    data = df.copy()
    
    # 1. Returns & Range
    data['Returns'] = data['Close'].pct_change()
    data['Range'] = (data['High'] - data['Low']) / data['Close']

    # 2. RSI (14-Day)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal_Line']

    # 4. Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(20).mean()
    std_20 = data['Close'].rolling(20).std()
    data['Upper_Band'] = data['SMA_20'] + (2 * std_20)
    data['Lower_Band'] = data['SMA_20'] - (2 * std_20)
    data['Pct_B'] = (data['Close'] - data['Lower_Band']) / (data['Upper_Band'] - data['Lower_Band'])

    # 5. Volatility (For Position Sizing)
    data['Rolling_Vol'] = data['Returns'].rolling(20).std() * np.sqrt(252)

    # 6. Lag Features (Shifted for Prediction)
    # We use Yesterday's data to predict Today
    features = ['Returns', 'Range', 'RSI', 'MACD_Hist', 'Pct_B']
    for feat in features:
        data[f'{feat}_Lag1'] = data[feat].shift(1)
        
    # Target: 1 if Next Close > Today Close (Binary Classification)
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.dropna()