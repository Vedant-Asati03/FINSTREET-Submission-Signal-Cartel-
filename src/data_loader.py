
import pandas as pd
import os
from fyers_apiv3 import fyersModel
from src.config import CLIENT_ID, ACCESS_TOKEN  # Imports keys from your config file

def fetch_fyers_data(ticker, start_date, end_date):
    """
    Fetches historical data using the Fyers API and returns a DataFrame.
    """
    try:
        # 1. Initialize Fyers Model
        # We read the token from config, or you can read from 'access_token.txt' if you prefer
        fyers = fyersModel.FyersModel(
            client_id=CLIENT_ID, 
            token=ACCESS_TOKEN, 
            log_path="./logs"
        )
        
        # 2. Define Data Request
        data_input = {
            "symbol": ticker,
            "resolution": "D",     # Daily
            "date_format": "1",    # YYYY-MM-DD
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": "1"
        }

        # 3. Fetch Data
        response = fyers.history(data=data_input)
        
        if response.get('s') != 'ok':
            print(f"❌ API Error: {response.get('message')}")
            return pd.DataFrame()

        # 4. Convert to DataFrame
        cols = ['epoch', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(response['candles'], columns=cols)
        
        # 5. Format Dates
        df['Date'] = pd.to_datetime(df['epoch'], unit='s').dt.date
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # 6. Clean up
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        
        return df

    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return pd.DataFrame()