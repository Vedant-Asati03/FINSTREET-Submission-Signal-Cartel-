# src/execution.py
from fyers_apiv3 import fyersModel

def execute_order(fyers, symbol, action, qty=1):
    """
    Constructs and sends a trade order to the Fyers API.
    """
    if action == "CASH":
        return
        
    # Map Action to Fyers Side (1=Buy, -1=Sell)
    side = 1 if action == "LONG" else -1
    
    # Construct the Order Payload (Standard Fyers Format)
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 2,           # Market Order
        "side": side,
        "productType": "INTRADAY",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY"
    }
    
    try:
        # ---------------------------------------------------------
        # Code is commented to avoid accidental real money trades
        # response = fyers.place_order(data=data)
        # ---------------------------------------------------------
        
        # PROOF OF LOGIC: Print the exact payload we WOULD send
        print(f"\n [EXECUTION TRIGGERED] Action: {action} | Symbol: {symbol}")
        print(f"API Payload: {data}")
        
    except Exception as e:
        print(f"❌ Execution Failed: {e}")
