import yfinance as yf

ticker = yf.Ticker('AAPL')  # Apple's stock symbol

# Fetching basic information
info = ticker.info

# Printing retrieved information
print("Ticker Info:")
for key, value in info.items():
    print(f"{key}: {value}")