import os
import requests
from dotenv import load_dotenv
import pandas as pd
import time
import yfinance as yf

# Load environment variables from .env file
load_dotenv()

# Get the Alpha Vantage API keys from environment variables
api_keys_str = os.getenv('ALPHA_VANTAGE_API_KEYS')
if not api_keys_str:
    print("ALPHA_VANTAGE_API_KEYS are not set. Please check your environment variables.")
    exit(1)

ALPHA_VANTAGE_API_KEYS = api_keys_str.split(',')

# Print part of the API keys for debugging
print(f"Using Alpha Vantage API keys: {[key[:4] + '...' + key[-4:] for key in ALPHA_VANTAGE_API_KEYS]}")

# Initialize API key index
api_key_index = 0

def get_next_api_key():
    global api_key_index
    api_key = ALPHA_VANTAGE_API_KEYS[api_key_index]
    api_key_index = (api_key_index + 1) % len(ALPHA_VANTAGE_API_KEYS)
    return api_key

def fetch_daily_data(symbol):
    print(f"Fetching data for {symbol} from Yahoo Finance...")
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")  # Fetch the last 5 days of data
        if hist.empty:
            print(f"No data found for {symbol}")
            return None

        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        volume = hist['Volume'].iloc[-1]

        return {
            'latest_close': latest_close,
            'previous_close': previous_close,
            'volume': volume
        }
    except Exception as e:
        print(f"Exception occurred while fetching data for {symbol}: {str(e)}")
        return None

def analyze_performance(symbols):
    performance_data = []

    for symbol in symbols:
        data = fetch_daily_data(symbol)
        if data:
            percent_change = ((data['latest_close'] - data['previous_close']) / data['previous_close']) * 100

            performance_data.append({
                'symbol': symbol,
                'latest_close': data['latest_close'],
                'previous_close': data['previous_close'],
                'percent_change': percent_change,
                'volume': data['volume']
            })

            print(f"Analyzed performance for {symbol}: {percent_change:.2f}% change, volume {data['volume']}")

        # To avoid hitting the API rate limit, sleep for 0.5 seconds between requests
        print(f"Sleeping for 0.5 seconds to avoid rate limit...")
        time.sleep(0.5)

    df = pd.DataFrame(performance_data)
    return df

def get_top_performers(df, n=5):
    top_gainers = df.nlargest(n, 'percent_change')
    top_losers = df.nsmallest(n, 'percent_change')
    top_volume = df.nlargest(n, 'volume')
    return top_gainers, top_losers, top_volume

if __name__ == "__main__":
    # Read the list of top 100 S&P 500 companies from the file
    with open('/Users/jose/Desktop/ExperimentalScripts/FINISHED/GroupMeBotRender/top100_companies.txt', 'r') as file:
        symbols = [line.strip() for line in file if line.strip()]

    # Batch processing to respect rate limits
    batch_size = 5
    all_performance_data = []

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i + batch_size]
        print(f"Analyzing batch: {batch_symbols}")
        df = analyze_performance(batch_symbols)
        if not df.empty:
            all_performance_data.append(df)
        
        # Sleep to respect the rate limit
        if i + batch_size < len(symbols):
            print("Sleeping for 1 seconds to respect rate limits...")
            time.sleep(1)

    if all_performance_data:
        final_df = pd.concat(all_performance_data)
        top_gainers, top_losers, top_volume = get_top_performers(final_df)

        print("📈 **Top Gainers:**")
        print(top_gainers[['symbol', 'percent_change', 'volume']])
        
        print("\n📉 **Top Losers:**")
        print(top_losers[['symbol', 'percent_change', 'volume']])
        
        print("\n🔝 **Top Volume:**")
        print(top_volume[['symbol', 'percent_change', 'volume']])
    else:
        print("Failed to fetch performance data.")