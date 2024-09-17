import requests
import bs4 as bs
import pandas as pd
from io import StringIO

def get_sp500_stocks():
    # URL of Wikipedia page containing S&P 500 companies
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Create a BeautifulSoup object to parse the HTML content
    soup = bs.BeautifulSoup(response.text, 'lxml')
    
    # Find the table containing the S&P 500 companies
    table = soup.find('table', {'class': 'wikitable sortable'})
    
    # Create a DataFrame from the table
    df = pd.read_html(StringIO(str(table)))[0]
    
    # Get the 'Symbol' column which contains the stock tickers
    tickers = df['Symbol'].tolist()
    print(f"Number of S&P 500 stocks: {len(tickers)}")
    
    return tickers

def get_nasdaq_stocks():
    # URL of Wikipedia page containing NASDAQ-100 companies
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Create a BeautifulSoup object to parse the HTML content
    soup = bs.BeautifulSoup(response.text, 'lxml')
    
    # Find the table containing the NASDAQ-100 companies
    table = soup.find('table', {'id': 'constituents'})
    
    # Create a DataFrame from the table
    df = pd.read_html(StringIO(str(table)))[0]
    
    # Get the 'Ticker' column which contains the stock tickers
    tickers = df['Ticker'].tolist()
    print(f"Number of NASDAQ-100 stocks: {len(tickers)}")
    
    return tickers

def save_stocks_to_file(filename='stock_tickers.txt'):
    # Get the S&P 500 and NASDAQ-100 stock tickers
    sp500_stocks = get_sp500_stocks()
    nasdaq_stocks = get_nasdaq_stocks()
    
    # Combine and remove duplicates
    all_stocks = list(set(sp500_stocks + nasdaq_stocks))
    
    print(f"Total number of unique stocks: {len(all_stocks)}")
    
    # Save the tickers to a text file
    with open(filename, 'w') as f:
        for stock in all_stocks:
            f.write(f"{stock}\n")
    
    print(f"S&P 500 and NASDAQ-100 stocks have been saved to {filename}")

if __name__ == "__main__":
    save_stocks_to_file()
