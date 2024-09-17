import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, StaleElementReferenceException
from datetime import datetime
from dotenv import load_dotenv
import logging
from bs4 import BeautifulSoup

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'output')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
SP500_FILE = os.path.join(INPUT_DIR, 'sp500_stocks.txt')

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set up logging
log_file = os.path.join(LOGS_DIR, 'groupme_bot.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'output')
SP500_FILE = os.path.join(INPUT_DIR, 'sp500_stocks.txt')

def load_sp500_companies(file_path=SP500_FILE):
    print(f"Attempting to load S&P 500 companies from: {file_path}")
    with open(file_path, 'r') as file:
        companies = set(company.strip().upper() for company in file if company.strip())
    return companies

def clean_ticker(ticker):
    return ticker[:-1] if ticker.endswith('D') else ticker

def save_tickers_to_file(tickers, week):
    filename = os.path.join(INPUT_DIR, f"{week.lower().replace(' ', '_')}_tickers.txt")
    with open(filename, 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")

def parse_earnings_data(html_file, sp500_companies):
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    earnings_data = {}
    rows = soup.find_all('tr', class_='tv-data-table__row tv-data-table__stroke tv-screener-table__result-row')
    
    weekly_tickers = []
    
    for row in rows:
        ticker_element = row.find('a', class_='tv-screener__symbol')
        if ticker_element:
            original_ticker = ticker_element.text.strip()
            weekly_tickers.append(original_ticker)
            cleaned_ticker = clean_ticker(original_ticker)
            if cleaned_ticker in sp500_companies:
                date_element = row.find('td', {'data-field-key': 'earnings_release_next_date'})
                timing_element = row.find('td', {'data-field-key': 'earnings_release_next_time'})
                
                date = date_element.text.strip() if date_element else 'Date not found'
                timing = timing_element.get('title', 'Timing not found') if timing_element else 'Timing not found'
                
                earnings_data[cleaned_ticker] = (date, timing)
    
    return earnings_data, weekly_tickers

def get_earnings_events(driver, week, sp500_companies):
    try:
        week_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//div[contains(@class, 'itemContent') and text()='{week}']")
            )
        )
        week_button.click()
        logger.info(f"Clicked '{week}' button")
        
        table = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "tv-data-table__tbody"))
        )
        logger.info("Table found")
        
        # Save the page source
        file_name = f"earnings_page_{week.lower().replace(' ', '_')}.html"
        file_path = os.path.join(INPUT_DIR, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        
        # Parse earnings data
        earnings_data, weekly_tickers = parse_earnings_data(file_path, sp500_companies)
        
        # Save weekly tickers
        save_tickers_to_file(weekly_tickers, week)
        logger.info(f"Saved {len(weekly_tickers)} tickers for {week}")
        
        return earnings_data
    
    except (TimeoutException, WebDriverException) as e:
        logger.error(f"Error occurred while getting earnings events for {week}: {e}")
        return {}

def save_earnings_to_file(earnings_events, week):
    filename = os.path.join(OUTPUT_DIR, f"earnings_{week.lower().replace(' ', '_')}.txt")
    with open(filename, 'w') as f:
        for ticker, (date_str, timing) in earnings_events.items():
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                day_of_week = date_obj.strftime('%A')
                f.write(f"{ticker}: {date_str} ({day_of_week}) - {timing}\n")
            except ValueError:
                f.write(f"{ticker}: {date_str} - {timing}\n")
    return filename

if __name__ == "__main__":
    load_dotenv()
    
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"SP500_FILE: {SP500_FILE}")
    
    sp500_companies = load_sp500_companies()
    logger.info(f"Loaded {len(sp500_companies)} companies from S&P 500 list")
    
    url = "https://www.tradingview.com/markets/stocks-usa/earnings/"
    logger.info(f"Opening URL: {url}")
    
    chrome_options = Options()
    # Add your Chrome options here
    
    with webdriver.Chrome(options=chrome_options) as driver:
        driver.set_page_load_timeout(30)
        driver.get(url)
        logger.info("Page loaded")
        
        this_week_events = get_earnings_events(driver, 'This Week', sp500_companies)
        this_week_file = save_earnings_to_file(this_week_events, 'This Week')
        
        next_week_events = get_earnings_events(driver, 'Next Week', sp500_companies)
        next_week_file = save_earnings_to_file(next_week_events, 'Next Week')
        
        logger.info(f"Earnings events saved to:")
        logger.info(f"This week: {this_week_file}")
        logger.info(f"Next week: {next_week_file}")
        
        print(f"Earnings events saved to:")
        print(f"This week: {this_week_file}")
        print(f"Next week: {next_week_file}")
        
        input("Press Enter to close the browser...")