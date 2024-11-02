import sys
import json
import asyncio
from datetime import datetime, timezone, timedelta
import time
from openai import OpenAI
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import httpx
from flask import Flask, request, jsonify
from typing import List, Dict, Union, Tuple
import logging
import traceback
from logging.handlers import RotatingFileHandler
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from asgiref.wsgi import WsgiToAsgi
from hypercorn.asyncio import serve
from hypercorn.config import Config
import html
import re
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
import requests
from functools import lru_cache
import random
import nest_asyncio
import aiohttp
from urllib.parse import urljoin

# Apply nest_asyncio to allow nested event loops (useful for certain environments)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get the directory where application.py resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Define paths to local files
ECONOMIC_CALENDAR_FILE = os.path.join(PROJECT_ROOT, "data", "input", "usd_us_events.json")
TOP_COMPANIES_FILE = os.path.join(PROJECT_ROOT, "data", "input", "sp500_stocks.txt")
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'output')

print(f"Economic Calendar File Path: {ECONOMIC_CALENDAR_FILE}")
print(f"Top Companies File Path: {TOP_COMPANIES_FILE}")
print(f"Input Directory: {INPUT_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, 'config', '.env'))

# Configuration settings - Removed from ENV and set as constants
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "logs", "groupme_bot.log")
LOG_LEVEL = "INFO"
RATE_LIMIT_PER_MINUTE = 50
RATE_LIMIT_PER_HOUR = 500
RATE_LIMIT_PER_DAY = 5000
MAX_PROMPT_LENGTH = 500

# Ensure logs directory exists
logs_dir = os.path.dirname(LOG_FILE_PATH)
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
def setup_logging():
    logger = logging.getLogger('groupme_bot')
    logger.setLevel(getattr(logging, LOG_LEVEL))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Optionally, add file handler
    file_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=10**6, backupCount=3)
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

# Retrieve API keys from environment variables
BOT_ID = os.getenv("BOT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROUPME_ACCESS_TOKEN = os.getenv("GROUPME_ACCESS_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

print(f"BOT_ID loaded: {BOT_ID is not None}")
print(f"OPENAI_API_KEY loaded: {OPENAI_API_KEY is not None}")
print(f"TAVILY_API_KEY loaded: {TAVILY_API_KEY is not None}")
print(f"GROUPME_ACCESS_TOKEN loaded: {GROUPME_ACCESS_TOKEN is not None}")
print(f"NEWS_API_KEY loaded: {NEWS_API_KEY is not None}")

# Remove any logging of the actual API key
print("OpenAI API Key loaded successfully.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Tavily client
from tavily import TavilyClient
tavily_client = TavilyClient(TAVILY_API_KEY)

# Initialize Flask application
application = Flask(__name__)

@application.route('/', methods=['GET', 'HEAD'])
def root():
    return "Welcome to the GroupMe Bot! The server is running."

@application.route('/test', methods=['GET'])
def test():
    return "Test route is working."

def get_rate_limit_key():
    return get_remote_address() or 'default'

limiter = Limiter(
    key_func=get_rate_limit_key,
    app=application,
    default_limits=[
        f"{RATE_LIMIT_PER_MINUTE} per minute",
        f"{RATE_LIMIT_PER_HOUR} per hour",
        f"{RATE_LIMIT_PER_DAY} per day"
    ],
    storage_uri="memory://"
)

executor = ThreadPoolExecutor(max_workers=10)

def split_message(text: str, limit: int = 1000) -> List[str]:
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break

        # Find the last occurrence of a sentence-ending punctuation or a newline within the limit
        split_index = max(
            text.rfind('.', 0, limit),
            text.rfind('!', 0, limit),
            text.rfind('?', 0, limit),
            text.rfind('\n', 0, limit)
        )

        if split_index == -1 or split_index == 0:
            # If no suitable split point is found, split at the last space
            split_index = text.rfind(' ', 0, limit)

        if split_index == -1:
            # If still no split point found, split at the limit
            split_index = limit

        parts.append(text[:split_index])
        text = text[split_index:].lstrip()
    return parts

async def upload_image_to_groupme(image_url: str) -> Union[str, None]:
    try:
        print(f"Downloading image from: {image_url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
        print("Image downloaded successfully")

        img = Image.open(BytesIO(response.content))
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        print("Image converted to JPEG")

        upload_url = 'https://image.groupme.com/pictures'
        headers = {
            'X-Access-Token': GROUPME_ACCESS_TOKEN,
            'Content-Type': 'image/jpeg'
        }

        print("Uploading image to GroupMe")
        async with httpx.AsyncClient() as client:
            response = await client.post(upload_url, content=img_byte_arr, headers=headers)
            response.raise_for_status()
        print("Image uploaded successfully")
        return response.json()['payload']['url']
    except Exception as e:
        print(f"Failed to upload image: {str(e)}")
        return None

# Add this constant to track which bot goes to which group
GROUP_BOTS = {
    "agent": os.getenv("AGENT_BOT_ID"),  # Bot ID for agent responses
    "market": os.getenv("MARKET_BOT_ID"), # Bot ID for market commands
    # Add other bot IDs as needed
}

async def send_message(text: str, image_url: Union[str, None] = None) -> None:
    """Simple message sender using single bot ID"""
    url = "https://api.groupme.com/v3/bots/post"
    data = {
        "bot_id": BOT_ID,  # Using the single bot ID
        "text": text,
        "attachments": []
    }
    if image_url:
        data["attachments"].append({"type": "image", "url": image_url})
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)

def get_help_message() -> str:
    help_message = """
🤖 AI Bot Commands 🤖

• !help - Show this help message
• !usage - Get OpenAI API usage summary
• !ai [prompt] - Generate a response using GPT-3.5 (default AI model)
• !ai4 [prompt] - Generate a response using GPT-4 with web search
• !image [prompt] - Generate an image using DALL-E
• !market - Get the full market summary
• !calendar - Get the economic calendar
• !stocks - Get the top stocks summary
• !news - Get the top business news
• !earnings - Get earnings calendar for this week and next week

Test Endpoints:
• /test_help - Test the help message
• /test_usage - Test the OpenAI API usage summary
• /test_ai?prompt=[Your prompt] - Test GPT-3.5 response
• /test_ai4?prompt=[Your prompt] - Test GPT-4 response with web search
• /test_image?prompt=[Your prompt] - Test image generation
• /test_market_summary - Test the full market summary
• /test_calendar - Test the economic calendar
• /test_stocks - Test the top stocks summary
• /test_news - Test the top business news
• /test_earnings - Test the earnings calendar

For any issues or feature requests, please contact the bot administrator.
"""
    return help_message

# --------------------------- Start of Market Summary Integration ---------------------------

def get_impact_emoji(impact):
    impact = impact.lower()
    if 'high' in impact:
        return '🔴'
    elif 'medium' in impact:
        return '🟠'
    else:
        return '🟢'

def get_event_emoji(event_name):
    event_name = event_name.lower()
    if 'gdp' in event_name:
        return '📊'
    elif 'unemployment' in event_name:
        return '👥'
    elif 'inflation' in event_name or 'cpi' in event_name or 'pce' in event_name:
        return '💹'
    elif 'interest rate' in event_name or 'fed' in event_name:
        return '🏦'
    elif 'manufacturing' in event_name:
        return '🏭'
    elif 'sales' in event_name:
        return '🛍️'
    elif 'oil' in event_name:
        return '🛢️'
    else:
        return '📅'

def get_economic_calendar(start_date, end_date):
    try:
        # Load the JSON data from local file
        with open(ECONOMIC_CALENDAR_FILE, 'r') as file:
            events = json.load(file)

        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Filter events within the date range
        filtered_events = [
            event for event in events
            if start_date <= datetime.strptime(event['date'], '%Y-%m-%d') <= end_date
        ]

        # Convert to DataFrame
        df = pd.DataFrame(filtered_events)
        df['Release Date'] = pd.to_datetime(df['date'])
        df['Release Time (ET)'] = df['time']
        df['Release Name'] = df['name']
        df['Impact'] = df['impactTitle']

        # Select and reorder columns
        df = df[['Release Date', 'Release Time (ET)', 'Release Name', 'Impact']]

        # Sort by date and time
        df.sort_values(by=['Release Date', 'Release Time (ET)'], inplace=True)

        return df
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=1)
def get_sp500_symbols() -> List[str]:
    try:
        # Load symbols from local file
        with open(TOP_COMPANIES_FILE, 'r') as file:
            symbols = [line.strip() for line in file.readlines() if line.strip()]
        return symbols
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
        return []

def get_stock_data(symbol: str) -> Union[Dict, None]:
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5d")  # Fetch the last 5 days of data
        if len(hist) < 2:
            logger.warning(f"Not enough data for symbol: {symbol}")
            return None
        latest_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2]
        percent_change = ((latest_close - previous_close) / previous_close) * 100
        volume = hist['Volume'].iloc[-1]
        return {
            "Symbol": symbol,
            "Latest Close": latest_close,
            "Previous Close": previous_close,
            "Percent Change": percent_change,
            "Volume": volume
        }
    except Exception as e:
        logger.error(f"Error fetching data for symbol {symbol}: {str(e)}")
        return None

def get_top_stocks(symbols: List[str], batch_size=50, max_workers=10) -> Tuple[pd.DataFrame, List[str]]:
    all_results = []
    failed_symbols = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(get_stock_data, symbol): symbol for symbol in symbols}

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Unexpected error for symbol {symbol}: {str(e)}")
                failed_symbols.append(symbol)

    df = pd.DataFrame(all_results)
    if df.empty:
        logger.warning("No stock data could be fetched.")
    else:
        logger.info(f"Fetched stock data for {len(df)} symbols.")
    return df, failed_symbols

def get_top_news() -> List[Dict]:
    try:
        url = f'https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Error fetching news: HTTP {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return []
        articles = response.json().get('articles', [])
        top_articles = [{'title': article['title'], 'url': article['url']} for article in articles[:5]]
        return top_articles
    except Exception as e:
        logger.error(f"Error fetching top news: {str(e)}")
        return []

def get_earnings_data() -> str:
    output = "📅 EARNINGS CALENDAR 📅\n\n"
    
    this_week_file = os.path.join(OUTPUT_DIR, 'earnings_this_week.txt')
    next_week_file = os.path.join(OUTPUT_DIR, 'earnings_next_week.txt')
    
    logger.debug(f"This week earnings file path: {this_week_file}")
    logger.debug(f"Next week earnings file path: {next_week_file}")
    
    # This Week
    output += "--- This Week ---\n"
    if os.path.exists(this_week_file):
        with open(this_week_file, 'r') as file:
            output += file.read() + "\n"
    else:
        logger.warning(f"Earnings data file not found: {this_week_file}")
        output += "Earnings data file not found for This Week.\n"
    
    # Next Week
    output += "\n--- Next Week ---\n"
    if os.path.exists(next_week_file):
        with open(next_week_file, 'r') as file:
            output += file.read() + "\n"
    else:
        logger.warning(f"Earnings data file not found: {next_week_file}")
        output += "Earnings data file not found for Next Week.\n"
    
    return output

def generate_market_summary() -> str:
    # Define date range for the week
    today = datetime.today()
    start_date = today.strftime('%Y-%m-%d')
    end_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')

    # Fetch all S&P 500 symbols from top100_companies.txt
    all_symbols = get_sp500_symbols()
    logger.info(f"Fetched {len(all_symbols)} S&P 500 symbols.")

    # Fetch economic calendar
    economic_calendar = get_economic_calendar(start_date, end_date)
    logger.info(f"Fetched economic calendar with {len(economic_calendar)} events.")

    # Fetch top stocks
    top_stocks_df, failed_symbols = get_top_stocks(all_symbols)
    if top_stocks_df.empty:
        logger.warning("Unable to fetch top stocks data.")
    else:
        logger.info(f"Successfully fetched data for {len(top_stocks_df)} stocks.")

    if failed_symbols:
        logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {', '.join(failed_symbols)}")

    # Fetch top news
    top_news = get_top_news()
    logger.info(f"Fetched {len(top_news)} top news articles.")

    # Fetch earnings data
    earnings_data = get_earnings_data()

    # Format and return the market summary
    market_summary = format_market_summary(economic_calendar, top_stocks_df, top_news, earnings_data)
    return market_summary

def shorten_url(url):
    domain = re.search(r'https?://(?:www\.)?([\w\-]+)\.', url)
    return domain.group(1) if domain else 'link'

def parse_time(time_str):
    try:
        return datetime.strptime(time_str, '%I:%M %p').time()
    except ValueError:
        try:
            return datetime.strptime(time_str, '%H:%M').time()
        except ValueError:
            return None

def format_market_summary(economic_calendar: pd.DataFrame, top_stocks: pd.DataFrame, top_news: List[Dict], earnings_data: str) -> str:
    output = "📊 MARKET SUMMARY 📊\n\n"

    # Process economic calendar
    economic_calendar['Release Date'] = pd.to_datetime(economic_calendar['Release Date'])
    economic_calendar['Release Time (ET)'] = economic_calendar['Release Time (ET)'].apply(parse_time)
    economic_calendar = economic_calendar.sort_values(['Release Date', 'Release Time (ET)'])

    # Helper function to abbreviate event names
    def abbreviate_event(event_name):
        words = event_name.split()
        if len(words) > 1:
            return words[0]
        return event_name

    # Helper function to group earnings
    def group_earnings(earnings_list):
        before_open = []
        after_close = []
        for ticker, time in earnings_list:
            if 'Before Open' in time:
                before_open.append(ticker)
            elif 'After Close' in time:
                after_close.append(ticker)
        
        result = []
        if before_open:
            result.append(f"📅 Before Open: {', '.join(before_open)}")
        if after_close:
            result.append(f"📅 After Close: {', '.join(after_close)}")
        return '; '.join(result)

    # Process earnings data
    earnings_dict = {}
    for line in earnings_data.split('\n'):
        if ':' in line:
            ticker, date_time = line.split(':', 1)
            date, time = date_time.strip().split(' ', 1)
            earnings_dict[date] = earnings_dict.get(date, []) + [(ticker.strip(), time)]

    # Combine and display events
    current_date = datetime.now().date()
    end_of_next_week = current_date + timedelta(days=(13 - current_date.weekday()))
    
    for date in pd.date_range(current_date, end_of_next_week):
        date_str = date.strftime('%m/%d (%a)')
        day_events = economic_calendar[economic_calendar['Release Date'].dt.date == date.date()]
        day_earnings = earnings_dict.get(date.strftime('%Y-%m-%d'), [])
        
        if not day_events.empty or day_earnings:
            output += f"{date_str}:\n"
            
            # Group events by time
            events_by_time = day_events.groupby('Release Time (ET)')
            for time, group in events_by_time:
                events = []
                for _, event in group.iterrows():
                    impact_emoji = get_impact_emoji(event['Impact'])
                    event_emoji = get_event_emoji(event['Release Name'])
                    abbreviated_name = abbreviate_event(event['Release Name'])
                    events.append(f"{impact_emoji}{event_emoji}{abbreviated_name}")
                output += f"  {', '.join(events)} ({time})\n"
            
            if day_earnings:
                grouped_earnings = group_earnings(day_earnings)
                output += f"  {grouped_earnings}\n"
            
            output += "\n"

        if date.date() == current_date + timedelta(days=(6 - current_date.weekday())):
            output += "--- NEXT WEEK ---\n"

    # Top Stocks
    output += "📈 TOP STOCKS\n"
    if not top_stocks.empty:
        categories = [
            ("Gainers", top_stocks.nlargest(5, 'Percent Change')),
            ("Losers", top_stocks.nsmallest(5, 'Percent Change')),
            ("Volume", top_stocks.nlargest(5, 'Volume'))
        ]
        for title, df in categories:
            stocks = [f"${row['Symbol']} {row['Percent Change']:.2f}%" for _, row in df.iterrows()]
            output += f"{title}: {', '.join(stocks)}\n"
    else:
        output += "No stock data available.\n"
    output += "\n"

    # Top News
    output += "📰 TOP NEWS\n"
    if top_news:
        for i, article in enumerate(top_news[:5], 1):
            title = article['title'][:50] + '...' if len(article['title']) > 50 else article['title']
            short_url = shorten_url(article['url'])
            escaped_title = html.escape(title)
            link = f'<a href="{article["url"]}">{short_url}</a>'
            output += f"{i}. {escaped_title} {link}\n"
    else:
        output += "No news articles available.\n"

    return output

def generate_economic_calendar() -> str:
    today = datetime.today()
    start_date = today.strftime('%Y-%m-%d')
    end_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')
    economic_calendar = get_economic_calendar(start_date, end_date)
    
    output = "📅 ECONOMIC CALENDAR 📅\n\n"
    if not economic_calendar.empty:
        events_by_day = economic_calendar.groupby('Release Date')
        for date, group in events_by_day:
            day_of_week = date.strftime('%A')
            output += f"--- {date.strftime('%Y-%m-%d')} ({day_of_week}) ---\n"
            
            # Group events by time and impact
            events_by_time_impact = group.groupby(['Release Time (ET)', 'Impact'])
            for (time, impact), subgroup in events_by_time_impact:
                impact_emoji = get_impact_emoji(impact)
                event_names = [get_event_emoji(row['Release Name']) + ' ' + row['Release Name'] for _, row in subgroup.iterrows()]
                consolidated_events = ', '.join(event_names)
                output += f"{impact_emoji}{consolidated_events} ({time})\n"
            output += "\n"
    else:
        output += "No economic events available.\n"
    return output

def generate_top_stocks() -> str:
    all_symbols = get_sp500_symbols()
    top_stocks_df, failed_symbols = get_top_stocks(all_symbols)
    
    output = "📈 TOP STOCKS 📈\n\n"
    if not top_stocks_df.empty:
        categories = [
            ("TOP GAINERS", top_stocks_df.nlargest(5, 'Percent Change')),
            ("TOP LOSERS", top_stocks_df.nsmallest(5, 'Percent Change')),
            ("HIGHEST VOLUME", top_stocks_df.nlargest(5, 'Volume'))
        ]
        for title, df in categories:
            output += f"--- {title} ---\n"
            for _, row in df.iterrows():
                output += f"${row['Symbol']}: {row['Percent Change']:.2f}% ({int(row['Volume']):,})\n"
            output += "\n"
    else:
        output += "No stock data available.\n"
    return output

def generate_top_news() -> str:
    top_news = get_top_news()
    
    output = "📰 TOP NEWS 📰\n\n"
    if top_news:
        for i, article in enumerate(top_news[:5], 1):
            output += f"{i}. {article['title']}\n   {article['url']}\n\n"
    else:
        output += "No news articles available.\n"
    return output

# --------------------------- End of Market Summary Integration ---------------------------

# Original bot functions
def validate_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        logger.warning(f"Prompt exceeded maximum length and was truncated: {prompt[:50]}...")
        return prompt[:MAX_PROMPT_LENGTH]
    return prompt

async def get_openai_usage() -> str:
    try:
        current_date = datetime.now(timezone.utc)
        date_str = current_date.strftime("%Y-%m-%d")
        usage_url = "https://api.openai.com/v1/usage"
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        params = {"date": date_str}
        async with httpx.AsyncClient() as client:
            response = await client.get(usage_url, headers=headers, params=params)
            response.raise_for_status()
            usage_data = response.json()

        total_tokens = sum(
            item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0)
            for item in usage_data.get('data', [])
        )
        gpt35_tokens = sum(
            item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0)
            for item in usage_data.get('data', []) if 'gpt-3.5' in item.get('snapshot_id', '')
        )
        gpt4_tokens = sum(
            item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0)
            for item in usage_data.get('data', []) if 'gpt-4' in item.get('snapshot_id', '')
        )
        dalle_images = sum(item.get('num_images', 0) for item in usage_data.get('dalle_api_data', []))

        gpt35_cost = (gpt35_tokens / 1000) * 0.002
        gpt4_cost = (gpt4_tokens / 1000) * 0.06
        dalle_cost = dalle_images * 0.02
        total_cost = gpt35_cost + gpt4_cost + dalle_cost

        return (
            f"OpenAI API Usage Summary for {date_str}:\n"
            f"Total tokens: {total_tokens:,}\n"
            f"GPT-3.5 tokens: {gpt35_tokens:,}\n"
            f"GPT-4 tokens: {gpt4_tokens:,}\n"
            f"DALL-E images: {dalle_images}\n"
            f"Estimated cost: ${total_cost:.2f}\n"
            f"(GPT-3.5: ${gpt35_cost:.2f}, GPT-4: ${gpt4_cost:.2f}, DALL-E: ${dalle_cost:.2f})"
        )

    except Exception as e:
        return f"Error fetching OpenAI usage: {str(e)}"

async def generate_image(prompt: str) -> Union[str, None]:
    try:
        logger.info(f"Attempting to generate image with prompt: {prompt}")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        logger.info(f"Image generated successfully: {image_url}")
        return image_url
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None

def search_web(query: str) -> Union[List[Dict], None]:
    try:
        search_result = tavily_client.search(query=query, search_depth="advanced")
        return search_result['results'][:3]  # Return top 3 results
    except Exception as e:
        print(f"Error searching web: {e}")
        return None

async def generate_ai_response(prompt: str, model: str = "gpt-3.5-turbo", use_web_search: bool = False) -> str:
    try:
        logger.debug(f"Generating AI response for prompt: '{prompt}', model: {model}, use_web_search: {use_web_search}")
        
        if use_web_search:
            logger.debug("Attempting web search")
            search_results = search_web(prompt)
            logger.debug(f"Web search results: {search_results}")
            limited_results = search_results[:3] if search_results else []
            search_content = "\n".join(
                [f"- {result['title']}: {result['content'][:200]}..." for result in limited_results]
            )
            logger.info(f"Web search results: {search_content}")
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant specialized in stocks, cryptocurrency, investing strategies, and money-making opportunities. Your role is to support and enhance discussions within a GroupMe chat focused on these topics. You have access to recent web information."
                        "Provide accurate and helpful responses based on the given web search results and your knowledge."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Web search results for '{prompt}':\n{search_content}"
                        if search_content else "No search results found."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are an AI assistant specialized in stocks, cryptocurrency, investing strategies, and money-making opportunities. Your role is to support and enhance discussions within a GroupMe chat focused on these topics."},
                {"role": "user", "content": prompt}
            ]

        logger.debug("Sending request to OpenAI API")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000 if model == "gpt-4" else 500
        )
        logger.debug(f"Received response from OpenAI API: {response}")

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Sorry, I couldn't generate a response at this time. Error: {str(e)}"

# Add this new function to handle agent communication
async def send_to_agent(message: str, user_id: str, username: str) -> str:
    try:
        logger.info(f"Attempting to send message to agent: {message}")
        async with aiohttp.ClientSession() as session:
            url = urljoin(AGENT_BASE_URL, f"{DEFAULT_AGENT}/message")
            logger.info(f"Agent URL: {url}")
            payload = {
                "text": message,
                "userId": user_id,
                "userName": username
            }
            logger.info(f"Payload: {payload}")
            
            async with session.post(url, json=payload) as response:
                logger.info(f"Agent response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Agent response: {data}")
                    return data.get('text', 'No response from agent')
                else:
                    logger.error(f"Agent error response: {await response.text()}")
                    return "Sorry, I'm having trouble processing your request."

    except Exception as e:
        logger.error(f"Error connecting to agent: {str(e)}")
        return "Sorry, I'm unable to connect to the agent service."

# Define bot IDs for different purposes
BOT_IDS = {
    "command": "8ca1dd7879b66b571b92f1d34e",  # Jarvis - for /commands
    "conversation": "cf5c9f00ca2c80f1651ba6b722"  # Eliza - for normal conversation
}

@application.route('/', methods=['POST'])
async def webhook():
    try:
        data = request.get_json()
        
        if data.get('sender_type') == 'bot':
            return jsonify({"success": True}), 200
            
        message = data.get('text', '').strip()
        user_id = data.get('user_id', 'unknown')
        username = data.get('name', 'unknown')
        
        # Route based on message type
        if message.startswith('!') or message.startswith('/'):
            # Commands go to Jarvis
            bot_id = BOT_IDS["command"]
            # Handle command processing...
            await send_message(bot_id, response)
        else:
            # Normal conversation goes to Eliza
            bot_id = BOT_IDS["conversation"]
            response = await send_to_agent(message, user_id, username)
            if response:
                await send_message(bot_id, response)
        
        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

async def send_message(bot_id: str, text: str, image_url: Union[str, None] = None) -> None:
    """Send message using specified bot ID"""
    url = "https://api.groupme.com/v3/bots/post"
    data = {
        "bot_id": bot_id,
        "text": text,
        "attachments": []
    }
    if image_url:
        data["attachments"].append({"type": "image", "url": image_url})
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)

# Error handler for rate limiting
@application.errorhandler(RateLimitExceeded)
def handle_rate_limit_exceeded(e):
    logger.warning(f"Rate limit exceeded: {str(e)}")
    return "Rate limit exceeded. Please try again later.", 429

@application.route('/test_help', methods=['GET'])
def test_help():
    help_message = get_help_message()
    return f"<pre>{help_message}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_usage', methods=['GET'])
async def test_usage():
    usage_info = await get_openai_usage()
    return f"<pre>{usage_info}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_ai', methods=['GET'])
async def test_ai():
    prompt = request.args.get('prompt', 'What is the capital of France?')
    response = await generate_ai_response(prompt, "gpt-3.5-turbo", False)
    return f"<pre>Prompt: {prompt}\n\nResponse: {response}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_ai4', methods=['GET'])
async def test_ai4():
    prompt = request.args.get('prompt', 'What are the latest developments in AI?')
    response = await generate_ai_response(prompt, "gpt-4", True)
    return f"<pre>Prompt: {prompt}\n\nResponse: {response}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_image', methods=['GET'])
async def test_image():
    prompt = request.args.get('prompt', 'A futuristic city skyline')
    image_url = await generate_image(prompt)
    if image_url:
        return f"<p>Image generated for prompt: '{prompt}'</p><img src='{image_url}' alt='Generated Image'>", 200, {'Content-Type': 'text/html; charset=utf-8'}
    else:
        return f"<pre>Failed to generate image for prompt: '{prompt}'</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_market_summary', methods=['GET'])
async def test_market_summary():
    try:
        market_summary = generate_market_summary()
        html_formatted_summary = market_summary.replace('\n', '<br>')
        return f"<pre>{html_formatted_summary}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error generating market summary: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@application.route('/test_calendar', methods=['GET'])
async def test_calendar():
    calendar = generate_economic_calendar()
    return f"<pre>{calendar}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_earnings', methods=['GET'])
async def test_earnings():
    earnings_data = get_earnings_data()
    return f"<pre>{earnings_data}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_stocks', methods=['GET'])
async def test_stocks():
    try:
        all_symbols = get_sp500_symbols()
        top_stocks_df, failed_symbols = get_top_stocks(all_symbols)
        
        output = "📈 TOP STOCKS 📈\n\n"
        if not top_stocks_df.empty:
            categories = [
                ("TOP GAINERS", top_stocks_df.nlargest(5, 'Percent Change')),
                ("TOP LOSERS", top_stocks_df.nsmallest(5, 'Percent Change')),
                ("HIGHEST VOLUME", top_stocks_df.nlargest(5, 'Volume'))
            ]
            for title, df in categories:
                output += f"--- {title} ---\n"
                for _, row in df.iterrows():
                    output += f"${row['Symbol']}: {row['Percent Change']:.2f}% ({int(row['Volume']):,})\n"
                output += "\n"
        else:
            output += "No stock data available.\n"
        
        return f"<pre>{output}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error in test_stocks: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@application.route('/test_news', methods=['GET'])
async def test_news():
    try:
        news = generate_top_news()
        return f"<pre>{news}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error in test_news: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

"""
Message Flow:
-------------
GroupMe Message
    │
    ▼
Is command? (!?)
    │
    ├──Yes──► Handle Command ──┐
    │                         │
    └──No───► Send to Agent   │
                │            │
                ▼            │
          Agent Processes    │
                │            │
                ▼            │
          Return Response    │
                │            │
                ▼            ▼
            Send to GroupMe

This flow diagram represents how messages are processed:
1. Message arrives from GroupMe
2. Check if it's a command (starts with !)
3. If yes: Handle internally with command processor
4. If no: Send to agent service for processing
5. Both paths eventually send response back to GroupMe
"""

if __name__ == "__main__":
    print("Starting Flask development server...")
    application.run(debug=True, host='0.0.0.0', port=5001)




