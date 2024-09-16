


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
from flask_cors import CORS
from typing import List, Dict, Union
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
import aiohttp
from functools import lru_cache
import random
import nest_asyncio
import psutil

# Load environment variables
load_dotenv()

# Configuration settings
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "groupme_bot.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Increased rate limits
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 50))  # Increased from 10 to 50
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", 500))     # Increased from 50 to 500
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", 5000))      # Increased from 200 to 5000
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", 500))

# Set up logging
def setup_logging():
    logger = logging.getLogger('groupme_bot')
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

BOT_ID = os.getenv("BOT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"Loaded API key: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROUPME_ACCESS_TOKEN = os.getenv("GROUPME_ACCESS_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

print(f"BOT_ID loaded: {BOT_ID is not None}")
print(f"OPENAI_API_KEY loaded: {OPENAI_API_KEY is not None}")
print(f"TAVILY_API_KEY loaded: {TAVILY_API_KEY is not None}")
print(f"GROUPME_ACCESS_TOKEN loaded: {GROUPME_ACCESS_TOKEN is not None}")
print(f"NEWS_API_KEY loaded: {NEWS_API_KEY is not None}")
print(f"FINNHUB_API_KEY loaded: {FINNHUB_API_KEY is not None}")

# Remove any logging of the actual API key
print("OpenAI API Key loaded successfully.")

# Set the OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

from tavily import TavilyClient
tavily_client = TavilyClient(TAVILY_API_KEY)

application = Flask(__name__)
CORS(application)

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

async def split_message(text: str, limit: int = 1000) -> List[str]:
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

async def send_message(bot_id: str, text: str) -> None:
    url = "https://api.groupme.com/v3/bots/post"
    message_parts = await split_message(text)
    
    async with httpx.AsyncClient() as client:
        for i, part in enumerate(message_parts):
            data = {
                "bot_id": bot_id,
                "text": part,
            }
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    response = await client.post(url, json=data)
                    response.raise_for_status()
                    print(f"Message part {i+1} sent successfully!")
                    break  # Exit the retry loop if successful
                except httpx.RequestError as e:
                    print(f"Error sending message part {i+1}: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(1)  # Wait before retrying
            
            # Wait between message parts to avoid rate limiting
            if i < len(message_parts) - 1:
                await asyncio.sleep(1)

def get_help_message() -> str:
    help_message = """
🤖 AI Bot Commands 🤖

• !help - Show this help message
• !usage - Get OpenAI API usage summary
• !ai [prompt] - Generate a response using GPT-3.5 (default AI model)
• !ai4 [prompt] - Generate a response using GPT-4 with web search
• !image [prompt] - Generate an image using DALL-E
• !market - Get a daily market summary

For any issues or feature requests, please contact the bot administrator.
"""
    return help_message

@application.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json(force=True)
        logger.info(f"Received message: {json.dumps(data)}")
        
        # Ignore messages sent by bots (including itself)
        if data.get('sender_type') == 'bot':
            logger.debug("Message sent by bot; ignoring.")
            return jsonify(success=True), 200
        
        text = data.get('text', '').lower().strip()
        
        if text == '!market':
            market_summary = get_top_movers()
            asyncio.run(send_message(BOT_ID, market_summary))
            return jsonify(success=True, message="Market summary sent to GroupMe"), 200
        
        # ... (handle other commands)
    
        return jsonify(success=True), 200
    
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from request")
        return jsonify(success=False, error="Invalid JSON"), 400
    except Exception as e:
        logger.error(f"Unexpected error in webhook: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(success=False, error="Internal server error"), 500

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

# Ensure logging is set to DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
                        "You are an advanced AI assistant with access to recent web information. "
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
                {"role": "system", "content": "You are a helpful AI assistant."},
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

@application.errorhandler(RateLimitExceeded)
def handle_rate_limit_exceeded(e):
    logger.warning(f"Rate limit exceeded: {str(e)}")
    return "Rate limit exceeded. Please try again later.", 429

# Create a ThreadPoolExecutor at the module level
executor = ThreadPoolExecutor(max_workers=10)

def get_stock_data(symbol, retries=3, delay=2):
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info:
                logging.warning(f"No info returned for {symbol}")
                return None

            current_price = info.get('regularMarketPrice')
            previous_close = info.get('regularMarketPreviousClose')
            volume = info.get('volume')

            if current_price is None or previous_close is None:
                logging.warning(f"Missing price data for {symbol}. Current: {current_price}, Previous: {previous_close}")
                return None

            percent_change = ((current_price - previous_close) / previous_close) * 100 if previous_close != 0 else 0

            logging.info(f"Successfully fetched data for {symbol}")
            return {
                'Symbol': symbol,
                'Percent Change': percent_change,
                'Volume': volume
            }
        except Exception as e:
            if attempt < retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}. Retrying...")
                time.sleep(delay)
                continue
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

async def fetch_stock_data(symbol):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, get_stock_data, symbol)
    return result
    # Make sure no lingering resources or sessions remain

async def get_top_stocks(symbols, batch_size=20):
    all_results = []
    failed_symbols = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        sem = asyncio.Semaphore(5)  # Adjust based on testing

        async def safe_fetch(symbol):
            async with sem:
                try:
                    result = await fetch_stock_data(symbol)
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logging.error(f"Error fetching data for {symbol}: {str(e)}")
                    failed_symbols.append(symbol)

        tasks = [asyncio.create_task(safe_fetch(symbol)) for symbol in batch]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)  # Short delay between batches

    df = pd.DataFrame(all_results)
    if df.empty:
        logging.warning("No stock data could be fetched.")
        return df, failed_symbols
    df = df.sort_values(by=['Volume', 'Percent Change'], ascending=False)
    return df, failed_symbols

def get_sp500_symbols():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(sp500_url)
    sp500_symbols = table[0]['Symbol'].tolist()
    return [symbol.replace('.', '-') for symbol in sp500_symbols]

def get_top_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}")
        return []
    try:
        articles = response.json().get('articles', [])
    except ValueError as e:
        print("Error parsing JSON response from NewsAPI:", e)
        return []
    top_articles = [{'title': article['title'], 'url': article['url']} for article in articles[:5]]
    return top_articles

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
    elif 'inflation' in event_name or 'cpi' in event_name:
        return '💹'
    elif 'interest rate' in event_name:
        return '🏦'
    else:
        return '📅'

def get_economic_calendar(start_date, end_date):
    try:
        # Load the JSON data
        with open('usd_us_events.json', 'r') as file:
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
        df = df.sort_values(['Release Date', 'Release Time (ET)'])
        
        return df
    except Exception as e:
        logger.error(f"Error processing economic calendar data: {e}")
        return pd.DataFrame()

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
    logging.info(f"Current memory usage: {mem:.2f} MB")

async def generate_market_summary():
    try:
        # Define date range for the week
        today = datetime.today()
        start_date = today.strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')
    
        # Fetch all S&P 500 symbols
        logging.info("Before fetching stock symbols")
        log_memory_usage()
        all_symbols = get_sp500_symbols()
        logging.info(f"Fetched {len(all_symbols)} S&P 500 symbols")
        log_memory_usage()
    
        # Fetch economic calendar
        economic_calendar = get_economic_calendar(start_date, end_date)
    
        # Fetch top stocks asynchronously
        top_stocks, failed_symbols = await get_top_stocks(all_symbols)
        logging.info("After fetching stock data")
        log_memory_usage()
        
        if top_stocks.empty:
            logging.warning("Unable to fetch stock data.")
            return "Unable to fetch stock data at this time."
    
        # Fetch top news
        top_news = get_top_news()
        logging.info(f"Fetched {len(top_news)} news articles")
    
        # Format output
        output = format_output(economic_calendar, pd.DataFrame(), top_stocks, top_news)
    
        logging.info(f"Generated summary: {output}")
        return output
    
    except Exception as e:
        logging.error(f"Error generating market summary: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Sorry, I couldn't generate a market summary at this time. Error: {str(e)}"

def format_output(economic_calendar, earnings_calendar, top_stocks, top_news):
    output = ""
    
    # Economic Calendar
    output += "Economic Calendar:\n"
    if not economic_calendar.empty:
        economic_calendar['Release Date'] = pd.to_datetime(economic_calendar['Release Date'])
        events_by_day = economic_calendar.groupby('Release Date')
        for date, group in events_by_day:
            output += f"{date.strftime('%Y-%m-%d')}:\n"
            for _, row in group.iterrows():
                impact_emoji = get_impact_emoji(row['Impact'])
                event_emoji = get_event_emoji(row['Release Name'])
                output += f"{impact_emoji} {event_emoji} {row['Release Name']} ({row['Release Time (ET)']})\n"
            output += "\n"
    else:
        output += "No economic events available.\n"
    
    # Top Stocks
    output += "\nTop Stocks Today:\n"
    if not top_stocks.empty:
        # Prepare data for three tables
        gainers = top_stocks.nlargest(10, 'Percent Change')
        losers = top_stocks.nsmallest(10, 'Percent Change')
        volume = top_stocks.nlargest(10, 'Volume')

        # Format data for tabulate
        gainers_data = [[row['Symbol'], f"{row['Percent Change']:.2f}%", f"{int(row['Volume']):,}"] for _, row in gainers.iterrows()]
        losers_data = [[row['Symbol'], f"{row['Percent Change']:.2f}%", f"{int(row['Volume']):,}"] for _, row in losers.iterrows()]
        volume_data = [[row['Symbol'], f"{row['Percent Change']:.2f}%", f"{int(row['Volume']):,}"] for _, row in volume.iterrows()]

        # Create tables
        gainers_table = tabulate(gainers_data, headers=['Top Gainers', '%', 'Volume'], tablefmt='pipe')
        losers_table = tabulate(losers_data, headers=['Top Losers', '%', 'Volume'], tablefmt='pipe')
        volume_table = tabulate(volume_data, headers=['Top Volume', '%', 'Shares'], tablefmt='pipe')

        output += gainers_table + "\n\n"
        output += volume_table + "\n\n"
        output += losers_table + "\n\n"
    else:
        output += "No top stocks data available.\n\n"
    
    # Top News
    output += "Top News Stories:\n"
    if top_news:
        for article in top_news:
            output += f"- {article['title']} ({article['url']})\n"
    else:
        output += "No news stories available.\n"
    
    return output

import os
import requests

def get_top_movers():
    url = f'https://finnhub.io/api/v1/stock/top-movers?exchange=US&token={FINNHUB_API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Error fetching top movers: {response.status_code}")
        return "Failed to fetch market data."

    data = response.json()
    
    top_gainers = data.get('gainers', [])[:5]  # Get top 5 gainers
    top_losers = data.get('losers', [])[:5]    # Get top 5 losers
    
    # Format the data
    gainers_text = '\n'.join([f"{idx+1}. {stock['symbol']} (+{stock['percent_change']}%)" for idx, stock in enumerate(top_gainers)])
    losers_text = '\n'.join([f"{idx+1}. {stock['symbol']} (-{stock['percent_change']}%)" for idx, stock in enumerate(top_losers)])
    
    market_summary = f"""
📈 **Top Gainers:**
{gainers_text}

📉 **Top Losers:**
{losers_text}
"""
    return market_summary

if not BOT_ID:
    logger.error("BOT_ID is not set or empty. Please check your environment variables.")
    sys.exit(1)

nest_asyncio.apply()

if __name__ == "__main__":
    application.run(debug=True)