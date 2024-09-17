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

# Apply nest_asyncio to allow nested event loops (useful for certain environments)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get the directory where application.py resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to local files
ECONOMIC_CALENDAR_FILE = os.path.join(BASE_DIR, "usd_us_events.json")
TOP_COMPANIES_FILE = os.path.join(BASE_DIR, "top100_companies.txt")

print(f"Economic Calendar File Path: {ECONOMIC_CALENDAR_FILE}")  # Debug statement
print(f"Top Companies File Path: {TOP_COMPANIES_FILE}")          # Debug statement

# Configuration settings - Removed from ENV and set as constants
LOG_FILE_PATH = "groupme_bot.log"
LOG_LEVEL = "INFO"
RATE_LIMIT_PER_MINUTE = 50
RATE_LIMIT_PER_HOUR = 500
RATE_LIMIT_PER_DAY = 5000
MAX_PROMPT_LENGTH = 500

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
    except httpx.RequestError as e:
        print(f"Failed to upload image: {str(e)}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

async def send_message(bot_id: str, text: str, image_url: Union[str, None] = None) -> None:
    url = "https://api.groupme.com/v3/bots/post"

    attachment = []
    if image_url:
        print(f"Attempting to upload image: {image_url}")
        groupme_image_url = await upload_image_to_groupme(image_url)
        if groupme_image_url:
            print(f"Image uploaded successfully to GroupMe: {groupme_image_url}")
            attachment = [{"type": "image", "url": groupme_image_url}]
        else:
            print("Failed to upload image, sending message without image.")

    message_parts = split_message(text)

    async with httpx.AsyncClient() as client:
        for i, part in enumerate(message_parts):
            escaped_part = html.escape(part)  # Escape special characters
            data = {
                "bot_id": bot_id,
                "text": escaped_part,
            }
            if i == 0 and attachment:
                data["attachments"] = attachment

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
                    logger.error(f"Full error details: {e.request.url}, {e.request.headers}, {e.request.content}")
                    if 'response' in locals():
                        logger.error(f"Response content: {response.content}")
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(1)  # Wait before retrying
                except Exception as e:
                    logger.error(f"Unexpected error in send_message: {str(e)}")
                    logger.error(traceback.format_exc())
                    break  # Exit the retry loop for unexpected errors

            # Wait between message parts, with a longer delay after the first part
            if i == 0:
                await asyncio.sleep(2)  # Longer delay after the first part
            elif i < len(message_parts) - 1:
                await asyncio.sleep(1)  # Shorter delay between subsequent parts

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
    elif 'inflation' in event_name or 'cpi' in event_name:
        return '💹'
    elif 'interest rate' in event_name:
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

def format_market_summary(economic_calendar: pd.DataFrame, top_stocks: pd.DataFrame, top_news: List[Dict]) -> str:
    output = "📊 MARKET SUMMARY 📊\n\n"

    # Economic Calendar
    output += "📅 ECONOMIC CALENDAR 📅\n"
    if not economic_calendar.empty:
        economic_calendar['Release Date'] = pd.to_datetime(economic_calendar['Release Date'])
        events_by_day = economic_calendar.groupby('Release Date')
        for date, group in events_by_day:
            output += f"--- {date.strftime('%Y-%m-%d')} ---\n"
            for _, row in group.iterrows():
                impact_emoji = get_impact_emoji(row['Impact'])
                event_emoji = get_event_emoji(row['Release Name'])
                output += f"{impact_emoji}{event_emoji} {row['Release Name']} ({row['Release Time (ET)']}) - {row['Impact']} Impact\n"
            output += "\n"
    else:
        output += "No economic events available.\n\n"

    # Top Stocks
    output += "📈 TOP STOCKS 📈\n"
    if not top_stocks.empty:
        categories = [
            ("TOP GAINERS", top_stocks.nlargest(5, 'Percent Change')),
            ("TOP LOSERS", top_stocks.nsmallest(5, 'Percent Change')),
            ("HIGHEST VOLUME", top_stocks.nlargest(5, 'Volume'))
        ]
        for title, df in categories:
            output += f"--- {title} ---\n"
            for _, row in df.iterrows():
                output += f"{row['Symbol']}: {row['Percent Change']:.2f}% ({int(row['Volume']):,})\n"
            output += "\n"
    else:
        output += "No stock data available.\n\n"

    # Top News
    output += "📰 TOP NEWS 📰\n"
    if top_news:
        for i, article in enumerate(top_news[:5], 1):
            output += f"{i}. {article['title']}\n   {article['url']}\n\n"
    else:
        output += "No news articles available.\n"

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

    # Format and return the market summary
    market_summary = format_market_summary(economic_calendar, top_stocks_df, top_news)
    return market_summary

def generate_economic_calendar() -> str:
    today = datetime.today()
    start_date = today.strftime('%Y-%m-%d')
    end_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')
    economic_calendar = get_economic_calendar(start_date, end_date)
    
    output = "📅 ECONOMIC CALENDAR 📅\n\n"
    if not economic_calendar.empty:
        events_by_day = economic_calendar.groupby('Release Date')
        for date, group in events_by_day:
            output += f"--- {date.strftime('%Y-%m-%d')} ---\n"
            for _, row in group.iterrows():
                impact_emoji = get_impact_emoji(row['Impact'])
                event_emoji = get_event_emoji(row['Release Name'])
                output += f"{impact_emoji}{event_emoji} {row['Release Name']} ({row['Release Time (ET)']}) - {row['Impact']} Impact\n"
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
                output += f"{row['Symbol']}: {row['Percent Change']:.2f}% ({int(row['Volume']):,})\n"
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

# Webhook route
@application.route('/', methods=['POST'])
@limiter.limit(
    f"{RATE_LIMIT_PER_MINUTE} per minute; {RATE_LIMIT_PER_HOUR} per hour; {RATE_LIMIT_PER_DAY} per day",
    key_func=get_remote_address
)
async def webhook():
    try:
        data = request.get_json(force=True)
        logger.info(f"Received message: {json.dumps(data)}")
        
        # Ignore messages sent by bots (including itself)
        if data.get('sender_type') == 'bot':
            logger.debug("Message sent by bot; ignoring.")
            return jsonify(success=True), 200
        
        # Process messages from users
        message = html.unescape(data.get('text', ''))
        logger.info(f"Processed message: {message}")

        if message.lower().startswith('!help'):
            await send_message(BOT_ID, get_help_message())
        elif message.lower().startswith('!usage'):
            usage_info = await get_openai_usage()
            await send_message(BOT_ID, usage_info)
        elif message.lower().startswith('!ai4'):
            prompt = validate_prompt(message[5:].strip())
            if prompt:
                logger.info(f"Generating AI4 response for prompt: '{prompt}'")
                response = await generate_ai_response(prompt, "gpt-4", True)
                logger.info(f"AI4 response generated: {response}")
                await send_message(BOT_ID, response)
            else:
                await send_message(BOT_ID, "Please provide a valid prompt.")
        elif message.lower().startswith('!ai'):
            prompt = validate_prompt(message[4:].strip())
            if prompt:
                logger.info(f"Generating AI response for prompt: '{prompt}'")
                response = await generate_ai_response(prompt, "gpt-3.5-turbo", False)
                logger.info(f"AI response generated: {response}")
                await send_message(BOT_ID, response)
            else:
                await send_message(BOT_ID, "Please provide a valid prompt.")
        elif message.lower().startswith('!image'):
            prompt = validate_prompt(message[7:].strip())
            if prompt:
                logger.info(f"Generating image for prompt: '{prompt}'")
                image_url = await generate_image(prompt)
                if image_url:
                    logger.info(f"Image generated successfully: {image_url}")
                    await send_message(BOT_ID, f"Here's your image for '{prompt}'", image_url)
                    logger.info("Message with image sent to GroupMe")
                else:
                    logger.error(f"Failed to generate image for '{prompt}'")
                    await send_message(BOT_ID, f"Sorry, I couldn't generate an image for '{prompt}'")
            else:
                logger.warning("Invalid prompt for image generation")
                await send_message(BOT_ID, "Please provide a valid prompt for image generation.")
        elif message.lower().startswith('!market'):
            logger.info("Processing !market command")
            market_summary = generate_market_summary()
            await send_message(BOT_ID, market_summary)
        elif message.lower().startswith('!calendar'):
            calendar = generate_economic_calendar()
            await send_message(BOT_ID, calendar)
        elif message.lower().startswith('!stocks'):
            stocks = generate_top_stocks()
            await send_message(BOT_ID, stocks)
        elif message.lower().startswith('!news'):
            news = generate_top_news()
            await send_message(BOT_ID, news)
        
        return jsonify(success=True), 200

    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from request")
        return jsonify(success=False, error="Invalid JSON"), 400
    except Exception as e:
        logger.error(f"Unexpected error in webhook: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify(success=False, error="Internal server error"), 500

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

@application.route('/test_stocks', methods=['GET'])
async def test_stocks():
    stocks = generate_top_stocks()
    return f"<pre>{stocks}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_news', methods=['GET'])
async def test_news():
    news = generate_top_news()
    return f"<pre>{news}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_calendar', methods=['GET'])
async def test_calendar():
    calendar = generate_economic_calendar()
    return f"<pre>{calendar}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_stocks', methods=['GET'])
async def test_stocks():
    stocks = generate_top_stocks()
    return f"<pre>{stocks}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

@application.route('/test_news', methods=['GET'])
async def test_news():
    news = generate_top_news()
    return f"<pre>{news}</pre>", 200, {'Content-Type': 'text/html; charset=utf-8'}

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    asgi_app = WsgiToAsgi(application)
    asyncio.run(serve(asgi_app, config))
