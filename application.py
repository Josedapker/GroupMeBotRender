import sys
import json
import asyncio
from datetime import datetime, timezone
import time
from openai import OpenAI
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import httpx
from flask import Flask, request, jsonify
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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tabulate import tabulate
import requests
import aiohttp
from functools import lru_cache

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

print(f"BOT_ID loaded: {BOT_ID is not None}")
print(f"OPENAI_API_KEY loaded: {OPENAI_API_KEY is not None}")
print(f"TAVILY_API_KEY loaded: {TAVILY_API_KEY is not None}")
print(f"GROUPME_ACCESS_TOKEN loaded: {GROUPME_ACCESS_TOKEN is not None}")
print(f"NEWS_API_KEY loaded: {NEWS_API_KEY is not None}")

# Remove any logging of the actual API key
print("OpenAI API Key loaded successfully.")

# Set the OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

from tavily import TavilyClient
tavily_client = TavilyClient(TAVILY_API_KEY)

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

@application.route('/', methods=['POST'])
@limiter.limit(
    f"{RATE_LIMIT_PER_MINUTE} per minute; {RATE_LIMIT_PER_HOUR} per hour; {RATE_LIMIT_PER_DAY} per day",
    key_func=get_rate_limit_key
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
            logger.info("Generating market summary")
            summary = await generate_market_summary()
            if summary:
                logger.info("Market summary generated")
                # Send the summary to GroupMe
                await send_message(BOT_ID, summary)
                return jsonify(success=True, message="Market summary sent to GroupMe"), 200
            else:
                return jsonify(success=False, message="Failed to generate market summary"), 500
        
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

@lru_cache(maxsize=100)
def get_stock_data(symbol, data):
    try:
        current_price = data.get('currentPrice')
        previous_close = data.get('previousClose')
        
        if current_price is None or previous_close is None:
            logging.warning(f"Missing price data for {symbol}. Current: {current_price}, Previous: {previous_close}")
            return None
        
        percent_change = ((current_price - previous_close) / previous_close) * 100
        volume = data.get('volume', 0)
        
        logging.info(f"Successfully fetched data for {symbol}")
        return {
            'Symbol': symbol,
            'Percent Change': percent_change,
            'Volume': volume
        }
    except Exception as e:
        logging.error(f"Error processing stock data for {symbol}: {str(e)}")
        return None

async def fetch_stock_data(session, symbol):
    try:
        async with session.get(f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}") as response:
            data = await response.json()
            return get_stock_data(symbol, data['quoteResponse']['result'][0])
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

async def get_top_stocks(symbols, batch_size=50):
    all_results = []
    failed_symbols = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [fetch_stock_data(session, symbol) for symbol in batch]
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if result:
                    all_results.append(result)
                else:
                    failed_symbols.append(result['Symbol'])
            
            await asyncio.sleep(1)  # Add a short delay between batches
    
    df = pd.DataFrame(all_results)
    if df.empty:
        logging.warning("No stock data could be fetched.")
        return df, failed_symbols
    df = df.sort_values(by=['Volume', 'Percent Change'], ascending=False)
    return df.head(10), failed_symbols

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
        with open('usd_us_events_high_medium.json', 'r') as file:
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
        print(f"Error processing economic calendar data: {e}")
        return pd.DataFrame()

async def generate_market_summary():
    try:
        # Define date range for the week
        today = datetime.today()
        start_date = today.strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=7)).strftime('%Y-%m-%d')

        # Fetch all S&P 500 symbols
        all_symbols = get_sp500_symbols()
        print(f"Fetched {len(all_symbols)} S&P 500 symbols")

        # Fetch economic calendar
        economic_calendar = get_economic_calendar(start_date, end_date)

        # Fetch top stocks (limit to 100 symbols for faster processing)
        top_stocks, failed_symbols = await get_top_stocks(all_symbols[:100])
        print(f"Fetched top stocks. DataFrame empty: {top_stocks.empty}")
        
        if top_stocks.empty:
            return "Unable to fetch stock data at this time."

        # Fetch top news
        top_news = get_top_news()
        print(f"Fetched {len(top_news)} news articles")

        # Format output
        output = "Daily Market Summary:\n\n"

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

        print(f"Generated summary: {output}")  # Add this line to print the summary
        return output

    except Exception as e:
        logger.error(f"Error generating market summary: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Sorry, I couldn't generate a market summary at this time. Error: {str(e)}"

if not BOT_ID:
    logger.error("BOT_ID is not set or empty. Please check your environment variables.")
    sys.exit(1)

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    asgi_app = WsgiToAsgi(application)
    asyncio.run(serve(asgi_app, config))
