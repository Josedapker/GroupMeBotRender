import sys
import json
import asyncio
import websockets
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
import requests
from logging.handlers import RotatingFileHandler
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from asgiref.wsgi import WsgiToAsgi
import traceback
from hypercorn.asyncio import serve
from hypercorn.config import Config

# Load environment variables
load_dotenv()

# Configuration settings
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "groupme_bot.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 10))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", 50))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", 200))
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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROUPME_ACCESS_TOKEN = os.getenv("GROUPME_ACCESS_TOKEN")

print(f"BOT_ID loaded: {BOT_ID is not None}")
print(f"OPENAI_API_KEY loaded: {OPENAI_API_KEY is not None}")
print(f"TAVILY_API_KEY loaded: {TAVILY_API_KEY is not None}")
print(f"GROUPME_ACCESS_TOKEN loaded: {GROUPME_ACCESS_TOKEN is not None}")

print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')[:5]}...")

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

limiter = Limiter(
    get_remote_address,
    app=application,
    default_limits=[f"{RATE_LIMIT_PER_DAY} per day", f"{RATE_LIMIT_PER_HOUR} per hour"],
    storage_uri="memory://"
)

def split_message(text: str, limit: int = 1000) -> List[str]:
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        part = text[:limit]
        last_space = part.rfind(' ')
        if last_space == -1:
            parts.append(part)
            text = text[limit:]
        else:
            parts.append(part[:last_space])
            text = text[last_space+1:]
    return parts

async def upload_image_to_groupme(image_url: str) -> str | None:
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

async def send_message(bot_id: str, text: str, image_url: str | None = None) -> None:
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
            data = {
                "bot_id": bot_id,
                "text": part,
                "attachments": attachment if i == 0 else []
            }

            try:
                response = await client.post(url, json=data)
                response.raise_for_status()
                print(f"Message part {i+1} sent successfully!")
            except httpx.RequestError as e:
                print(f"Error sending message part {i+1}: {str(e)}")
            
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

For any issues or feature requests, please contact the bot administrator.
"""
    return help_message

@application.route('/', methods=['POST'])
def webhook():
    data = request.get_json()
    
    # Process the incoming message
    if data['name'] != 'GroupMe Bot':  # Prevent the bot from responding to itself
        message = data.get('text', '').lower()
        if message.startswith('!help'):
            send_message_sync(BOT_ID, get_help_message())
        elif message.startswith('!usage'):
            usage_info = get_openai_usage_sync()
            send_message_sync(BOT_ID, usage_info)
        elif message.startswith('!ai4'):
            prompt = validate_prompt(message[5:].strip())
            if prompt:
                logger.info(f"Generating AI4 response for prompt: '{prompt}'")
                response = generate_ai_response_sync(prompt, "gpt-4", True)
                logger.info(f"AI4 response generated: {response}")
                send_message_sync(BOT_ID, response)
            else:
                send_message_sync(BOT_ID, "Please provide a valid prompt.")
        elif message.startswith('!ai'):
            prompt = validate_prompt(message[4:].strip())
            if prompt:
                logger.info(f"Generating AI response for prompt: '{prompt}'")
                response = generate_ai_response_sync(prompt, "gpt-3.5-turbo", False)
                logger.info(f"AI response generated: {response}")
                send_message_sync(BOT_ID, response)
            else:
                send_message_sync(BOT_ID, "Please provide a valid prompt.")
        elif message.startswith('!image'):
            prompt = validate_prompt(message[7:].strip())
            if prompt:
                logger.info(f"Generating image for prompt: '{prompt}'")
                image_url = generate_image(prompt)
                if image_url:
                    logger.info(f"Image generated successfully: {image_url}")
                    send_message_sync(BOT_ID, f"Image for '{prompt}':", image_url)
                    logger.info("Message with image sent to GroupMe")
                else:
                    logger.error(f"Failed to generate image for '{prompt}'")
                    send_message_sync(BOT_ID, f"Failed to generate image for '{prompt}'")
            else:
                logger.warning("Invalid prompt for image generation")
                send_message_sync(BOT_ID, "Please provide a valid prompt for image generation.")
        
        return jsonify(success=True), 200
    
    return jsonify(success=True), 200

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
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }

        params = {"date": date_str}
        async with httpx.AsyncClient() as client:
            response = await client.get(usage_url, headers=headers, params=params)
            response.raise_for_status()
            usage_data = response.json()

        total_tokens = sum(item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0) 
                           for item in usage_data.get('data', []))
        gpt35_tokens = sum(item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0) 
                           for item in usage_data.get('data', []) if 'gpt-3.5' in item.get('snapshot_id', ''))
        gpt4_tokens = sum(item.get('n_context_tokens_total', 0) + item.get('n_generated_tokens_total', 0) 
                          for item in usage_data.get('data', []) if 'gpt-4' in item.get('snapshot_id', ''))
        dalle_images = sum(item.get('num_images', 0) for item in usage_data.get('dalle_api_data', []))

        gpt35_cost = (gpt35_tokens / 1000) * 0.002
        gpt4_cost = (gpt4_tokens / 1000) * 0.06
        dalle_cost = dalle_images * 0.02
        total_cost = gpt35_cost + gpt4_cost + dalle_cost

        return f"OpenAI API Usage Summary for {date_str}:\n" \
               f"Total tokens: {total_tokens:,}\n" \
               f"GPT-3.5 tokens: {gpt35_tokens:,}\n" \
               f"GPT-4 tokens: {gpt4_tokens:,}\n" \
               f"DALL-E images: {dalle_images}\n" \
               f"Estimated cost: ${total_cost:.2f}\n" \
               f"(GPT-3.5: ${gpt35_cost:.2f}, GPT-4: ${gpt4_cost:.2f}, DALL-E: ${dalle_cost:.2f})"

    except Exception as e:
        return f"Error fetching OpenAI usage: {str(e)}"

def generate_image(prompt: str) -> str | None:
    try:
        logger.info(f"Attempting to generate image with prompt: {prompt}")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        logger.info(f"Image generation response: {response}")
        return response.data[0].url
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None

def search_web(query: str) -> List[Dict] | None:
    try:
        search_result = tavily_client.search(query=query, search_depth="advanced")
        return search_result['results'][:3]  # Return top 3 results
    except Exception as e:
        print(f"Error searching web: {e}")
        return None

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
            search_content = "\n".join([f"- {result['title']}: {result['content'][:200]}..." for result in limited_results])
            logger.info(f"Web search results: {search_content}")
            
            messages = [
                {"role": "system", "content": "You are an advanced AI assistant with access to recent web information. Provide accurate and helpful responses based on the given web search results and your knowledge."},
                {"role": "user", "content": f"Web search results for '{prompt}':\n{search_content}" if search_content else "No search results found."},
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

def send_message_sync(bot_id: str, text: str, image_url: str | None = None) -> None:
    asyncio.run(send_message(bot_id, text, image_url))

def get_openai_usage_sync() -> str:
    return asyncio.run(get_openai_usage())

def generate_ai_response_sync(prompt: str, model: str = "gpt-3.5-turbo", use_web_search: bool = False) -> str:
    return asyncio.run(generate_ai_response(prompt, model, use_web_search))

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    asgi_app = WsgiToAsgi(application)
    asyncio.run(serve(asgi_app, config))