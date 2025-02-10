from openai import AsyncOpenAI
import asyncio
import aiohttp
from web_search_agent import WebSearchAgent
from langsmith import traceable
import logging
from typing import List, Dict, Optional
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@traceable(name="call_openai")
async def call_openai_async(
        messages: List[Dict[str, str]],
        model: str = "gpt-4",  # Updated default model
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
) -> Optional[str]:
    """
    Sends a request to the OpenAI Chat API and returns the answer.
    Includes rate limiting handling and retries.
    """
    if not api_key:
        raise ValueError("OpenAI API key is required")

    client = AsyncOpenAI(api_key=api_key)  # Use AsyncOpenAI
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            # Simplified message formatting
            formatted_messages = [
                {
                    "role": msg["role"],
                    "content": str(msg["content"])
                }
                for msg in messages
                if msg.get("content")
            ]

            if not formatted_messages:
                raise ValueError("No valid messages to send")

            kwargs = {
                "model": model,
                "messages": formatted_messages,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content if response.choices else ""

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"OpenAI API error after {max_retries} retries: {e}")
                return None
            logger.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(base_delay * (2 ** attempt))
    return None


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid and has a proper scheme."""
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False


@traceable(name="perform_search")
async def perform_search_async(query: str) -> List[str]:
    """
    Searching with Web Search Agent and returning list of valid URLs.
    """
    try:
        agent = WebSearchAgent(max_results=5)
        result = await agent.get_response(query=query)

        if result['status'] == 'success':
            urls = []
            summary_parts = result['response'].split('\n\n')

            for part in summary_parts:
                if 'Source:' in part:
                    url = part.split('Source:')[-1].strip()
                    if url and is_valid_url(url):
                        urls.append(url)

            logger.info(f"Found {len(urls)} valid search results")
            return urls

        logger.error(f"Search failed: {result.get('error', 'Unknown error')}")
        return []

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return []


@traceable(name="fetch_webpage")
async def fetch_webpage_text_async(session: aiohttp.ClientSession, url: str) -> str:
    """
    Asynchronously loads webpages with proper error handling and encoding detection.
    """
    if not is_valid_url(url):
        logger.warning(f"Invalid URL skipped: {url}")
        return ""

    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status == 200:
                # Try different encodings
                try:
                    content = await resp.read()
                    # Try UTF-8 first
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        # Try Latin-1 as fallback
                        text = content.decode('latin-1')
                    except UnicodeDecodeError:
                        # Try with errors ignored as last resort
                        text = content.decode('utf-8', errors='ignore')
                return text
            else:
                logger.warning(f"Error loading {url}: {resp.status}")
                return ""
    except asyncio.TimeoutError:
        logger.warning(f"Timeout while fetching {url}")
        return ""
    except Exception as e:
        logger.error(f"Error reading webpage from {url}: {e}")
        return ""


@traceable(name="evaluate_page_usefulness")
async def is_page_useful_async(user_query: str, page_text: str, api_key: str) -> str:
    """
    Checks whether the webpage is useful with OpenAI.
    Returns "Yes" or "No".
    """
    if not page_text.strip():
        return "No"

    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, "
        "determine if the webpage contains information relevant and useful for addressing the query. "
        "Respond with exactly one word: 'Yes' or 'No'."
    )
    messages = [
        {"role": "system", "content": "You are a strict evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {user_query}\nWebpage Content: {page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_openai_async(messages=messages, api_key=api_key)
    return "Yes" if response and response.strip().lower().startswith("yes") else "No"


@traceable(name="extract_relevant_context")
async def extract_relevant_context_async(user_query: str,
                                       search_query: str,
                                       page_text: str,
                                       api_key: str) -> str:
    """
    Extracts relevant information from the webpage text
    """
    if not page_text.strip():
        return ""

    prompt = (
        "You are an expert in extracting relevant information. Given the user's query, the search query that led to "
        "this page, and the webpage content, extract all relevant information. Return only the relevant text."
    )
    messages = [
        {"role": "system", "content": "You are a precise information extractor."},
        {
            "role": "user",
            "content": (
                f"User Query: {user_query}\n"
                f"Search Query: {search_query}\n"
                f"Webpage Content: {page_text[:20000]}\n\n{prompt}"
            )
        }
    ]
    response = await call_openai_async(messages=messages, api_key=api_key)
    return response.strip() if response else ""