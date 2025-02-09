from openai import OpenAI
import asyncio
import aiohttp
from web_search_agent import WebSearchAgent
from langsmith import traceable
import logging
from typing import List, Dict, Optional
from random import uniform
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchRateLimiter:
    def __init__(self):
        self.last_success = datetime.now() - timedelta(minutes=5)
        self.cooldown_period = timedelta(minutes=1)
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

    async def wait_if_needed(self):
        now = datetime.now()
        time_since_last = now - self.last_success

        if self.consecutive_failures >= self.max_consecutive_failures:
            # Extend cooldown period after multiple failures
            await asyncio.sleep(60 * (2 ** (self.consecutive_failures - self.max_consecutive_failures)))
            self.consecutive_failures = 0

        elif time_since_last < self.cooldown_period:
            delay = (self.cooldown_period - time_since_last).total_seconds()
            await asyncio.sleep(delay + uniform(1, 5))

    def record_success(self):
        self.last_success = datetime.now()
        self.consecutive_failures = 0

    def record_failure(self):
        self.consecutive_failures += 1


# Global rate limiter instance
rate_limiter = SearchRateLimiter()

@traceable(name="call_openai")
async def call_openai_async(
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
) -> Optional[str]:
    """
    Sending request to OpenAI Chat API and returning the answer.
    Includes rate limiting handling and retries.
    """
    if not api_key:
        raise ValueError("OpenAI API key is required")

    client = OpenAI(api_key=api_key)
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            # Format messages according to the new API format
            formatted_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if not content:  # Skip empty messages
                    continue

                formatted_msg = {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": str(content)
                        }
                    ]
                }
                formatted_messages.append(formatted_msg)

            if not formatted_messages:
                raise ValueError("No valid messages to send")

            # Prepare request kwargs
            kwargs = {
                "model": model,
                "messages": formatted_messages,
                "response_format": {"type": "text"}
            }

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            # Make the API call
            response = await client.chat.completions.create(**kwargs)

            # Extract and return the content
            message = response.choices[0].message
            return message.content if message.content else ""

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"OpenAI API error after {max_retries} retries: {str(e)}")
                return None

            logger.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            await asyncio.sleep(base_delay * (2 ** attempt))

    return None


@traceable(name="perform_search")
async def perform_search_async(query: str, max_retries: int = 3) -> List[str]:
    """
    Searching with DuckDuckGo and returning list of URLs.
    Uses WebSearchAgent with improved rate limit handling.

    Args:
        query: Search query string
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        List of URLs from search results
    """
    global rate_limiter

    for attempt in range(max_retries):
        try:
            # Check and wait for rate limit
            await rate_limiter.wait_if_needed()

            # Create new WebSearchAgent instance with increased delay
            agent = WebSearchAgent(
                max_results=5,
                min_request_interval=5 + uniform(1, 3)  # Random delay between 5-8 seconds
            )

            # Get search response
            result = agent.get_response(
                query=query,
                search_type='text'  # Explicitly use text search
            )

            if 'Ratelimit' in str(result.get('error', '')):
                rate_limiter.record_failure()
                raise Exception("Rate limit reached")

            if result['status'] == 'success':
                rate_limiter.record_success()

                # Extract URLs from the response
                summary_parts = result['response'].split('\n\n')
                urls = []

                for part in summary_parts:
                    if 'Source:' in part:
                        url = part.split('Source:')[-1].strip()
                        if url:
                            urls.append(url)

                logger.info(f"Found {len(urls)} search results")
                if urls:
                    logger.info(f"Search successful: {len(urls)} results found")
                return urls

            else:
                rate_limiter.record_failure()
                logger.warning(f"Search attempt {attempt + 1} failed: {result.get('error', 'Unknown error')}")
                if attempt == max_retries - 1:
                    logger.error("All search attempts failed")
                    return []
                continue

        except Exception as e:
            rate_limiter.record_failure()
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Error during search with DuckDuckGo after {max_retries} attempts: {str(e)}")
                return []

            # Calculate delay for next attempt
            delay = (60 * (2 ** attempt)) + uniform(1, 10)
            logger.info(f"Waiting {delay:.2f} seconds before next attempt...")
            await asyncio.sleep(delay)

    return []

@traceable(name="fetch_webpage")
async def fetch_webpage_text_async(session: aiohttp.ClientSession, url: str) -> str:
    """
    Asynchronously loads webpages with proper error handling
    """
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status == 200:
                return await resp.text()
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
    return response.strip() if response in ["Yes", "No"] else "No"

@traceable(name="extract_relevant_context")
async def extract_relevant_context_async(user_query: str,
                                       search_query: str,
                                       page_text: str,
                                       api_key: str) -> str:
    """
    Extracts relevant information from the webpage text
    """
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
