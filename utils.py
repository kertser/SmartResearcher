from openai import OpenAI
import asyncio
import aiohttp
from web_search_agent import WebSearchAgent
from langsmith import traceable
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@traceable(name="call_openai")
async def call_openai_async(
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
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

    client = OpenAI(api_key=api_key)
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            # Format messages into the structure expected by your API client.
            formatted_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if not content:
                    continue
                formatted_messages.append({
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": str(content)
                        }
                    ]
                })

            if not formatted_messages:
                raise ValueError("No valid messages to send")

            kwargs = {
                "model": model,
                "messages": formatted_messages,
                "response_format": {"type": "text"}
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            response = await client.chat.completions.create(**kwargs)
            # Ensure we have a valid response:
            if not response.choices or len(response.choices) == 0:
                raise ValueError("No choices returned from OpenAI API")
            message = response.choices[0].message
            # Depending on the client, message may be a dict or an object.
            content = message.get("content", "") if isinstance(message, dict) else message.content
            return content if content else ""
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"OpenAI API error after {max_retries} retries: {e}")
                return None
            logger.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(base_delay * (2 ** attempt))
    return None



@traceable(name="perform_search")
async def perform_search_async(query: str) -> List[str]:
    """
    Searching with DuckDuckGo and returning list of URLs.
    Uses AsyncDDGS for better performance and rate limit handling.

    Args:
        query: Search query string

    Returns:
        List of URLs from search results
    """
    try:
        # Create search agent instance
        agent = WebSearchAgent(max_results=5)

        # Get search response asynchronously
        result = await agent.get_response(query=query)

        if result['status'] == 'success':
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
                logger.info(f"Search successful with {len(urls)} results")

            return urls

        else:
            logger.error(f"Search failed: {result.get('error', 'Unknown error')}")
            return []

    except Exception as e:
        logger.error(f"Error during search with DuckDuckGo: {str(e)}")
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
    # Strip, lower-case, and then compare to handle extra whitespace/punctuation
    res = response.strip().lower() if response else ""
    return "Yes" if res.startswith("yes") else "No"


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
