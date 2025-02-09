import openai
import asyncio
import aiohttp
from duckduckgo_search import DDGS
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
async def call_openai_async(messages: List[Dict[str, str]],
                          model: str = "gpt-4o-mini",
                          api_key: Optional[str] = None) -> Optional[str]:
    """
    Sending request to OpenAI Chat API and returning the answer.
    """
    if not api_key:
        raise ValueError("OpenAI API key is required")

    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            api_key=api_key,
            timeout=30
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return None

@traceable(name="perform_search")
async def perform_search_async(query: str) -> List[str]:
    """
    Searching with DuckDuckGo and returning list of URLs.
    """
    try:
        results = list(DDGS().text(query, max_results=10))
        urls = [result['href'] for result in results]
        logger.info(f"Found {len(urls)} search results")
        return urls
    except Exception as e:
        logger.error(f"Error during search with DuckDuckGo: {e}")
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
    response = await call_openai_async(messages, api_key=api_key)
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
    response = await call_openai_async(messages, api_key=api_key)
    return response.strip() if response else ""
