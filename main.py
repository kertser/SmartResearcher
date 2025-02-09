import aiohttp
import os
import asyncio
import logging
from langgraph.graph import StateGraph
from utils import perform_search_async, fetch_webpage_text_async, is_page_useful_async, extract_relevant_context_async
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class State:
    def __init__(self, user_query: str, api_key: Optional[str] = None):
        self.user_query: str = user_query
        self.search_results: List[str] = []
        self.relevant_texts: List[str] = []
        self.api_key: Optional[str] = api_key


async def search_step(state: State) -> State:
    """Searches following the user request"""
    state.search_results = await perform_search_async(state.user_query)
    logger.info(f"Found {len(state.search_results)} search results")
    return state


async def process_single_url(session: aiohttp.ClientSession, url: str, query: str, api_key: str) -> Optional[str]:
    """Process a single URL with proper error handling"""
    try:
        page_text = await fetch_webpage_text_async(session, url)
        if not page_text:
            return None

        is_useful = await is_page_useful_async(query, page_text, api_key)
        if is_useful == "Yes":
            return await extract_relevant_context_async(query, query, page_text, api_key)
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
    return None


async def process_results_step(state: State) -> State:
    """Filtering links of the provided urls"""
    timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Process URLs concurrently with proper timeout handling
        tasks = [
            process_single_url(session, url, state.user_query, state.api_key)
            for url in state.search_results
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out errors and None results
            valid_results = [
                text for text in results
                if text and isinstance(text, str)
            ]
            state.relevant_texts.extend(valid_results)
            logger.info(f"Processed {len(valid_results)} relevant results")
        except asyncio.TimeoutError:
            logger.error("Processing timeout occurred")
        except Exception as e:
            logger.error(f"Error during processing: {e}")

    return state


def load_keys(keys: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize environment and load API keys"""
    if keys is None:
        keys = {}

    # Load environment variables
    load_dotenv()

    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")

    keys['openai_api_key'] = api_key
    return keys


async def async_main() -> None:
    try:
        # Load API keys
        keys = load_keys()

        # Initialize workflow
        workflow = StateGraph(State)
        workflow.add_node("search", search_step)
        workflow.add_node("process_results", process_results_step)
        workflow.set_entry_point("search")
        workflow.add_edge("search", "process_results")
        workflow.finalize()

        # Get user input
        user_query = input("Enter research query: ").strip()
        if not user_query:
            raise ValueError("Query cannot be empty")

        # Run workflow
        app = workflow.compile()
        result = await app.invoke(State(user_query, api_key=keys['openai_api_key']))

        # Display results
        print("\n=== Final Report ===\n")
        if result.relevant_texts:
            print("\n".join(result.relevant_texts))
        else:
            print("No relevant information found.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(async_main())