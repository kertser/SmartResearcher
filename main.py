import aiohttp
import os
import asyncio
from langgraph.graph import StateGraph
from langsmith import traceable
from utils import perform_search_async, fetch_webpage_text_async, is_page_useful_async, extract_relevant_context_async
from dotenv import load_dotenv
import logging
from typing import TypedDict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define the state structure as a TypedDict
class ResearchState(TypedDict):
    user_query: str
    search_results: List[str]
    relevant_texts: List[str]
    api_key: Optional[str]


@traceable(name="search_step")
async def search_step(state: dict) -> dict:
    """Searches following the user request"""
    search_results = await perform_search_async(state["user_query"])
    logger.info(f"Found {len(search_results)} search results")
    # Return only the updates to the state
    return {"search_results": search_results}


async def process_single_url(session: aiohttp.ClientSession, url: str, query: str, api_key: str) -> str | None:
    """Process a single URL with proper error handling"""
    try:
        page_text = await fetch_webpage_text_async(session, url)
        if page_text:
            is_useful = await is_page_useful_async(query, page_text, api_key)
            if is_useful == "Yes":
                return await extract_relevant_context_async(query, query, page_text, api_key)
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
    return None


@traceable(name="process_results_step")
async def process_results_step(state: dict) -> dict:
    """Filtering links of the provided urls"""
    relevant_texts = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        # Process URLs concurrently
        tasks = []
        for url in state["search_results"]:
            if state["api_key"]:  # Ensure we have an API key
                tasks.append(process_single_url(session, url, state["user_query"], state["api_key"]))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out errors and None results
            relevant_texts.extend([text for text in results if isinstance(text, str) and text])
            logger.info(f"Processed {len(relevant_texts)} relevant results")
        except Exception as e:
            logger.error(f"Error in processing results: {e}")

    # Return only the updates to the state
    return {"relevant_texts": relevant_texts}


def initialize_workflow() -> StateGraph:
    """Initialize and configure the workflow"""
    # Initialize with TypedDict instead of class
    workflow = StateGraph(ResearchState)

    # Add nodes - using the correct method signature
    workflow.add_node(
        "search",
        action=search_step,
        metadata={"description": "Performs search based on user query"}
    )
    workflow.add_node(
        "process_results",
        action=process_results_step,
        metadata={"description": "Processes and filters search results"}
    )

    # Set entry point
    workflow.set_entry_point("search")

    # Add edge
    workflow.add_edge("search", "process_results")

    # Set end node
    workflow.set_finish_point("process_results")

    return workflow


def load_keys() -> dict:
    """Initialize environment and API keys"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return {"openai_api_key": api_key}


async def async_main() -> None:
    try:
        # Load API keys
        keys = load_keys()

        # Get user input
        user_query = input("Enter research query: ").strip()
        if not user_query:
            raise ValueError("Query cannot be empty")

        # Initialize the state as a dictionary
        initial_state = {
            "user_query": user_query,
            "search_results": [],
            "relevant_texts": [],
            "api_key": keys["openai_api_key"]
        }

        # Initialize and run workflow
        workflow = initialize_workflow()
        app = workflow.compile()
        result = await app.ainvoke(initial_state)

        # Display results
        print("\n=== Final Report ===\n")
        if result["relevant_texts"]:
            print("\n".join(result["relevant_texts"]))
        else:
            print("No relevant information found.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(async_main())
