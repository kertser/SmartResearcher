import aiohttp
import os
import asyncio
from langgraph.graph import StateGraph
from langsmith import traceable
from langsmith.client import Client
from graphviz import Digraph
from utils import perform_search_async, fetch_webpage_text_async, is_page_useful_async, extract_relevant_context_async
from dotenv import load_dotenv
import logging
from typing import TypedDict, List, Optional
import shutil

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


def check_graphviz_installed() -> bool:
    """Check if Graphviz is installed and accessible"""
    return shutil.which('dot') is not None


async def create_visualization(client: Client, project_name: str) -> None:
    """Create visualization if Graphviz is available"""
    if not check_graphviz_installed():
        logger.warning("Graphviz is not installed or not in PATH. Skipping visualization.")
        logger.info("To install Graphviz:\n"
                    "1. Download from https://graphviz.gitlab.io/download/\n"
                    "2. Add the bin directory to your system's PATH\n"
                    "3. Restart your terminal/IDE")
        return

    try:
        # Retrieve runs associated with the specified project
        runs = list(client.list_runs(project_name=project_name, is_root=True))

        if not runs:
            logger.warning(
                f"No runs found for project '{project_name}'. Ensure that the project name is correct and that runs have been logged.")
            return

        latest_run = runs[0]  # Get the latest run
        child_runs = list(client.list_runs(parent_run=latest_run.id))

        # Create a graph
        dot = Digraph()
        dot.attr(rankdir='TB')  # Top to bottom direction

        # Add the root node
        dot.node(str(latest_run.id), latest_run.name)

        # Add child nodes and edges
        for child_run in child_runs:
            dot.node(str(child_run.id), child_run.name)
            dot.edge(str(latest_run.id), str(child_run.id))

        # Render flowchart
        dot.render("flowchart", format="png", view=True)
        logger.info("Visualization created successfully")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")


@traceable(name="search_step")
async def search_step(state: dict, **kwargs) -> dict:
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
async def process_results_step(state: dict, **kwargs) -> dict:
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
    workflow = StateGraph(ResearchState)

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

    workflow.set_entry_point("search")
    workflow.add_edge("search", "process_results")
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
        client = Client()

        # Get user input
        user_query = input("Enter research query: ").strip()
        if not user_query:
            raise ValueError("Query cannot be empty")

        # Initialize the state
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

        # Try to create visualization
        await create_visualization(client, "SmartResearch")

        # Display results regardless of visualization success
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
