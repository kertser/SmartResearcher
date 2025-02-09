import aiohttp
import os
from langgraph.graph import StateGraph
from utils import perform_search_async, fetch_webpage_text_async, is_page_useful_async, extract_relevant_context_async
from dotenv import load_dotenv


class State:
    def __init__(self, user_query):
        self.user_query = user_query
        self.search_results = []
        self.relevant_texts = []


async def search_step(state):
    """Searches following the user request"""
    state.search_results = await perform_search_async(state.user_query)
    return state


async def process_results_step(state):
    """Filtering links of the provided urls"""
    async with aiohttp.ClientSession() as session:
        for url in state.search_results:
            page_text = await fetch_webpage_text_async(session, url)
            if page_text and await is_page_useful_async(state.user_query, page_text) == "Yes":
                state.relevant_texts.append(
                    await extract_relevant_context_async(state.user_query, state.user_query, page_text))
    return state


workflow = StateGraph(State)
workflow.add_node("search", search_step)
workflow.add_node("process_results", process_results_step)
workflow.set_entry_point("search")
workflow.add_edge("search", "process_results")
workflow.finalize()


async def async_main():
    user_query = input("Enter research query: ").strip()
    app = workflow.compile()
    result = await app.invoke(State(user_query))
    print("\n=== Final Report ===\n")
    print("\n".join(result.relevant_texts))


def load_keys(keys=None):
    # Initialize environment and API
    load_dotenv()
    keys['openai_api_key'] = os.getenv("OPENAI_API_KEY")
    if not keys['openai_api_key']:
        raise ValueError("OpenAI API key not found")
    return keys


if __name__ == "__main__":
    import asyncio
    asyncio.run(async_main())
