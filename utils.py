import openai
import asyncio
import aiohttp
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph
from langsmith import traceable

OPENAI_API_KEY = "your_openai_api_key"

@traceable
async def call_openai_async(messages, model="gpt-4o-mini"):
    """
    Sending request to OpenAI Chat API and returning the answer.
    """
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            api_key=OPENAI_API_KEY
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

@traceable
async def perform_search_async(query):
    """
    Searching with DuckDuckGo and returning list of URLs.
    """
    try:
        results = list(DDGS().text(query, max_results=10))
        return [result['href'] for result in results]
    except Exception as e:
        print(f"Error during search with DuckDuckGo: {e}")
        return []

@traceable
async def fetch_webpage_text_async(session, url):
    """
    Asynchronously loads webpages
    """
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                print(f"Error loading {url}: {resp.status}")
                return ""
    except Exception as e:
        print(f"Error reading webpage from {url}: {e}")
        return ""

@traceable
async def is_page_useful_async(user_query, page_text):
    """
    Checks whether the webpage is useful with OpenAI.
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
    response = await call_openai_async(messages)
    return response.strip() if response in ["Yes", "No"] else "No"

@traceable
async def extract_relevant_context_async(user_query, search_query, page_text):
    """
    Extracts relevant information from the webpage text
    """
    prompt = (
        "You are an expert in extracting relevant information. Given the user's query, the search query that led to "
        "this page, and the webpage content, extract all relevant information. Return only the relevant text."
    )
    messages = [
        {"role": "system", "content": "You are a precise information extractor."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\nWebpage Content: {page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_openai_async(messages)
    return response.strip() if response else ""
