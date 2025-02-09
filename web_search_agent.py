import aiohttp
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """
    Simplified web search agent using the Bing Web Search API.
    """

    def __init__(self, subscription_key: str, max_results: int = 10):
        """
        Initialize the web search agent.

        Args:
            subscription_key: Your Bing Web Search API subscription key.
            max_results: Maximum number of search results to retrieve.
        """
        self.subscription_key = subscription_key
        self.max_results = max_results
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"

    async def get_response(
            self,
            query: str,
            context: Optional[str] = None,
            max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get web search response for a query asynchronously using Bing Web Search API.

        Args:
            query: User's question.
            context: Optional context to enhance the query.
            max_results: Maximum number of search results to retrieve.

        Returns:
            Dictionary containing response and metadata.
        """
        try:
            max_results = max_results if max_results is not None else self.max_results
            search_query = f"{context} {query}" if context else query

            params = {
                "q": search_query,
                "count": max_results,
                "offset": 0,
                "mkt": "en-US",
                "safeSearch": "Moderate"
            }
            headers = {
                "Ocp-Apim-Subscription-Key": self.subscription_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.endpoint, headers=headers, params=params) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Bing Search API returned status {resp.status}: {error_text}")
                        return {
                            'status': 'error',
                            'response': f"Bing Search API error {resp.status}: {error_text}",
                            'source': 'web_search',
                            'search_query': search_query
                        }
                    data = await resp.json()

                    # Extract the list of results from the "webPages" object.
                    results = data.get("webPages", {}).get("value", [])
                    if not results:
                        return {
                            'status': 'no_results',
                            'response': "No search results found.",
                            'source': 'web_search',
                            'search_query': search_query
                        }

                    summary = self.summarize_results(results)
                    return {
                        'status': 'success',
                        'response': summary,
                        'source': 'web_search',
                        'search_query': search_query,
                        'result_count': len(results)
                    }

        except Exception as e:
            logger.error(f"General error: {e}")
            return {
                'status': 'error',
                'response': f"Failed to process request: {e}",
                'source': 'web_search',
                'error': str(e)
            }

    @staticmethod
    def summarize_results(results: List[Dict[str, Any]]) -> str:
        """
        Generate a summary from search results by mapping Bing result fields
        to a standard format (title, snippet, link).

        Args:
            results: A list of dictionaries representing Bing search results.

        Returns:
            A formatted summary string of the top results.
        """
        if not results:
            return "No results found."

        MAX_RESULTS_TO_SUMMARIZE = 5
        summary_parts = []
        for result in results[:MAX_RESULTS_TO_SUMMARIZE]:
            # Bing returns "name" as the title, "snippet" as the summary, and "url" as the link.
            title = result.get("name", "").strip()
            snippet = result.get("snippet", "").strip()
            link = result.get("url", "").strip()

            if title and snippet:
                summary_parts.append(f"• {title}\n  {snippet}\n  Source: {link}")
            elif title:
                summary_parts.append(f"• {title}\n  Source: {link}")
            elif snippet:
                summary_parts.append(f"• {snippet}\n  Source: {link}")

        return "\n\n".join(summary_parts) if summary_parts else "No relevant information found."
