from duckduckgo_search import DDGS
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """
    Simplified web search agent using DDGS for DuckDuckGo searches.
    """

    def __init__(self, max_results: int = 10):
        """
        Initialize the web search agent
        Args:
            max_results: Maximum number of search results to retrieve
        """
        self.max_results = max_results

    async def get_response(
            self,
            query: str,
            context: Optional[str] = None,
            max_results: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Get web search response for a query asynchronously.

        Args:
            query: User's question
            context: Optional context to enhance the query
            max_results: Maximum number of search results to retrieve

        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Use instance max_results if not specified
            max_results = max_results if max_results is not None else self.max_results

            # Combine query and context if provided
            search_query = f"{context} {query}" if context else query

            async with DDGS() as ddgs:
                try:
                    # Get search results from DuckDuckGo
                    results = [r async for r in ddgs.text(
                        search_query,
                        region='wt-wt',
                        safesearch='moderate',
                        timelimit='y',
                        max_results=max_results
                    )]

                    if not results:
                        return {
                            'status': 'no_results',
                            'response': "No search results found.",
                            'source': 'web_search',
                            'search_query': search_query
                        }

                    # Generate a summary from the search results
                    summary = self.summarize_results(results)

                    return {
                        'status': 'success',
                        'response': summary,
                        'source': 'web_search',
                        'search_query': search_query,
                        'result_count': len(results)
                    }

                except Exception as e:
                    logger.error(f"Search error: {str(e)}")
                    return {
                        'status': 'error',
                        'response': f"Failed to perform web search: {str(e)}",
                        'source': 'web_search',
                        'search_query': search_query,
                        'error': str(e)
                    }

        except Exception as e:
            logger.error(f"General error: {str(e)}")
            return {
                'status': 'error',
                'response': f"Failed to process request: {str(e)}",
                'source': 'web_search',
                'error': str(e)
            }

    @staticmethod
    def summarize_results(results: List[Dict[str, str]]) -> str:
        """
        Generate a summary from search results
        """
        if not results:
            return "No results found."

        # Limit the number of results to summarize
        MAX_RESULTS_TO_SUMMARIZE = 5
        summary_parts = []

        for result in results[:MAX_RESULTS_TO_SUMMARIZE]:
            title = result.get("title", "").strip()
            body = result.get("body", "").strip()
            link = result.get("link", "").strip()

            if title and body:
                summary_parts.append(f"• {title}\n  {body}\n  Source: {link}")
            elif title:
                summary_parts.append(f"• {title}\n  Source: {link}")
            elif body:
                summary_parts.append(f"• {body}\n  Source: {link}")

        return "\n\n".join(summary_parts) if summary_parts else "No relevant information found."
