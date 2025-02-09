from duckduckgo_search import DDGS
from typing import Dict, Optional, List
import logging
import time
from random import uniform

logger = logging.getLogger(__name__)


class WebSearchAgent:
    def __init__(self, max_results: int = 10, min_request_interval: float = 5.0):
        self.ddgs = DDGS()
        self.max_results = max_results
        self._last_request_time = 0
        self.min_request_interval = min_request_interval

    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests with randomization"""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < self.min_request_interval:
            delay = self.min_request_interval - time_since_last_request + uniform(0, 2)
            time.sleep(delay)

        self._last_request_time = time.time()

    def get_response(
            self,
            query: str,
            context: Optional[str] = None,
            max_results: Optional[int] = None,
            search_type: str = 'text'
    ) -> Dict[str, str]:
        try:
            max_results = max_results if max_results is not None else self.max_results
            search_query = f"{context} {query}" if context else query

            # Wait before making request
            self._wait_for_rate_limit()

            try:
                # Perform the search with explicit parameters
                results = list(self.ddgs.text(
                    keywords=search_query,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit='y',
                    max_results=max_results,
                    backend='lite'  # Try lite backend first
                ))

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
                error_msg = str(e)
                logger.error(f"Search error: {error_msg}")
                return {
                    'status': 'error',
                    'response': f"Failed to perform web search: {error_msg}",
                    'source': 'web_search',
                    'search_query': search_query,
                    'error': error_msg
                }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"General error: {error_msg}")
            return {
                'status': 'error',
                'response': f"Failed to process request: {error_msg}",
                'source': 'web_search',
                'error': error_msg
            }

    @staticmethod
    def summarize_results(results: List[Dict[str, str]]) -> str:
        """
        Generate a summary from search results

        Args:
            results: List of search result dictionaries

        Returns:
            Summary string
        """
        if not results:
            return "No results found."

        # Limit the number of results to summarize to prevent overly long summaries
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


if __name__ == "__main__":
    # Test the agent directly
    agent = WebSearchAgent()

    # Test web search (no context provided)
    result = agent.get_response(
        query="What is the current temperature in Tel Aviv?"
    )
    print("Direct query result:")
    print(result['response'])
    print("\n" + "-" * 80 + "\n")

    # Test search with context
    result = agent.get_response(
        query="temperature and weather conditions",
        context="Tel Aviv today"
    )
    print("Query with context result:")
    print(result['response'])
