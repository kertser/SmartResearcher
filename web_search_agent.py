import logging
from typing import Dict, Optional, List, Any
from googlesearch import search
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """
    Web search agent using the googlesearch-python library.
    """

    def __init__(self, max_results: int = 10):
        """
        Initialize the web search agent.

        Args:
            max_results: Maximum number of search results to retrieve.
        """
        self.max_results = max_results

    async def get_response(
            self,
            query: str,
            context: Optional[str] = None,
            max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get web search response for a query using Google Search.

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

            # Since googlesearch is synchronous, we run it in a ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                results = await loop.run_in_executor(
                    executor,
                    lambda: list(search(
                        search_query,
                        num_results=max_results,
                        lang="en"
                    ))
                )

            if not results:
                return {
                    'status': 'no_results',
                    'response': "No search results found.",
                    'source': 'web_search',
                    'search_query': search_query
                }

            # Format results into the expected structure
            formatted_results = []
            for url in results:
                # Note: The googlesearch library only provides URLs
                # We create a simplified result structure
                formatted_results.append({
                    "url": url,
                    "name": url,  # Using URL as name since title is not available
                    "snippet": ""  # Snippet is not available in basic googlesearch
                })

            summary = self.summarize_results(formatted_results)
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
        Generate a summary from search results.

        Args:
            results: A list of dictionaries representing Google search results.

        Returns:
            A formatted summary string of the top results.
        """
        if not results:
            return "No results found."

        MAX_RESULTS_TO_SUMMARIZE = 5
        summary_parts = []
        for result in results[:MAX_RESULTS_TO_SUMMARIZE]:
            url = result.get("url", "").strip()
            summary_parts.append(f"• Source: {url}")

        return "\n\n".join(summary_parts) if summary_parts else "No relevant information found."


def main():
    """
    Main function to run the web search agent.
    """
    logging.basicConfig(level=logging.INFO)
    agent = WebSearchAgent()

    # Example usage
    query = "Pirates vs Climate Change"
    response = asyncio.run(agent.get_response(query))
    print(response)


if __name__ == "__main__":
    main()
