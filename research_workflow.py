from typing import List, Optional, TypedDict

# Define the state structure as a TypedDict
class ResearchState(TypedDict):
    user_query: str
    search_results: List[str]
    relevant_texts: List[str]
    api_key: Optional[str]
