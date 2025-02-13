from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ResearchPhase(Enum):
    THESIS = "thesis"
    ANTITHESIS = "antithesis"
    SYNTHESIS = "synthesis"

@dataclass
class Source:
    url: str
    content: str
    relevance_score: float
    citations: List[str]

@dataclass
class ResearchPoint:
    statement: str
    evidence: List[Source]
    confidence_score: float
    iteration: int
    status: str  # "pending", "approved", "disapproved"

@dataclass
class ResearchResult:
    thesis: ResearchPoint
    antithesis: ResearchPoint
    synthesis: Optional[str] = None
    references: List[str] = None

# Define the state structure as a TypedDict
class ResearchState(TypedDict):
    user_query: str
    search_results: List[str]
    relevant_texts: List[str]
    api_key: Optional[str]
