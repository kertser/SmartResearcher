class IterativeResearcher:
    def __init__(self, max_iterations: int = 3, confidence_threshold: float = 0.8):
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

    async def research_point(self, statement: str, phase: ResearchPhase) -> ResearchPoint:
        """
        Iteratively researches a thesis or antithesis until approved or max iterations reached
        """
        research_point = ResearchPoint(
            statement=statement,
            evidence=[],
            confidence_score=0.0,
            iteration=0,
            status="pending"
        )

        while (research_point.iteration < self.max_iterations and
               research_point.status == "pending"):
            # Gather new evidence
            new_sources = await self._gather_evidence(statement)

            # Evaluate evidence
            confidence = await self._evaluate_evidence(new_sources)

            # Update research point
            research_point.evidence.extend(new_sources)
            research_point.confidence_score = confidence
            research_point.iteration += 1

            # Check if approved/disapproved
            if confidence >= self.confidence_threshold:
                research_point.status = "approved"
            elif confidence < 0.2:  # Clear contradiction
                research_point.status = "disapproved"

        return research_point