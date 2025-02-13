class DialecticGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_dialectic_points(self, research_topic: str) -> tuple[str, str]:
        """
        Uses OpenAI to generate thesis and antithesis from research topic

        Example:
        Topic: "Impact of social media on mental health"
        Thesis: "Social media usage significantly deteriorates mental health"
        Antithesis: "Social media provides valuable social support and connection"
        """
        prompt = f"""
        Given the research topic: "{research_topic}"
        Generate:
        1. A clear thesis statement that takes a specific position
        2. An antithesis statement that challenges the thesis

        Format: Return only the thesis and antithesis, separated by |
        """
        # Implementation using OpenAI API
        return thesis, antithesis