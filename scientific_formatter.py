class ScientificFormatter:
    def format_research_output(self,
                               topic: str,
                               thesis: ResearchPoint,
                               antithesis: ResearchPoint,
                               synthesis: str) -> str:
        """
        Formats the research output in scientific paper format
        """
        template = """
        # {title}

        ## Abstract
        {abstract}

        ## Introduction
        {introduction}

        ## Thesis
        {thesis_section}

        ## Antithesis
        {antithesis_section}

        ## Synthesis
        {synthesis_section}

        ## Conclusion
        {conclusion}

        ## References
        {references}
        """

        # Implementation to fill template sections
        return formatted_output