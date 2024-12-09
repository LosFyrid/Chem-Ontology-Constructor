from .base_finder import BaseFinder, FinderState

class EvidenceFinder(BaseFinder):
    """Agent for analyzing evidence gaps in property values"""
    
    def analyze(self, state: FinderState) -> FinderState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze evidence gaps in property values from different sources."),
            ("user", """
            Class Information:
            {info}
            
            Focus on:
            1. Properties with multiple source values
            2. Conflicting experimental evidence
            3. Measurement inconsistencies
            4. Methodology differences
            
            For each gap, identify:
            - Property name
            - Different values and their sources
            - Potential reasons for discrepancy
            - Impact on chemical understanding
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(info=state["collected_info"]))
        state["gaps"] = parse_evidence_gaps(response.content)
        return state 