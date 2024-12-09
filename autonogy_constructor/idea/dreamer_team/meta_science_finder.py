from .base_finder import BaseFinder, FinderState
from langchain.prompts import ChatPromptTemplate

class MetaScienceFinder(BaseFinder):
    """Agent for analyzing deep structural differences between domains"""
    
    def analyze(self, state: FinderState) -> FinderState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze fundamental structural differences between domains."),
            ("user", """
            Domain Ontologies:
            {ontology1}
            {ontology2}
            
            Analyze:
            1. Core assumptions and biases
            2. Knowledge organization patterns
            3. Reasoning frameworks
            4. Validation approaches
            
            For each difference:
            - Identify structural pattern
            - Analyze historical development
            - Evaluate scientific implications
            - Suggest cross-domain insights
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(
            ontology1=state["source_domain"],
            ontology2=state["target_domain"]
        ))
        state["gaps"] = parse_meta_gaps(response.content)
        return state 