from .base_finder import BaseFinder, FinderState
from langchain.prompts import ChatPromptTemplate

class MethodologyFinder(BaseFinder):
    """Agent for analyzing methodology transfer opportunities"""
    
    def analyze(self, state: FinderState) -> FinderState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze methodology transfer opportunities between domains."),
            ("user", """
            Domain Ontologies:
            {ontology1}
            {ontology2}
            
            Focus on:
            1. Successful methodologies in source domain
            2. Similar challenges in target domain
            3. Adaptation requirements
            4. Potential benefits and risks
            
            For each opportunity:
            - Identify methodology
            - Analyze transfer feasibility
            - Suggest adaptation strategy
            - Predict potential impact
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(
            ontology1=state["source_domain"],
            ontology2=state["target_domain"]
        ))
        state["gaps"] = parse_methodology_gaps(response.content)
        return state 