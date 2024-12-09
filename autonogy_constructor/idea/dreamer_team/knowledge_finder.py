from .base_finder import BaseFinder, FinderState
from langchain.prompts import ChatPromptTemplate

class KnowledgeFinder(BaseFinder):
    """Agent for analyzing knowledge inconsistencies"""
    
    def analyze(self, state: FinderState) -> FinderState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze knowledge inconsistencies in chemical concepts."),
            ("user", """
            Class Information:
            {info}
            
            Focus on:
            1. Inconsistent class definitions
            2. Conflicting property restrictions
            3. Contradictory relationships
            4. Logical conflicts in chemical knowledge
            
            For each inconsistency:
            - Identify conflicting elements
            - Analyze chemical basis of conflict
            - Propose potential resolutions
            - Suggest experimental validation
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(info=state["collected_info"]))
        state["gaps"] = parse_knowledge_gaps(response.content)
        return state 