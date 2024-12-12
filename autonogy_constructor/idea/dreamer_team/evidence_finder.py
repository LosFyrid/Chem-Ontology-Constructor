from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
import json
from .base_finder import BaseFinder, FinderState

class EvidenceFinder(BaseFinder):
    """Agent for analyzing evidence gaps in property values"""
    
    def _get_analysis_prompt(self) -> ChatPromptTemplate:
        """Get prompt for evidence gap analysis"""
        return ChatPromptTemplate.from_messages([
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
            - Property name and type
            - Different values from different sources
            - Potential reasons for discrepancy
            - Impact on chemical understanding
            
            Format your response as a JSON list of gaps, each with:
            - property_name: name of the property
            - value_conflicts: list of conflicting values and their sources
            - reason: potential reason for discrepancy
            - impact: impact on chemical understanding
            - confidence: confidence score (0-1)
            """)
        ])
    
    def _get_ideation_prompt(self) -> ChatPromptTemplate:
        """Get prompt for research idea generation"""
        return ChatPromptTemplate.from_messages([
            ("system", "Generate research ideas to resolve evidence gaps."),
            ("user", """
            Evidence Gaps:
            {gaps}
            
            For each gap, propose a research idea that:
            1. Addresses the specific value conflict
            2. Designs experiments to resolve discrepancies
            3. Considers methodology standardization
            4. Has clear validation criteria
            
            Format your response as a JSON list of ideas, each with:
            - gap_addressed: which evidence gap this addresses
            - research_question: clear problem statement
            - methodology: proposed experimental approach
            - expected_outcome: what we expect to learn
            - validation_criteria: how to verify results
            """)
        ])
    
    def _parse_gaps(self, content: str) -> List[Dict]:
        """Parse evidence gaps from LLM response"""
        try:
            gaps = json.loads(content)
            # Validate required fields
            required_fields = ["property_name", "value_conflicts", "reason", "impact", "confidence"]
            for gap in gaps:
                if not all(field in gap for field in required_fields):
                    raise ValueError(f"Missing required fields in gap: {gap}")
            return gaps
        except Exception as e:
            print(f"Error parsing evidence gaps: {e}")
            return []
    
    def _parse_ideas(self, content: str) -> List[Dict]:
        """Parse research ideas from LLM response"""
        try:
            ideas = json.loads(content)
            # Validate required fields
            required_fields = ["gap_addressed", "research_question", "methodology", 
                             "expected_outcome", "validation_criteria"]
            for idea in ideas:
                if not all(field in idea for field in required_fields):
                    raise ValueError(f"Missing required fields in idea: {idea}")
            return ideas
        except Exception as e:
            print(f"Error parsing research ideas: {e}")
            return []
    
    def analyze(self, state: FinderState) -> FinderState:
        prompt = self._get_analysis_prompt()
        
        response = self.llm.invoke(prompt.format_messages(info=state["collected_info"]))
        state["gaps"] = self._parse_gaps(response.content)
        return state 