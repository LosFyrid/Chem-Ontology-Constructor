from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
import json
from .base_finder import BaseFinder, FinderState

class MetaScienceFinder(BaseFinder):
    """Agent for analyzing deep structural differences between domains"""
    
    def _get_analysis_prompt(self) -> ChatPromptTemplate:
        """Get prompt for meta-science gap analysis"""
        return ChatPromptTemplate.from_messages([
            ("system", "Analyze fundamental structural differences between domains."),
            ("user", """
            Source Domain:
            {source_domain}
            
            Target Domain:
            {target_domain}
            
            Analyze:
            1. Core assumptions and biases
            2. Knowledge organization patterns
            3. Reasoning frameworks
            4. Validation approaches
            
            Format your response as a JSON list of gaps, each with:
            - difference_type: type of structural difference
            - source_pattern: pattern in source domain
            - target_pattern: pattern in target domain
            - implications: scientific implications
            - potential_synthesis: possible integration approach
            - confidence: confidence score (0-1)
            """)
        ])
    
    def _get_ideation_prompt(self) -> ChatPromptTemplate:
        """Get prompt for research idea generation"""
        return ChatPromptTemplate.from_messages([
            ("system", "Generate research ideas based on meta-science analysis."),
            ("user", """
            Meta-Science Gaps:
            {gaps}
            
            For each gap, propose a research idea that:
            1. Bridges the structural difference
            2. Creates new theoretical insights
            3. Has practical applications
            4. Advances both domains
            
            Format your response as a JSON list of ideas, each with:
            - gap_addressed: which meta-science gap this addresses
            - research_question: clear problem statement
            - theoretical_framework: proposed synthesis
            - practical_applications: concrete use cases
            - expected_impact: anticipated contributions
            """)
        ])
    
    def _parse_gaps(self, content: str) -> List[Dict]:
        """Parse meta-science gaps from LLM response"""
        try:
            gaps = json.loads(content)
            required_fields = ["difference_type", "source_pattern", "target_pattern",
                             "implications", "potential_synthesis", "confidence"]
            for gap in gaps:
                if not all(field in gap for field in required_fields):
                    raise ValueError(f"Missing required fields in gap: {gap}")
            return gaps
        except Exception as e:
            print(f"Error parsing meta-science gaps: {e}")
            return []
    
    def _parse_ideas(self, content: str) -> List[Dict]:
        """Parse research ideas from LLM response"""
        try:
            ideas = json.loads(content)
            required_fields = ["gap_addressed", "research_question", "theoretical_framework",
                             "practical_applications", "expected_impact"]
            for idea in ideas:
                if not all(field in idea for field in required_fields):
                    raise ValueError(f"Missing required fields in idea: {idea}")
            return ideas
        except Exception as e:
            print(f"Error parsing research ideas: {e}")
            return [] 