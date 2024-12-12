from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
import json
from .base_finder import BaseFinder, FinderState

class KnowledgeFinder(BaseFinder):
    """Agent for analyzing knowledge inconsistencies"""
    
    def _get_analysis_prompt(self) -> ChatPromptTemplate:
        """Get prompt for knowledge gap analysis"""
        return ChatPromptTemplate.from_messages([
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
            - Consider resolution approaches
            - Evaluate impact on understanding
            
            Format your response as a JSON list of gaps, each with:
            - conflict_type: type of inconsistency
            - elements: list of conflicting elements
            - chemical_basis: chemical reason for conflict
            - impact: impact on chemical understanding
            - confidence: confidence score (0-1)
            """)
        ])
    
    def _get_ideation_prompt(self) -> ChatPromptTemplate:
        """Get prompt for research idea generation"""
        return ChatPromptTemplate.from_messages([
            ("system", "Generate research ideas to resolve knowledge inconsistencies."),
            ("user", """
            Knowledge Gaps:
            {gaps}
            
            For each gap, propose a research idea that:
            1. Addresses the specific inconsistency
            2. Has clear theoretical foundation
            3. Can be experimentally validated
            4. Resolves chemical understanding
            
            Format your response as a JSON list of ideas, each with:
            - gap_addressed: which knowledge gap this addresses
            - research_question: clear problem statement
            - theoretical_basis: underlying chemical theory
            - experimental_approach: validation method
            - expected_resolution: how this resolves the inconsistency
            """)
        ])
    
    def _parse_gaps(self, content: str) -> List[Dict]:
        """Parse knowledge gaps from LLM response"""
        try:
            gaps = json.loads(content)
            required_fields = ["conflict_type", "elements", "chemical_basis", 
                             "impact", "confidence"]
            for gap in gaps:
                if not all(field in gap for field in required_fields):
                    raise ValueError(f"Missing required fields in gap: {gap}")
            return gaps
        except Exception as e:
            print(f"Error parsing knowledge gaps: {e}")
            return []
    
    def _parse_ideas(self, content: str) -> List[Dict]:
        """Parse research ideas from LLM response"""
        try:
            ideas = json.loads(content)
            required_fields = ["gap_addressed", "research_question", "theoretical_basis",
                             "experimental_approach", "expected_resolution"]
            for idea in ideas:
                if not all(field in idea for field in required_fields):
                    raise ValueError(f"Missing required fields in idea: {idea}")
            return ideas
        except Exception as e:
            print(f"Error parsing research ideas: {e}")
            return []