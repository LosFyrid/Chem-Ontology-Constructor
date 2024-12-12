from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
import json
from .base_finder import BaseFinder, FinderState

class MethodologyFinder(BaseFinder):
    """Agent for analyzing methodology transfer opportunities"""
    
    def _get_analysis_prompt(self) -> ChatPromptTemplate:
        """Get prompt for methodology gap analysis"""
        return ChatPromptTemplate.from_messages([
            ("system", "Analyze methodology transfer opportunities between domains."),
            ("user", """
            Source Domain:
            {source_domain}
            
            Target Domain:
            {target_domain}
            
            Focus on:
            1. Successful methodologies in source domain
            2. Similar challenges in target domain
            3. Adaptation requirements
            4. Transfer feasibility
            
            Format your response as a JSON list of gaps, each with:
            - methodology_name: name of the methodology
            - source_usage: how it's used in source domain
            - target_challenge: related challenge in target domain
            - adaptation_needs: what needs to be modified
            - feasibility: feasibility score (0-1)
            - potential_impact: expected impact of transfer
            """)
        ])
    
    def _get_ideation_prompt(self) -> ChatPromptTemplate:
        """Get prompt for research idea generation"""
        return ChatPromptTemplate.from_messages([
            ("system", "Generate research ideas for methodology transfer."),
            ("user", """
            Methodology Gaps:
            {gaps}
            
            For each gap, propose a research idea that:
            1. Adapts the methodology for the new domain
            2. Addresses key challenges
            3. Maintains scientific rigor
            4. Has clear success criteria
            
            Format your response as a JSON list of ideas, each with:
            - gap_addressed: which methodology gap this addresses
            - research_question: clear problem statement
            - adaptation_strategy: how to modify the methodology
            - validation_approach: how to verify success
            - expected_benefits: anticipated improvements
            """)
        ])
    
    def _parse_gaps(self, content: str) -> List[Dict]:
        """Parse methodology gaps from LLM response"""
        try:
            gaps = json.loads(content)
            required_fields = ["methodology_name", "source_usage", "target_challenge",
                             "adaptation_needs", "feasibility", "potential_impact"]
            for gap in gaps:
                if not all(field in gap for field in required_fields):
                    raise ValueError(f"Missing required fields in gap: {gap}")
            return gaps
        except Exception as e:
            print(f"Error parsing methodology gaps: {e}")
            return []
    
    def _parse_ideas(self, content: str) -> List[Dict]:
        """Parse research ideas from LLM response"""
        try:
            ideas = json.loads(content)
            required_fields = ["gap_addressed", "research_question", "adaptation_strategy",
                             "validation_approach", "expected_benefits"]
            for idea in ideas:
                if not all(field in idea for field in required_fields):
                    raise ValueError(f"Missing required fields in idea: {idea}")
            return ideas
        except Exception as e:
            print(f"Error parsing research ideas: {e}")
            return [] 