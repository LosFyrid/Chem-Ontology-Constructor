from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
import json
from .base_finder import BaseFinder, FinderState

class MethodologyFinder(BaseFinder):
    """Agent for analyzing methodology transfer opportunities between domains"""
    
    def _get_analysis_prompt(self) -> ChatPromptTemplate:
        """Get prompt for methodology gap analysis"""
        return ChatPromptTemplate.from_messages([
            ("system", "你是一位专家，请分析两个本体之间方法论的差异及潜在的迁移可能性。"),
            ("user", """
源本体：
{source_domain}

目标本体：
{target_domain}

请重点关注以下方面：
1. 源本体中成功应用的方法论及其优势
2. 目标本体面临的具体挑战以及方法论上的不足
3. 针对目标本体的改进和方法论迁移需求
4. 现有方法论在新领域中的适应性

请以JSON格式返回分析结果，每个项目应包含：
- methodology_name: 方法论名称
- source_usage: 在源本体中的应用情况
- target_challenge: 目标本体面临的挑战
- adaptation_needs: 需要做出的调整
- feasibility: 迁移可行性（0-1）
- potential_impact: 预期的改善效果
""")
        ])
    
    def _get_ideation_prompt(self) -> ChatPromptTemplate:
        """Get prompt for research idea generation"""
        return ChatPromptTemplate.from_messages([
            ("system", "你是一位专家，请基于上述方法论差异分析提出改进思路及研究想法。"),
            ("user", """
Methodology Gaps:
{gaps}

请提出研究想法，要求：
1. 针对现有方法论的不足提出改进方案
2. 评估方法论迁移的可行性和预期效果
3. 给出明确的验证和评估方案

请以JSON格式返回每个研究想法，格式包括：
- gap_addressed: 针对的具体方法论差异
- research_question: 清晰的研究问题描述
- adaptation_strategy: 方法论的改进策略
- validation_approach: 如何验证改进方案
- expected_benefits: 预期改进效果
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