from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
import json
from .base_finder import BaseFinder, FinderState

class MetaScienceFinder(BaseFinder):
    """Agent for analyzing deep structural differences between domains"""
    
    def _get_analysis_prompt(self) -> ChatPromptTemplate:
        """Get prompt for meta-science gap analysis"""
        return ChatPromptTemplate.from_messages([
            ("system", "你是一位专家，请对比分析两个本体之间的深层结构差异以及可能存在的偏见。"),
            ("user", """
源本体：
{source_domain}

目标本体：
{target_domain}

请针对以下方面进行分析：
1. 本体整体知识表示和内在逻辑的差异
2. 可能存在的结构性偏见和假设差异
3. 知识组织模式的不同
4. 两个本体在信息表达上的内在逻辑差异

请以JSON格式返回每个差异项，格式应包括：
- difference_type: 差异类型描述
- source_pattern: 源本体中的相关模式
- target_pattern: 目标本体中的相关模式
- implications: 差异可能带来的科学影响
- potential_synthesis: 针对该差异建议的整合思路
- confidence: 置信度（0-1）
""")
        ])
    
    def _get_ideation_prompt(self) -> ChatPromptTemplate:
        """Get prompt for research idea generation"""
        return ChatPromptTemplate.from_messages([
            ("system", "你是一位专家，请基于上述元领域差异分析提出解决该差异的研究想法。"),
            ("user", """
Meta-Science Gaps:
{gaps}

请提出研究想法，要求：
1. 针对识别的深层结构差异给出新的整合理论
2. 提出可能化解或利用这些差异的实验思路
3. 强调理论新颖性和跨领域整合可能性

请以JSON格式返回每个研究想法，格式包括：
- gap_addressed: 针对的元领域差异
- research_question: 清晰的研究问题描述
- theoretical_framework: 整合理论构想
- practical_applications: 可能的应用场景
- expected_impact: 预期影响
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