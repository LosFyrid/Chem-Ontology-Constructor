"""
查询智能优化器 - QueryRefiner

重构为支持细粒度工具调用级别的决策和hints生成。
为每个工具-参数组合提供具体的行动指导和替代方案。
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from .ontology_tools import OntologyTools
from .schemas import ValidationReport, ValidationClassification, ToolCallClassification, RefinerDecision, ToolCallHint
# Moved to function-level imports to avoid circular dependency

logger = logging.getLogger(__name__)


class QueryRefiner:
    """查询智能优化器 - 细粒度版本
    
    职责：
    1. 分析每个工具调用的分类结果
    2. 为每个工具-参数组合生成具体的hints
    3. 使用get_class_richness_info评估替代类
    4. 提供整体的行动决策
    """
    
    def __init__(self, ontology_tools: OntologyTools):
        """初始化QueryRefiner
        
        Args:
            ontology_tools: 本体工具实例，用于评估类的丰富度
        """
        self.ontology_tools = ontology_tools
        
    def propose_next_action(self, state, validation_report: ValidationReport) -> RefinerDecision:
        """基于细粒度验证结果决定下一步行动
        
        Args:
            state: 当前查询状态
            validation_report: 包含每个工具调用分类的验证报告
            
        Returns:
            RefinerDecision: 包含整体行动和每个工具调用的具体hints
        """
        try:
            logger.info(f"[QueryRefiner] 开始分析 {len(validation_report.tool_classifications)} 个工具调用")
            
            retry_count = state.get("retry_count", 0)
            
            # 为每个工具调用生成hints
            tool_call_hints = []
            for tool_classification in validation_report.tool_classifications:
                hint = self._generate_tool_call_hint(state, tool_classification)
                if hint:
                    tool_call_hints.append(hint)
            
            # 基于个别hints决定整体行动
            overall_action = self._determine_overall_action(validation_report, tool_call_hints, retry_count)
            
            # 生成决策推理
            reason = self._generate_decision_reason(validation_report, tool_call_hints, overall_action, retry_count)
            
            decision = RefinerDecision(
                overall_action=overall_action,
                reason=reason,
                tool_call_hints=tool_call_hints
            )
            
            logger.info(f"[QueryRefiner] 决策完成: {overall_action}, 生成了 {len(tool_call_hints)} 个工具hints")
            
            return decision
            
        except Exception as e:
            logger.error(f"[QueryRefiner] 决策过程出错: {e}")
            return RefinerDecision(
                overall_action="terminate",
                reason=f"Decision process failed: {e}",
                tool_call_hints=[]
            )
    
    def _generate_tool_call_hint(self, state, tool_classification: ToolCallClassification) -> Optional[ToolCallHint]:
        """为单个工具调用生成hint
        
        Args:
            state: 当前状态
            tool_classification: 工具调用分类结果
            
        Returns:
            ToolCallHint或None
        """
        try:
            classification = tool_classification.classification
            tool = tool_classification.tool
            class_name = tool_classification.class_name
            
            # 获取已尝试过的类列表
            tried_classes = self._get_tried_classes_for_tool(state, tool)
            tried_classes_str = ", ".join(tried_classes) if tried_classes else "none"
            
            # 基于分类决定行动
            if classification == ValidationClassification.SUFFICIENT:
                # 结果充分，无需额外行动
                return None
            
            elif classification == ValidationClassification.INSUFFICIENT_PROPERTIES:
                # 缺乏属性信息，建议使用更详细的工具
                return ToolCallHint(
                    tool=tool,
                    class_name=class_name,
                    action="replace_tool",
                    hint=f"Try more detailed tools like parse_class_definition or get_class_properties for class '{class_name}'",
                    alternative_tools=["parse_class_definition", "get_class_properties", "get_related_classes"]
                )
            
            elif classification == ValidationClassification.INSUFFICIENT:
                # 一般信息不足，尝试替代类或更多工具
                return ToolCallHint(
                    tool=tool,
                    class_name=class_name,
                    action="replace_class",
                    hint=f"Try different classes from available options. Previously tried classes for {tool}: {tried_classes_str}. Find semantically similar but different classes.",
                    alternative_tools=["parse_class_definition", "get_class_properties"]
                )
            
            elif classification == ValidationClassification.NO_RESULTS:
                # 无结果，优先尝试替代类
                return ToolCallHint(
                    tool=tool,
                    class_name=class_name,
                    action="replace_class",
                    hint=f"No results found for '{class_name}'. Try different classes from available options. Previously tried: {tried_classes_str}. Look for related or alternative class names.",
                    alternative_tools=["parse_class_definition"] if not tried_classes else []
                )
            
            elif classification == ValidationClassification.ERROR:
                # 执行错误，跳过或尝试简单工具
                return ToolCallHint(
                    tool=tool,
                    class_name=class_name,
                    action="skip",
                    hint=f"Execution error occurred with {tool}({class_name}). Consider skipping or using simpler tools.",
                    alternative_tools=["get_class_info"]  # 最简单的工具
                )
            
            return None
            
        except Exception as e:
            logger.error(f"[QueryRefiner] 生成工具hint失败: {e}")
            return None
    
    def _get_tried_classes_for_tool(self, state, tool_name: str) -> List[str]:
        """获取特定工具已经尝试过的类列表
        
        Args:
            state: 当前状态
            tool_name: 工具名称
            
        Returns:
            已尝试过的类名列表
        """
        tried_calls = state.get("tried_tool_calls", {})
        tried_classes = []
        
        for call_info in tried_calls.values():
            if call_info.get("tool") == tool_name:
                params = call_info.get("params", {})
                class_name = params.get("class_names") or params.get("class_name")
                if class_name and class_name not in tried_classes:
                    tried_classes.append(class_name)
        
        return tried_classes
    
    def _get_alternative_classes(self, state, original_class: str, aggressive: bool = False) -> List[str]:
        """获取替代类，按丰富度排序
        
        Args:
            state: 当前状态
            original_class: 原始类名
            aggressive: 是否使用更激进的搜索策略
            
        Returns:
            按丰富度排序的替代类列表
        """
        try:
            # 从refined_classes或available_classes中获取候选
            refined_classes = state.get("refined_classes", [])
            available_classes = state.get("available_classes", [])
            
            if aggressive and len(refined_classes) < 20:
                # 激进模式：使用更大的候选集
                candidate_pool = available_classes[:100]  # 限制范围避免过慢
            else:
                candidate_pool = refined_classes
            
            # 移除原始类
            candidate_pool = [cls for cls in candidate_pool if cls != original_class]
            
            if not candidate_pool:
                return []
            
            # 使用丰富度评估前10个候选
            top_candidates = candidate_pool
            ranked_candidates, _ = self._rank_classes_by_richness_simple(state, top_candidates)
            
            # 返回前3-5个最佳选择
            return ranked_candidates[:5] if aggressive else ranked_candidates[:3]
            
        except Exception as e:
            logger.error(f"[QueryRefiner] 获取替代类失败: {e}")
            return []
    
    def _rank_classes_by_richness_simple(self, state, candidates: List[str]) -> Tuple[List[str], Dict]:
        """简化版的丰富度排序"""
        # Local import to avoid circular dependency
        from .workflow_utils import generate_tool_call_signature, has_tool_call_been_tried
        
        if not candidates:
            return [], {}
        
        try:
            scored_candidates = []
            
            for class_name in candidates:
                # 检查缓存
                tool_params = {"class_name": class_name}
                if has_tool_call_been_tried(state, "get_class_richness_info", tool_params):
                    signature = generate_tool_call_signature("get_class_richness_info", tool_params)
                    tried_calls = state.get("tried_tool_calls", {})
                    richness_info = tried_calls[signature]["result"]
                else:
                    richness_info = self.ontology_tools.get_class_richness_info(class_name)
                
                score = richness_info.get("richness_score", 0.0)
                scored_candidates.append((class_name, score))
            
            # 按分数排序
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            ranked_classes = [item[0] for item in scored_candidates]
            
            stats = {
                "evaluated_count": len(candidates),
                "top_score": scored_candidates[0][1] if scored_candidates else 0.0
            }
            
            return ranked_classes, stats
            
        except Exception as e:
            logger.error(f"[QueryRefiner] 简化丰富度排序失败: {e}")
            return candidates, {"error": str(e)}
    
    def _determine_overall_action(self, validation_report: ValidationReport, 
                                tool_call_hints: List[ToolCallHint], retry_count: int) -> str:
        """基于工具调用hints确定整体行动"""
        
        overall_classification = validation_report.overall_classification
        
        # 如果所有工具调用都成功，继续
        if overall_classification == ValidationClassification.SUFFICIENT:
            return "continue"
        
        # 如果重试次数过多，终止
        if retry_count >= 3:
            return "terminate"
        
        # 检查是否有有效的hints
        actionable_hints = [h for h in tool_call_hints if h.action != "skip"]
        
        if not actionable_hints:
            # 没有可行的hints，可能需要扩展搜索
            return "expand" if retry_count < 2 else "terminate"
        
        # 有hints可以尝试
        return "retry"
    
    def _generate_decision_reason(self, validation_report: ValidationReport, 
                                tool_call_hints: List[ToolCallHint],
                                overall_action: str, retry_count: int) -> str:
        """生成决策推理说明"""
        
        total_tools = len(validation_report.tool_classifications)
        actionable_hints = len([h for h in tool_call_hints if h.action != "skip"])
        
        if overall_action == "continue":
            return f"All {total_tools} tool calls were sufficient, continuing workflow"
        
        elif overall_action == "retry":
            return f"Generated {actionable_hints} actionable hints for {total_tools} tool calls (retry #{retry_count + 1})"
        
        elif overall_action == "expand":
            return f"Insufficient hints generated, expanding search space (retry #{retry_count + 1})"
        
        elif overall_action == "terminate":
            if retry_count >= 3:
                return f"Maximum retry limit reached ({retry_count}), terminating"
            else:
                return f"No viable improvement options found, terminating"
        
        return f"Action: {overall_action}, retry count: {retry_count}"