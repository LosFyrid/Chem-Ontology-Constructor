"""查询工作流工具函数

此模块包含用于查询工作流状态管理的工具函数，包括：
- 工具调用签名生成和记录
- LLM停滞检测
- 状态管理相关的辅助函数
"""

from typing import Dict, List, Literal, Optional, Any, Union, Set
from datetime import datetime
import hashlib
import json
import logging
from .stategraph import QueryState

logger = logging.getLogger(__name__)

def generate_tool_call_signature(tool_name: str, params: Dict) -> str:
    """生成标准化的工具调用签名
    
    Args:
        tool_name: 工具函数名
        params: 参数字典
        
    Returns:
        唯一的调用签名字符串
    """
    # 标准化参数字典：排序并序列化
    normalized_params = json.dumps(params, sort_keys=True, ensure_ascii=False)
    
    # 创建签名字符串
    signature_content = f"{tool_name}:{normalized_params}"
    
    # 生成哈希以确保签名长度可控
    signature_hash = hashlib.md5(signature_content.encode('utf-8')).hexdigest()
    
    return f"{tool_name}_{signature_hash[:8]}"

def record_tool_call(state: QueryState, tool_name: str, params: Dict, result: Any) -> Dict:
    """记录工具调用到tried_tool_calls中
    
    Args:
        state: 当前QueryState
        tool_name: 工具名
        params: 参数
        result: 调用结果
        
    Returns:
        更新后的state片段
    """
    signature = generate_tool_call_signature(tool_name, params)
    
    # 初始化tried_tool_calls如果不存在
    tried_calls = state.get("tried_tool_calls", {})
    
    # 记录这次调用
    tried_calls[signature] = {
        "tool": tool_name,
        "params": params,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "retry_count": state.get("retry_count", 0)
    }
    
    return {"tried_tool_calls": tried_calls}

def has_tool_call_been_tried(state: QueryState, tool_name: str, params: Dict) -> bool:
    """检查特定工具调用是否已被尝试过
    
    Args:
        state: 当前QueryState
        tool_name: 工具名
        params: 参数
        
    Returns:
        是否已尝试过
    """
    signature = generate_tool_call_signature(tool_name, params)
    tried_calls = state.get("tried_tool_calls", {})
    return signature in tried_calls

def detect_stagnation(state: QueryState) -> bool:
    """检测LLM是否在实体选择上停滞
    
    Args:
        state: 当前QueryState
        
    Returns:
        bool: 是否检测到停滞
    """
    iteration_history = state.get("iteration_history", [])
    if len(iteration_history) < 2:
        return False
    
    # 获取最近两次的实体选择
    current_entities = set(state.get("refined_classes", [])[:5])  # 比较前5个
    previous_entities = set()
    
    # 从历史中查找上一次的refined_classes
    for i in range(len(iteration_history) - 1, -1, -1):
        if "refined_classes" in iteration_history[i]:
            previous_entities = set(iteration_history[i]["refined_classes"][:5])
            break
    
    if not previous_entities:
        return False
    
    # 计算相似度
    intersection = len(current_entities.intersection(previous_entities))
    union = len(current_entities.union(previous_entities))
    
    if union == 0:
        return False
    
    similarity = intersection / union
    
    # 如果相似度 > 0.8，认为是停滞
    return similarity > 0.8

def handle_stagnation_with_entity_matcher(state: QueryState, entity_matcher, ontology_tools) -> Dict:
    """处理LLM停滞，使用EntityMatcher和丰富度评估获取新候选
    
    Args:
        state: 当前QueryState
        entity_matcher: EntityMatcher实例
        ontology_tools: OntologyTools实例
        
    Returns:
        包含新候选类的状态更新
    """
    try:
        logger.info("[StagnationHandler] 检测到LLM停滞，启动EntityMatcher获取新候选")
        
        # 获取原始查询
        original_query = state.get("query", "")
        if not original_query:
            return {}
        
        # 使用EntityMatcher获取新的候选类
        new_candidates = entity_matcher.extract_ranked_candidate_classes(
            query=original_query, 
            top_k=30  # 获取更多候选进行丰富度评估
        )
        
        if not new_candidates:
            logger.warning("[StagnationHandler] EntityMatcher未返回新候选")
            return {}
        
        # 使用丰富度评估对新候选进行排序
        logger.info(f"[StagnationHandler] 开始评估 {len(new_candidates)} 个新候选的丰富度")
        
        scored_candidates = []
        for candidate in new_candidates:
            richness_info = ontology_tools.get_class_richness_info(candidate)
            score = richness_info.get("richness_score", 0.0)
            scored_candidates.append((candidate, score))
        
        # 按丰富度排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        ranked_candidates = [item[0] for item in scored_candidates]
        
        # 记录这次操作到tried_tool_calls
        stagnation_record = record_tool_call(
            state, 
            "handle_stagnation", 
            {"method": "entity_matcher_with_richness"}, 
            {
                "new_candidates_count": len(new_candidates),
                "avg_richness_score": sum(item[1] for item in scored_candidates) / len(scored_candidates),
                "top_candidates": ranked_candidates[:10]
            }
        )
        
        logger.info(f"[StagnationHandler] 成功获取 {len(ranked_candidates)} 个按丰富度排序的候选")
        
        # 返回状态更新，强制注入最佳候选
        return {
            "refined_classes": ranked_candidates,
            "stagnation_handled": True,
            "stagnation_method": "entity_matcher_with_richness",
            **stagnation_record
        }
        
    except Exception as e:
        logger.error(f"[StagnationHandler] 处理停滞失败: {e}")
        return {"stagnation_error": str(e)}

def get_tool_call_history(state: QueryState, tool_name: Optional[str] = None) -> List[Dict]:
    """获取工具调用历史
    
    Args:
        state: 当前QueryState
        tool_name: 可选的工具名过滤器
        
    Returns:
        工具调用历史列表
    """
    tried_calls = state.get("tried_tool_calls", {})
    
    if not tried_calls:
        return []
    
    history = []
    for signature, call_info in tried_calls.items():
        if tool_name is None or call_info.get("tool") == tool_name:
            history.append({
                "signature": signature,
                **call_info
            })
    
    # 按时间戳排序
    history.sort(key=lambda x: x.get("timestamp", ""))
    return history

def clear_tool_call_history(state: QueryState, tool_name: Optional[str] = None) -> Dict:
    """清除工具调用历史
    
    Args:
        state: 当前QueryState
        tool_name: 可选的工具名过滤器，如果为None则清除所有
        
    Returns:
        更新后的state片段
    """
    tried_calls = state.get("tried_tool_calls", {})
    
    if tool_name is None:
        # 清除所有
        return {"tried_tool_calls": {}}
    else:
        # 只清除特定工具的记录
        filtered_calls = {
            signature: call_info 
            for signature, call_info in tried_calls.items()
            if call_info.get("tool") != tool_name
        }
        return {"tried_tool_calls": filtered_calls}