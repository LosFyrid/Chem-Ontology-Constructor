from typing import Dict, List, TypedDict, Annotated, Optional, Any
from langgraph.graph.message import AnyMessage, add_messages


# 导入LangGraph相关组件
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 导入Query工作流
from .query_team.query_workflow import create_query_graph, QueryState

class DreamerState(TypedDict):
    """Dreamer团队状态"""
    # 输入
    ontology: Any  # 主要本体
    additional_ontologies: Optional[List[Any]]  # 用于跨领域分析的其他本体
    
    # 分析
    analysis_type: str  # "single_domain" 或 "cross_domain"
    domain_analysis: Optional[Dict]  # 领域结构分析结果
    gap_analysis: Dict  # 研究空白分析
    research_ideas: List[Dict]  # 生成的研究创意
    
    # 评价与改进
    critic_feedback: Optional[Dict]  # 来自Critic Team的反馈
    idea_versions: Optional[List[Dict]]  # 记录创意的不同版本
    
    # 查询管理
    pending_queries: Optional[List[Dict]]  # 等待Query Team处理的查询
    query_results: Optional[Dict]  # Query Team返回的查询结果
    information_needs: Optional[List[Dict]]  # 已识别的信息需求
    
    # 工作流状态管理
    stage: str  # 当前阶段
    previous_stage: Optional[str]  # 上一阶段
    status: str  # 状态：initialized, processing, waiting_for_query, waiting_for_critic, error, completed
    
    # 系统
    messages: Annotated[List[AnyMessage], add_messages]  # 系统消息


class StateManager:
    def __init__(self, query_manager=None):
        """初始化Dreamer团队状态管理器"""
        self.state: DreamerState = {
            "ontology": None,
            "additional_ontologies": None,
            "analysis_type": "single_domain",
            "domain_analysis": None,
            "gap_analysis": {},
            "research_ideas": [],
            "critic_feedback": None,
            "idea_versions": [],
            "pending_queries": [],
            "query_results": {},
            "information_needs": [],
            "stage": "initialized",
            "previous_stage": None,
            "status": "initialized",
            "messages": []
        }
        
        # 使用外部注入的查询管理器或创建新的
        self.query_manager = query_manager or QueryManager()
        
    def submit_query(self, query_text: str,
                    priority: str = "normal", 
                    originating_stage: str = None) -> str:
        """提交查询并返回查询ID"""
        if originating_stage is None:
            originating_stage = self.state.get("stage", "unknown")
            
        # 准备查询上下文
        query_context = {
            "ontology": self.state.get("ontology"),
            "additional_ontology": self.state.get("additional_ontologies", [None])[0],
            "originating_team": "dreamer",
            "originating_stage": originating_stage,
        }
        
        # 使用查询管理器提交查询
        query_id = self.query_manager.submit_query(
            query_text=query_text,
            query_context=query_context,
            priority=priority
        )
        
        # 订阅查询结果
        self.query_manager.subscribe(query_id, self._handle_query_result)
        
        # 更新状态
        pending_queries = self.state.get("pending_queries", [])
        pending_queries.append({"query_id": query_id, "query": query_text})
        
        self.update_state({
            "pending_queries": pending_queries,
            "status": "waiting_for_query"
        })
        
        return query_id
        
    def _handle_query_result(self, query_id: str, result: Dict) -> None:
        """处理查询结果的回调函数"""
        # 更新状态
        self.process_query_results({query_id: result})
        
    def process_query_results(self, results: Dict[str, Dict]) -> None:
        """处理查询结果并更新状态"""
        if not results:
            return
        
        # 更新状态中的查询结果
        query_results = self.state.get("query_results", {})
        query_results.update(results)
        
        # 查找并移除已完成的查询
        pending_queries = self.state.get("pending_queries", [])
        completed_query_ids = set(results.keys())
        pending_queries = [q for q in pending_queries if q.get("query_id") not in completed_query_ids]
        
        # 更新状态
        self.update_state({
            "query_results": query_results,
            "pending_queries": pending_queries,
            "status": "processing" if not pending_queries else "waiting_for_query"
        })
        
        # 添加状态更新消息
        for query_id in results:
            self.add_message({
                "role": "system", 
                "content": f"查询 {query_id} 已完成并处理结果"
            })

def create_state_manager() -> StateManager:
    """创建并返回一个StateManager实例"""
    return StateManager()

