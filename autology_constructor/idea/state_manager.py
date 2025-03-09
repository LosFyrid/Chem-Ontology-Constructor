from typing import Dict, List, TypedDict, Literal, Annotated, Optional, Any
from langgraph.graph.message import AnyMessage, add_messages


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
    def __init__(self):
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
    
    def update_state(self, updates: Dict) -> None:
        """更新状态"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in self.state and isinstance(self.state[key], dict):
                self.state[key].update(value)
            elif isinstance(value, list) and key in self.state and isinstance(self.state[key], list):
                # 对于列表类型，考虑是追加还是替换
                if key in ["messages", "pending_queries", "information_needs"]:
                    self.state[key].extend(value)
                else:
                    self.state[key] = value
            else:
                self.state[key] = value
                
        # 自动更新previous_stage
        if "stage" in updates and updates["stage"] != self.state.get("previous_stage"):
            self.state["previous_stage"] = updates.get("previous_stage", self.state.get("stage"))
    
    def handle_error(self, error: Exception) -> None:
        """处理错误，记录错误信息"""
        error_msg = f"Error in {self.state.get('stage')}: {str(error)}"
        self.state['status'] = 'error'
        self.add_message({"role": "system", "content": error_msg})
    
    def add_message(self, message: Dict[str, str]) -> None:
        """添加消息到消息历史"""
        if isinstance(message, str):
            message = {"role": "system", "content": message}
        self.state['messages'].append(message)
    
    def get_state(self) -> DreamerState:
        """获取当前状态"""
        return self.state
    
    def reset(self) -> None:
        """重置状态"""
        self.__init__()
    
    def process_query_results(self, results: Dict) -> None:
        """处理Query Team返回的查询结果"""
        if not results:
            return
            
        self.state["query_results"] = results
        self.add_message({"role": "system", "content": "收到Query Team查询结果"})
        
        # 清空pending_queries
        self.state["pending_queries"] = []
        
        # 更新状态
        self.state["status"] = "processing"
    
    def process_critic_feedback(self, feedback: Dict) -> None:
        """处理Critic Team返回的反馈"""
        if not feedback:
            return
            
        self.state["critic_feedback"] = feedback
        self.add_message({"role": "system", "content": "收到Critic Team反馈"})
        
        # 更新状态
        self.state["status"] = "processing"
        self.state["stage"] = "received_feedback"


def create_state_manager() -> StateManager:
    """创建并返回一个StateManager实例"""
    return StateManager()