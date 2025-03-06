from typing import TypedDict, Literal, Dict, Optional, List, Union, Any, Annotated
from langgraph.graph.message import AnyMessage, add_messages

class WorkflowState(TypedDict):
    """工作流状态"""
    # Control
    stage: str
    previous_stage: Optional[str]
    status: str
    
    # Input
    source_ontology: Any
    target_ontology: Optional[Any]
    analysis_type: str
    
    # Context（统一存放所有子模块共同使用的信息）
    shared_context: Dict  # 示例键： "ontology_analysis", "research_ideas", "gap_analysis", "query" (包含 query_requests, query_results, interface), "evaluations"
    
    # Error Handling & 控制信息
    error: Optional[str]
    needs_improvement: bool
    
    # 系统消息
    messages: Annotated[list[AnyMessage], add_messages]

class StateManager:
    def __init__(self):
        """初始化状态管理器"""
        self.state = {
            # Control
            "stage": "querying",
            "previous_stage": None,
            "status": "initialized",
            
            # Input
            "source_ontology": None,
            "target_ontology": None,
            "analysis_type": "single_domain",
            
            # Context
            "shared_context": {},
            "ontology_analysis": {},
            "research_ideas": [],
            "current_idea": None,
            "idea_evaluations": [],
            "gap_analysis": {},
            
            # Query Management
            "query_interface": None,
            "query_requests": [],
            "query_results": {},
            
            # Error Handling
            "error": None,
            "needs_improvement": False,
            
            # System
            "messages": []
        }
    
    def update_state(self, updates: Dict) -> None:
        """更新状态"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in self.state and isinstance(self.state[key], dict):
                self.state[key].update(value)
            else:
                self.state[key] = value
    
    def prepare_team_state(self, team: str) -> Dict:
        """准备特定team的状态"""
        if team == "dream":
            return {
                "current_class": self.state["current_class"],
                "target_class": self.state["target_class"],
                "collected_info": {
                    "ontology_info": self.state["ontology_info"],
                    "class_info": self.state["class_info"]
                },
                "gap_analysis": self.state["gap_analysis"],
                "query_interface": self.state["query_interface"],
                "analysis_type": self.state["analysis_type"]
            }
        elif team == "critic":
            return {
                "research_idea": self.state["current_idea"],
                "context": {
                    "current_class": self.state["current_class"],
                    "target_class": self.state["target_class"],
                    "ontology_info": self.state["ontology_info"],
                    "class_info": self.state["class_info"]
                },
                "query_interface": self.state["query_interface"]
            }
        return {}
    
    def handle_team_result(self, team: str, result: Dict) -> None:
        """处理team的执行结果"""
        if team == "dream":
            if result.get("research_ideas"):
                new_idea = result["research_ideas"][-1]
                self.state["current_idea"] = new_idea
                self.state["research_ideas"].append(new_idea)
            
            # 更新gap分析结果
            for gap_type in ["evidence_gaps", "knowledge_gaps", "methodology_gaps", "meta_science_gaps"]:
                if gap_type in result:
                    self.state["gap_analysis"][gap_type].extend(result[gap_type])
                    
        elif team == "critic":
            if result.get("evaluation"):
                evaluation = result["evaluation"]
                self.state["idea_evaluations"].append(evaluation)
                self.state["needs_improvement"] = evaluation.get("needs_improvement", False)
                
                # 更新当前想法的评估结果
                if self.state["current_idea"]:
                    self.state["current_idea"]["evaluation"] = evaluation
    
    def handle_query_result(self, result: Dict) -> None:
        """处理查询结果"""
        if result.get("status") == "success":
            self.state["query_results"].update(result.get("results", {}))
            
            # 更新相关信息
            if "ontology_info" in result.get("results", {}):
                self.state["ontology_info"].update(result["results"]["ontology_info"])
            if "class_info" in result.get("results", {}):
                self.state["class_info"].update(result["results"]["class_info"])
    
    def handle_error(self, error: Exception, context: Dict) -> None:
        """处理错误"""
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "stage": self.state["stage"]
        }
        self.state["error"] = error_info
        self.state["messages"].append(f"Error in {self.state['stage']}: {str(error)}")
    
    def rollback_to_previous(self) -> None:
        """回滚到前一个状态"""
        if self.state["previous_stage"]:
            self.state["stage"] = self.state["previous_stage"]
            self.state["messages"].append(f"Rolled back to {self.state['previous_stage']}")
    
    def add_message(self, message: str) -> None:
        """添加系统消息"""
        self.state["messages"].append(message)

def create_state_manager() -> StateManager:
    """创建状态管理器实例"""
    return StateManager()