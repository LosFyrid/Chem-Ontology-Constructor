from typing import TypedDict, List, Dict, Any, NotRequired, Literal, Union
from langgraph.graph import Graph, StateGraph
from .query_team.query_workflow import create_query_graph, QueryState
from .dreamer_team.dreamer_workflow import create_dreamer_graph, DreamerState
from .critic_team.critic_workflow import create_critic_graph, CriticState

class MainState(TypedDict):
    """Main workflow state"""
    # Current processing stage
    stage: Literal["querying", "dreaming", "critiquing"]  # "querying", "dreaming", "critiquing"
    
    # Subgraph states
    query_state: NotRequired[Union[QueryState, None]]
    dreamer_state: NotRequired[Union[DreamerState, None]]
    critic_state: NotRequired[Union[CriticState, None]]
    
    # Shared data
    ontology: Any  # The ontology being analyzed
    current_class: str  # Current class being processed
    target_class: str  # Target class for cross-domain analysis (optional)
    research_ideas: List[Dict]  # Generated ideas
    evaluations: List[Dict]  # Idea evaluations
    
    # System state
    messages: List[str]  # System messages
    done: bool

def create_main_graph(ontology) -> Graph:
    """Create main workflow graph"""
    workflow = StateGraph(MainState)
    
    # Create subgraphs
    query_graph = create_query_graph(ontology)
    dreamer_graph = create_dreamer_graph()
    critic_graph = create_critic_graph()
    
    # Add subgraphs as nodes
    workflow.add_node("prepare_query", prepare_query_state)
    workflow.add_node("query", query_graph)
    workflow.add_node("handle_query_results", handle_query_results)
    workflow.add_node("dream", dreamer_graph)
    workflow.add_node("handle_dreamer_results", handle_dreamer_results)
    workflow.add_node("critique", critic_graph)
    workflow.add_node("handle_critic_results", handle_critic_results)
    workflow.add_node("check_query_needed", check_query_needed)
    workflow.add_node("check_ideas", check_ideas)
    workflow.add_node("end", None)  # 束节点
    
    # Define state transformations
    def prepare_query_state(state: MainState) -> MainState:
        """Prepare state for query subgraph"""
        state["query_state"] = {
            "query": f"Get all information about class {state['current_class']}",
            "requester": state["stage"],
            "context": {
                "current_class": state["current_class"],
                "target_class": state.get("target_class"),
                "purpose": "idea_generation"
            }
        }
        return state
    
    def handle_query_results(state: MainState) -> MainState:
        """Process query results and update dreamer state"""
        if state["query_state"]["status"] == "success":
            return {
                "dreamer_state": {
                    "current_class": state["current_class"],
                    "target_class": state.get("target_class"),
                    "collected_info": state["query_state"]["results"]
                }
            }
        return {}  # 如果没有更新,返回空字典
    
    def handle_dreamer_results(state: MainState) -> MainState:
        """Process dreamer results and update critic state"""
        if state["dreamer_state"]["research_ideas"]:
            # Update research ideas and prepare for critique
            state["research_ideas"].extend(state["dreamer_state"]["research_ideas"])
            # 更新critic状态，包含所有必要字段
            state["critic_state"].update({
                "research_idea": state["dreamer_state"]["research_ideas"][-1],
                "context": {
                    "current_class": state["current_class"],
                    "target_class": state.get("target_class")
                },
                "feasibility_score": 0.0,
                "novelty_score": 0.0,
                "practicality_score": 0.0,
                "verification_results": {},
                "evaluations": [],
                "suggestions": [],
                "query_requests": [],
                "messages": [],
                "done": False
            })
        return state
    
    def handle_critic_results(state: MainState) -> MainState:
        """Process critic results and prepare feedback for dreamer"""
        if state["critic_state"]["done"]:
            current_idea = state["critic_state"]["research_idea"]
            
            # 保存评估结果
            evaluation = {
                "idea": current_idea,
                "scores": {
                    "feasibility": state["critic_state"]["feasibility_score"],
                    "novelty": state["critic_state"]["novelty_score"],
                    "practicality": state["critic_state"]["practicality_score"]
                },
                "suggestions": state["critic_state"]["suggestions"],
                "verification": state["critic_state"]["verification_results"]
            }
            state["evaluations"].append(evaluation)
            
            # 检查是否需要���进
            avg_score = sum(evaluation["scores"].values()) / len(evaluation["scores"])
            if avg_score < 0.7:  # 设置一个质量阈值
                # 准备反馈给dreamer的信息
                state["dreamer_state"].update({
                    "feedback": {
                        "original_idea": current_idea,
                        "gap_type": current_idea["gap_type"],  # 保持gap类型不变
                        "evaluation": evaluation,
                        "improvement_needed": True
                    }
                })
                state["stage"] = "dreaming"  # 返回到dreamer改进
            else:
                # 想法质量达标，保存并继续
                state["research_ideas"].append(current_idea)
                
        return state
    
    def prepare_improvement(state: MainState) -> MainState:
        """Prepare dreamer state for idea improvement"""
        if state["stage"] == "dreaming" and state["dreamer_state"].get("feedback"):
            feedback = state["dreamer_state"]["feedback"]
            # 确保改进基于原始的gap
            state["dreamer_state"].update({
                "current_class": state["current_class"],
                "target_class": state.get("target_class"),
                "improvement_context": {
                    "original_gap": feedback["gap_type"],
                    "critic_feedback": feedback["evaluation"],
                    "suggestions": feedback["evaluation"]["suggestions"]
                }
            })
        return state
    
    def check_ideas(state: MainState) -> MainState:
        """Check remaining ideas and prepare state for next iteration"""
        return state
    
    # Add edges with state transformations
    workflow.add_edge("prepare_query", "query")
    workflow.add_edge("query", "handle_query_results")
    workflow.add_edge("handle_query_results", "dream")
    workflow.add_edge("dream", "handle_dreamer_results")
    workflow.add_edge("handle_dreamer_results", "critique")
    workflow.add_edge("critique", "handle_critic_results")
    
    # Add conditional edges
    def check_improvement_needed(state: MainState) -> str:
        """Check if idea needs improvement"""
        if (state["stage"] == "dreaming" and 
            state["dreamer_state"].get("feedback", {}).get("improvement_needed", False)):
            return "needs_improvement"
        return "no_improvement"
    
    def check_query_needed(state: MainState) -> bool:
        """Check if more queries are needed"""
        return bool(state["dreamer_state"].get("query_requests") or 
                   state["critic_state"].get("query_requests"))
    
    def check_ideas_remaining(state: MainState) -> bool:
        """Check if there are more ideas to evaluate"""
        return bool(state["research_ideas"])
    
    # 添加改进循环
    workflow.add_conditional_edges(
        "handle_critic_results",
        check_improvement_needed,
        {
            "needs_improvement": "prepare_improvement",
            "no_improvement": "check_query_needed"
        }
    )
    
    workflow.add_edge("prepare_improvement", "dream")  # 改进后重新生成想法
    
    workflow.add_conditional_edges(
        "check_query_needed",
        check_query_needed,
        {
            True: "prepare_query",
            False: "check_ideas"
        }
    )
    
    workflow.add_conditional_edges(
        "check_ideas",
        check_ideas_remaining,
        {
            True: "critique",
            False: "end"
        }
    )
    
    workflow.add_node("check_ideas", check_ideas)
    
    # Set entry and exit points
    workflow.set_entry_point("prepare_query")
    workflow.set_finish_point("end")
    
    # 验证图的完整性
    app = workflow.compile()
    return app

def validate_state(state: MainState) -> bool:
    """验证状态的完整性和正确性"""
    try:
        required_fields = ["stage", "ontology", "current_class"]
        if not all(field in state for field in required_fields):
            return False
            
        # 验证stage的值
        if state["stage"] not in ["querying", "dreaming", "critiquing"]:
            return False
            
        # 验证子状态的完整性
        if state["stage"] == "querying" and "query_state" not in state:
            return False
        if state["stage"] == "dreaming" and "dreamer_state" not in state:
            return False
        if state["stage"] == "critiquing" and "critic_state" not in state:
            return False
            
        return True
    except Exception:
        return False 