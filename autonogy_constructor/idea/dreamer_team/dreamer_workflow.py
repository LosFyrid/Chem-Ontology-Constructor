from typing import Dict, List, TypedDict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph

from .evidence_finder import EvidenceFinder
from .knowledge_finder import KnowledgeFinder
from .methodology_finder import MethodologyFinder
from .meta_science_finder import MetaScienceFinder

class DreamerState(TypedDict):
    """Dreamer team state"""
    # Input
    current_class: str
    target_class: str  # 跨领域分析的目标类
    collected_info: Dict
    
    # Gap Analysis Results
    evidence_gaps: List[Dict]
    knowledge_gaps: List[Dict]
    methodology_gaps: List[Dict]
    meta_science_gaps: List[Dict]
    
    # Ideas
    research_ideas: List[Dict]
    
    # Query Management
    query_requests: List[Dict]
    
    # System
    messages: List[BaseMessage]
    done: bool

def create_dreamer_graph() -> Graph:
    """Create dreamer team workflow"""
    workflow = StateGraph(DreamerState)
    
    # Create finder subgraphs
    evidence_graph = EvidenceFinder().create_finder_graph()
    knowledge_graph = KnowledgeFinder().create_finder_graph()
    methodology_graph = MethodologyFinder().create_finder_graph()
    meta_science_graph = MetaScienceFinder().create_finder_graph()
    
    # Add subgraphs
    workflow.add_node("evidence", evidence_graph)
    workflow.add_node("knowledge", knowledge_graph)
    workflow.add_node("methodology", methodology_graph)
    workflow.add_node("meta_science", meta_science_graph)
    
    # Add edges based on whether cross-domain analysis is needed
    def route_analysis(state: DreamerState) -> str:
        if state["target_class"]:
            return "methodology"  # 有目标类时进行跨领域分析
        else:
            return "evidence"  # 否则进行单领域分析
    
    workflow.add_conditional_edges(
        "start",
        route_analysis,
        {
            "evidence": "evidence",
            "methodology": "methodology"
        }
    )
    
    # Add other edges
    workflow.add_edge("evidence", "knowledge")
    workflow.add_edge("methodology", "meta_science")
    
    return workflow.compile() 