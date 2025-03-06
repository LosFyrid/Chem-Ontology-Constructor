from typing import Dict, List, TypedDict, Literal, Annotated, Optional, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from autology_constructor.idea.query_team.ontology_tools import OntologyAnalyzer
from .evidence_finder import EvidenceFinder
from .knowledge_finder import KnowledgeFinder
from .methodology_finder import MethodologyFinder
from .meta_science_finder import MetaScienceFinder
from autology_constructor.idea.query_team.utils import parse_json
class DreamerState(TypedDict):
    """Dreamer团队状态"""
    # Input
    source_ontology: Any
    target_ontology: Optional[Any]
    analysis_type: str
    domain_analysis: Dict
    
    # Analysis
    gaps: Dict
    research_ideas: List[Dict]
    
    # State Management
    stage: str
    previous_stage: Optional[str]
    status: str
    
    # System
    messages: Annotated[list[AnyMessage], add_messages]

def create_dreamer_graph() -> Graph:
    workflow = StateGraph(DreamerState)
    
    def analyze_domain(state: DreamerState) -> Dict:
        """分析领域并生成研究想法"""
        try:
            analyzer = OntologyAnalyzer()
            # 分析领域结构
            domain_analysis = analyzer.analyze_domain_structure(state["source_ontology"])
            domain_analysis["key_concepts"] = analyzer.find_key_concepts(state["source_ontology"])
            
            if state["analysis_type"] == "cross_domain":
                domain_analysis["cross_domain"] = analyzer.compare_domains(
                    state["source_ontology"],
                    state["target_ontology"]
                )
            
            # 识别研究机会：调用四个finder
            finders = {
                "evidence": EvidenceFinder(),
                "knowledge": KnowledgeFinder(),
                "methodology": MethodologyFinder(),
                "meta_science": MetaScienceFinder()
            }
            
            gaps = {}
            research_ideas = []
            for finder_type, finder in finders.items():
                if finder_type in ["methodology", "meta_science"] and state["analysis_type"] != "cross_domain":
                    continue
                gaps[finder_type] = finder.analyze_gaps(domain_analysis)
                research_ideas.extend(finder.generate_ideas(gaps[finder_type]))
            
            # 将初步结果写入状态
            return {
                "stage": "dreaming",
                "previous_stage": state.get("stage"),
                "status": "success",
                "gap_analysis": gaps,
                "research_ideas": research_ideas,
                "messages": [
                    f"Generated {len(research_ideas)} research ideas",
                    f"Analyzed {len(gaps)} gap categories"
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": str(e),
                "messages": [f"Domain analysis failed: {str(e)}"]
            }
    
    # 以下子图调用保持不变
    evidence_graph = EvidenceFinder().create_finder_graph()
    knowledge_graph = KnowledgeFinder().create_finder_graph()
    methodology_graph = MethodologyFinder().create_finder_graph()
    meta_science_graph = MetaScienceFinder().create_finder_graph()
    
    async def assess_information_sufficiency(state: DreamerState) -> Dict:
        # 此处逻辑保持简单，若分析结果缺失则发起查询请求
        analysis = state.get("ontology_analysis", {})
        if not analysis:
            return {
                "query_requests": [{
                    "query": "Analyze domain structure",
                    "requester": "dream",
                    "priority": 1,
                    "context": {
                        "source_ontology": state["source_ontology"],
                        "target_ontology": state.get("target_ontology")
                    }
                }],
                "messages": ["Need domain structure analysis"]
            }
        required_keys = ["core_concepts", "key_patterns", "research_opportunities"]
        if state["analysis_type"] == "cross_domain":
            required_keys.extend(["analogies", "transfer_opportunities"])
        missing = [k for k in required_keys if k not in analysis]
        if missing:
            return {
                "query_requests": [{
                    "query": f"Complete analysis for {aspect}",
                    "requester": "dream",
                    "priority": 1,
                    "context": {"missing_aspects": missing}
                } for aspect in missing],
                "messages": [f"Missing analysis aspects: {', '.join(missing)}"]
            }
        return {"can_proceed": True}
    
    workflow.add_node("assess_info", assess_information_sufficiency)
    
    def route_after_assessment(state: DreamerState) -> Literal["start_analysis", "need_info"]:
        return "start_analysis" if state.get("can_proceed", False) else "need_info"
    
    def route_analysis(state: DreamerState) -> Literal["evidence", "methodology"]:
        # 根据是否存在 target_ontology 判断单领域还是跨领域分析
        if state.get("target_ontology"):
            return "methodology"
        return "evidence"
    
    workflow.add_edge(START, "assess_info")
    workflow.add_conditional_edges(
        "assess_info",
        route_after_assessment,
        {
            "start_analysis": "router",
            "need_info": END
        }
    )
    
    workflow.add_node("router", route_analysis)
    workflow.add_conditional_edges(
        "router",
        route_analysis,
        {
            "evidence": "evidence",
            "methodology": "methodology"
        }
    )
    
    workflow.add_edge("evidence", "knowledge")
    workflow.add_edge("methodology", "meta_science")
    workflow.add_edge("knowledge", END)
    workflow.add_edge("meta_science", END)
    
    return workflow.compile() 