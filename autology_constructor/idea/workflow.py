from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import Graph, StateGraph, END, START
from autology_constructor.idea.query_team.query_workflow import create_query_graph
from autology_constructor.idea.dreamer_team.dreamer_workflow import create_dreamer_graph
from autology_constructor.idea.critic_team.critic_workflow import create_critic_graph
from autology_constructor.idea.state_manager import WorkflowState
from autology_constructor.idea.query_team.ontology_tools import OntologyAnalyzer
from autology_constructor.idea.dreamer_team.evidence_finder import EvidenceFinder
from autology_constructor.idea.dreamer_team.knowledge_finder import KnowledgeFinder
from autology_constructor.idea.dreamer_team.methodology_finder import MethodologyFinder
from autology_constructor.idea.dreamer_team.meta_science_finder import MetaScienceFinder
import os

def create_workflow(ontology_folder: str) -> Graph:
    """
    创建主工作流
    根据给定文件夹扫描.owl本体文件：
      - 只有1个文件：进入单领域分析模式（single_domain）
      - 2个及以上：进入多领域分析模式（cross_domain）
    """
    # 扫描文件夹中的.owl文件
    owl_files = [f for f in os.listdir(ontology_folder) if f.endswith('.owl')]
    if not owl_files:
        raise ValueError("指定文件夹中未找到owl本体文件")
        
    def load_ontology(file_path: str):
        # TODO: 根据实际项目实现本体加载逻辑，此处占位返回文件路径
        return file_path

    if len(owl_files) == 1:
        source_ontology = load_ontology(os.path.join(ontology_folder, owl_files[0]))
        target_ontology = None
        analysis_type = "single_domain"
    else:
        source_ontology = load_ontology(os.path.join(ontology_folder, owl_files[0]))
        target_ontology = load_ontology(os.path.join(ontology_folder, owl_files[1]))
        analysis_type = "cross_domain"
    
    workflow = StateGraph(WorkflowState)
    
    # 创建子图
    query_graph = create_query_graph()  # query team子图
    dream_graph = create_dreamer_graph()
    critic_graph = create_critic_graph()
    analyzer = OntologyAnalyzer()
    
    def initialize_state(state: WorkflowState) -> Dict:
        """初始化工作流状态"""
        return {
            "stage": "querying",
            "previous_stage": None,
            "source_ontology": source_ontology,
            "target_ontology": target_ontology,
            "analysis_type": analysis_type,
            "shared_context": {
                "ontology_analysis": {},
                "research_ideas": [],
                "gap_analysis": {},
                "query": {
                    "interface": None,
                    "requests": [],
                    "results": {}
                },
                "evaluations": []
            },
            "error": None,
            "needs_improvement": False,
            "messages": []
        }

    def route_execution(state: WorkflowState) -> str:
        """路由执行流程"""
        # 检查是否有查询请求
        if state.get("query_requests", []):
            return "query"
            
        # 根据当前阶段决定下一步
        stage_map = {
            "querying": "dream",
            "dreaming": "critic",
            "critiquing": "dream" if state.get("needs_improvement") else "end"
        }
        return stage_map[state["stage"]]
    
    def handle_query_result(state: WorkflowState) -> Dict:
        """处理查询结果"""
        updates = {
            "status": state["query_state"].get("status", "error"),
            "stage": state["previous_stage"] if state["query_state"].get("status") == "success" else "error",
            "previous_stage": state.get("stage"),
            "messages": []
        }
        
        if state["query_state"].get("status") == "success":
            # 更新查询结果
            if "ontology_analysis" in state["query_state"].get("results", {}):
                updates["ontology_analysis"] = state["query_state"]["results"]["ontology_analysis"]
                updates["messages"].append("Successfully updated ontology analysis")
                
            # 更新查询结果
            updates["query_results"] = state["query_state"].get("results", {})
            updates["messages"].append("Query completed successfully")
        else:
            # 处理错误情况
            updates["error"] = state["query_state"].get("error", "Unknown error in query")
            updates["messages"].append(f"Query failed: {updates['error']}")
        
        return updates

    def prepare_dream_state(state: WorkflowState) -> Dict:
        """准备 dream 状态"""
        domain_analysis = analyzer.analyze_domain_structure(state["source_ontology"])
        domain_analysis["key_concepts"] = analyzer.find_key_concepts(state["source_ontology"])
        
        if state["analysis_type"] == "cross_domain":
            domain_analysis["cross_domain"] = analyzer.compare_domains(
                state["source_ontology"],
                state["target_ontology"]
            )
        
        # 2. 基于分析结果识别研究机会
        finders = {
            "evidence": EvidenceFinder(),
            "knowledge": KnowledgeFinder(),
            "methodology": MethodologyFinder(),
            "meta_science": MetaScienceFinder()
        }
        
        gaps = {}
        research_ideas = []
        for finder_type, finder in finders.items():
            # 针对跨领域分析，仅调用方法论和元科学finder
            if finder_type in ["methodology", "meta_science"] and state["analysis_type"] != "cross_domain":
                continue
            # 每个finder先进行差距分析
            gaps[finder_type] = finder.analyze_gaps(domain_analysis)
            # 根据差距利用LLM生成研究想法，generate_ideas方法统一在BaseFinder中实现
            research_ideas.extend(finder.generate_ideas(gaps[finder_type]))
        
        return {
            "dream_state": {
                "source_ontology": state["source_ontology"],
                "target_ontology": state.get("target_ontology"),
                "analysis_type": state["analysis_type"],
                "domain_analysis": domain_analysis,
                "gaps": gaps
            }
        }
    
    def handle_dream_result(state: WorkflowState) -> Dict:
        """处理 dream 结果"""
        dream_state = state["dream_state"]
        return {
            "stage": "critiquing",
            "previous_stage": state.get("stage"),
            "status": dream_state.get("status", "success"),
            "research_ideas": dream_state.get("research_ideas", []),
            "gap_analysis": dream_state.get("gap_analysis", {}),
            "messages": dream_state.get("messages", ["Dream phase completed"]),
            "error": dream_state.get("error")
        }

    def handle_critic_result(state: WorkflowState) -> Dict:
        """处理 critic 结果"""
        critic_state = state.get("critic_state", {})
        return {
            "stage": "dreaming" if critic_state.get("needs_improvement") else "end",
            "previous_stage": state.get("stage"),
            "status": critic_state.get("status", "success"),
            "evaluations": critic_state.get("evaluations", []),
            "needs_improvement": critic_state.get("needs_improvement", False),
            "messages": critic_state.get("messages", []),
            "error": critic_state.get("error"),
            "research_ideas": state.get("research_ideas", []) if critic_state.get("needs_improvement") else []
        }

    # 添加节点
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("router", route_execution)
    workflow.add_node("query", query_graph)
    workflow.add_node("dream", dream_graph)
    workflow.add_node("critic", critic_graph)
    
    # 添加边
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "router")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "router",
        route_execution,
        {
            "query": "query",
            "dream": "dream",
            "critic": "critic",
            "end": END
        }
    )
    
    # 添加返回边
    workflow.add_edge("query", "router")
    workflow.add_edge("dream", "router")
    workflow.add_edge("critic", "router")
    
    return workflow.compile()




