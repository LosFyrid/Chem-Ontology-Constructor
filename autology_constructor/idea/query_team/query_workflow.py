from typing import Dict, List, TypedDict, Literal, Annotated, Optional, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from .ontology_tools import OntologyTools
from autology_constructor.idea.state_manager import StateManager, QueryState
import json
from autology_constructor.idea.query_team.query_agents import QueryParserAgent, StrategyPlannerAgent, ToolExecutorAgent, SparqlExpertAgent, ValidationAgent



def create_query_graph() -> Graph:
    """创建查询工作流"""
    query_builder = StateGraph(QueryState)
    
    # 初始化所有Agent
    parser = QueryParserAgent()
    strategist = StrategyPlannerAgent()
    tool_executor = ToolExecutorAgent()
    sparql_expert = SparqlExpertAgent()
    validator = ValidationAgent()

    # 定义状态节点
    def parse_node(state): 
        return parser(state)
    
    def strategy_node(state):
        return strategist.decide_strategy(state["request_processed"])
    
    def tool_exec_node(state):
        if state["strategy"] == "tool_based":
            plan = _generate_query_plan(state["request_processed"])
            results = tool_executor.execute_plan(plan, state["ontology"])
            return {"raw_results": results}
    
    def sparql_node(state):
        if state["strategy"] == "sparql":
            sparql = sparql_expert.generate_sparql(state["request_processed"])
            # 模拟执行SPARQL
            return {"raw_results": {"sparql": sparql}}
    
    def validate_node(state):
        return validator.validate(state["raw_results"])

    # 构建工作流
    query_builder.add_node("parse", parse_node)
    query_builder.add_node("strategy", strategy_node)
    query_builder.add_node("tool", tool_exec_node)
    query_builder.add_node("sparql", sparql_node)
    query_builder.add_node("validate", validate_node)

    # 设置路由逻辑
    def route_strategy(state):
        return "tool" if state["strategy"] == "tool_based" else "sparql"
    
    query_builder.add_edge(START, "parse")
    query_builder.add_edge("parse", "strategy")
    query_builder.add_conditional_edges(
        "strategy",
        route_strategy,
        {"tool": "tool", "sparql": "sparql"}
    )
    query_builder.add_edge("tool", "validate")
    query_builder.add_edge("sparql", "validate")
    query_builder.add_edge("validate", END)

    return query_builder.compile()

def _generate_query_plan(standardized_query: dict) -> list:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the standardized JSON query description provided, generate a sequence of tool function calls to answer the query. Each step should be a JSON object with keys 'tool' and 'params'. The available tools are: get_class_info, get_class_properties, get_property_restrictions, get_parents, get_children, get_ancestors, get_descendants, get_related_classes, get_disjoint_classes, parse_hierarchy_structure. Answer with a JSON array representing the sequence of steps. Only output the JSON, nothing else."),
        ("user", "Standardized query description:\n{standardized_query}")
    ])
    response = ChatOpenAI(temperature=0).invoke(prompt.format_messages(standardized_query=json.dumps(standardized_query, ensure_ascii=False)))
    try:
        return json.loads(response.content.strip())
    except Exception:
        raise ValueError("Failed to parse query plan from JSON.")

