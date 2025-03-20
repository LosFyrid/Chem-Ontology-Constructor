from typing import Dict, List, TypedDict, Literal, Annotated, Optional, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from .ontology_tools import OntologyTools
from .query_agents import QueryParserAgent, StrategyPlannerAgent, ToolExecutorAgent, SparqlExpertAgent, ValidationAgent

class QueryState(TypedDict):
    """查询团队状态"""
    # Input
    query: str  # 自然语言查询
    source_ontology: Any  # 使用主工作流中的source_ontology
    query_type: str  # 查询类型
    templated_query: str  # 模板化查询
    query_strategy: Literal["tool_sequence", "SPARQL"]  # 查询策略
    additional_ontology: Optional[Any]  # 额外本体（用于跨域查询）
    originating_team: str  # 发起查询的团队
    originating_stage: str  # 发起查询的阶段
    
    # Query Management
    query_results: Dict  # 查询结果
    normalized_query: Optional[Dict]  # 标准化的查询结构
    execution_plan: Optional[List[Dict]]  # 执行计划
    status: str  # 状态
    stage: str  # 当前阶段
    previous_stage: Optional[str]  # 上一阶段
    
    # System
    messages: Annotated[list[AnyMessage], add_messages]

def create_query_graph() -> Graph:
    """创建查询工作流"""
    workflow = StateGraph(QueryState)
    
    # 初始化代理和工具
    parser_agent = QueryParserAgent()
    strategy_agent = StrategyPlannerAgent()
    tool_agent = ToolExecutorAgent()
    sparql_agent = SparqlExpertAgent()
    validator_agent = ValidationAgent()
    
    # 节点实现
    def normalize_query(state: QueryState) -> Dict:
        """解析并标准化查询"""
        try:
            query = state["query"]
            
            # 如果已经有模板化查询，使用模板化查询
            templated_query = state.get("templated_query")
            if templated_query and templated_query.strip():
                input_query = templated_query
            else:
                input_query = query
                
            # 使用解析代理处理查询
            query_state = {"natural_query": input_query}
            normalized = parser_agent(query_state)
            
            return {
                "normalized_query": normalized,
                "status": "parsing_complete",
                "stage": "normalized",
                "previous_stage": state.get("stage"),
                "messages": [f"Query normalized: {input_query}"]
            }
        except Exception as e:
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": str(e),
                "messages": [f"Query normalization failed: {str(e)}"]
            }
    
    def determine_strategy(state: QueryState) -> Dict:
        """确定查询执行策略"""
        try:
            # 如果已经指定策略，使用指定策略
            if state.get("query_strategy"):
                strategy = state["query_strategy"]
            else:
                # 使用策略规划器确定最佳策略
                strategy = strategy_agent.decide_strategy(state["normalized_query"])
                
            return {
                "query_strategy": strategy,
                "status": "strategy_determined",
                "stage": "strategy",
                "previous_stage": state.get("stage"),
                "messages": [f"Query strategy determined: {strategy}"]
            }
        except Exception as e:
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": str(e),
                "messages": [f"Strategy determination failed: {str(e)}"]
            }
    
    def execute_query(state: QueryState) -> Dict:
        """执行查询"""
        try:
            strategy = state.get("query_strategy", "tool_sequence")
            normalized_query = state["normalized_query"]
            source_ontology = state["source_ontology"]
            
            if strategy == "tool_sequence":
                # 为工具序列生成执行计划
                execution_plan = _generate_tool_plan(normalized_query)
                
                # 执行工具调用
                results = tool_agent.execute_plan(execution_plan, source_ontology)
                
                return {
                    "execution_plan": execution_plan,
                    "query_results": results,
                    "status": "executed",
                    "stage": "executed",
                    "previous_stage": state.get("stage"),
                    "messages": ["Tool-based query executed successfully"]
                }
            elif strategy == "SPARQL":
                # 生成SPARQL查询
                sparql_query = sparql_agent.generate_sparql(normalized_query)
                
                # 执行SPARQL查询
                results = _execute_sparql(sparql_query, source_ontology)
                
                return {
                    "query_results": results,
                    "sparql_query": sparql_query,
                    "status": "executed",
                    "stage": "executed",
                    "previous_stage": state.get("stage"),
                    "messages": ["SPARQL query executed successfully"]
                }
            else:
                raise ValueError(f"Unsupported query strategy: {strategy}")
                
        except Exception as e:
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": str(e),
                "messages": [f"Query execution failed: {str(e)}"]
            }
    
    def validate_results(state: QueryState) -> Dict:
        """验证查询结果"""
        try:
            results = state["query_results"]
            
            # 使用验证代理验证结果
            validation = validator_agent.validate(results)
            
            if validation.get("valid", False):
                return {
                    "validation": validation,
                    "status": "success",
                    "stage": "validated",
                    "previous_stage": state.get("stage"),
                    "messages": ["Results validation successful"]
                }
            else:
                return {
                    "validation": validation,
                    "status": "warning",
                    "stage": "validation_warning",
                    "previous_stage": state.get("stage"),
                    "messages": [f"Results validation warning: {validation.get('message', 'Unknown issue')}"]
                }
        except Exception as e:
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": str(e),
                "messages": [f"Results validation failed: {str(e)}"]
            }
    
    def _generate_tool_plan(normalized_query: Dict) -> List[Dict]:
        """为标准化查询生成工具执行计划"""
        # 这里可以使用LLM生成更复杂的执行计划
        # 简化实现：直接从查询类型映射到工具
        query_type = normalized_query.get("type", "unknown")
        target = normalized_query.get("target", "")
        
        if query_type == "class_info":
            return [{"tool": "get_class_info", "params": {"class_name": target}}]
        elif query_type == "property_info":
            property_name = normalized_query.get("property", "")
            return [
                {"tool": "get_class_properties", "params": {"class_name": target}},
                {"tool": "get_property_restrictions", "params": {"class_name": target, "property_name": property_name}}
            ]
        elif query_type == "hierarchy":
            return [
                {"tool": "get_parents", "params": {"class_name": target}},
                {"tool": "get_children", "params": {"class_name": target}}
            ]
        elif query_type == "domain_analysis":
            return [{"tool": "parse_hierarchy_structure", "params": {}}]
        else:
            # 默认情况
            return [{"tool": "execute_query", "params": {"query": normalized_query.get("natural_query", "")}}]
    
    def _execute_sparql(sparql_query: str, ontology: Any) -> Dict:
        """执行SPARQL查询"""
        tools = OntologyTools(ontology)
        return tools.execute_sparql(sparql_query)
    
    # 添加节点
    workflow.add_node("normalize", normalize_query)
    workflow.add_node("strategy", determine_strategy)
    workflow.add_node("execute", execute_query)
    workflow.add_node("validate", validate_results)
    
    # 添加边
    workflow.add_edge(START, "normalize")
    workflow.add_edge("normalize", "strategy")
    workflow.add_edge("strategy", "execute")
    workflow.add_edge("execute", "validate")
    workflow.add_edge("validate", END)
    
    # 错误处理
    def should_end(state: QueryState) -> bool:
        return state.get("status") == "error"
        
    workflow.add_conditional_edges(
        "normalize",
        should_end,
        {
            True: END,
            False: "strategy"
        }
    )
    
    workflow.add_conditional_edges(
        "strategy",
        should_end,
        {
            True: END,
            False: "execute"
        }
    )
    
    workflow.add_conditional_edges(
        "execute",
        should_end,
        {
            True: END,
            False: "validate"
        }
    )
    
    return workflow.compile() 