from typing import Dict, List, TypedDict, Literal, Annotated, Optional, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from .ontology_tools import OntologyTools
from autology_constructor.idea.state_manager import StateManager, QueryState
import json



def create_query_graph() -> Graph:
    """创建查询工作流"""
    workflow = StateGraph(QueryState)
    tools = OntologyTools(None)  # 初始化工具
    llm = ChatOpenAI(temperature=0)
    
    def execute_query(state: QueryState) -> Dict:
        """执行查询"""
        try:
            current_query = state["current_query"]
            natural_query = current_query["request_natural_language"]
            tools.onto = state["ontology"]  # 设置当前本体
            
            # 获取可用的本体类列表，用于辅助自然语言转换
            available = [cls.name for cls in tools.onto.classes()]
            available_classes = ", ".join(available)
            
            # Process natural language query into standardized JSON query description with ontology context
            standardized_query = _process_natural_language(natural_query, available_classes)
            current_query["request_processed"] = standardized_query
            
            # 根据标准化描述判断查询策略
            strategy = _decide_query_strategy(json.dumps(standardized_query, ensure_ascii=False))
            current_query["query_strategy_chosen"] = strategy

            results = {}
            if strategy == "tool_based":
                # Generate query plan (sequence of tool function calls) based on the standardized query
                tool_plan = _generate_query_plan(standardized_query)
                current_query["query_plan_tool"] = tool_plan
                aggregated_results = []
                for step in tool_plan:
                    tool_name = step.get("tool")
                    params = step.get("params", {})
                    tool_func = getattr(tools, tool_name, None)
                    if tool_func is None:
                        aggregated_results.append({"step": step, "result": f"Tool {tool_name} not found."})
                    else:
                        result = tool_func(**params)
                        aggregated_results.append({"step": step, "result": result})
                results = aggregated_results
            elif strategy == "sparql":
                sparql_query = _generate_sparql_query(json.dumps(standardized_query, ensure_ascii=False))
                current_query["query_sparql"] = sparql_query
                # 执行SPARQL查询，此处模拟返回结果
                results = {"sparql_result": "模拟SPARQL查询结果"}
            
            current_query["raw_query_output"] = results
            processed_response = _post_process_results(results)
            current_query["processed_response"] = processed_response
            
            return {
                "status": "success",
                "stage": "querying",
                "previous_stage": state.get("stage"),
                "messages": [f"Query executed successfully: {json.dumps(standardized_query, ensure_ascii=False)}"]
            }
        except Exception as e:
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": str(e),
                "messages": [f"Query execution failed: {str(e)}"]
            }
    
    def _parse_query_type(query: str) -> str:
        """解析查询类型"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Determine the type of ontology query."),
            ("user", """
            Query: {query}
            
            Classify as one of:
            - class_info: basic class information
            - property_info: property details
            - hierarchy: class hierarchy
            - semantic: semantic relationships
            - domain_analysis: complete domain analysis
            
            Return only the type.
            """)
        ])
        
        response = llm.invoke(prompt.format_messages(query=query))
        return response.content.strip().lower()
    
    def _extract_class_name(query: str) -> str:
        """从查询中提取类名"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract the class name from the query."),
            ("user", "Extract the main class name from: {query}")
        ])
        
        response = llm.invoke(prompt.format_messages(query=query))
        return response.content.strip()
    
    def _extract_property_name(query: str) -> str:
        """从查询中提取属性名"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract the property name from the query."),
            ("user", "Extract the property name from: {query}")
        ])
        
        response = llm.invoke(prompt.format_messages(query=query))
        return response.content.strip()
    
    def validate_results(state: QueryState) -> Dict:
        """验证查询结果"""
        try:
            results = state["query_results"]
            
            # 验证结果完整性
            if not results:
                raise ValueError("Empty query results")
                
            # 验证结果格式
            if not isinstance(results, dict):
                raise ValueError("Invalid results format")
                
            return {
                "status": "success",
                "stage": "validated",
                "previous_stage": state.get("stage"),
                "messages": ["Results validation successful"]
            }
        except Exception as e:
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": str(e),
                "messages": [f"Results validation failed: {str(e)}"]
            }
    
    # 新增辅助函数，修改 _process_natural_language 接受 available_classes
    def _process_natural_language(nl: str, available_classes: str) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Convert the natural language query into a standardized JSON query description. The JSON must strictly follow this schema: {\"query_type\": string (one of 'class_info', 'property_info', 'hierarchy', 'semantic', 'domain_analysis'), \"class_name\": string, \"property_name\": string}. If a field is not applicable, set it to an empty string. Consider the following available ontology classes: " + available_classes),
            ("user", "Please convert the following natural language query into a JSON formatted query description strictly following the schema above:\n{nl}")
        ])
        response = llm.invoke(prompt.format_messages(nl=nl))
        try:
            return json.loads(response.content.strip())
        except Exception:
            raise ValueError("Failed to parse standardized JSON query description.")

    def _decide_query_strategy(processed_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Select the appropriate query strategy based on the query content."),
            ("user", "Based on the standardized query description below, decide the query strategy. Answer with only 'tool_based' or 'sparql':\n{processed_query}")
        ])
        response = llm.invoke(prompt.format_messages(processed_query=processed_query))
        strategy = response.content.strip().lower()
        if strategy not in ["tool_based", "sparql"]:
            strategy = "tool_based"
        return strategy

    def _generate_query_plan(standardized_query: dict) -> list:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Based on the standardized JSON query description provided, generate a sequence of tool function calls to answer the query. Each step should be a JSON object with keys 'tool' and 'params'. The available tools are: get_class_info, get_class_properties, get_property_restrictions, get_parents, get_children, get_ancestors, get_descendants, get_related_classes, get_disjoint_classes, parse_hierarchy_structure. Answer with a JSON array representing the sequence of steps. Only output the JSON, nothing else."),
            ("user", "Standardized query description:\n{standardized_query}")
        ])
        response = llm.invoke(prompt.format_messages(standardized_query=json.dumps(standardized_query, ensure_ascii=False)))
        try:
            return json.loads(response.content.strip())
        except Exception:
            raise ValueError("Failed to parse query plan from JSON.")

    def _generate_sparql_query(processed_query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate the corresponding SPARQL query for the given standardized query description."),
            ("user", "Please generate the SPARQL query for the following standardized query description:\n{processed_query}")
        ])
        response = llm.invoke(prompt.format_messages(processed_query=processed_query))
        return response.content.strip()

    def _post_process_results(raw: Any) -> str:
        import json
        try:
            return json.dumps(raw, ensure_ascii=False)
        except Exception:
            return str(raw)
    
    # 添加节点
    workflow.add_node("execute", execute_query)
    workflow.add_node("validate", validate_results)
    
    # 添加边
    workflow.add_edge(START, "execute")
    workflow.add_edge("execute", "validate")
    workflow.add_edge("validate", END)
    
    return workflow.compile() 