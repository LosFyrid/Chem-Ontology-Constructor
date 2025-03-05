from typing import Dict, List, TypedDict, Literal, Annotated, Optional, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from .ontology_tools import OntologyTools

class QueryState(TypedDict):
    """查询团队状态"""
    # Input
    query: str
    source_ontology: Any  # 使用主工作流中的source_ontology
    
    # Query Management
    query_results: Dict
    status: str
    stage: str  # 添加stage字段
    previous_stage: Optional[str]  # 添加previous_stage字段
    
    # System
    messages: Annotated[list[AnyMessage], add_messages]

def create_query_graph() -> Graph:
    """创建查询工作流"""
    workflow = StateGraph(QueryState)
    tools = OntologyTools(None)  # 初始化工具
    llm = ChatOpenAI(temperature=0)
    
    def execute_query(state: QueryState) -> Dict:
        """执行查询"""
        try:
            query = state["query"]
            source_ontology = state["source_ontology"]
            tools.onto = source_ontology  # 设置当前本体
            
            # 解析查询类型
            query_type = _parse_query_type(query)
            
            # 根据查询类型执行相应的查询
            results = {}
            if query_type == "class_info":
                class_name = _extract_class_name(query)
                results = tools.get_class_info(class_name)
            elif query_type == "property_info":
                class_name = _extract_class_name(query)
                prop_name = _extract_property_name(query)
                results = {
                    "properties": tools.get_class_properties(class_name),
                    "restrictions": tools.get_property_restrictions(class_name, prop_name)
                }
            elif query_type == "hierarchy":
                class_name = _extract_class_name(query)
                results = {
                    "parents": tools.get_parents(class_name),
                    "children": tools.get_children(class_name),
                    "ancestors": tools.get_ancestors(class_name),
                    "descendants": tools.get_descendants(class_name)
                }
            elif query_type == "semantic":
                class_name = _extract_class_name(query)
                results = {
                    "related_classes": tools.get_related_classes(class_name),
                    "disjoint_classes": tools.get_disjoint_classes(class_name)
                }
            elif query_type == "domain_analysis":
                results = tools.parse_hierarchy_structure()
            
            return {
                "query_results": results,
                "status": "success",
                "stage": "querying",
                "previous_stage": state.get("stage"),
                "messages": [f"Query executed successfully: {query}"]
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
    
    # 添加节点
    workflow.add_node("execute", execute_query)
    workflow.add_node("validate", validate_results)
    
    # 添加边
    workflow.add_edge(START, "execute")
    workflow.add_edge("execute", "validate")
    workflow.add_edge("validate", END)
    
    return workflow.compile() 