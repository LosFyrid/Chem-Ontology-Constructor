from typing import TypedDict, List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from .ontology_tools import OntologyTools
from .utils import (
    parse_json,
    format_owlready2_value,
    format_sparql_results,
    format_query_results,
    format_class_info,
    format_restrictions,
    format_hierarchy
)

class QueryState(TypedDict):
    """Query state that supports interaction with other subgraphs"""
    # Input fields
    query: str  # Query from other agents
    requester: str  # Which agent made the query
    context: Dict[str, Any]  # Context from requester
    
    # Processing fields
    query_type: str  # "atomic" or "sparql"
    atomic_plan: List[Dict]  # Execution plan
    sparql_query: str  # SPARQL query
    
    # Output fields
    results: Dict  # Query results
    messages: List[BaseMessage]  # System messages
    status: str  # "processing", "success", "error"
    error: str  # Error message if any
    done: bool

def create_tool_executor(ontology) -> Dict[str, ToolExecutor]:
    """Create tool executor with ontology tools"""
    tools_instance = OntologyTools(ontology)
    return {
        "ontology_tools": ToolExecutor(
            tools=[
                tools_instance.get_class_info,
                tools_instance.get_information_sources,
                tools_instance.get_information_by_source,
                tools_instance.get_class_properties,
                tools_instance.get_property_restrictions,
                tools_instance.get_property_values,
                tools_instance.get_parents,
                tools_instance.get_children,
                tools_instance.get_ancestors,
                tools_instance.get_descendants,
                tools_instance.get_related_classes,
                tools_instance.get_property_path,
                tools_instance.get_semantic_similarity,
                tools_instance.parse_class_definition,
                tools_instance.parse_property_definition,
                tools_instance.parse_hierarchy_structure
            ]
        )
    }

def create_query_graph(ontology) -> Graph:
    """Create query subgraph that can be used by other agents"""
    workflow = StateGraph(QueryState)
    tools = create_tool_executor(ontology)
    
    # 1. Query Analysis Node
    def analyze_query(state: QueryState) -> QueryState:
        """Analyze query type and generate execution plan"""
        try:
            llm = ChatOpenAI(temperature=0)
            
            # Analyze query type
            type_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert in ontology querying.
                Analyze if the query can be handled using atomic operations or needs SPARQL.
                
                Available atomic operations:
                - Basic info: get_class_info, get_information_sources
                - Properties: get_class_properties, get_property_values
                - Hierarchy: get_parents, get_children, get_ancestors
                - Semantic: get_related_classes, get_semantic_similarity
                - Parsing: parse_class_definition, parse_property_definition
                """),
                ("user", "Query: {query}\nRespond with 'atomic' or 'sparql'")
            ])
            
            response = llm.invoke(type_prompt.format_messages(query=state["query"]))
            query_type = response.content.strip().lower()
            state["query_type"] = query_type
            
            if query_type == "atomic":
                state["atomic_plan"] = _generate_atomic_plan(state["query"], llm)
            else:
                state["sparql_query"] = _generate_sparql_query(state["query"], llm)
                
            return state
        except Exception as e:
            state["messages"].append(f"Error in query analysis: {str(e)}")
            state["done"] = True
            return state
    
    # 2. Atomic Query Node
    def atomic_query(state: QueryState) -> QueryState:
        """Execute atomic operations query"""
        try:
            results = {}
            for step in state["atomic_plan"]:
                try:
                    result = tools["ontology_tools"].invoke(
                        step["operation"],
                        **step["args"]
                    )
                    results[f"step_{len(results)}"] = {
                        "operation": step["operation"],
                        "result": result,
                        "description": step["description"]
                    }
                except Exception as e:
                    results[f"step_{len(results)}"] = {
                        "operation": step["operation"],
                        "error": str(e)
                    }
            state["results"] = results
            return state
        except Exception as e:
            state["messages"].append(f"Error in atomic query: {str(e)}")
            state["done"] = True
            return state
    
    # 3. SPARQL Query Node
    def sparql_query(state: QueryState) -> QueryState:
        """Execute SPARQL query"""
        try:
            results = list(tools["ontology_tools"].onto.world.sparql(
                state["sparql_query"]
            ))
            state["results"] = format_sparql_results(results)
            return state
        except Exception as e:
            state["messages"].append(f"Error in SPARQL query: {str(e)}")
            state["done"] = True
            return state
    
    # 4. Result Formatting Node
    def format_results(state: QueryState) -> QueryState:
        """Format query results"""
        try:
            llm = ChatOpenAI(temperature=0)
            format_prompt = ChatPromptTemplate.from_messages([
                ("system", "Format the query results in a clear and readable way."),
                ("user", """
                Original query: {query}
                Query type: {query_type}
                Results: {results}
                
                Format these results in a clear, structured way that directly answers the query.
                """)
            ])
            
            response = llm.invoke(format_prompt.format_messages(
                query=state["query"],
                query_type=state["query_type"],
                results=state["results"]
            ))
            
            state["results"] = {
                "raw": state["results"],
                "formatted": response.content
            }
            state["done"] = True
            return state
        except Exception as e:
            state["messages"].append(f"Error in result formatting: {str(e)}")
            state["done"] = True
            return state
    
    # Helper functions
    def _generate_atomic_plan(query: str, llm: ChatOpenAI) -> List[Dict]:
        plan_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a plan using atomic operations to answer the query."),
            ("user", """Query: {query}
            
            Return a JSON list of operations, each with:
            - operation: tool name
            - args: arguments
            - description: why this operation is needed
            """)
        ])
        
        plan_response = llm.invoke(plan_prompt.format_messages(query=query))
        return parse_json(plan_response.content)
    
    def _generate_sparql_query(query: str, llm: ChatOpenAI) -> str:
        sparql_prompt = ChatPromptTemplate.from_messages([
            ("system", """Convert the query to SPARQL.
            Remember owlready2's SPARQL requirements:
            1. No prefix definitions needed
            2. Variables must start with '?'
            3. Use basic SPARQL 1.1 features
            """),
            ("user", "Query: {query}")
        ])
        
        sparql_response = llm.invoke(sparql_prompt.format_messages(query=query))
        return sparql_response.content.strip()
    
    # Build workflow
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("atomic", atomic_query)
    workflow.add_node("sparql", sparql_query)
    workflow.add_node("format", format_results)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "analyze",
        lambda state: state["query_type"],
        {
            "atomic": "atomic",
            "sparql": "sparql"
        }
    )
    
    # Add other edges
    workflow.add_edge("atomic", "format")
    workflow.add_edge("sparql", "format")
    
    # Set entry and exit points
    workflow.set_entry_point("analyze")
    workflow.set_finish_point("format")
    
    return workflow.compile() 