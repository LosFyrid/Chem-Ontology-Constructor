from typing import Dict, List, Literal, Optional, Any, Union
from typing_extensions import Annotated, TypedDict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from .ontology_tools import OntologyTools, SparqlExecutionError
from .query_agents import QueryParserAgent, StrategyPlannerAgent, ToolPlannerAgent, ToolExecutorAgent, SparqlExpertAgent, ValidationAgent
from .query_manager import Query, QueryStatus
from .utils import format_sparql_error, format_sparql_results, extract_variables_from_sparql
from .schemas import NormalizedQuery, ToolPlan, ValidationReport
from autology_constructor.idea.common.llm_provider import get_cached_default_llm

class QueryState(TypedDict):
    """查询团队状态"""
    # Input
    query: str  # 自然语言查询
    source_ontology: Any  # 使用主工作流中的source_ontology
    query_type: str  # 查询类型
    query_strategy: Optional[Literal["tool_sequence", "SPARQL"]]  # 查询策略
    additional_ontology: Optional[Any]  # 额外本体（用于跨域查询）
    originating_team: str  # 发起查询的团队
    originating_stage: str  # 发起查询的阶段
    available_classes: List[str]  # Add available classes from cache
    
    # Query Management
    query_results: Dict  # 查询结果
    normalized_query: Optional[NormalizedQuery]  # 标准化的查询结构
    execution_plan: Optional[ToolPlan]  # 执行计划
    validation_report: Optional[ValidationReport]  # Added field for report
    sparql_query: Optional[str]  # Generated SPARQL query
    status: str  # 状态
    stage: str  # 当前阶段
    previous_stage: Optional[str]  # 上一阶段
    error: Optional[str]  # Add error field for better tracking
    
    # System
    messages: Annotated[list[AnyMessage], add_messages]

def create_query_graph() -> Graph:
    """创建查询工作流"""

    workflow = StateGraph(QueryState)

    # Get the default LLM instance
    try:
        default_model = get_cached_default_llm()
    except Exception as e:
        print(f"Critical Error: Failed to initialize default LLM: {e}")
        # Decide how to handle this - maybe raise the error or use a fallback
        raise RuntimeError("LLM initialization failed, cannot create query graph.") from e

    # Instantiate agents with the default model
    parser_agent = QueryParserAgent(model=default_model)
    strategy_agent = StrategyPlannerAgent(model=default_model)
    tool_planner_agent = ToolPlannerAgent(model=default_model)
    tool_agent = ToolExecutorAgent(model=default_model)
    sparql_agent = SparqlExpertAgent(model=default_model)
    validator_agent = ValidationAgent(model=default_model)
    ontology_tools_instance = OntologyTools(None)
    
    # 节点实现
    def normalize_query(state: QueryState) -> Dict:
        """解析并标准化查询，使用缓存的类名"""
        try:
            query = state["query"]
            available_classes = state["available_classes"]
            # Prepare state for parser agent, including available classes
            parser_state = {
                "natural_query": query,
                "available_classes": available_classes
            }
            # Use parser agent
            normalized_result = parser_agent(parser_state)
            # Check if parsing resulted in an error reported by the agent
            if isinstance(normalized_result, dict) and normalized_result.get("error"):
                    raise ValueError(f"Query parsing failed: {normalized_result.get('error')}")
            elif not isinstance(normalized_result, NormalizedQuery):
                    # Should not happen if agent works correctly, but good to check
                    raise TypeError(f"Query parser returned unexpected type: {type(normalized_result)}")
            return {
                "normalized_query": normalized_result,
                "status": "parsing_complete",
                "stage": "normalized",
                "previous_stage": state.get("stage"),
                "messages": [SystemMessage(content=f"Query normalized: {query}")]
            }
        except Exception as e:
            error_message = f"Query normalization failed: {str(e)}"
            print(error_message)
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": error_message,
                "messages": [SystemMessage(content=error_message)]
            }
    
    def determine_strategy(state: QueryState) -> Dict:
        """确定查询执行策略"""
        try:
            # If strategy is already provided (e.g., via context), use it.
            # Otherwise, use the strategy agent.
            strategy = state.get("query_strategy")
            if not strategy:
                normalized_query_obj = state.get("normalized_query")
                if not normalized_query_obj or not isinstance(normalized_query_obj, NormalizedQuery):
                    raise ValueError("NormalizedQuery object is missing or invalid, cannot determine strategy.")
                
                # Use strategy planner agent
                strategy = strategy_agent.decide_strategy(normalized_query_obj.model_dump())
                # Basic validation of strategy output
                if strategy not in ["tool_sequence", "SPARQL"]:
                     print(f"Warning: Strategy agent returned unsupported strategy '{strategy}'. Defaulting to tool_sequence.")
                     strategy = "tool_sequence"

            return {
                "query_strategy": strategy,
                "status": "strategy_determined",
                "stage": "strategy",
                "previous_stage": state.get("stage"),
                "messages": [SystemMessage(content=f"Query strategy determined: {strategy}")]
            }
        except Exception as e:
            error_message = f"Strategy determination failed: {str(e)}"
            print(error_message)
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": error_message,
                "messages": [SystemMessage(content=error_message)]
            }
    
    def execute_query(state: QueryState) -> Dict:
        """执行查询 (工具序列或SPARQL)"""
        try:
            strategy = state.get("query_strategy")
            normalized_query_obj = state["normalized_query"]
            source_ontology = state["source_ontology"]

            if not strategy or not source_ontology or not isinstance(normalized_query_obj, NormalizedQuery):
                 raise ValueError("Missing strategy, source ontology, or invalid NormalizedQuery object for execution.")

            # Ensure the ontology tools instance has the correct ontology
            ontology_tools_instance.onto = source_ontology
            # Also set ontology for the ToolExecutorAgent's internal tools reference
            tool_agent.tools = ontology_tools_instance

            if strategy == "tool_sequence":
                # Generate execution plan using the new ToolPlannerAgent
                plan_result = tool_planner_agent.generate_plan(normalized_query_obj, ontology_tools_instance)
                
                # Check if plan generation resulted in an error
                if isinstance(plan_result, dict) and plan_result.get("error"):
                    raise ValueError(f"Failed to generate tool plan: {plan_result.get('error')}")
                elif not isinstance(plan_result, Union[ToolPlan, Dict]):
                     raise TypeError(f"Tool planner returned unexpected type: {type(plan_result)}")
                
                # Execute the plan using ToolExecutorAgent
                execution_results = tool_agent.execute_plan(plan_result, source_ontology) # Pass ontology just in case

                # Check for errors during execution
                execution_errors = [step["error"] for step in execution_results if "error" in step]
                if execution_errors:
                    print(f"Errors during tool execution: {execution_errors}")
                    # Decide how to handle partial success/failure - here we just store all results

                return {
                    "execution_plan": plan_result,
                    "query_results": {"results": execution_results}, # Wrap tool results for consistency
                    "status": "executed",
                    "stage": "executed",
                    "previous_stage": state.get("stage"),
                    "messages": [SystemMessage(content="Tool-based query executed.")]
                }
            elif strategy == "SPARQL":
                # Generate SPARQL query using SparqlExpertAgent
                sparql_query_str = sparql_agent.generate_sparql(normalized_query_obj.model_dump())

                # Execute SPARQL using the robust OntologyTools.execute_sparql
                results = ontology_tools_instance.execute_sparql(sparql_query_str)
                
                # Check for errors returned by execute_sparql
                if isinstance(results, dict) and results.get("error"):
                    raise SparqlExecutionError(f"SPARQL execution failed: {results.get('error')}. Query: {results.get('query')}")

                return {
                    "query_results": results, # Already formatted by execute_sparql
                    "sparql_query": sparql_query_str,
                    "execution_plan": None, # Explicitly set plan to None for SPARQL path
                    "status": "executed",
                    "stage": "executed",
                    "previous_stage": state.get("stage"),
                    "messages": [SystemMessage(content="SPARQL query executed successfully.")]
                }
            else:
                raise ValueError(f"Unsupported query strategy: {strategy}")

        except Exception as e:
            error_message = f"Query execution failed: {str(e)}"
            print(error_message)
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": error_message,
                "messages": [SystemMessage(content=error_message)]
            }
    
    def validate_results(state: QueryState) -> Dict:
        """验证查询结果"""
        try:
            results_to_validate = state.get("query_results")
            normalized_query_obj = state.get("normalized_query")

            if not results_to_validate or not isinstance(results_to_validate, dict):
                 print("Warning: Skipping validation due to missing or malformed results.")
                 return {"status": state.get("status", "executed"), "stage": "validated", "validation_report": None}

            if results_to_validate.get("error"):
                 print(f"Skipping validation because previous step failed: {results_to_validate.get('error')}")
                 # Propagate error status if validation is reached after an error
                 return { 
                    "status": "error", 
                    "stage": "validation_skipped_due_to_error",
                    "error": results_to_validate.get("error"),
                    "validation_report": None, # Set report to None on error skip
                    "previous_stage": state.get("stage"),
                    "messages": [SystemMessage(content="Validation skipped due to prior error.")]
                 }

            # Prepare query context for validation agent
            query_context = {}
            if isinstance(normalized_query_obj, NormalizedQuery):
                 query_context = {
                     "intent": normalized_query_obj.intent,
                     "target": ", ".join(normalized_query_obj.target_entities),
                     # Add more context from normalized_query if needed
                 }
            query_context["query"] = state.get("query")
            query_context["type"] = state.get("query_type", "unknown")
            query_context["strategy"] = state.get("query_strategy")

            # Use validation agent
            validation_result = validator_agent.validate(results_to_validate, query_context)

            if isinstance(validation_result, dict) and validation_result.get("error"):
                 raise ValueError(f"Validation agent failed: {validation_result.get('error')}")
            elif not isinstance(validation_result, ValidationReport):
                 raise TypeError(f"Validation agent returned unexpected type: {type(validation_result)}")

            # Determine final status based on validation
            final_status = "success" if validation_result.valid else "warning"
            validation_message = validation_result.message

            return {
                "validation_report": validation_result,
                "status": final_status,
                "stage": "validated",
                "previous_stage": state.get("stage"),
                "messages": [SystemMessage(content=f"Results validation {final_status}: {validation_message}")]
            }
        except Exception as e:
            error_message = f"Results validation failed: {str(e)}"
            print(error_message)
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": error_message,
                "validation_report": None, # Ensure report is None on error
                "messages": [SystemMessage(content=error_message)]
            }
    
    # Add nodes
    workflow.add_node("normalize", normalize_query)
    workflow.add_node("strategy", determine_strategy)
    workflow.add_node("execute", execute_query)
    workflow.add_node("validate", validate_results)
    
    # Define conditional edges for error handling and branching
    def decide_next_node(state: QueryState):
        if state.get("status") == "error":
            print(f"Workflow ending due to error at stage: {state.get('stage')}, Error: {state.get('error')}")
            return END
        current_stage = state.get("stage")
        if current_stage == "normalized":
            return "strategy"
        elif current_stage == "strategy":
            return "execute"
        elif current_stage == "executed":
             return "validate"
        elif current_stage == "validated":
             return END
        # Add a fallback or default case if needed, though END is often suitable
        print(f"Warning: Unexpected state '{current_stage}' reached. Ending workflow.")
        return END
    
    # Add edges using the conditional logic
    workflow.add_edge(START, "normalize")
    workflow.add_conditional_edges("normalize", decide_next_node)
    workflow.add_conditional_edges("strategy", decide_next_node)
    workflow.add_conditional_edges("execute", decide_next_node)
    workflow.add_conditional_edges("validate", decide_next_node)
    
    # Compile the graph (ensure this is done correctly)
    # Consider compiling outside if graph needs further modification (e.g., persistence)
    compiled_graph = workflow.compile()
    return compiled_graph 