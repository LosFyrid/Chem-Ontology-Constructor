from typing import Dict, List, Literal, Optional, Any, Union
from typing_extensions import Annotated, TypedDict
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from .ontology_tools import OntologyTools, SparqlExecutionError
from .query_agents import QueryParserAgent, StrategyPlannerAgent, ToolPlannerAgent, ToolExecutorAgent, SparqlExpertAgent, ValidationAgent, HypotheticalDocumentAgent, ResultFormatterAgent
from .query_manager import Query, QueryStatus
from .utils import format_sparql_error, format_sparql_results, extract_variables_from_sparql
from .schemas import NormalizedQuery, ToolPlan, ValidationReport
from autology_constructor.idea.common.llm_provider import get_cached_default_llm
from config.settings import OntologySettings

class QueryState(TypedDict):
    """查询团队状态"""
    # Input
    query: str  # 自然语言查询
    source_ontology: OntologySettings  # 使用OntologySettings类型，而不是Any
    query_type: str  # 查询类型
    query_strategy: Optional[Literal["tool_sequence", "SPARQL"]]  # 查询策略
    originating_team: str  # 发起查询的团队
    originating_stage: str  # 发起查询的阶段
    available_classes: List[str]  # Add available classes from cache
    available_data_properties: List[str]  
    available_object_properties: List[str]
    
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
    
    # Retry and Feedback
    retry_count: Optional[int]  # 重试计数
    force_strategy: Optional[str]  # 强制使用不同的策略
    hypothetical_document: Optional[Dict]  # 假设性文档（由化学专家生成）
    validation_history: Optional[List]  # 验证报告历史
    formatted_results: Optional[Dict]  # 格式化后的结果
    iteration_history: Optional[List[Dict]] # ADD: To store history of each iteration
    
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
    hypothetical_document_agent = HypotheticalDocumentAgent(model=default_model)
    result_formatter_agent = ResultFormatterAgent(model=default_model)
    
    # 节点实现
    def normalize_query(state: QueryState) -> Dict:
        """解析并标准化查询，使用缓存的类名"""
        retry_count = state.get("retry_count",0)
        try:
            query = state["query"]
            available_classes = state["available_classes"]
            available_data_properties = state["available_data_properties"]
            available_object_properties = state["available_object_properties"]
            # Prepare state for parser agent, including available classes
            if state.get("validation_report"):
                enhanced_feedback = getattr(state.get("validation_report"), "improvement_suggestions")
            else:
                enhanced_feedback = None
            parser_state = {
                "natural_query": query,
                "available_classes": available_classes,
                "available_data_properties": available_data_properties,
                "available_object_properties": available_object_properties,
                "enhanced_feedback": enhanced_feedback,
                "hypothetical_document": state.get("hypothetical_document")
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
                "messages": [SystemMessage(content=f"Query normalized: {query}")],
                "retry_count": retry_count + 1
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
            ontology_settings = state["source_ontology"]

            if not strategy or not ontology_settings or not isinstance(normalized_query_obj, NormalizedQuery):
                 raise ValueError("Missing strategy, ontology settings, or invalid NormalizedQuery object.")

            # 验证ontology_settings类型
            if not isinstance(ontology_settings, OntologySettings):
                raise TypeError(f"source_ontology must be an OntologySettings instance, got {type(ontology_settings).__name__}")

            # 创建OntologyTools实例
            ontology_tools = OntologyTools(ontology_settings)
            
            # 设置工具代理的OntologyTools实例
            tool_agent.set_ontology_tools(ontology_tools)
            
            if strategy == "tool_sequence":
                # 生成执行计划
                plan_result = tool_planner_agent.generate_plan(normalized_query_obj, ontology_tools)
                
                # 检查计划生成是否出错
                if isinstance(plan_result, dict) and plan_result.get("error"):
                    raise ValueError(f"Failed to generate tool plan: {plan_result.get('error')}")
                elif not isinstance(plan_result, Union[ToolPlan, Dict]):
                     raise TypeError(f"Tool planner returned unexpected type: {type(plan_result)}")
                
                # 执行计划 - 已经在上面设置了OntologyTools
                execution_results = tool_agent.execute_plan(plan_result)

                # 处理结果
                return {
                    "execution_plan": plan_result,
                    "query_results": {"results": execution_results}, # Wrap tool results for consistency
                    "status": "executed",
                    "stage": "executed",
                    "previous_stage": state.get("stage"),
                    "messages": [SystemMessage(content="Tool-based query executed.")]
                }
                
            elif strategy == "SPARQL":
                # 生成SPARQL查询
                sparql_query_str = sparql_agent.generate_sparql(normalized_query_obj.model_dump())

                # 使用创建的OntologyTools实例执行SPARQL
                results = ontology_tools.execute_sparql(sparql_query_str)
                
                # 错误检查
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
        """验证查询结果并记录迭代历史"""
        try:
            results_to_validate = state.get("query_results")
            normalized_query_obj = state.get("normalized_query")

            # Initialize or retrieve history
            iteration_history = state.get("iteration_history", [])

            if not results_to_validate or not isinstance(results_to_validate, dict):
                print("Warning: Skipping validation due to missing or malformed results.")
                return {"status": state.get("status", "executed"), "stage": "validated", "validation_report": None, "iteration_history": iteration_history}

            if results_to_validate.get("error"):
                print(f"Skipping validation because previous step failed: {results_to_validate.get('error')}")
                return {
                    "status": "error",
                    "stage": "validation_skipped_due_to_error",
                    "error": results_to_validate.get("error"),
                    "validation_report": None,
                    "previous_stage": state.get("stage"),
                    "messages": [SystemMessage(content="Validation skipped due to prior error.")],
                    "iteration_history": iteration_history
                }

            # Prepare query context for validation agent
            query_context = {}
            if isinstance(normalized_query_obj, NormalizedQuery):
                query_context = {
                    "intent": normalized_query_obj.intent,
                    "relevant_entities": ", ".join(normalized_query_obj.relevant_entities),
                    "relevant_properties": ", ".join(normalized_query_obj.relevant_properties),
                }
            query_context["query"] = state.get("query")
            query_context["type"] = state.get("query_type", "unknown")
            query_context["strategy"] = state.get("query_strategy")

            # Use validation agent
            validation_result = validator_agent.validate(results_to_validate, query_context)

            if isinstance(validation_result, dict) and validation_result.get("error"):
                print(f"Validation Report: {validation_result.get('validation_report')}")
                raise ValueError(f"Validation agent failed: {validation_result.get('error')}")
            elif not isinstance(validation_result, ValidationReport):
                raise TypeError(f"Validation agent returned unexpected type: {type(validation_result)}")

            # Create a snapshot of the current iteration
            current_iteration_snapshot = {
                "retry_count": state.get("retry_count", 0),
                "hypothetical_document": state.get("hypothetical_document"),
                "normalized_query": state.get("normalized_query"),
                "query_strategy": state.get("query_strategy"),
                "execution_plan": state.get("execution_plan"),
                "sparql_query": state.get("sparql_query"),
                "query_results": state.get("query_results"),
                "validation_report": validation_result,
                "timestamp": datetime.now().isoformat(),
                "messages": state.get("messages")
            }
            iteration_history.append(current_iteration_snapshot)

            # Determine final status based on validation
            final_status = "success" if validation_result.valid else "warning"
            validation_message = validation_result.message

            return {
                "validation_report": validation_result,
                "status": final_status,
                "stage": "validated",
                "previous_stage": state.get("stage"),
                "messages": [SystemMessage(content=f"Results validation {final_status}: {validation_message}")],
                "iteration_history": iteration_history  # Pass the updated history
            }
        except Exception as e:
            error_message = f"Results validation failed: {str(e)}"
            print(error_message)
            # Also update history on error if possible
            iteration_history = state.get("iteration_history", [])
            iteration_history.append({
                "error": error_message,
                "stage": "validate_results",
                "timestamp": datetime.now().isoformat()
            })
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": error_message,
                "validation_report": None,
                "messages": [SystemMessage(content=error_message)],
                "iteration_history": iteration_history
            }
    
    def generate_hypothetical_document(state: QueryState) -> Dict:
        """从专业化学家角度生成假设性答案，帮助查询标准化"""
        try:
            query = state.get("query")
            validation_history = state.get("validation_history", [])
            
            if not query:
                raise ValueError("Cannot generate hypothetical document: query is missing")
            
            # 使用HypotheticalDocumentAgent生成假设性文档
            hypothetical_doc = hypothetical_document_agent.generate_hypothetical_document(
                query=query, 
                validation_history=validation_history
            )
            
            # 更新状态
            return {
                "hypothetical_document": hypothetical_doc,
                "status": "hypothetical_generated",
                "stage": "hypothetical_generated",
                "previous_stage": state.get("stage"),
                "messages": [SystemMessage(content=f"Generated hypothetical document to aid in query understanding: {hypothetical_doc.get('interpretation', '')[:100]}...")]
            }
        except Exception as e:
            error_message = f"Hypothetical document generation failed: {str(e)}"
            print(error_message)
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": error_message,
                "messages": [SystemMessage(content=error_message)]
            }
    
    def format_results(state: QueryState) -> Dict:
        """格式化查询结果为用户友好的形式"""
        try:
            query = state.get("query")
            results = state.get("query_results")
            normalized_query_obj = state.get("normalized_query")
            
            if not query or not results:
                raise ValueError("Cannot format results: query or results is missing")
            
            # 准备query_context
            query_context = {}
            if isinstance(normalized_query_obj, NormalizedQuery):
                query_context = {
                    "intent": normalized_query_obj.intent,
                    "relevant_entities": ", ".join(normalized_query_obj.relevant_entities),
                    "relevant_properties": ", ".join(normalized_query_obj.relevant_properties),
                }
            
            # 使用ResultFormatterAgent格式化结果
            formatted_results = result_formatter_agent.format_results(
                query=query,
                results=results,
                query_context=query_context
            )
            
            # 更新状态
            return {
                "formatted_results": formatted_results,
                "status": "completed",
                "stage": "completed",
                "previous_stage": state.get("stage"),
                "messages": [SystemMessage(content=f"Results formatted: {formatted_results.get('summary', '')}")]
            }
        except Exception as e:
            error_message = f"Results formatting failed: {str(e)}"
            print(error_message)
            return {
                "status": "error",
                "stage": "error",
                "previous_stage": state.get("stage"),
                "error": error_message,
                "messages": [SystemMessage(content=error_message)]
            }
    
    # Add nodes
    workflow.add_node("normalize", normalize_query)
    workflow.add_node("strategy", determine_strategy)
    workflow.add_node("execute", execute_query)
    workflow.add_node("validate", validate_results)
    workflow.add_node("generate_hypothetical_document", generate_hypothetical_document)  # 新增节点
    workflow.add_node("format_results", format_results)  # 新增节点
    
    # Define conditional edges for error handling and branching
    def decide_next_node(state: QueryState):
        # 检查是否存在错误状态
        if state.get("status") == "error":
            print(f"Workflow ending due to error at stage: {state.get('stage')}, Error: {state.get('error')}")
            return END

        # 获取当前阶段和重试计数
        current_stage = state.get("stage")
        current_state = state.get("status")
        retry_count = state.get("retry_count", 0)
        print(f"Retry count: {retry_count}")
        # 对于验证反馈阶段，实现重试逻辑
        if current_stage == "validated" and current_state == "warning":
            print(f"In retry logic, Retry count: {retry_count}")
            if retry_count <= 2:
                # 前两次重试，回到标准化阶段
                return "normalize"
            elif retry_count == 3:
                # 第三次重试，尝试假设性文档生成
                return "generate_hypothetical_document"
            else:
                # 超出重试次数，终止工作流
                print(f"Exceeded maximum retry attempts ({retry_count})")
                return "format_results"

        # 标准流程节点决策
        if current_stage == "normalized":
            return "strategy"
        elif current_stage == "strategy":
            return "execute"
        elif current_stage == "executed":
            return "validate"
        elif current_stage == "validated":
            return "format_results"
        elif current_stage == "hypothetical_generated":
            return "normalize"  # 生成假设性文档后返回到标准化阶段
        elif current_stage == "completed":
            return END

        # 默认情况（包括其他未明确处理的状态）
        print(f"Warning: Unexpected state '{current_stage}' reached. Ending workflow.")
        return END
    
    # Add edges using the conditional logic
    workflow.add_edge(START, "generate_hypothetical_document")
    workflow.add_conditional_edges("normalize", decide_next_node)
    workflow.add_conditional_edges("strategy", decide_next_node)
    workflow.add_conditional_edges("execute", decide_next_node)
    workflow.add_conditional_edges("validate", decide_next_node)
    workflow.add_conditional_edges("generate_hypothetical_document", decide_next_node)  # 新增边
    workflow.add_conditional_edges("format_results", decide_next_node)  # 新增边
    
    # 编译工作流
    compiled_graph = workflow.compile()
    return compiled_graph 