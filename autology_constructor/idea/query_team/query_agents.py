from typing import Dict, List, Any, Union
from autology_constructor.idea.common.base_agent import AgentTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
import json
import inspect
import re

from .ontology_tools import OntologyTools   
from .utils import parse_json
from config.settings import OntologySettings

# Import Pydantic models
from .schemas import NormalizedQuery, ToolCallStep, ValidationReport, DimensionReport, ToolPlan

class ToolPlannerAgent(AgentTemplate):
    """Generates a tool execution plan based on a normalized query using an LLM."""
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """You are an expert planner for ontology tool execution.
Given a normalized query description and a list of available tools with their descriptions, create a sequential execution plan (a list of JSON objects) to fulfill the query.
Each step in the plan should be a JSON object with 'tool' (the tool name) and 'params' (a dictionary of parameters for the tool).
Only use the provided tools. Ensure the parameters match the tool's requirements based on its description.
Output ONLY the JSON list of plan steps, without any other text or explanation.

Available tools:
{tool_descriptions}
"""
        super().__init__(
            model=model,
            name="ToolPlannerAgent",
            system_prompt=system_prompt,
            tools=[] # This agent plans, it doesn't execute tools directly
        )

    def _get_tool_descriptions(self, tool_instance: OntologyTools) -> str:
        """Generates formatted descriptions of OntologyTools methods."""
        descriptions = []
        # Ensure tool_instance is not None
        if tool_instance is None:
            return "No tool instance provided."
            
        for name, method in inspect.getmembers(tool_instance, predicate=inspect.ismethod):
            # Exclude private methods, constructor, and potentially the main execute_sparql if planning should use finer tools
            if not name.startswith("_") and name not in ["__init__", "execute_sparql"]: 
                try:
                    sig = inspect.signature(method)
                    doc = inspect.getdoc(method)
                    desc = f"- {name}{sig}: {doc if doc else 'No description available.'}"
                    descriptions.append(desc)
                except ValueError: # Handles methods without signatures like built-ins if any sneak through
                    descriptions.append(f"- {name}(...): No signature/description available.")
        return "\n".join(descriptions) if descriptions else "No tools available."

    def generate_plan(self, normalized_query: Union[Dict, NormalizedQuery], ontology_tools: OntologyTools) -> Union[ToolPlan, Dict]:
        """Generates the tool execution plan."""
        if not normalized_query:
             return {"error": "Cannot generate plan from missing normalized query."}
        # Check for error dictionary explicitly
        if isinstance(normalized_query, dict) and normalized_query.get("error"):
             return {"error": f"Cannot generate plan from invalid normalized query: {normalized_query.get('error', 'Unknown error')}"}

        tool_descriptions_str = self._get_tool_descriptions(ontology_tools)
        
        # Prepare prompt using system prompt as template
        formatted_system_prompt = self.system_prompt.format(tool_descriptions=tool_descriptions_str)
        
        # Handle normalized_query being either Dict or Pydantic model for prompt
        try:
            if isinstance(normalized_query, NormalizedQuery):
                normalized_query_str = normalized_query.model_dump_json(indent=2)
            else: # Assume it's a Dict
                normalized_query_str = json.dumps(normalized_query, indent=2, ensure_ascii=False)
        except Exception as dump_error:
             return {"error": f"Failed to serialize normalized query for planning: {dump_error}"}

        user_message = f"""Generate an execution plan for the following normalized query:
{normalized_query_str}

Output the plan as a JSON list of steps matching the ToolCallStep structure."""

        messages = [
            ("system", formatted_system_prompt),
            ("user", user_message)
        ]
        
        try:
            # Use the helper method to get the structured LLM
            structured_llm = self._get_structured_llm(ToolPlan)
            plan: ToolPlan = structured_llm.invoke(messages)

            # Basic validation: check if it's a list (LangChain should handle Pydantic validation)
            if not isinstance(plan, ToolPlan):
                # This case might indicate an issue with the LLM or LangChain's parsing
                raise ValueError("LLM did not return a list structure as expected for the plan.")

            # Further optional validation: Ensure all items are ToolCallStep (Pydantic handles this)
            # Optional: Check if tool names are valid based on ontology_tools? Maybe too strict here.

            return plan # Return the list of Pydantic models

        except Exception as e:
            # Catch errors during structured output generation/parsing or validation
            error_msg = f"Failed to generate or parse structured tool plan: {str(e)}"
            print(f"{error_msg}") # Log the error
            # Consider logging the raw response if available and helpful for debugging
            # raw_response = getattr(e, 'response', None) # Example, actual attribute might differ
            # print(f"Raw LLM response (if available): {raw_response}")
            return {"error": error_msg} # Return error dictionary

class QueryParserAgent(AgentTemplate):
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """You are an expert ontology query parser. Your task is to convert natural language queries into a structured format.
1. Strictly adhere to the NormalizedQuery JSON schema for the output.
2. Refer to the provided list of available ontology classes to identify entities.
3. Refer to the provided lists of data properties and object properties to identify property relationships.
4. Note that there are SourcedInformation objects that provide additional metadata. When queries involve concepts like "source", "description", or "definition", consider that these information are not related to relations."""

        super().__init__(
            model=model,
            name="QueryParserAgent",
            system_prompt=system_prompt,
            tools=[] # No tools needed for parsing itself
        )
        # Configure LLM for structured output immediately using helper
        try:
            self.structured_llm = self._get_structured_llm(NormalizedQuery)
        except RuntimeError as e:
            print(f"Error initializing structured LLM for QueryParserAgent: {e}")
            self.structured_llm = None # Ensure it's None if setup fails

    def __call__(self, state: Dict) -> Union[NormalizedQuery, Dict]:
        if not self.structured_llm:
             # This check is now more robust based on __init__
             return {"error": "QueryParserAgent LLM not configured for structured output during init."}

        natural_query = state.get("natural_query")
        available_classes = state.get("available_classes", [])
        available_data_properties = state.get("available_data_properties", [])
        available_object_properties = state.get("available_object_properties", [])
        enhanced_feedback = state.get("enhanced_feedback")  # 获取增强反馈

        if not natural_query:
            return {"error": "Natural query missing in input state."}

        prompt_messages = self._create_prompt_messages(
            natural_query, 
            available_classes,
            available_data_properties,
            available_object_properties,
            enhanced_feedback  # 传递增强反馈
        )
        print(prompt_messages)
        try:
            # Use the structured LLM instance created in __init__
            response: NormalizedQuery = self.structured_llm.invoke(prompt_messages)
            return response
        except Exception as e:
            error_msg = f"Failed to get structured output for query parsing: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

    def _create_prompt_messages(self, query: str, available_classes: List[str],
                              available_data_properties: List[str] = None,  # Added: parameter for data properties
                              available_object_properties: List[str] = None,  # Added: parameter for object properties
                              enhanced_feedback: str = None  # Added: parameter for enhanced feedback
                             ) -> List[tuple[str, str]]:
        
        # Default to empty lists if None
        available_data_properties = available_data_properties or []
        available_object_properties = available_object_properties or []
        
        class_list_str = ", ".join(available_classes) if available_classes else "No available class information provided."
        data_prop_list_str = ", ".join(available_data_properties) if available_data_properties else "No available data property information provided."  # Added: format data properties
        obj_prop_list_str = ", ".join(available_object_properties) if available_object_properties else "No available object property information provided."  # Added: format object properties
        
        # Updated user prompt to include all lists
        user_content = f"""Available classes: {class_list_str}
Available data properties: {data_prop_list_str}
Available object properties: {obj_prop_list_str}

Please analyze the following query and convert it into the NormalizedQuery JSON format:
Query: {query}"""

        # 添加增强反馈（如果有）
        if enhanced_feedback:
            user_content += f"\n\n--- VALIDATION FEEDBACK ---\n{enhanced_feedback}\n---"

        user_content += "\n\nOutput *only* the JSON object conforming to the NormalizedQuery schema."
        
        return [
            ("system", self.system_prompt),
            ("user", user_content)
        ]

class StrategyPlannerAgent(AgentTemplate):
    """Select the optimal execution strategy (tool_sequence/SPARQL) based on the query characteristics."""
    def __init__(self, model: BaseLanguageModel):
        super().__init__(
            model=model,
            name="StrategyPlannerAgent",
            system_prompt="""You are an expert strategy planner. Based on the standardized query characteristics, select the optimal execution strategy: 'tool_sequence' or 'SPARQL'.

Available Strategies:
1.  **tool_sequence**: Utilizes a sequence of pre-defined atomic operations (wrapped owlready2 functions) to retrieve relevant information from the ontology by combining these operations.
2.  **SPARQL**: Converts the natural language query into a SPARQL query and executes it directly against the ontology to retrieve information.

Instructions:
- Prefer the 'tool_sequence' strategy for most queries.
- Use the 'SPARQL' strategy ONLY when the query is complex and naturally suited for a SPARQL query (e.g., involves complex graph patterns, aggregations, or specific SPARQL features not easily replicated by tool sequences).

Output ONLY the selected strategy name ('tool_sequence' or 'SPARQL').""",
            tools=[]
        )
    
    def decide_strategy(self, standardized_query: Dict) -> str:
        # Construct the user message content
        user_content = f"""Standardized query:
{json.dumps(standardized_query, indent=2, ensure_ascii=False)}

Based on the query characteristics and the available strategies described in the system prompt, please select the optimal strategy ('tool_sequence' or 'SPARQL'). Output ONLY the selected strategy name."""

        # Create the messages list including the system prompt
        messages = [
            ("system", self.system_prompt),
            ("user", user_content)
        ]

        # Invoke the model with the structured messages
        response = self.model_instance.invoke(messages)
        # Ensure the response content is stripped and lowercased
        strategy = response.content.strip().lower()

        # Basic validation to ensure it's one of the expected strategies
        if strategy not in ['tool_sequence', 'sparql']:
            print(f"Warning: StrategyPlannerAgent returned an unexpected strategy: '{strategy}'. Defaulting to 'tool_sequence'.")
            # Consider raising an error or logging more formally depending on desired robustness
            return 'tool_sequence' # Or handle the unexpected output appropriately

        return strategy

class ToolExecutorAgent(AgentTemplate):
    """Execute the tool call sequence according to the query plan."""
    def __init__(self, model: BaseLanguageModel):
        # 不预先创建OntologyTools实例
        self.ontology_tools_instance = None
        super().__init__(
            model=model,
            name="ToolExecutorAgent",
            system_prompt="Execute the tool call sequence according to the query plan.",
            tools=[] # Let's keep this empty for AgentTemplate's init, as we call methods directly
        )
    
    def set_ontology_tools(self, ontology_tools: OntologyTools) -> None:
        """设置OntologyTools实例
        
        Args:
            ontology_tools: 预配置好的OntologyTools实例
        """
        self.ontology_tools_instance = ontology_tools
    
    def execute_plan(self, plan: ToolPlan) -> List[Dict]:
        """执行工具调用序列
        
        Args:
            plan: 执行计划
        """
        # 验证OntologyTools实例是否已设置
        if self.ontology_tools_instance is None:
            return [{"error": "OntologyTools instance not set. Call set_ontology_tools() before executing plan."}]
        
        results = []
        for step in plan.steps: # Iterate over ToolCallStep objects
            tool_name = step.tool
            params = step.params
            try:
                # 使用实例直接调用方法
                tool_method = getattr(self.ontology_tools_instance, tool_name, None)
                if not tool_method or not callable(tool_method):
                    results.append({
                        "error": f"Tool '{tool_name}' not found or not callable in OntologyTools",
                        "step_tool": tool_name,
                        "step_params": params
                    })
                    continue

                # 执行工具方法
                result = tool_method(**params)
                results.append({
                    "tool": tool_name, # Changed 'step' to 'tool' for clarity
                    "params": params,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "error": f"Error executing tool '{tool_name}': {str(e)}",
                    "tool": tool_name,
                    "params": params
                })
        return results

class SparqlExpertAgent(AgentTemplate):
    """Convert the standardized query into correct SPARQL syntax."""
    def __init__(self, model: BaseLanguageModel):
        super().__init__(
            model=model,
            name="SparqlExpertAgent",
            system_prompt="Convert the standardized query into correct SPARQL syntax.",
            tools=[]
        )
    
    def generate_sparql(self, query_desc: Dict) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Please generate the SPARQL statement for the following query:\\n{query}")
        ])
        response = self.model_instance.invoke(prompt.format_messages(
            query=json.dumps(query_desc, ensure_ascii=False)
        ))
        return response.content

"""
You are an expert specializing in validating query results for an ontology system. You need to evaluate the quality of the query results across multiple dimensions: completeness, consistency, and accuracy.

Provide a detailed assessment and specific reasoning for each dimension.

When validation fails, provide specific improvement suggestions addressing:
- Entity recognition issues
- Property selection problems
- Query formulation concerns
- Strategy selection considerations

Your validation result MUST strictly follow the ValidationReport JSON schema format, which includes fields for improvement suggestions and issue aspects.
"""

class ValidationAgent(AgentTemplate):
    """验证查询结果质量并提供改进建议的专家代理"""
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """
You are an expert specializing in validating query results for an ontology system. You need to evaluate the quality of the query results across multiple dimensions: completeness, consistency, and accuracy.

Provide a detailed assessment and specific reasoning for each dimension.

When validation fails, provide specific improvement suggestions.

Your validation result MUST strictly follow the ValidationReport JSON schema format, which includes fields for improvement suggestions and issue aspects.
"""



        super().__init__(
            model=model,
            name="ValidationAgent",
            system_prompt=system_prompt,
            tools=[]
        )
        # Configure LLM for structured output using helper
        try:
            self.structured_llm = self._get_structured_llm(ValidationReport)
        except RuntimeError as e:
            print(f"Error initializing structured LLM for ValidationAgent: {e}")
            self.structured_llm = None

    def validate(self, results: Any, query_context: Dict = None) -> Union[ValidationReport, Dict]:
        """执行结果验证，并提供改进建议
        
        Args:
            results: 查询结果
            query_context: 可选的查询上下文信息
        
        Returns:
            Union[ValidationReport, Dict]: 验证结果，包含valid, details, message等字段，以及improvement_suggestions
        """
        if not self.structured_llm:
             return {"error": "ValidationAgent LLM not configured for structured output during init."}

        # Basic check for empty results
        if not results:
            # Return an error dict, not a ValidationReport, as validation cannot proceed.
            return {"error": "Validation failed: Cannot validate empty result set."}

        # Serialize results for the prompt. Handle potential errors.
        try:
            results_str = json.dumps(results, indent=2, ensure_ascii=False, default=str) # Added default=str for broader serialization
            print(results_str)
        except Exception as e:
            return {"error": f"Validation failed: Could not serialize results for LLM prompt - {str(e)}"}

        # Build the prompt parts
        user_prompt = f"""Please validate the following query results:

```json
{results_str}
```

"""

        if query_context:
            user_prompt += f"""
Validation Context Information:
- Query: "{query_context.get('query', 'Unknown')}"
- Intent: {query_context.get('intent', 'Unknown')}
- Type: {query_context.get('type', 'Unknown')}
- Strategy: {query_context.get('strategy', 'Unknown')}
- Relevant entities: {query_context.get('relevant_entities', 'Unknown')}
- Relevant properties: {query_context.get('relevant_properties', 'Unknown')}
"""

        user_prompt += """
Evaluate based on completeness, consistency, and accuracy.

If validation fails, provide:
1. A list of specific text suggestions for improvement in the "improvement_suggestions" field
2. A list of corresponding issue aspects (like "entity_recognition", "property_selection", etc.) in the "issue_aspects" field

Your response must be a ValidationReport object with these fields if validation fails.
"""

        try:
            # Use the structured LLM instance created in __init__
            validation_report: ValidationReport = self.structured_llm.invoke([
                ("system", self.system_prompt),
                ("user", user_prompt)
            ])
            return validation_report
        except Exception as e:
            # Catch errors from structured output process
            error_msg = f"Failed to get or parse structured validation report: {str(e)}"
            print(error_msg)
            # Consider logging raw response if available
            return {"error": error_msg}

class HypotheticalDocumentAgent(AgentTemplate):
    """从专业化学家角度生成假设性答案，帮助查询标准化"""
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """You are an expert chemist who specializes in chemistry knowledge representation. 
Your task is to help clarify and interpret chemistry queries that have been difficult to process.

When presented with an ambiguous or failed chemistry query, you should:
1. Interpret what the query is trying to ask from a chemistry expert's perspective
2. Generate a "hypothetical answer" - what a complete and accurate answer would look like
3. Identify the key chemistry concepts, relationships, and properties that would be needed

Do NOT concern yourself with ontology structures, classes, or implementation details.
Focus ONLY on creating a chemistry expert's interpretation of the question and ideal answer."""

        super().__init__(
            model=model,
            name="HypotheticalDocumentAgent",
            system_prompt=system_prompt,
            tools=[]
        )

    def generate_hypothetical_document(self, query: str, validation_history: Any = None) -> Dict:
        """Generate a hypothetical answer from a chemistry expert perspective.
        
        Args:
            query: The natural language query
            validation_history: Previous validation reports to learn from
            
        Returns:
            Dict containing hypothetical answer and key concepts
        """
        # Format validation history info if available
        validation_info = ""
        if validation_history:
            validation_info = "Previous validation issues:\n"
            if isinstance(validation_history, list):
                for i, report in enumerate(validation_history):
                    if hasattr(report, 'message'):
                        validation_info += f"- Attempt {i+1}: {report.message}\n"
            elif hasattr(validation_history, 'message'):
                validation_info += f"- {validation_history.message}\n"
        
        # Create the prompt
        user_prompt = f"""As a chemistry expert, please help clarify this chemistry query that has been difficult to interpret:

"{query}"

{validation_info}

Please provide:

1. A CHEMISTRY EXPERT'S INTERPRETATION of what this query is really asking about. 
   Explain the query from a chemistry perspective, clarifying any ambiguities.

2. A HYPOTHETICAL IDEAL ANSWER that would fully address this query.
   What would a complete and accurate response look like?
   Include all relevant chemistry information that should appear in the answer.

3. KEY CHEMISTRY CONCEPTS that are essential to understanding this query:
   - Main chemical entities/substances involved
   - Important properties or relationships being asked about
   - Chemistry-specific terminology that needs to be understood

Please format your response as a JSON object with these sections:
"interpretation": Your chemistry expert's understanding of the query
"hypothetical_answer": What a complete answer would look like
"key_concepts": List of essential chemistry concepts, entities and properties
"""

        # Call the model
        response = self.model_instance.invoke([
            ("system", self.system_prompt),
            ("user", user_prompt)
        ])
        
        # Process the response
        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # If can't parse as JSON, extract structured information using regex
            # or return a formatted version of the raw response
            print("Warning: Could not parse hypothetical document response as JSON")
            return {
                "interpretation": "Could not parse structured response.",
                "hypothetical_answer": response.content,
                "key_concepts": []
            }

class ResultFormatterAgent(AgentTemplate):
    """Formats query results into concise, organized information points."""
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """You are an expert at distilling complex chemistry query results into clear, concise information points.

Your task is to:
1. Analyze the provided query and its results
2. Extract the most relevant information that directly addresses the query
3. Present this information as a well-organized set of key points
4. Eliminate redundancy and irrelevant details
5. Ensure technical accuracy while making the information accessible

When formatting results:
- Start with the most important findings that directly answer the query
- Group related information together logically
- Use consistent, precise terminology
- Highlight quantitative data, relationships, and definitive facts
- Include important qualifiers or context when necessary"""

        super().__init__(
            model=model,
            name="ResultFormatterAgent",
            system_prompt=system_prompt,
            tools=[]
        )

    def format_results(self, query: str, results: Dict, query_context: Dict = None) -> Dict:
        """Format query results into organized information points.
        
        Args:
            query: The original natural language query
            results: The query results to format
            query_context: Additional context about the query
            
        Returns:
            Dict containing formatted results with key points
        """
        # Format context information
        context_info = ""
        if query_context:
            if query_context.get('intent'):
                context_info += f"Query intent: {query_context.get('intent')}\n"
            if query_context.get('relevant_entities'):
                context_info += f"Relevant entities: {query_context.get('relevant_entities')}\n"
            if query_context.get('relevant_properties'):
                context_info += f"Relevant properties: {query_context.get('relevant_properties')}\n"
        
        # Try to convert results to string if not already
        results_str = ""
        try:
            if isinstance(results, str):
                results_str = results
            elif isinstance(results, dict):
                results_str = json.dumps(results, indent=2, ensure_ascii=False, default=str)
            else:
                results_str = str(results)
        except:
            results_str = "Error: Could not format results as string"
        
        # Create the prompt
        user_prompt = f"""Please format the following chemistry query results into clear, concise information points:

ORIGINAL QUERY:
"{query}"

{context_info}

QUERY RESULTS:
{results_str}

Please extract the most relevant information that directly addresses the query, and present it as:
1. A short summary (1-2 sentences) that directly answers the main question
2. A set of key information points, organized logically
3. Any important relationships or patterns found in the data

Format your response as a JSON object with:
"summary": A direct answer to the query
"key_points": An array of important information points
"relationships": Any significant relationships or patterns (if applicable)
"""

        # Call the model
        response = self.model_instance.invoke([
            ("system", self.system_prompt),
            ("user", user_prompt)
        ])
        
        # Process the response
        try:
            formatted_result = json.loads(response.content)
            return formatted_result
        except json.JSONDecodeError:
            # If can't parse as JSON, return a simple structure with the raw content
            return {
                "summary": "Could not generate structured summary.",
                "key_points": [response.content],
                "relationships": []
            }
