from typing import Dict, List, Any, Union
from autology_constructor.idea.common.base_agent import AgentTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
import json
import inspect
import re

from .ontology_tools import OntologyTools   
from .utils import parse_json

# Import Pydantic models
from .schemas import NormalizedQuery, ToolCallStep, ValidationReport, DimensionReport

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

    def generate_plan(self, normalized_query: Union[Dict, NormalizedQuery], ontology_tools: OntologyTools) -> Union[List[ToolCallStep], Dict]:
        """Generates the tool execution plan."""
        if not normalized_query:
             return {"error": "Cannot generate plan from missing normalized query."}
        # Check for error dictionary explicitly
        if isinstance(normalized_query, dict) and normalized_query.get("error"):
             return {"error": f"Cannot generate plan from invalid normalized query: {normalized_query.get('error')}"}

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
            structured_llm = self._get_structured_llm(List[ToolCallStep])
            plan: List[ToolCallStep] = structured_llm.invoke(messages)

            # Basic validation: check if it's a list (LangChain should handle Pydantic validation)
            if not isinstance(plan, list):
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
    """You are an expert ontology query parser. Your task is to convert natural language queries into a structured format.
1. Strictly adhere to the NormalizedQuery JSON schema for the output.
2. Refer to the provided list of available ontology classes to identify entities."""
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """You are an expert ontology query parser. Your task is to convert natural language queries into a structured format.
1. Strictly adhere to the NormalizedQuery JSON schema for the output.
2. Refer to the provided list of available ontology classes to identify entities."""
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

        if not natural_query:
            return {"error": "Natural query missing in input state."}

        prompt_messages = self._create_prompt_messages(natural_query, available_classes)
        
        try:
            # Use the structured LLM instance created in __init__
            response: NormalizedQuery = self.structured_llm.invoke(prompt_messages)
            return response
        except Exception as e:
            error_msg = f"Failed to get structured output for query parsing: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

    def _create_prompt_messages(self, query: str, available_classes: List[str]) -> List[tuple[str, str]]:
        class_list_str = ", ".join(available_classes) if available_classes else "No available class information provided."
        
        # Updated user prompt to reinforce structured output format
        user_content = f"""Available classes: {class_list_str}

Please analyze the following query and convert it into the NormalizedQuery JSON format:
Query: {query}

Output *only* the JSON object conforming to the NormalizedQuery schema."""
        
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
            system_prompt="Select the optimal execution strategy (tool_sequence/SPARQL) based on the query characteristics.",
            tools=[]
        )
    
    def decide_strategy(self, standardized_query: Dict) -> str:
        prompt = f"""Standardized query:
{json.dumps(standardized_query, indent=2)}

Please select a strategy:"""
        response = self.llm.invoke(prompt)
        return response.content.strip().lower()

class ToolExecutorAgent(AgentTemplate):
    """Execute the tool call sequence according to the query plan."""
    def __init__(self, model: BaseLanguageModel):
        # We need an OntologyTools instance for the 'tools' argument of AgentTemplate
        # Passing None initially, will be set in execute_plan. This might need adjustment
        # depending on how AgentTemplate uses self.tools in its __init__.
        # Let's instantiate it here for now.
        self.ontology_tools_instance = OntologyTools(None)
        super().__init__(
            model=model,
            name="ToolExecutorAgent",
            system_prompt="Execute the tool call sequence according to the query plan.",
            # Pass the *instance* of OntologyTools, not the class itself
            # AgentTemplate expects a list of tool callables or LangChain tools
            # We might need to adjust how tools are passed or used in AgentTemplate
            # For now, let's assume AgentTemplate doesn't strictly require LangChain tools in __init__
            # And we primarily use self.ontology_tools_instance directly here.
            tools=[] # Let's keep this empty for AgentTemplate's init, as we call methods directly
        )
    
    def execute_plan(self, plan: List[ToolCallStep], ontology: Any) -> List[Dict]:
        """Execute the tool call sequence according to the query plan."""
        # Use the ontology_tools_instance created in __init__
        self.ontology_tools_instance.onto = ontology  # 注入当前本体
        results = []
        for step in plan: # Iterate over ToolCallStep objects
            tool_name = step.tool
            params = step.params
            try:
                # Use the instance directly
                tool_method = getattr(self.ontology_tools_instance, tool_name, None)
                if not tool_method or not callable(tool_method):
                    results.append({
                        "error": f"Tool '{tool_name}' not found or not callable in OntologyTools",
                        "step_tool": tool_name,
                        "step_params": params
                    })
                    continue

                # Execute the tool method with its parameters
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
        response = self.llm.invoke(prompt.format_messages(
            query=json.dumps(query_desc, ensure_ascii=False)
        ))
        return response.content

class ValidationAgent(AgentTemplate):
    """You are an expert specializing in validating query results. You need to evaluate the quality of the query results across multiple dimensions: completeness, consistency, and accuracy.
Provide a detailed assessment and specific reasoning for each dimension.
Your validation result MUST strictly follow the ValidationReport JSON schema format."""
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """
        You are an expert specializing in validating query results. You need to evaluate the quality of the query results across multiple dimensions: completeness, consistency, and accuracy.
        Provide a detailed assessment and specific reasoning for each dimension.
        Your validation result MUST strictly follow the ValidationReport JSON schema format.
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
        """Execute result validation
        
        Args:
            results: Query results
            query_context: Optional query context information
        
        Returns:
            Union[ValidationReport, Dict]: Validation result, containing valid, details, message, etc. fields
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
        except Exception as e:
            return {"error": f"Validation failed: Could not serialize results for LLM prompt - {str(e)}"}

        # Build the prompt parts
        prompt_parts = [f"Please validate the following query results:\\n\\n```json\\n{results_str}\\n```"]

        if query_context:
            prompt_parts.append(f"""
Validation Context Information:
- Query Intent: {query_context.get('intent', 'Unknown')}
- Query Type: {query_context.get('type', 'Unknown')}
- Query Target: {query_context.get('target', 'Unknown')}
""")

        prompt_parts.append("""
Please validate based on the dimensions of completeness, consistency, and accuracy, providing detailed reasoning.
Provide a score (1-5) for each dimension.
Return a single JSON object strictly conforming to the ValidationReport JSON schema.
""")

        prompt = "\n\n".join(prompt_parts)

        try:
            # Use the structured LLM instance created in __init__
            validation_report: ValidationReport = self.structured_llm.invoke(prompt)
            return validation_report
        except Exception as e:
            # Catch errors from structured output process
            error_msg = f"Failed to get or parse structured validation report: {str(e)}"
            print(error_msg)
            # Consider logging raw response if available
            return {"error": error_msg}
