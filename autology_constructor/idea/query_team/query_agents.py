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
from .schemas import NormalizedQuery, ToolCallStep, ValidationReport, DimensionReport, ToolPlan, ExtractedProperties, NormalizedQueryBody

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
        '''
        原始部分指令
        1. Strictly adhere to the NormalizedQueryBody JSON schema for the output.
2. Refer to the provided list of available ontology classes to identify 
entities for the 'relevant_entities' field. First, parse the natural language 
query to identify initial candidate entities.
3. If 'HYPOTHETICAL DOCUMENT INSIGHTS' are provided, you MUST then use them to 
refine and expand your initial findings:
    - Use the 'Expert Interpretation of the Query' and 'Hypothetical Answer' to 
    better understand the user's true intent and the full scope of information 
    needed. This can help confirm or adjust the initially identified entities 
    and intent.
    - Critically examine the 'Key Chemistry Concepts Identified' list. These 
    are expert-identified chemical entities. Carefully check if 
    `available_classes` contain terms that are identical, synonymous, or 
    semantically very similar to these `Key Chemistry Concepts Identified`. 
    Prioritize including such identified classes from `available_classes` in 
    the 'relevant_entities' field, potentially adding to or replacing entities 
    derived solely from the natural language query if these expert concepts 
    offer a more accurate or complete representation.
    - Your goal is to make 'relevant_entities' comprehensive and accurate by 
    first performing a primary analysis of the user's query, and then 
    augmenting this with the expert insights from the hypothetical document.
4. Note that there are SourcedInformation objects that provide additional 
metadata. When queries involve concepts like "source", "description", or 
"definition", consider that these information are not related to relations.
        '''
        system_prompt_main_body = """You are an expert ontology query parser. Your task is to convert natural language queries into a structured format representing the main body of the query (intent, entities, filters, type).

**Instructions for Identifying `relevant_entities`:**
1.  **Be Comprehensive**: Your primary goal is to be comprehensive. It is better to include moderately related entities than to miss important ones.
2.  **Strictly Match Names**: Every entry in the `relevant_entities` field MUST EXACTLY match a name from the provided `available_classes` list. Do not alter names or add entries not present in the list.
3.  **Find All Variants**: The `available_classes` list may contain similar or related terms (e.g., abbreviations and full names, or just case variations). Make sure to identify all relevant classes that correspond to the concepts in the query.

**General Workflow:**
1.  Strictly adhere to the NormalizedQueryBody JSON schema for the output.
2.  Refer to the provided list of `available_classes` to identify entities for the `relevant_entities` field.
3.  If 'HYPOTHETICAL DOCUMENT INSIGHTS' are provided, you MUST use them to refine and expand your initial findings to better understand the user's true intent and the full scope of information needed.
4.  Note that concepts like "source", "description", or "definition" in a query often refer to `SourcedInformation` metadata, not relationships between chemical entities.
"""
        system_prompt_properties = """You are an expert ontology query parser specializing in identifying relevant properties.
Given a natural language query, the already identified main query body (intent, entities), and available property lists:
1. Strictly adhere to the ExtractedProperties JSON schema for the output, focusing ONLY on the 'relevant_properties' field.
2. Refer to the provided lists of data properties and object properties to identify property relationships.
3. If 'HYPOTHETICAL DOCUMENT INSIGHTS' are provided, you MUST deeply analyze them:
    - Use the 'Expert Interpretation of the Query' and 'Hypothetical Answer', and 'Key Chemistry Concepts Identified' in conjunction with the 'main_query_body' to infer the properties needed to satisfy the query and connect the identified entities.
    - The goal is to identify all properties essential for answering the query comprehensively, as suggested by the expert insights and the query's intent.
"""
        # Store system prompts for later use in creating full ChatPromptTemplate messages
        self.system_prompt_main_body = system_prompt_main_body
        self.system_prompt_properties = system_prompt_properties

        super().__init__(
            model=model,
            name="QueryParserAgent",
            system_prompt="This main system prompt is not directly used for LLM calls in QueryParserAgent, as it's split into two parts.", # This is a placeholder.
            tools=[] # No tools needed for parsing itself
        )
        # Configure LLM for structured output immediately using helper
        try:
            self.main_body_llm = self._get_structured_llm(NormalizedQueryBody)
            self.properties_llm = self._get_structured_llm(ExtractedProperties)
        except RuntimeError as e:
            print(f"Error initializing structured LLMs for QueryParserAgent: {e}")
            self.main_body_llm = None
            self.properties_llm = None

    def _create_main_query_body_prompt(self, query: str, available_classes: List[str],
                                      enhanced_feedback: str = None,
                                      hypothetical_document: Dict = None
                                      ) -> List[tuple[str, str]]:
        class_list_str = ", ".join(available_classes) if available_classes else "No available class information provided."

        user_content = f"Please analyze the following query and convert it into the NormalizedQueryBody JSON format:\nQuery: {query}"

        if enhanced_feedback:
            print(f"Enhanced Feedback Got ")
            user_content += f"\n\n--- VALIDATION FEEDBACK ---\n{enhanced_feedback}\n---"

        if hypothetical_document:
            interpretation = hypothetical_document.get("interpretation")
            hypo_answer = hypothetical_document.get("hypothetical_answer")
            key_concepts_list = hypothetical_document.get("key_concepts")

            hypo_content_parts = ["\n\n--- HYPOTHETICAL DOCUMENT INSIGHTS (Expert Chemist's Perspective) ---"]
            has_hypo_info = False
            if interpretation:
                hypo_content_parts.append(f"\nExpert Interpretation of the Query:\n{interpretation}")
                has_hypo_info = True
            if hypo_answer:
                hypo_content_parts.append(f"\nHypothetical Answer:\n{hypo_answer}")
                has_hypo_info = True
            if key_concepts_list and isinstance(key_concepts_list, list) and any(key_concepts_list):
                filtered_key_concepts = [str(kc) for kc in key_concepts_list if kc and str(kc).strip()]
                if filtered_key_concepts:
                    key_concepts_str = "\n- ".join(filtered_key_concepts)
                    hypo_content_parts.append(f"\nKey Chemistry Concepts Identified (use these to guide 'relevant_entities'):\n- {key_concepts_str}")
                    has_hypo_info = True
            
            if has_hypo_info:
                user_content += "".join(hypo_content_parts)
                user_content += "\n--- END OF HYPOTHETICAL DOCUMENT INSIGHTS ---"

        user_content += f"\n\nAvailable classes: {class_list_str}"
        user_content += "\n\nOutput *only* the JSON object conforming to the NormalizedQueryBody schema."
        
        return [
            ("system", self.system_prompt_main_body),
            ("user", user_content)
        ]

    def _create_extract_properties_prompt(self, query: str, main_query_body: NormalizedQueryBody,
                                        available_data_properties: List[str] = None,
                                        available_object_properties: List[str] = None,
                                        enhanced_feedback: str = None,
                                        hypothetical_document: Dict = None
                                        ) -> List[tuple[str, str]]:
        available_data_properties = available_data_properties or []
        available_object_properties = available_object_properties or []
        
        data_prop_list_str = ", ".join(available_data_properties) if available_data_properties else "No available data property information provided."
        obj_prop_list_str = ", ".join(available_object_properties) if available_object_properties else "No available object property information provided."
        
        main_body_context_str = (
            f"Previously identified query body:\n"
            f"Intent: {main_query_body.intent}\n"
            f"Relevant Entities: {', '.join(main_query_body.relevant_entities) if main_query_body.relevant_entities else 'None'}\n"
            f"Filters: {main_query_body.filters if main_query_body.filters else 'None'}\n"
            f"Query Type Suggestion: {main_query_body.query_type_suggestion if main_query_body.query_type_suggestion else 'None'}"
        )

        user_content = (
            f"Based on the original query, the following identified query body, and available property lists, please extract the relevant properties.\n"
            f"Original Query: {query}\n\n"
            f"{main_body_context_str}"
        )

        if enhanced_feedback: # This might be less relevant here but kept for consistency
            user_content += f"\n\n--- VALIDATION FEEDBACK (primarily for overall query, consider if it implies property needs) ---\n{enhanced_feedback}\n---"

        if hypothetical_document:
            interpretation = hypothetical_document.get("interpretation")
            hypo_answer = hypothetical_document.get("hypothetical_answer")
            key_concepts_list = hypothetical_document.get("key_concepts")

            hypo_content_parts = ["\n\n--- HYPOTHETICAL DOCUMENT INSIGHTS (Expert Chemist's Perspective) ---"]
            has_hypo_info = False
            if interpretation:
                hypo_content_parts.append(f"\nExpert Interpretation of the Query:\n{interpretation}")
                has_hypo_info = True
            if hypo_answer:
                hypo_content_parts.append(f"\nHypothetical Answer (consider what properties are needed to construct this answer):\n{hypo_answer}")
                has_hypo_info = True
            if key_concepts_list and isinstance(key_concepts_list, list) and any(key_concepts_list):
                filtered_key_concepts = [str(kc) for kc in key_concepts_list if kc and str(kc).strip()]
                if filtered_key_concepts:
                    key_concepts_str = "\n- ".join(filtered_key_concepts)
                    hypo_content_parts.append(f"\nKey Chemistry Concepts Identified (these entities might be linked by properties you need to find):\n- {key_concepts_str}")
                    has_hypo_info = True
            
            if has_hypo_info:
                user_content += "".join(hypo_content_parts)
                user_content += "\n--- END OF HYPOTHETICAL DOCUMENT INSIGHTS ---"
        
        user_content += f"\n\nAvailable data properties: {data_prop_list_str}"
        user_content += f"\nAvailable object properties: {obj_prop_list_str}"
        user_content += "\n\nOutput *only* the JSON object conforming to the ExtractedProperties schema (i.e., a JSON with a single key 'relevant_properties' which is a list of strings)."
        
        return [
            ("system", self.system_prompt_properties),
            ("user", user_content)
        ]

    def _generate_main_query_body(self, state: Dict) -> Union[NormalizedQueryBody, Dict]:
        if not self.main_body_llm:
            return {"error": "QueryParserAgent Main Body LLM not configured."}

        natural_query = state.get("natural_query")
        available_classes = state.get("available_classes", [])
        enhanced_feedback = state.get("enhanced_feedback")
        hypothetical_document = state.get("hypothetical_document")

        if not natural_query:
            return {"error": "Natural query missing for main body generation."}

        prompt_messages = self._create_main_query_body_prompt(
            natural_query, 
            available_classes,
            enhanced_feedback,
            hypothetical_document
        )
        
        try:
            response: NormalizedQueryBody = self.main_body_llm.invoke(prompt_messages)
            return response
        except Exception as e:
            error_msg = f"Failed to get structured output for query body: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

    def _extract_relevant_properties(self, state: Dict, main_query_body: NormalizedQueryBody) -> Union[ExtractedProperties, Dict]:
        if not self.properties_llm:
            return {"error": "QueryParserAgent Properties LLM not configured."}

        natural_query = state.get("natural_query")
        available_data_properties = state.get("available_data_properties", [])
        available_object_properties = state.get("available_object_properties", [])
        enhanced_feedback = state.get("enhanced_feedback") # May be less relevant here
        hypothetical_document = state.get("hypothetical_document")

        if not natural_query: # Should be caught earlier, but good practice
            return {"error": "Natural query missing for properties extraction."}

        prompt_messages = self._create_extract_properties_prompt(
            natural_query,
            main_query_body,
            available_data_properties,
            available_object_properties,
            enhanced_feedback,
            hypothetical_document
        )
        
        try:
            response: ExtractedProperties = self.properties_llm.invoke(prompt_messages)
            return response
        except Exception as e:
            error_msg = f"Failed to get structured output for relevant properties: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

    def __call__(self, state: Dict) -> Union[NormalizedQuery, Dict]:
        if not self.main_body_llm or not self.properties_llm:
             return {"error": "QueryParserAgent LLMs not properly configured during init."}

        # Step 1: Generate the main query body
        main_body_result = self._generate_main_query_body(state)
        if isinstance(main_body_result, dict) and main_body_result.get("error"):
            return {"error": f"Failed during main query body generation: {main_body_result.get('error')}"}
        if not isinstance(main_body_result, NormalizedQueryBody): # Should be caught by _generate_main_query_body
            return {"error": "Main query body generation returned unexpected type."}

        # Step 2: Extract relevant properties (Temporarily disabled)
        # properties_result = self._extract_relevant_properties(state, main_body_result)
        # if isinstance(properties_result, dict) and properties_result.get("error"):
        #     return {"error": f"Failed during relevant properties extraction: {properties_result.get('error')}"}
        # if not isinstance(properties_result, ExtractedProperties): # Should be caught by _extract_relevant_properties
        #     return {"error": "Relevant properties extraction returned unexpected type."}
            
        # Step 3: Combine results into a NormalizedQuery object
        try:
            final_normalized_query = NormalizedQuery(
                intent=main_body_result.intent,
                relevant_entities=main_body_result.relevant_entities,
                filters=main_body_result.filters,
                query_type_suggestion=main_body_result.query_type_suggestion,
                relevant_properties=[] # Temporarily set to empty list
                # relevant_properties=properties_result.relevant_properties # Original line
            )
            return final_normalized_query
        except Exception as e: # Catch potential Pydantic validation errors if fields are missing/wrong type after all
            error_msg = f"Failed to combine query body and properties into NormalizedQuery: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

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
            return {"error": error_msg, "validation_report": validation_report}

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

Please filter out irrelevant content and extract both highly relevant information and moderately relevant information that could be expanded into the answer. Present it as:
1. A short summary (1-2 sentences) that directly answers the main question
2. A set of key information points, organized logically
3. Any important relationships or patterns found in the data

When the output information in the QUERY RESULTS is associated with a DOI in the same sourcedInformation, please include the DOI reference in your output for proper citation.

Format your response as a JSON object with:
"summary": A direct answer to the query
"key_points": An array of important information points
"relationships": Any significant relationships or patterns (if applicable)

DO NOT wrap your response in ```json```.
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
