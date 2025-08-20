from typing import Dict, List, Any, Union, Optional
from autology_constructor.idea.common.base_agent import AgentTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
import json
import inspect
import re

from .ontology_tools import OntologyTools   
from .utils import parse_json
from config.settings import OntologySettings
from .entity_matcher import EntityMatcher

# Import Pydantic models
from .schemas import NormalizedQuery, ToolCallStep, ValidationReport, DimensionReport, ToolPlan, ExtractedProperties, NormalizedQueryBody, ValidationClassification, ToolCallClassification

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
            if not name.startswith("_") and name not in ["__init__", "execute_sparql", "get_class_richness_info"]: 
                try:
                    sig = inspect.signature(method)
                    doc = inspect.getdoc(method)
                    desc = f"- {name}{sig}: {doc if doc else 'No description available.'}"
                    descriptions.append(desc)
                except ValueError: # Handles methods without signatures like built-ins if any sneak through
                    descriptions.append(f"- {name}(...): No signature/description available.")
        
        # Join descriptions with separator lines for better readability
        separator = "\n" + "-" * 80 + "\n"
        return separator.join(descriptions) if descriptions else "No tools available."

    def generate_plan(self, normalized_query: Union[Dict, NormalizedQuery], ontology_tools: OntologyTools, tool_hints: List = None) -> Union[ToolPlan, Dict]:
        """Generates the tool execution plan with optional tool hints for refinement."""
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

        # NEW: 构建包含tool hints的用户消息
        user_message = f"""Generate an execution plan for the following normalized query:
{normalized_query_str}

Output the plan as a JSON list of steps matching the ToolCallStep structure."""

        # NEW: 如果有tool hints，添加到用户消息中
        if tool_hints:
            hints_text = []
            for hint in tool_hints:
                hints_text.append(f"- For tool '{hint.tool}' on class '{hint.class_name}': {hint.hint}")
                if hint.alternative_tools:
                    hints_text.append(f"  Consider these alternative tools: {', '.join(hint.alternative_tools)}")
            
            hints_section = "\n".join(hints_text)
            user_message += f"""

IMPORTANT REFINEMENT HINTS:
{hints_section}

Please incorporate these hints when generating the tool plan. Use the suggested alternative tools or modifications."""

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

            # NEW: 验证和自动修正toolcall参数中的类名
            if isinstance(normalized_query, NormalizedQuery) and normalized_query.relevant_entities:
                plan = self._validate_and_fix_plan_entities(plan, normalized_query.relevant_entities)

            return plan # Return the validated and corrected plan

        except Exception as e:
            # Catch errors during structured output generation/parsing or validation
            error_msg = f"Failed to generate or parse structured tool plan: {str(e)}"
            print(f"{error_msg}") # Log the error
            # Consider logging the raw response if available and helpful for debugging
            # raw_response = getattr(e, 'response', None) # Example, actual attribute might differ
            # print(f"Raw LLM response (if available): {raw_response}")
            return {"error": error_msg} # Return error dictionary

    def _validate_and_fix_plan_entities(self, plan: ToolPlan, available_entities: List[str]) -> ToolPlan:
        """验证并修正执行计划中的实体参数"""
        from .workflow_utils import auto_fix_entity_mismatch
        
        corrected_steps = []
        total_corrections = 0
        
        for step in plan.steps:
            corrected_step = step
            
            # 检查params中是否包含class_names参数
            if hasattr(step, 'params') and isinstance(step.params, dict):
                step_params = step.params.copy()
                
                # 提取类名参数（可能的字段名）
                class_name_fields = ['class_names', 'class_name', 'entity_names', 'entity_name']
                
                for field_name in class_name_fields:
                    if field_name in step_params:
                        original_value = step_params[field_name]
                        
                        # 转换为字符串列表
                        if isinstance(original_value, str):
                            target_entities = [original_value]
                            is_single_value = True
                        elif isinstance(original_value, list):
                            target_entities = original_value
                            is_single_value = False
                        else:
                            continue  # 跳过非字符串/列表类型
                        
                        # 使用工具函数进行修正
                        corrected_entities, has_corrections, correction_mapping = auto_fix_entity_mismatch(
                            target_entities=target_entities,
                            primary_classes=available_entities,
                            fallback_classes=None,  # planner场景不需要fallback
                            min_score_threshold=0.3,
                            top_k=1
                        )
                        
                        if has_corrections:
                            # 将修正后的值放回params
                            if is_single_value and corrected_entities:
                                step_params[field_name] = corrected_entities[0]
                            else:
                                step_params[field_name] = corrected_entities
                                
                            total_corrections += len(correction_mapping)
                            print(f"[ToolPlannerAgent] Fixed {field_name} in {step.tool}: {correction_mapping}")
                
                # 创建修正后的step（如果有修改的话）
                if step_params != step.params:
                    corrected_step = ToolCallStep(tool=step.tool, params=step_params)
            
            corrected_steps.append(corrected_step)
        
        if total_corrections > 0:
            print(f"[ToolPlannerAgent] Applied {total_corrections} entity corrections to tool plan")
            return ToolPlan(steps=corrected_steps)
        
        return plan

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
2.  **Strictly Match Names**: Every entry in the `relevant_entities` field MUST EXACTLY match a name from the provided `available_classes` list. Do not alter names or add entries not present in the list. Prefer the full snake_case name. If an abbreviation is used, you must also specify the full name in a comment or parentheses.
3.  **Find All Variants**: The `available_classes` list may contain similar or related terms (e.g., abbreviations and full names, or just case variations). Make sure to identify all relevant classes that correspond to the concepts in the query. 
3.1. To be comprehensive, you MUST identify and include all relevant forms found in the list. For instance, if the query mentions a concept represented by an abbreviation, you must also include its full name if it exists in the list.
3.2. If an abbreviation is certain but could correspond to multiple possible full names in the available_classes list, you MUST include all of those potential full names.

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
                                      hypothetical_document: Dict = None,
                                      class_hints: List = None  # NEW: 添加class_hints参数
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

        # NEW: 处理class hints
        print(f"[DEBUG-CLASS-HINTS] Processing class_hints: {class_hints}")
        print(f"[DEBUG-CLASS-HINTS] Type of class_hints: {type(class_hints)}")
        
        if class_hints:
            hints_text = []
            print(f"[DEBUG-CLASS-HINTS] class_hints is truthy, length: {len(class_hints) if hasattr(class_hints, '__len__') else 'No length'}")
            
            # 安全地迭代class_hints
            try:
                for i, hint in enumerate(class_hints):
                    print(f"[DEBUG-CLASS-HINTS] Processing hint {i}: {hint}, type: {type(hint)}")
                    
                    if hint is None:
                        print(f"[DEBUG-CLASS-HINTS] Hint {i} is None, skipping")
                        continue
                        
                    try:
                        # 安全地访问hint的属性
                        class_name = getattr(hint, 'class_name', None)
                        hint_text = getattr(hint, 'hint', None)
                        
                        print(f"[DEBUG-CLASS-HINTS] Hint {i} - class_name: {class_name}, hint_text: {hint_text}")
                        
                        if class_name is not None and hint_text is not None:
                            hints_text.append(f"- Previous attempt with class '{class_name}' had issues: {hint_text}")
                            print(f"[DEBUG-CLASS-HINTS] Successfully processed hint {i}")
                        else:
                            print(f"[DEBUG-CLASS-HINTS] Hint {i} missing required attributes")
                            
                    except Exception as hint_error:
                        print(f"[DEBUG-CLASS-HINTS] Error accessing attributes of hint {i}: {hint_error}")
                        continue
                        
            except Exception as iterate_error:
                print(f"[DEBUG-CLASS-HINTS] Error iterating through class_hints: {iterate_error}")
                hints_text = []
            
            print(f"[DEBUG-CLASS-HINTS] Final hints_text: {hints_text}")
            
            if hints_text:
                hints_section = "\n".join(hints_text)
                user_content += f"""

--- CLASS SELECTION HINTS ---
{hints_section}

Please consider these hints when selecting relevant_entities. Try to choose different classes that avoid the mentioned issues.
--- END OF CLASS SELECTION HINTS ---"""
                print(f"[DEBUG-CLASS-HINTS] Added hints section to user_content")
            else:
                print(f"[DEBUG-CLASS-HINTS] No valid hints to add")
        else:
            print(f"[DEBUG-CLASS-HINTS] class_hints is falsy")

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
        class_hints = state.get("class_hints", [])  # NEW: 获取class hints

        if not natural_query:
            return {"error": "Natural query missing for main body generation."}

        prompt_messages = self._create_main_query_body_prompt(
            natural_query, 
            available_classes,
            enhanced_feedback,
            hypothetical_document,
            class_hints  # NEW: 传递class hints
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

    def _needs_entity_refinement(self, normalized_query_body: NormalizedQueryBody, available_classes: List[str]) -> bool:
        """检查是否需要进行实体refinement (用于内部检查)
        
        Args:
            normalized_query_body: 标准化查询主体
            available_classes: 可用类列表
            
        Returns:
            是否需要refinement
        """
        if not normalized_query_body or not normalized_query_body.relevant_entities:
            return False
        
        entity_matcher = EntityMatcher(available_classes)
        return entity_matcher.needs_refinement(normalized_query_body.relevant_entities)

    def __call__(self, state: Dict) -> Union[NormalizedQuery, Dict]:
        if not self.main_body_llm or not self.properties_llm:
             return {"error": "QueryParserAgent LLMs not properly configured during init."}

        # Step 1: Generate the main query body
        main_body_result = self._generate_main_query_body(state)
        if isinstance(main_body_result, dict) and main_body_result.get("error"):
            return {"error": f"Failed during main query body generation: {main_body_result.get('error')}"}
        if not isinstance(main_body_result, NormalizedQueryBody): # Should be caught by _generate_main_query_body
            return {"error": "Main query body generation returned unexpected type."}

        # Step 1.5: 内部实体验证和程序化重试机制（保留为兜底机制）
        # 仅在使用refined_classes时启动内部重试，确保workflow层面的refine_entities节点能正常工作
        available_classes = state.get("available_classes", [])
        original_available_classes = state.get("original_available_classes", available_classes)
        
        # 判断当前是否使用的是refined_classes（通过比较两个类集合是否不同）
        using_refined_classes = (available_classes != original_available_classes and 
                               len(available_classes) < len(original_available_classes))
        
        print(f"[QueryParserAgent] Using refined classes: {using_refined_classes}, "
              f"available: {len(available_classes)}, original: {len(original_available_classes)}")
        
        # 内部重试触发条件：仅在使用refined_classes且实体在其中找不到时才触发
        if (using_refined_classes and 
            self._needs_entity_refinement(main_body_result, available_classes)):
            
            print("[QueryParserAgent] Internal fallback refinement triggered - entities not found in refined class set")
            print(f"[QueryParserAgent] Using programmatic entity matching instead of LLM retry")
            
            # 使用新的通用工具进行程序化实体修正，不再调用LLM
            from .workflow_utils import auto_fix_entity_mismatch
            
            corrected_entities, has_corrections, correction_mapping = auto_fix_entity_mismatch(
                target_entities=main_body_result.relevant_entities,
                primary_classes=available_classes,
                fallback_classes=original_available_classes,
                min_score_threshold=0.1,
                top_k=1
            )
            
            if has_corrections:
                print(f"[QueryParserAgent] Programmatic corrections applied: {correction_mapping}")
                
                # 直接更新main_body_result的relevant_entities，不重新调用LLM
                main_body_result = NormalizedQueryBody(
                    intent=main_body_result.intent,
                    relevant_entities=corrected_entities,
                    filters=main_body_result.filters,
                    query_type_suggestion=main_body_result.query_type_suggestion
                )
                
                print(f"[QueryParserAgent] Internal refinement successful via programmatic matching")
            else:
                print(f"[QueryParserAgent] No corrections needed or no suitable matches found")

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
    """查询结果分类器 - 对每个工具调用进行分类"""
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """
You are an expert classifier for ontology query results. Your job is to classify each tool call result.

Classification Categories:
- "sufficient": Good results with adequate information
- "insufficient_properties": Results exist but lack property/relationship details (ONLY for basic tools like get_class_info)
- "insufficient": Results are incomplete or inadequate
- "no_results": No meaningful results returned
- "error": Execution failed or returned errors

IMPORTANT CLASSIFICATION RULES:
1. If a tool call used detailed property tools (get_class_properties, parse_class_definition), do NOT classify as "insufficient_properties"
2. For detailed property tools, use only "sufficient", "insufficient", "no_results", or "error"
3. Use "insufficient_properties" ONLY when basic tools like get_class_info were used and more detailed property information could be obtained

For each tool call in the results, provide:
1. Tool name (e.g., "get_class_properties")
2. Class name parameter (e.g., "ChemicalCompound") 
3. Classification from the 5 categories above
4. Brief reason (1 sentence)

Your response MUST include:
- tool_classifications: list of individual tool call classifications
- message: brief summary message describing the overall assessment

Keep your output simple and structured.
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
        """对查询结果中的每个工具调用进行分类
        
        Args:
            results: 查询结果，预期包含多个工具调用的结果
            query_context: 查询上下文信息
        
        Returns:
            ValidationReport: 包含每个工具调用的分类结果
        """
        if not self.structured_llm:
             return {"error": "ValidationAgent LLM not configured for structured output during init."}

        # Basic check for empty results
        if not results:
            return ValidationReport(
                valid=False,
                tool_classifications=[],
                message="No query results to validate"
            )

        # Serialize results for analysis
        try:
            results_str = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            return ValidationReport(
                valid=False,
                tool_classifications=[],
                message=f"Failed to serialize results: {str(e)}"
            )

        # Extract tool call information from results
        tool_call_info = self._extract_tool_call_info(results)

        # Build the classification prompt
        user_prompt = f"""Analyze these query results and classify each tool call:

Query Results:
```json
{results_str}
```

Tool Calls Found:
{self._format_tool_call_info(tool_call_info)}
"""

        if query_context:
            user_prompt += f"""
Query Context:
- Original Query: "{query_context.get('query', 'Unknown')}"
- Intent: {query_context.get('intent', 'Unknown')}"
- Target Entities: {query_context.get('relevant_entities', 'Unknown')}"
"""

        user_prompt += """
For each tool call, classify the result quality and provide:
1. tool: The tool function name
2. class_name: The class parameter that was passed
3. classification: One of [sufficient, insufficient_properties, insufficient, no_results, error]
4. reason: Brief explanation (one sentence)

Also determine the overall classification (use the worst individual classification).

Output a ValidationReport with:
- valid: boolean (true if overall classification is sufficient)
- overall_classification: the worst classification from all tool calls
- tool_classifications: list of individual tool call classifications
- message: brief summary message describing the overall results quality
"""

        try:
            validation_report: ValidationReport = self.structured_llm.invoke([
                ("system", self.system_prompt),
                ("user", user_prompt)
            ])
            
            # 直接返回LLM分类结果，不进行后处理
            return validation_report
            
        except Exception as e:
            error_msg = f"Failed to classify tool call results: {str(e)}"
            print(error_msg)
            return ValidationReport(
                valid=False,
                overall_classification=ValidationClassification.ERROR,
                tool_classifications=[],
                message=error_msg
            )
    
    def _extract_tool_call_info(self, results: Any) -> List[Dict]:
        """从结果中提取工具调用信息 - 基于标准格式"""
        tool_calls = []
        
        if isinstance(results, dict) and "results" in results and isinstance(results["results"], list):
            # 标准的ToolExecutorAgent结果格式
            for result_item in results["results"]:
                if isinstance(result_item, dict) and "tool" in result_item:
                    tool_calls.append({
                        "tool": result_item.get("tool", "unknown"),
                        "params": result_item.get("params", {}),
                        "result": result_item.get("result"),
                        "error": result_item.get("error")
                    })
        
        return tool_calls
    
    
    def _format_tool_call_info(self, tool_call_info: List[Dict]) -> str:
        """格式化工具调用信息用于LLM分析"""
        if not tool_call_info:
            return "No tool calls identified"
        
        formatted = []
        for i, call in enumerate(tool_call_info, 1):
            # 支持多种参数名称格式
            params = call.get("params", {})
            class_name = (params.get("class_name") or 
                         params.get("class_names") or 
                         params.get("classes") or 
                         "unknown")
            formatted.append(f"{i}. {call['tool']}(class_name='{class_name}')")
        
        return "\n".join(formatted)
    
    
    

class HypotheticalDocumentAgent(AgentTemplate):
    """增强版化学专家 - 具备工具调用能力"""
    def __init__(self, model: BaseLanguageModel):
        system_prompt = """You are an expert chemist with access to ontology tools for clarifying ambiguous queries.

Your task is to help interpret chemistry queries that have been difficult to process, especially those containing abbreviations or ambiguous terms.

Available Tools:
1. search_classes(query): Search for classes containing the query term
2. get_class_info(class_name): Get basic information about a class
3. get_class_richness_info(class_name): Evaluate how much information a class contains

Workflow for Ambiguous Queries:
1. Detect abbreviations or ambiguous terms in the query
2. Use search_classes to find potential full names or related classes
3. Use get_class_richness_info to evaluate which classes have the most information
4. Use get_class_info to confirm existence and get basic details
5. Generate a chemistry expert's interpretation based on your findings

When creating hypothetical documents, focus on:
- Chemistry expert interpretation of what the query is really asking
- Hypothetical ideal answer with rich chemistry information  
- Key chemistry concepts identified through tool exploration

Do NOT concern yourself with ontology structures or implementation details.
Focus on creating accurate chemistry interpretations."""

        super().__init__(
            model=model,
            name="HypotheticalDocumentAgent",
            system_prompt=system_prompt,
            tools=[]  # Tools will be set dynamically
        )
        
        # Store ontology tools instance for tool calling
        self.ontology_tools = None

    def set_ontology_tools(self, ontology_tools):
        """设置本体工具实例"""
        self.ontology_tools = ontology_tools

    def generate_hypothetical_document(self, query: str, validation_history: Any = None) -> Dict:
        """Generate enhanced hypothetical answer with tool-assisted analysis.
        
        Args:
            query: The natural language query
            validation_history: Previous validation reports to learn from
            
        Returns:
            Dict containing hypothetical answer and key concepts
        """
        if not self.ontology_tools:
            print("[HypotheticalDocumentAgent] Warning: No ontology tools available, using basic mode")
            return self._generate_basic_hypothetical_document(query, validation_history)
        
        # Enhanced workflow with tool assistance
        try:
            # Step 1: Analyze query for abbreviations and ambiguous terms
            analysis_result = self._analyze_query_with_tools(query)
            
            # Step 2: Generate hypothetical document incorporating tool findings
            return self._generate_enhanced_hypothetical_document(query, analysis_result, validation_history)
            
        except Exception as e:
            print(f"[HypotheticalDocumentAgent] Tool-assisted analysis failed: {e}, falling back to basic mode")
            return self._generate_basic_hypothetical_document(query, validation_history)

    def _analyze_query_with_tools(self, query: str) -> Dict:
        """Use tools to analyze query for abbreviations and find relevant classes"""
        
        analysis = {
            "detected_terms": [],
            "class_candidates": {},
            "richness_evaluations": {}
        }
        
        # Simple term detection (could be enhanced with NLP)
        potential_terms = self._extract_potential_terms(query)
        
        for term in potential_terms:
            try:
                # Search for classes containing this term
                search_results = self._safe_search_classes(term)
                
                if search_results:
                    analysis["class_candidates"][term] = search_results[:5]  # Top 5 results
                    
                    # Evaluate richness of top candidates
                    richness_scores = {}
                    for class_name in search_results[:3]:  # Evaluate top 3
                        richness_info = self._safe_get_richness_info(class_name)
                        if richness_info:
                            richness_scores[class_name] = richness_info.get("richness_score", 0.0)
                    
                    analysis["richness_evaluations"][term] = richness_scores
                    analysis["detected_terms"].append(term)
                    
            except Exception as e:
                print(f"[HypotheticalDocumentAgent] Error analyzing term '{term}': {e}")
                continue
        
        return analysis

    def _extract_potential_terms(self, query: str) -> List[str]:
        """Extract potential chemistry terms, abbreviations, and entities from query"""
        import re
        
        # Simple extraction - could be enhanced
        words = re.findall(r'\b[A-Z][A-Za-z]*\b', query)  # Capitalized words
        abbreviations = re.findall(r'\b[A-Z]{2,6}\b', query)  # 2-6 letter abbreviations
        
        # Combine and deduplicate
        potential_terms = list(set(words + abbreviations))
        
        # Filter out common English words
        common_words = {'The', 'What', 'How', 'Where', 'When', 'Why', 'And', 'Or', 'But', 'For', 'Of', 'In', 'On', 'At', 'To', 'From', 'With', 'By'}
        potential_terms = [term for term in potential_terms if term not in common_words]
        
        return potential_terms

    def _safe_search_classes(self, term: str) -> List[str]:
        """Safely search for classes, handling errors"""
        try:
            # Use ontology_tools to search for classes
            # This is a simplified implementation - the actual search method may vary
            all_classes = getattr(self.ontology_tools.onto_settings.ontology, 'classes', lambda: [])()
            
            matching_classes = []
            term_lower = term.lower()
            
            for cls in all_classes:
                if hasattr(cls, 'name') and term_lower in cls.name.lower():
                    matching_classes.append(cls.name)
                    
            return matching_classes[:10]  # Return top 10 matches
            
        except Exception as e:
            print(f"[HypotheticalDocumentAgent] Search failed for term '{term}': {e}")
            return []

    def _safe_get_richness_info(self, class_name: str) -> Optional[Dict]:
        """Safely get class richness information"""
        try:
            return self.ontology_tools.get_class_richness_info(class_name)
        except Exception as e:
            print(f"[HypotheticalDocumentAgent] Richness evaluation failed for '{class_name}': {e}")
            return None

    def _generate_enhanced_hypothetical_document(self, query: str, analysis: Dict, validation_history: Any = None) -> Dict:
        """Generate hypothetical document using tool analysis results"""
        
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

        # Format tool analysis results
        tool_findings = ""
        if analysis["detected_terms"]:
            tool_findings += "Tool Analysis Findings:\n"
            for term in analysis["detected_terms"]:
                candidates = analysis["class_candidates"].get(term, [])
                if candidates:
                    tool_findings += f"- '{term}' found in classes: {', '.join(candidates[:3])}\n"
                    
                    richness = analysis["richness_evaluations"].get(term, {})
                    if richness:
                        best_class = max(richness.items(), key=lambda x: x[1])
                        tool_findings += f"  Most information-rich: {best_class[0]} (score: {best_class[1]:.2f})\n"

        # Create the enhanced prompt
        user_prompt = f"""As a chemistry expert with tool analysis support, please clarify this query:

"{query}"

{validation_info}

{tool_findings}

Based on both your chemistry expertise and the tool findings above, provide:

1. CHEMISTRY EXPERT'S INTERPRETATION: Your understanding of what this query is asking, incorporating insights from the tool analysis about relevant classes and their information richness.

2. HYPOTHETICAL IDEAL ANSWER: What would a complete, accurate response look like, utilizing the most information-rich classes identified.

3. KEY CHEMISTRY CONCEPTS: Essential concepts for understanding this query, prioritizing those found in information-rich ontology classes.

Format as JSON:
"interpretation": Your chemistry expert's understanding
"hypothetical_answer": Complete ideal answer
"key_concepts": List of essential chemistry concepts (prioritize tool-identified rich classes)
"""

        # Call the model
        response = self.model_instance.invoke([
            ("system", self.system_prompt),
            ("user", user_prompt)
        ])
        
        # Process the response
        try:
            result = json.loads(response.content)
            # Add tool analysis metadata
            result["tool_analysis"] = analysis
            return result
        except json.JSONDecodeError:
            print("[HypotheticalDocumentAgent] Warning: Could not parse enhanced response as JSON")
            return {
                "interpretation": "Could not parse structured response.",
                "hypothetical_answer": response.content,
                "key_concepts": [],
                "tool_analysis": analysis
            }

    def _generate_basic_hypothetical_document(self, query: str, validation_history: Any = None) -> Dict:
        """Fallback method for basic hypothetical document generation"""
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
        
        # Create the prompt without tool assistance
        user_prompt = f"""As a chemistry expert, please help clarify this chemistry query:

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
