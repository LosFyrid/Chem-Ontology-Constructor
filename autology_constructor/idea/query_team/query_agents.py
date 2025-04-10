from typing import Dict, List, Any
from src.agents.base_agent import AgentTemplate
from langchain.prompts import ChatPromptTemplate
import json
import inspect
import re

from .ontology_tools import OntologyTools   
from .utils import parse_json

class ToolPlannerAgent(AgentTemplate):
    """Generates a tool execution plan based on a normalized query using an LLM."""
    def __init__(self):
        system_prompt = """You are an expert planner for ontology tool execution.
Given a normalized query description and a list of available tools with their descriptions, create a sequential execution plan (a list of JSON objects) to fulfill the query.
Each step in the plan should be a JSON object with 'tool' (the tool name) and 'params' (a dictionary of parameters for the tool).
Only use the provided tools. Ensure the parameters match the tool's requirements based on its description.
Output ONLY the JSON list of plan steps, without any other text or explanation.

Available tools:
{tool_descriptions}
"""
        super().__init__(
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

    def generate_plan(self, normalized_query: Dict, ontology_tools: OntologyTools) -> List[Dict]:
        """Generates the tool execution plan."""
        if not normalized_query or isinstance(normalized_query, dict) and normalized_query.get("error"):
             return [{"error": "Cannot generate plan from invalid or missing normalized query."}]
             
        tool_descriptions_str = self._get_tool_descriptions(ontology_tools)
        
        # Prepare prompt using system prompt as template
        # Assuming AgentTemplate stores the raw system_prompt string
        # If AgentTemplate pre-formats the prompt, adjust accordingly
        formatted_system_prompt = self.system_prompt.format(tool_descriptions=tool_descriptions_str)
        
        user_message = f"""Generate an execution plan for the following normalized query:
{json.dumps(normalized_query, indent=2, ensure_ascii=False)}

Output the plan as a JSON list of steps."""

        # Combine into messages for LLM invocation (adapt based on AgentTemplate's LLM interface)
        messages = [
            ("system", formatted_system_prompt),
            ("user", user_message)
        ]
        
        # Assuming self.llm.invoke can handle a list of messages
        response = self.llm.invoke(messages) 

        try:
            # Parse the response content, assuming it's a JSON list
            # Use existing parse_json from utils which includes basic error handling
            # Let's enhance parsing robustness slightly here
            raw_content = response.content
            cleaned_content = re.sub(r"```json\n?(.*?)\n?```", r"\1", raw_content, flags=re.DOTALL).strip()
            
            plan = json.loads(cleaned_content) # Try parsing cleaned content

            if isinstance(plan, list):
                # Basic validation: check if items are dicts with 'tool'
                if all(isinstance(step, dict) and 'tool' in step for step in plan):
                    return plan
                elif not plan: # Empty list is a valid plan (no tools needed)
                    return []
            # If parsing or validation fails, return error
            error_msg = "Failed to generate a valid plan JSON list (non-list or invalid step format)"
            print(f"{error_msg}. Raw response: {raw_content}")
            return [{"error": error_msg, "raw_response": raw_content}]
        except json.JSONDecodeError:
             # Attempt to find JSON within the string if direct parsing fails
            match = re.search(r'\[\s*\{.*?\}\s*\]', raw_content, re.DOTALL) # Look specifically for list of objects
            if match:
                try:
                    plan = json.loads(match.group(0))
                    if isinstance(plan, list) and all(isinstance(step, dict) and 'tool' in step for step in plan):
                         print("Warning: Had to extract JSON from raw response.")
                         return plan
                except json.JSONDecodeError:
                     pass # Fall through to error
            error_msg = "Failed to parse plan JSON list from LLM response"
            print(f"{error_msg}. Raw response: {raw_content}")
            return [{"error": error_msg, "raw_response": raw_content}]
        except Exception as e:
            error_msg = f"Error parsing generated plan: {str(e)}"
            print(f"{error_msg}. Raw response: {getattr(response, 'content', str(response))}")
            return [{"error": error_msg, "raw_response": getattr(response, 'content', str(response))}]

class QueryParserAgent(AgentTemplate):
    """自然语言查询解析器 (无工具版本)"""
    def __init__(self):
        system_prompt = """你是本体查询解析专家，负责将自然语言查询转换为结构化格式。注意：
1. 必须严格遵循输出JSON格式。
2. 请参考下面提供的可用本体类列表来识别实体。"""
        super().__init__(
            system_prompt=system_prompt,
            tools=[] # 无工具
        )

    def __call__(self, state: Dict) -> Dict:
        natural_query = state.get("natural_query")
        available_classes = state.get("available_classes", []) # Get classes from state

        if not natural_query:
            return {"error": "Natural query missing in input state."}

        prompt_messages = self._create_prompt_messages(natural_query, available_classes)
        
        response = self.llm.invoke(prompt_messages)
        return self._parse_response(response.content)

    def _create_prompt_messages(self, query: str, available_classes: List[str]) -> List[tuple[str, str]]:
        class_list_str = ", ".join(available_classes) if available_classes else "无可用类信息"
        
        user_content = f"可用类: {class_list_str}\n\n请转换查询：{query}\n\n输出必须是 JSON 格式。"
        
        return [
            ("system", self.system_prompt),
            ("user", user_content)
        ]

    def _parse_response(self, raw: str) -> Dict:
        try:
            cleaned_raw = re.sub(r"```json\n?(.*?)\n?```", r"\1", raw, flags=re.DOTALL).strip()
            return json.loads(cleaned_raw)
        except json.JSONDecodeError as e:
            match = re.search(r'\{.*\}|\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    extracted_json = match.group(0)
                    return json.loads(extracted_json)
                except json.JSONDecodeError as inner_e:
                    error_msg = f"解析失败 (直接解析: {e}; 提取后解析: {inner_e})"
                    print(f"{error_msg}. Raw response: {raw}")
                    return {"error": error_msg, "raw_response": raw}
            else:
                error_msg = f"解析失败，未找到有效的 JSON 结构: {e}"
                print(f"{error_msg}. Raw response: {raw}")
                return {"error": error_msg, "raw_response": raw}
        except Exception as e:
             error_msg = f"解析时发生未知错误: {str(e)}"
             print(f"{error_msg}. Raw response: {raw}")
             return {"error": error_msg, "raw_response": raw}

class StrategyPlannerAgent(AgentTemplate):
    """查询策略规划器"""
    def __init__(self):
        super().__init__(
            system_prompt="根据查询特征选择最佳执行策略（tool_based/sparql）",
            tools=[]
        )
    
    def decide_strategy(self, standardized_query: Dict) -> str:
        prompt = f"""标准化查询：
{json.dumps(standardized_query, indent=2)}

请选择策略："""
        response = self.llm.invoke(prompt)
        return response.content.strip().lower()

class ToolExecutorAgent(AgentTemplate):
    """工具执行专家"""
    def __init__(self):
        super().__init__(
            system_prompt="根据查询计划执行工具调用序列",
            tools=OntologyTools(None)  # 继承工具集
        )
    
    def execute_plan(self, plan: List[Dict], ontology: Any) -> List[Dict]:
        """执行工具调用计划"""
        self.tools.onto = ontology  # 注入当前本体
        results = []
        for step in plan:
            try:
                tool = getattr(self.tools, step["tool"], None)
                if not tool:
                    results.append({"error": f"工具 {step['tool']} 不存在"})
                    continue
                result = tool(**step.get("params", {}))
                results.append({
                    "step": step["tool"],
                    "params": step.get("params"),
                    "result": result
                })
            except Exception as e:
                results.append({
                    "error": str(e),
                    "step": step
                })
        return results

class SparqlExpertAgent(AgentTemplate):
    """SPARQL生成专家"""
    def __init__(self):
        super().__init__(
            system_prompt="将标准化查询转换为正确的SPARQL语法",
            tools=[]
        )
    
    def generate_sparql(self, query_desc: Dict) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "请为以下查询生成SPARQL语句：\n{query}")
        ])
        response = self.llm.invoke(prompt.format_messages(
            query=json.dumps(query_desc, ensure_ascii=False)
        ))
        return response.content

class ValidationAgent(AgentTemplate):
    """结果验证专家"""
    def __init__(self):
        system_prompt = """
        你是一个专门验证查询结果的专家。你需要从多个维度评估查询结果的质量：
        
        1. 完整性：
           - 结果是否包含所有必要信息？
           - 是否有缺失的字段或数据？
           - 查询结果是否足够详细？
        
        2. 一致性：
           - 结果内部是否存在矛盾？
           - 数据格式是否统一？
           - 不同结果项之间是否保持一致的结构？
        
        3. 准确性：
           - 结果是否符合查询意图？
           - 内容是否正确？
           - 是否存在明显错误？
        
        对每个维度进行详细评估，并提供具体分析理由。
        你的验证结果应以JSON格式返回，包含以下字段：
        - valid: 布尔值，表示结果是否有效
        - details: 包含各维度验证结果的列表
        - message: 总体评估结论
        """
        super().__init__(system_prompt=system_prompt, tools=[])
    
    def validate(self, results: Any, query_context: Dict = None) -> Dict:
        """执行结果验证
        
        Args:
            results: 查询结果
            query_context: 可选的查询上下文信息
        
        Returns:
            Dict: 验证结果，包含valid, details, message等字段
        """
        # 基础格式验证
        if not results:
            return {"valid": False, "message": "空结果集", "details": []}
        
        # 构建验证提示
        prompt = f"""
        请验证以下查询结果:
        
        {json.dumps(results, indent=2, ensure_ascii=False)}
        """
        
        # 如果有查询上下文，添加到提示中
        if query_context:
            prompt += f"""
            验证上下文信息:
            - 查询意图: {query_context.get('intent', '未知')}
            - 查询类型: {query_context.get('type', '未知')}
            - 查询目标: {query_context.get('target', '未知')}
            """
        
        prompt += """
        请从完整性、一致性和准确性三个维度进行验证，并给出详细理由。
        对每个维度评分（1-5分，5分为最佳），并提供总体评估。
        
        返回JSON格式，包含以下结构:
        {
            "valid": true/false,  // 结果是否有效
            "details": [  // 各维度验证结果
                {
                    "dimension": "completeness",
                    "score": 4,  // 1-5分
                    "valid": true,  // 该维度是否通过
                    "message": "详细评估..."
                },
                // 其他维度...
            ],
            "message": "总体评估结论"
        }
        """
        
        response = self.llm.invoke(prompt)
        
        try:
            # 尝试解析JSON响应
            validation = parse_json(response.content)
            
            # 确保结果格式正确
            if "valid" not in validation:
                validation["valid"] = False
                validation["message"] = "验证结果格式不完整"
                
            if "details" not in validation:
                validation["details"] = []
                
            return validation
        except Exception as e:
            # 如果解析失败，使用简单的文本分析
            text = response.content.lower()
            valid = "valid" in text and "true" in text and "invalid" not in text
            
            # 创建基本验证结果
            return {
                "valid": valid,
                "details": [
                    {
                        "dimension": "general",
                        "valid": valid,
                        "message": response.content[:200] + "..."
                    }
                ],
                "message": f"无法解析详细验证结果: {str(e)[:100]}"
            }
