from typing import Dict, List, Any
from src.agents.base_agent import AgentTemplate
from langchain.prompts import ChatPromptTemplate
import json

from .ontology_tools import OntologyTools   
from .utils import parse_json

class QueryParserAgent(AgentTemplate):
    """自然语言查询解析器 (无工具版本)"""
    def __init__(self):
        system_prompt = """你是本体查询解析专家，负责将自然语言查询转换为结构化格式。注意：
1. 必须严格遵循输出JSON格式
2. 只使用可用的本体类：{available_classes}"""
        super().__init__(
            system_prompt=system_prompt,
            tools=[]  # 无工具
        )

    def __call__(self, state: Dict) -> Dict:
        prompt = self._create_prompt(state["natural_query"])
        response = self.llm.invoke(prompt)
        return self._parse_response(response.content)

    def _create_prompt(self, query: str) -> str:
        return f"请转换查询：{query}"

    def _parse_response(self, raw: str) -> Dict:
        try:
            return json.loads(raw)
        except:
            return {"error": "解析失败"}

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
