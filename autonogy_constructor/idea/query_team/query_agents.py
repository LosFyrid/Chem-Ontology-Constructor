from typing import Dict, List, Any
from src.agents.base_agent import AgentTemplate
from langchain.prompts import ChatPromptTemplate
import json

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
        super().__init__(
            system_prompt="验证查询结果的完整性和有效性",
            tools=[]
        )
    
    def validate(self, results: Any) -> Dict:
        """执行结果验证"""
        # 基础格式验证
        if not results:
            return {"valid": False, "message": "空结果集"}
        
        # 调用LLM进行语义验证
        prompt = ChatPromptTemplate.from_messages([
            ("system", "请验证以下查询结果是否符合预期："),
            ("user", """根据原始查询需求，判断结果是否：
1. 包含所有必需字段
2. 数据格式正确
3. 逻辑自洽

结果数据：
{results}

返回JSON验证报告，包含：
- valid: 是否有效
- missing_fields: 缺失字段
- format_errors: 格式错误
- logical_issues: 逻辑问题""")
        ])
        
        response = self.llm.invoke(prompt.format_messages(
            results=json.dumps(results, ensure_ascii=False)
        ))
        return parse_json(response.content)
