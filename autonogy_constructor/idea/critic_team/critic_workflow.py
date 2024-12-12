from typing import TypedDict, List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph
import json

class CriticState(TypedDict):
    """Critic team state"""
    # Input
    research_idea: Dict  # 待评估的研究想法
    context: Dict  # 相关上下文（当前类、目标类等）
    
    # Evaluation Results
    feasibility_score: float  # 科学可行性评分
    novelty_score: float     # 创新性评分
    practicality_score: float  # 实验可行性评分
    verification_results: Dict  # 知识验证结果
    
    # Final Results
    evaluations: List[Dict]  # 所有评估结果
    suggestions: List[str]   # 改进建议
    
    # Query Management
    query_requests: List[Dict]  # 需要查询的信息
    
    # System
    messages: List[BaseMessage]
    done: bool

def create_critic_graph() -> Graph:
    """Create critic team workflow"""
    workflow = StateGraph(CriticState)
    llm = ChatOpenAI(temperature=0)
    
    # 1. 科学可行性评估
    def assess_feasibility(state: CriticState) -> CriticState:
        """Assess scientific feasibility"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a rigorous scientific reviewer."),
            ("user", """
            Research Idea:
            {idea}
            
            Evaluate the scientific feasibility considering:
            1. Theoretical foundation
            2. Chemical principles
            3. Methodology soundness
            4. Potential challenges
            
            Provide:
            1. Score (0-1)
            2. Detailed analysis
            3. Specific concerns
            4. Suggestions for improvement
            
            Format as JSON with:
            - score: float
            - analysis: string
            - concerns: list of strings
            - suggestions: list of strings
            """)
        ])
        
        response = llm.invoke(prompt.format_messages(idea=state["research_idea"]))
        result = parse_evaluation(response.content)
        
        state["feasibility_score"] = result["score"]
        state["evaluations"].append({
            "type": "feasibility",
            "result": result
        })
        
        return state
    
    # 2. 创新性评估
    def assess_novelty(state: CriticState) -> CriticState:
        """Assess novelty and innovation"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in research innovation assessment."),
            ("user", """
            Research Idea:
            {idea}
            
            Evaluate the novelty considering:
            1. Originality of approach
            2. Potential impact
            3. Advancement over existing work
            4. Cross-domain insights
            
            Provide:
            1. Score (0-1)
            2. Innovation analysis
            3. Similar existing work
            4. Unique contributions
            
            Format as JSON with:
            - score: float
            - analysis: string
            - similar_work: list of strings
            - contributions: list of strings
            """)
        ])
        
        response = llm.invoke(prompt.format_messages(idea=state["research_idea"]))
        result = parse_evaluation(response.content)
        
        state["novelty_score"] = result["score"]
        state["evaluations"].append({
            "type": "novelty",
            "result": result
        })
        
        return state
    
    # 3. 实验可行性评估
    def assess_practicality(state: CriticState) -> CriticState:
        """Assess experimental practicality"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an experienced experimental chemist."),
            ("user", """
            Research Idea:
            {idea}
            
            Evaluate the experimental practicality considering:
            1. Required resources
            2. Technical challenges
            3. Time and cost
            4. Safety considerations
            
            Provide:
            1. Score (0-1)
            2. Practical analysis
            3. Resource requirements
            4. Risk assessment
            
            Format as JSON with:
            - score: float
            - analysis: string
            - requirements: list of strings
            - risks: list of strings
            """)
        ])
        
        response = llm.invoke(prompt.format_messages(idea=state["research_idea"]))
        result = parse_evaluation(response.content)
        
        state["practicality_score"] = result["score"]
        state["evaluations"].append({
            "type": "practicality",
            "result": result
        })
        
        return state
    
    # 4. 知识验证
    def verify_knowledge(state: CriticState) -> CriticState:
        """Verify knowledge accuracy"""
        # 检查是否需要额外信息
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are verifying the knowledge foundations of a research idea."),
            ("user", """
            Research Idea:
            {idea}
            
            Context:
            {context}
            
            Identify:
            1. Key knowledge claims
            2. Required verifications
            3. Missing information
            4. Potential conflicts
            
            Format as JSON with:
            - claims: list of knowledge claims
            - verifications_needed: list of needed checks
            - missing_info: list of required information
            - potential_conflicts: list of concerns
            """)
        ])
        
        response = llm.invoke(prompt.format_messages(
            idea=state["research_idea"],
            context=state["context"]
        ))
        verification_needs = parse_verification(response.content)
        
        # 如果需要额外信息，生成查询请求
        if verification_needs["missing_info"]:
            state["query_requests"].extend([
                {
                    "query": info,
                    "purpose": "verification"
                }
                for info in verification_needs["missing_info"]
            ])
        
        state["verification_results"] = verification_needs
        return state
    
    # 5. 综合评估
    def synthesize_evaluation(state: CriticState) -> CriticState:
        """Synthesize all evaluations"""
        # 计算总分
        scores = {
            "feasibility": state["feasibility_score"],
            "novelty": state["novelty_score"],
            "practicality": state["practicality_score"]
        }
        
        # 生成最终建议
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize evaluations into final recommendations."),
            ("user", """
            Evaluations:
            {evaluations}
            
            Verification Results:
            {verification}
            
            Provide:
            1. Overall assessment
            2. Key strengths
            3. Major concerns
            4. Improvement suggestions
            
            Format as JSON with:
            - assessment: string
            - strengths: list of strings
            - concerns: list of strings
            - suggestions: list of strings
            """)
        ])
        
        response = llm.invoke(prompt.format_messages(
            evaluations=state["evaluations"],
            verification=state["verification_results"]
        ))
        synthesis = parse_synthesis(response.content)
        
        state["suggestions"] = synthesis["suggestions"]
        state["done"] = True
        return state
    
    # Build workflow
    workflow.add_node("feasibility", assess_feasibility)
    workflow.add_node("novelty", assess_novelty)
    workflow.add_node("practicality", assess_practicality)
    workflow.add_node("verification", verify_knowledge)
    workflow.add_node("synthesis", synthesize_evaluation)
    
    # Add edges
    workflow.add_edge("feasibility", "novelty")
    workflow.add_edge("novelty", "practicality")
    workflow.add_edge("practicality", "verification")
    
    # Add conditional edges
    def needs_verification(state: CriticState) -> bool:
        return bool(state["query_requests"])
    
    workflow.add_conditional_edges(
        "verification",
        needs_verification,
        {
            True: "query",  # 需要验证时转到查询子图
            False: "synthesis"  # 不需要验证时进行综合评估
        }
    )
    
    workflow.add_edge("synthesis", "end")
    
    # Set entry and exit
    workflow.set_entry_point("feasibility")
    workflow.set_finish_point("end")
    
    return workflow.compile()

def parse_evaluation(content: str) -> Dict:
    """Parse evaluation result"""
    try:
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing evaluation: {e}")
        return {"score": 0, "analysis": "Error in evaluation"}

def parse_verification(content: str) -> Dict:
    """Parse verification needs"""
    try:
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing verification: {e}")
        return {"missing_info": []}

def parse_synthesis(content: str) -> Dict:
    """Parse synthesis result"""
    try:
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing synthesis: {e}")
        return {"suggestions": []} 