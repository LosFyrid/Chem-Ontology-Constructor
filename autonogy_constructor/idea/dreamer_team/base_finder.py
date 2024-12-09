from typing import Dict, List, TypedDict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph, StateGraph

class FinderState(TypedDict):
    """Base state for gap finder agents"""
    # Input
    current_class: str
    target_class: str  # 用于跨领域分析
    collected_info: Dict
    
    # Analysis
    gaps: List[Dict]
    ideas: List[Dict]
    
    # Query Management
    query_requests: List[Dict]
    query_results: Dict
    
    # Reflection
    needs_more_info: bool
    reflection_notes: List[str]
    
    # System
    messages: List[BaseMessage]
    done: bool

class BaseFinder:
    """Base class for gap finder agents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
    
    def create_finder_graph(self) -> Graph:
        """Create base finder workflow"""
        workflow = StateGraph(FinderState)
        
        # 1. Gap Analysis Node
        def analyze_gaps(state: FinderState) -> FinderState:
            """Analyze gaps based on current information"""
            response = self.llm.invoke(self._get_analysis_prompt().format_messages(
                info=state["collected_info"]
            ))
            state["gaps"] = self._parse_gaps(response.content)
            return state
        
        # 2. Reflection Node
        def reflect(state: FinderState) -> FinderState:
            """Reflect on analysis and decide if more info is needed"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Reflect on the current analysis."),
                ("user", """
                Current Information:
                {info}
                
                Gaps Found:
                {gaps}
                
                Consider:
                1. Are the gaps well-supported?
                2. Do we need more details?
                3. Are there unclear points?
                
                Clearly state if more information is needed.
                """)
            ])
            
            response = self.llm.invoke(prompt.format_messages(
                info=state["collected_info"],
                gaps=state["gaps"]
            ))
            
            reflection = response.content
            state["reflection_notes"].append(reflection)
            state["needs_more_info"] = "need more information" in reflection.lower()
            
            if state["needs_more_info"]:
                state["query_requests"].append(self._generate_query(reflection))
            
            return state
        
        # 3. Idea Generation Node
        def generate_ideas(state: FinderState) -> FinderState:
            """Generate ideas based on gaps"""
            response = self.llm.invoke(self._get_ideation_prompt().format_messages(
                gaps=state["gaps"]
            ))
            state["ideas"] = self._parse_ideas(response.content)
            return state
        
        # Build workflow
        workflow.add_node("analyze", analyze_gaps)
        workflow.add_node("reflect", reflect)
        workflow.add_node("generate", generate_ideas)
        
        # Add edges
        workflow.add_edge("analyze", "reflect")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "reflect",
            lambda state: state["needs_more_info"],
            {
                True: "query",  # 需要查询时转到查询子图
                False: "generate"  # 不需要查询时生成想法
            }
        )
        
        workflow.add_edge("generate", "end")
        
        # Set entry and exit
        workflow.set_entry_point("analyze")
        workflow.set_finish_point("end")
        
        return workflow.compile()
    
    def _get_analysis_prompt(self) -> ChatPromptTemplate:
        """Get gap analysis prompt"""
        raise NotImplementedError
    
    def _get_ideation_prompt(self) -> ChatPromptTemplate:
        """Get idea generation prompt"""
        raise NotImplementedError
    
    def _parse_gaps(self, content: str) -> List[Dict]:
        """Parse gaps from LLM response"""
        raise NotImplementedError
    
    def _parse_ideas(self, content: str) -> List[Dict]:
        """Parse ideas from LLM response"""
        raise NotImplementedError