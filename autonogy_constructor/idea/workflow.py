from typing import TypedDict, List, Dict, Any
from langgraph.graph import Graph, StateGraph
from .query_team.query_workflow import create_query_graph, QueryState
from .dreamer_team.dreamer_workflow import create_dreamer_graph, DreamerState
from .critic_team.critic_workflow import create_critic_graph, CriticState

class MainState(TypedDict):
    """Main workflow state"""
    # Current processing stage
    stage: str  # "querying", "dreaming", "critiquing"
    
    # Subgraph states
    query_state: QueryState
    dreamer_state: DreamerState
    critic_state: CriticState
    
    # Shared data
    ontology: Any  # The ontology being analyzed
    current_class: str  # Current class being processed
    research_ideas: List[Dict]  # Generated ideas
    evaluations: List[Dict]  # Idea evaluations
    
    # System state
    messages: List[str]  # System messages
    done: bool

def create_main_graph(ontology) -> Graph:
    """Create main workflow graph"""
    workflow = StateGraph(MainState)
    
    # Create subgraphs
    query_graph = create_query_graph(ontology)
    dreamer_graph = create_dreamer_graph()
    critic_graph = create_critic_graph()
    
    # Add subgraphs as nodes
    workflow.add_node("query", query_graph)
    workflow.add_node("dream", dreamer_graph)
    workflow.add_node("critique", critic_graph)
    
    # Define state transformations
    def prepare_query_state(state: MainState) -> MainState:
        """Prepare state for query subgraph"""
        state["query_state"] = {
            "query": f"Get all information about class {state['current_class']}",
            "requester": state["stage"],
            "context": {
                "current_class": state["current_class"],
                "purpose": "idea_generation"
            }
        }
        return state
    
    def handle_query_results(state: MainState) -> MainState:
        """Process query results and update shared state"""
        if state["query_state"]["status"] == "success":
            # Update shared data with query results
            state["dreamer_state"]["class_info"] = state["query_state"]["results"]
        return state
    
    # Add edges with state transformations
    workflow.add_edge("prepare_query", "query")
    workflow.add_edge("query", "handle_results")
    workflow.add_edge("handle_results", "dream")
    workflow.add_edge("dream", "critique")
    
    # Add conditional returns to query
    workflow.add_conditional_edges(
        "critique",
        lambda state: "query" if state["needs_more_info"] else "end",
        {
            True: "prepare_query",
            False: "end"
        }
    )
    
    return workflow.compile() 