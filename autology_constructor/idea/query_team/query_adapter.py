from typing import Dict, Optional, Any, List, Literal
from .schemas import Query, QueryStatus

class QueryToStateAdapter:
    """将Query转换为QueryState的转换器"""
    def transform(self, query: Query) -> Dict:
        """将Query对象转换为QueryState字典"""
        return {
            "query": query.natural_query,
            "source_ontology": query.query_context.get("ontology"),
            "query_type": query.query_context.get("query_type", "unknown"),
            "query_strategy": None,
            "originating_team": query.originating_team,
            "originating_stage": query.query_context.get("originating_stage", "unknown"),
            "query_results": {},
            "normalized_query": None,
            "execution_plan": None,
            "status": "initialized",
            "stage": "initialized",
            "previous_stage": None,
            "messages": []
        }

class StateToQueryAdapter:
    """将QueryState转换回Query的转换器"""
    def transform(self, state: Dict, query: Query) -> None:
        """将QueryState的状态更新到Query对象"""
        query.result = state["query_results"]
        if state["status"] == "error":
            query.status = QueryStatus.FAILED
            query.error = state.get("error", "Unknown error")
        else:
            query.status = QueryStatus.COMPLETED 