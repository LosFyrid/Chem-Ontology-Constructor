from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import uuid
from datetime import datetime

class NormalizedQuery(BaseModel):
    """Represents the structured understanding of a natural language query."""
    intent: str = Field(description="The main goal or action of the query, e.g., 'find information', 'compare entities', 'get property'.")
    relevant_entities: List[str] = Field(default_factory=list, description="The primary entities or concepts the query is about. The names in the list must be present in the available classes.")
    relevant_properties: List[str] = Field(default_factory=list, description="List of specific property names mentioned or relevant to the query.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filtering conditions to apply, where keys are property names and values are the filter criteria.")
    query_type_suggestion: Optional[str] = Field(default=None, description="A suggested type for the query based on the parsing, e.g., 'fact-finding', 'comparison', 'definition'.")

    @field_validator('relevant_properties', 'relevant_entities', mode='before')
    @classmethod
    def convert_none_to_empty_list(cls, value):
        if value is None:
            return []
        return value

class NormalizedQueryBody(BaseModel):
    """Represents the main body of a structured query, excluding properties."""
    intent: str = Field(description="The main goal or action of the query, e.g., 'find information', 'compare entities', 'get property'.")
    relevant_entities: List[str] = Field(default_factory=list, description="The primary entities or concepts the query is about. The names in the list must be present in the available classes.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filtering conditions to apply, where keys are property names and values are the filter criteria.")
    query_type_suggestion: Optional[str] = Field(default=None, description="A suggested type for the query based on the parsing, e.g., 'fact-finding', 'comparison', 'definition'.")

    @field_validator('relevant_entities', mode='before')
    @classmethod
    def convert_none_to_empty_list(cls, value):
        if value is None:
            return []
        return value

class ToolCallStep(BaseModel):
    """Represents a single step in a tool execution plan."""
    tool: str = Field(description="The name of the tool to be called. Must be one of the available OntologyTools methods.")
    params: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of parameters required to call the specified tool.")

class ToolPlan(BaseModel):
    """Represents the planned sequence of tool calls."""
    steps: List[ToolCallStep] = Field(default_factory=list, description="The sequence of tool calls to execute.")
            


class DimensionReport(BaseModel):
    """Represents the validation result for a specific dimension."""
    dimension: str = Field(description="The dimension being evaluated, e.g., 'completeness', 'consistency', 'accuracy'.")
    # score: Optional[int] = Field(default=None, description="The score for this dimension (typically 1-5).")
    # valid: bool = Field(description="Whether the result passed validation for this dimension.")
    message: str = Field(description="Detailed assessment or reasoning for this dimension's validation outcome.")

class ValidationReport(BaseModel):
    """Represents the overall validation report for a query result."""
    valid: bool = Field(description="Overall assessment of whether the query result is valid.")
    details: List[DimensionReport] = Field(default_factory=list, description="A list of detailed validation results for each assessed dimension.")
    message: str = Field(description="A concluding summary message about the overall validation result.")
    improvement_suggestions: Optional[List[str]] = Field(default=None, description="Text suggestions for improving the query if validation failed.")
    # issue_aspects: Optional[List[str]] = Field(default=None, description="The aspects of the query that need improvement (e.g., 'entity_recognition', 'property_selection', 'strategy').")

class QueryStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Query(BaseModel):
    """查询请求"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    natural_query: str

    # 元数据
    originating_team: str  # dreamer, critic等
    originating_agent: str  # 发起查询的agent
    priority: str = "normal"  # high, normal, low

    # 查询上下文
    query_context: Dict[str, Any] = Field(default_factory=dict)

    # 状态跟踪
    created_at: datetime = Field(default_factory=datetime.now)
    status: QueryStatus = QueryStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None

class ExtractedProperties(BaseModel):
    """Represents a list of relevant properties extracted from a query."""
    relevant_properties: List[str] = Field(default_factory=list, description="List of specific property names identified from the query and available property lists.")

    @field_validator('relevant_properties', mode='before')
    @classmethod
    def convert_none_to_empty_list(cls, value):
        if value is None:
            return []
        return value 