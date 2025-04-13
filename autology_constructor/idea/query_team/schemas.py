from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class NormalizedQuery(BaseModel):
    """Represents the structured understanding of a natural language query."""
    intent: str = Field(description="The main goal or action of the query, e.g., 'find information', 'compare entities', 'get property'.")
    target_entities: List[str] = Field(default_factory=list, description="The primary entities or concepts the query is about.")
    properties: List[str] = Field(default_factory=list, description="Specific property names mentioned or relevant to the query.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filtering conditions to apply, where keys are property names and values are the filter criteria.")
    query_type_suggestion: Optional[str] = Field(default=None, description="A suggested type for the query based on the parsing, e.g., 'fact-finding', 'comparison', 'definition'.")

class ToolCallStep(BaseModel):
    """Represents a single step in a tool execution plan."""
    tool: str = Field(description="The name of the tool to be called. Must be one of the available OntologyTools methods.")
    params: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of parameters required to call the specified tool.")

class DimensionReport(BaseModel):
    """Represents the validation result for a specific dimension."""
    dimension: str = Field(description="The dimension being evaluated, e.g., 'completeness', 'consistency', 'accuracy'.")
    score: Optional[int] = Field(default=None, description="The score for this dimension (typically 1-5).")
    valid: bool = Field(description="Whether the result passed validation for this dimension.")
    message: str = Field(description="Detailed assessment or reasoning for this dimension's validation outcome.")

class ValidationReport(BaseModel):
    """Represents the overall validation report for a query result."""
    valid: bool = Field(description="Overall assessment of whether the query result is valid.")
    details: List[DimensionReport] = Field(default_factory=list, description="A list of detailed validation results for each assessed dimension.")
    message: str = Field(description="A concluding summary message about the overall validation result.") 