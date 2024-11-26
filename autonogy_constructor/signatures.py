import dspy

from autonogy_constructor.base_data_structures import OntologyElements, OntologyProperties


class ExtractOntologyElements(dspy.Signature):
    """Analyze the provided text from research papers in the field of chemistry to identify all chemistry-related entities, hierarchical relationships and disjoint relationships to construct an ontological framework.

    You are an expert ontology engineer. Your task is to carefully analyze the text and extract key ontological elements following these steps:

    1. Extract Chemistry-Related Entities:
      - Identify all significant chemistry concepts that should be represented as entities
      - Look for nouns, proper nouns, and technical terms representing:
        * Chemical compounds, molecules, and substances
        * Reaction types and processes
        * Chemical properties and characteristics
        * Laboratory equipment and procedures
      - Ensure each entity name is specific and meaningful
      - Avoid redundancy by consolidating similar concepts

    2. Identify Hierarchical Relationships:
      - Carefully analyze the text for "is-a" relationships between entities
      - Look for phrases indicating classification or categorization
      - Ensure the hierarchy is logically consistent
      - Document the evidence supporting each relationship

    3. Determine Disjoint Relationships:
      - Identify mutually exclusive classes that cannot share instances
      - Look for explicit statements of incompatibility
      - Verify each disjoint relationship is properly justified

    Remember: The goal is to create a precise and well-structured ontological framework that accurately represents the chemistry domain knowledge.
    """

    text: str = dspy.InputField(
        desc="a paragraph of text to extract entities and their relationships to form an ontology"
    )
    ontology_elements: OntologyElements = dspy.OutputField(
        desc="List representation of the ontology entities, their hierarchy and disjointness extracted from the text."
    )

class ExtractOntologyProperties(dspy.Signature):
    """Analyze the provided text from research papers in the field of chemistry to identify data properties and object properties within an ontological framework.

    You are an expert ontology engineer. Your task is to carefully analyze the text and extract ontological properties following these steps:

    1. Identify Data Properties:
      - Look for value-based attributes that describe entities:
        * Numerical measurements and quantities
        * Descriptive characteristics
        * Physical or chemical properties
      - For each property:
        * Determine the owner entity from existing_ontology_elements
        * Identify specific values if present
        * Document the context and meaning
        * Validate owner entity exists in the ontology
      - Ensure properties are attributes, not entities themselves

    2. Establish Object Properties:
      - Identify relationships between entities by looking for:
        * Verbs and action phrases
        * Structural relationships
        * Functional dependencies
      - For each relationship:
        * Define clear domain (source) entities from existing_ontology_elements
        * Specify range (target) entities from existing_ontology_elements
        * Determine if it's a universal or existential restriction
        * Verify domain and range entities exist in the ontology
        * Validate the relationship direction and semantics
        * Consider complex domain/range expressions using union/intersection

    Remember: The goal is to create a comprehensive property structure that captures all meaningful relationships and characteristics in the chemistry domain while maintaining consistency with existing entities.
    """

    text: str = dspy.InputField(
        desc="a paragraph of text to extract data and object properties for the ontology"
    )
    existing_ontology_elements: OntologyElements = dspy.InputField(
        desc="List representation of the ontology entities extracted from the text"
    )
    ontology_properties: OntologyProperties = dspy.OutputField(
        desc="List representation of the data and object properties extracted from the text"
    )

class Assess(dspy.Signature):
    """Assess the quality of an ontology or a part of an ontology along the specified dimension."""

    assessed_text: str = dspy.InputField(
        desc="The text that is used to construct ontology"
    )
    assessment_ontology: str = dspy.InputField(
        desc="Structured text representation of the ontology extracted from the text"
    )
    assessment_criteria: str = dspy.InputField(
        desc="The criteria of dimension of ontology quality to be assessed"
    )
    assessment_score: int = dspy.OutputField(
        desc="Score with extreme rigor - only award full points when the ontology achieves perfect alignment with assessment criteria and would be deemed flawless by expert chemists"
    )
    assessment_reason: str = dspy.OutputField(
        desc="Leave this field empty if the ontology receives the full score for this criterion. Otherwise, explain what criteria are not met and propose improvements."
    )
