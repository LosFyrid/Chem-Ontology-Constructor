import dspy

from autonogy_constructor.base_data_structures import Ontology


class ExtractOntologyElements(dspy.Signature):
    """Analyze the provided text from research papers in the field of chemistry to identify all chemistry-related entities, data properties, and object properties within an ontological framework.

    Follow these Step-by-Step Analysis:

    1. Extract Chemistry-Related Entities:
      - Identify all significant nouns, proper nouns, and technical terminologies that represent chemistry-related concepts, such as molecules, reactions, compounds, processes, or any substantial entities.
      - Ensure that you capture entities across different levels of detail, from broad chemical categories to specific molecular structures, to create a comprehensive representation of the subject matter.
      - Choose names for entities that are specific enough to indicate their meaning without additional context, avoiding overly generic terms.
      - Consolidate similar entities to avoid redundancy, ensuring each represents a distinct concept at appropriate granularity levels.

    2. Identify Data Properties:
      - Extract attributes or characteristics of the identified entities that can be classified as data properties, ensuring they are value-based and not entities themselves.
      - Clearly define each data property, ensuring it accurately describes an attribute of an entity.

    3. Establish Object Properties:
      - Carefully examine the text to identify all relationships between entities, ensuring each relationship is correctly captured with accurate details about the interactions.
      - Analyze the context and interactions between the identified entities to determine how they are interconnected, focusing on actions, associations, dependencies, or similarities.
      - Clearly define the relationships, ensuring accurate directionality that reflects the logical or functional dependencies among entities.

    Objective: Produce a detailed and comprehensive ontology that captures the full spectrum of chemistry-related entities, data properties, and object properties mentioned in the text, along with their interrelations, reflecting both broad concepts and intricate details specific to the chemistry domain.

    """

    text: str = dspy.InputField(
        desc="a paragraph of text to extract entities, data properties, object properties and class hierachy to form an ontology"
    )
    ontology: Ontology = dspy.OutputField(
        desc="List representation of the ontology extracted from the text."
    )


class Assess(dspy.Signature):
    """Assess the quality of a ontology along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_ontology = dspy.InputField()
    assessment_criteria = dspy.InputField()
    assessment_score: int = dspy.OutputField(
        desc="Score with extreme rigor - only award full points when the ontology achieves perfect alignment with assessment criteria and would be deemed flawless by expert chemists"
    )
    assessment_reason: str = dspy.OutputField(
        desc="Leave this field empty if the ontology receives the full score for this criterion. Otherwise, explain what criteria are not met and propose improvements."
    )
