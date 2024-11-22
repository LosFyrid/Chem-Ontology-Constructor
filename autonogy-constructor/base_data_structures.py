from pydantic import BaseModel, Field
from typing import List

class DataProperty(BaseModel):
    """Extract data properties from text for ontology creation. These should be value-based properties, not entities."""
    name: str = Field(description="Provide a clear and concise name for the data property.")
    information: str = Field(description="Offer a complete and comprehensive sentence detailing the data property from the text.")

class Entity(BaseModel):
    """Extract entities from text for ontology creation, including classes and individuals, excluding data properties."""
    name: str = Field(description="Provide a clear and concise name for the entity. Use the format [Full name]([Abbreviation]) if applicable.")
    information: str = Field(description="Offer a complete and comprehensive sentence detailing the entity from the text.")

class ObjectProperty(BaseModel):
    """Extract object properties from text for ontology creation."""
    name: str = Field(description="Provide a clear and concise name for the object property. It should be separated by underscores between words.")
    domain: str = Field(
        description="Domain entity name of the object property, it should be an existing entity in the Entity list"
    )
    range: str = Field(
        description="Range entity name of the object property, it should be an existing entity in the Entity list"
    )
    restriction: str = Field(
        description="Specify 'only' to indicate a universal restriction (owl:allValuesFrom), meaning all possible values of the property must belong to the specified range. Specify 'some' to indicate an existential restriction (owl:someValuesFrom), meaning at least one value of the property must belong to the specified range."

    )
    information: str = Field(
        description="Offer a complete and comprehensive sentence detailing the object property from the text."
    )

class EntityHierarchy(BaseModel):
    """Extract subclass-superclass relationship of entities in the ontology."""
    subclass: str = Field(description="Provide a clear and concise name for the subclass.")
    superclass: str = Field(description="Provide a clear and concise name for the superclass.")
    information: str = Field(description="Offer a complete and comprehensive sentence detailing the subclass-superclass relationship from the text.")

class Ontology(BaseModel):
    """Graph representation of the ontology for text."""

    entities: List[Entity] = Field(
        description="List of entities in the knowledge graph"
    )
    data_properties: List[DataProperty] = Field(
        description="List of data properties in the ontology"
    )
    object_properties: List[ObjectProperty] = Field(
        description="List of object properties in the ontology"
    )