from pydantic import BaseModel, Field
from typing import List, Literal, Union

class DataProperty(BaseModel):
    """Extract data properties from text for ontology creation. These should be value-based properties, not entities. Do not miss any fields."""
    name: str = Field(description="Provide a clear and concise name for the data property.")
    values: Union[dict, None] = Field(
        default=None,
        description="A dictionary of values as values and their owner classes for the data property as keys. Owner classes should be existing entities in the Entity list. If the data property has no value in context, specify 'None'."
    )
    information: Union[str, None] = Field(
        default=None,
        description="Offer a complete and comprehensive sentence detailing the data property from the text."
    )

class Entity(BaseModel):
    """Extract entities from text for ontology creation, including classes and individuals, excluding data properties. Then regard all entities as classes. Do not miss any fields."""
    name: str = Field(description="Provide a clear and concise name for the entity. Use the format [Full name]([Abbreviation]) if applicable.")
    information: Union[str, None] = Field(
        default=None,
        description="Offer a complete and comprehensive sentence detailing the entity from the text."
    )

class Domain(BaseModel):
    """Describe the domain of an object property. Do not miss any fields."""
    existence: Literal[True, False, ""] = Field(
        default=False,
        description="Specify 'True' if it need domain to clarify the object property, otherwise specify 'False'.'False' means entity field and type field are both filled with None"
    )
    entity: Union[str, None] = Field(
        default=None,
        description="Domain entity name of the object property meaning the owner class of the property. It should be an existing entity in the Entity list with domain_type as 'single'. If one entity cannot describe the domain, use multiple entities combined with domain_type to specify their relationship. If the existence field is 'False', specify 'None'."
    )
    type: Union[Literal['union', 'intersection', 'single'], None] = Field(
        default=None,
        description="Specify the domain type. 'union' means union of multiple domain entities, 'intersection' means intersection of multiple domain entities, and 'single' means single domain entity. If the existence field is 'False', specify 'None'."
    )

class Range(BaseModel):
    """Describe the range of an object property. Do not miss any fields."""
    existence: Literal[True, False, ""] = Field(
        default=False,
        description="Specify 'True' if the range need exist, otherwise specify 'False'."
    )
    entity: Union[str, None] = Field(
        default=None,
        description="Range entity name of the object property meaning the value class of the property, it should be existing entities in the Entity list with range_type as 'single'. If the range has multiple entities, use 'union' to indicate union and 'intersection' to indicate intersection. If the existence field is 'False', specify 'None'."
    )
    type: Union[Literal['union', 'intersection', 'single'], None] = Field(
        default=None,
        description="Specify the range type. 'union' means union of multiple range entities, 'intersection' means intersection of multiple range entities, and 'single' means single range entity. If the existence field is 'False', specify 'None'."
    )

class ObjectProperty(BaseModel):
    """Extract object properties from text for ontology creation. Do not miss any fields."""
    name: str = Field(description="Provide a clear and concise name for the object property. It should be separated by underscores between words.")
    domain: Union[Domain, None] = Field(
        default=None,
        description="Describe the domain of the object property. If the domain existence field is 'False', specify 'None'."
    )
    range: Union[Range, None] = Field(
        default=None,
        description="Describe the range of the object property. If the range existence field is 'False', specify 'None'."
    )
    restriction: Literal['only', 'some', ""] = Field(
        default='some',
        description="Specify 'only' to indicate a universal restriction (owl:allValuesFrom), meaning all possible values of the property must belong to the specified range. Specify 'some' to indicate an existential restriction (owl:someValuesFrom), meaning at least one value of the property must belong to the specified range."
    )
    information: Union[str, None] = Field(
        default=None,
        description="Offer a complete and comprehensive sentence detailing the object property from the text."
    )

class Hierarchy(BaseModel):
    """Extract subclass-superclass relationship of entities in the ontology. Do not miss any fields."""
    subclass: str = Field(
        default='',
        description="Subclass name, it should be an existing entity in the Entity list"
    )
    superclass: str = Field(
        default='',
        description="Superclass name, it should be an existing entity in the Entity list"
    )
    information: Union[str, None] = Field(
        default=None,
        description="Offer a complete and comprehensive sentence detailing the subclass-superclass relationship from the text."
    )

class Disjointness(BaseModel):
    """Extract disjoint class relationships in the ontology. Disjoint classes means nothing belongs to both classes. Do not miss any fields."""
    class1: str = Field(
        default='',
        description="First class name, it should be an existing entity in the Entity list"
    )
    class2: str = Field(
        default='',
        description="Second class name, it should be an existing entity in the Entity list"
    )

class Ontology(BaseModel):
    """List representation of the ontology for text. In this context, entity means same as class."""

    entities: List[Entity] = Field(
        description="List of entities in the knowledge graph"
    )
    hierarchy: Union[List[Hierarchy], None] = Field(
        default=None,
        description="List of subclass-superclass relationships in the ontology. If there is no subclass-superclass relationship, specify 'None'."
    )
    disjointness: Union[List[Disjointness], None] = Field(
        default=None,
        description="List of disjoint class relationships in the ontology. If there is no disjoint class relationship, specify 'None'."
    )
    data_properties: Union[List[DataProperty], None] = Field(
        default=None,
        description="List of data properties in the ontology. If there is no data property, specify 'None'."
    )
    object_properties: List[ObjectProperty] = Field(
        description="List of object properties in the ontology"
    )