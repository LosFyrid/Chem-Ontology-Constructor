from typing import TypedDict, List, Annotated, Tuple, Dict, Union
from operator import add

from owlready2 import *

from config import settings
from concept_extractor import IndividualMetaData, ClassMetaData,parse_llm_output
from src.ontology.preprocess import create_metadata_properties

class BuilderTeamState(TypedDict):
    ontology: owlready2.namespace.Ontology
    classes: Annotated[List[ClassMetaData], add]
    individuals: Annotated[List[IndividualMetaData], add]
    individual_classifications: Annotated[
        List[Dict[
            str,
            Union[IndividualMetaData, ClassMetaData]
        ]],
        add
    ]
    data_properties: Annotated[
        List[
            Tuple[
                Union[IndividualMetaData, ClassMetaData],
                Dict[str, str]
            ]
        ],
        add
    ]
    object_properties: Annotated[
        List[
            Tuple[
                Union[IndividualMetaData, ClassMetaData],
                str,
                Union[IndividualMetaData, ClassMetaData]
            ]
        ],
        add
    ]

def create_classes(state: BuilderTeamState) -> BuilderTeamState:
    """
    Convert class definitions from ClassMetaData list into actual classes in the ontology and create data properties
    
    Args:
        state: BuilderTeamState - Current state of the builder team
        
    Returns:
        BuilderTeamState - Updated state
    """
    ontology = state["ontology"]
    
    for class_meta in state["classes"]:
        # Check if class already exists
        name = class_meta.name.replace(" ", "_").lower()
        if not ontology[name] in ontology.classes():
            print(f"Class: {name} does not exist, creating...")
            # Create new class in ontology
            with ontology:
                new_class = types.new_class(name, (Thing,))
                
                # Create embedding annotation property
                new_class.embedding = class_meta.embedding
                
                # Create location annotation property
                new_class.location = [f"doi: {class_meta.location[0]} - page: {class_meta.location[1]}"]
                
                # Create information annotation property
                new_class.information = [class_meta.information]
        else:
            print(f"Class: {name} already exists")
            with ontology:
                class_to_update = ontology[name]
                print(class_to_update)
                class_to_update.location.append(f"doi: {class_meta.location[0]} - page: {class_meta.location[1]}")
                class_to_update.information.append(class_meta.information)
                
                
    return state

def create_individuals(state: BuilderTeamState) -> BuilderTeamState:
    ontology = state["ontology"]
    
    for individual_meta in state["individuals"]:
        name = individual_meta.name.replace(" ", "_").lower()
        if not ontology[name] in ontology.individuals():
            print(f"Individual: {name} does not exist, creating...")
            with ontology:
                new_individual = Thing(name)
                
                # Create embedding annotation property
                new_individual.embedding = individual_meta.embedding
                
                # Create location annotation property
                new_individual.location = [f"doi: {individual_meta.location[0]} - page: {individual_meta.location[1]}"]
                
                # Create information annotation property
                new_individual.information = [individual_meta.information]
        else:
            print(f"Individual: {name} already exists")
            with ontology:
                individual_to_update = ontology[name]
                individual_to_update.location.append(f"doi: {individual_meta.location[0]} - page: {individual_meta.location[1]}")
                individual_to_update.information.append(individual_meta.information)
                
    return state

output = """
Name: Electron donation
Classification: class
Justification: It refers to a process that can occur in various contexts and with different groups or atoms.
Information: Electron donation affects chemical shifts by influencing electron distribution.

Name: Electron withdrawal
Classification: class
Justification: It is a process that can occur with different groups or atoms, affecting chemical shifts.
Information: Electron withdrawal can occur through conjugation or inductive effects, impacting chemical shifts.

Name: Chemical shift
Classification: class
Justification: It is a measurable property that can vary across different compounds and conditions.
Information: Chemical shifts are influenced by electron donation and withdrawal.

Name: Benzene ring
Classification: class
Justification: It is a structural motif that can be part of various compounds.
Information: Benzene rings with substituents can show different chemical shifts.

Name: Conjugation
Classification: class
Justification: It is a process involving π bonds that can occur in various chemical contexts.
Information: Conjugation affects electron distribution through π bonds.

Name: Inductive effect
Classification: class
Justification: It is a process involving σ bonds that can occur in various chemical contexts.
Information: Inductive effects result from electron withdrawal or donation through σ bonds.

Name: Nitro group
Classification: individual
Justification: It is a specific functional group known for its electron-withdrawing properties.
Information: The nitro group is a powerful electron-withdrawing group by conjugation.

Name: Carbonyl group
Classification: class
Justification: It is a functional group that can be part of various compounds.
Information: Carbonyl groups can withdraw electrons by conjugation.

Name: Nitrile group
Classification: class
Justification: It is a functional group that can be part of various compounds.
Information: Nitrile groups can withdraw electrons by conjugation.

Name: CF3 group
Classification: individual
Justification: It is a specific functional group known for its inductive electron-withdrawing properties.
Information: The CF3 group is an important example of inductive electron withdrawal.

Name: Halogens
Classification: class
Justification: It includes elements like F, Cl, Br, and I, which can participate in electron donation and withdrawal.
Information: Halogens can balance electron withdrawal by inductive effect and donation of lone pairs by conjugation.

Name: Alkyl groups
Classification: class
Justification: It refers to a category of groups that can donate electrons inductively.
Information: Alkyl groups are weak inductive electron donors.

Name: Amino group
Classification: individual
Justification: It is a specific functional group known for its electron-donating properties.
Information: The amino group is the best electron donor by conjugation.

Name: NO2 group
Classification: individual
Justification: It is a specific functional group known for its electron-withdrawing properties.
Information: The NO2 group is the best electron withdrawer.

Name: NH2 group
Classification: individual
Justification: It is a specific functional group known for its electron-donating properties.
Information: The NH2 group is the best electron donor.

Name: Lone pairs
Classification: class
Justification: It refers to pairs of electrons that can be involved in electron donation.
Information: Lone pairs on halogens, O, and N affect electron donation.

Name: p block elements
Classification: class
Justification: It includes elements like F, Cl, Br, and I, which have lone pairs in p orbitals.
Information: The p block elements have lone pairs in p orbitals affecting electron donation.
"""

onto_path.append(settings.ONTOLOGY_CONFIG["ontology_directory_path"])
ontology = get_ontology("http://www.test.org/chem_ontologies/chem_ontology.owl").load(only_local=True)
meta = ontology.get_namespace("http://www.test.org/chem_ontologies/meta/")


create_metadata_properties(ontology, meta)

class_concepts, individual_concepts = parse_llm_output(output)

test_state = BuilderTeamState(ontology=ontology, classes=class_concepts, individuals=individual_concepts, individual_classifications=[], data_properties=[], object_properties=[])

create_classes(test_state)

create_individuals(test_state)

ontology.save()
