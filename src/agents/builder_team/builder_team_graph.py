from typing import TypedDict, List, Annotated, Tuple, Dict, Union
from operator import add

from owlready2 import *

from config import settings
from concept_extractor import IndividualMetaData, ClassMetaData,parse_llm_output, llm
from src.ontology.preprocess import create_metadata_properties
from src.agents.builder_team.individual_classifier import generate_prompt, generate_relationship_prompt

class BuilderTeamState(TypedDict):
    ontology: Annotated[owlready2.namespace.Ontology, settings.ONTOLOGY_CONFIG["ontology"]]
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
    class_namespace = settings.ONTOLOGY_CONFIG["classes"]

    for class_meta in state["classes"]:
        # Check if class already exists
        name = class_meta.name.replace(" ", "_").lower()
        if not class_namespace[name] in ontology.classes():
            print(f"Class: {name} does not exist, creating...")
            # Create new class in ontology
            with class_namespace:
                new_class = types.new_class(name, (Thing,))
                
                new_class.embedding = class_meta.embedding
                
                new_class.location = [f"doi: {class_meta.location[0]} - page: {class_meta.location[1]}"]
                
                new_class.information = [class_meta.information]
        else:
            print(f"Class: {name} already exists")
            with class_namespace:
                class_to_update = class_namespace[name]
                print(class_to_update)
                class_to_update.location.append(f"doi: {class_meta.location[0]} - page: {class_meta.location[1]}")
                class_to_update.information.append(class_meta.information)
                
                
    return state

def create_individuals(state: BuilderTeamState) -> BuilderTeamState:
    ontology = state["ontology"]
    individual_namespace = settings.ONTOLOGY_CONFIG["individuals"]
    for individual_meta in state["individuals"]:
        name = individual_meta.name.replace(" ", "_").lower()
        if not individual_namespace[name] in ontology.individuals():
            print(f"Individual: {name} does not exist, creating...")
            with individual_namespace:
                new_individual = Thing(name)
                
                # Create embedding annotation property
                new_individual.embedding = individual_meta.embedding
                
                # Create location annotation property
                new_individual.location = [f"doi: {individual_meta.location[0]} - page: {individual_meta.location[1]}"]
                
                # Create information annotation property
                new_individual.information = [individual_meta.information]
        else:
            print(f"Individual: {name} already exists")
            with individual_namespace:
                individual_to_update = individual_namespace[name]
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

context = """
###### How electron donation and withdrawal change chemical shifts  \nWe can get an idea of the effect of electron distribution by looking at a series of benzene rings\nwith the same substituent in the 1 and 4 positions. This pattern makes all four hydrogens on\nthe ring identical. Here are a few compounds listed in order of chemical shift: largest shift\n(lowest fi eld; most deshielded) fi rst. Conjugation is shown by the usual curly arrows, and\ninductive effects by a straight arrow by the side of the group. Only one hydrogen atom and\none set of arrows are shown.  \nConjugation, as discussed in\nChapter 7, is felt through π bonds,\nwhile inductive effects are the\nresult of electron withdrawal or\ndonation felt simply by polarization\nof the σ bonds of the molecule.\nSee p. 135.  \nthe effect of electron-withdrawing groups\nby conjugation  \nby inductive effects  \n**H**  \n**O**  \n**O**  \n**HO**  \n**N**  \nδH 8.48 δH 8.10 **C** δH 8.10 δH 8.07 δH 7.78  \n**N**  \n**O**  \n**O**  \n**OH**  \n**C**  \n**N**  \n**O**  \n**H**  \n**F** **F**  \n**F**  \nThe largest shifts come from groups that withdraw electrons by conjugation. Nitro is the\nmost powerful—this should not surprise you as we saw the same in non-aromatic compounds\nin both [13]C and [1]H NMR spectra. Then come the carbonyl and nitrile group followed by groups\nshowing simple inductive withdrawal. CF3 is an important example of this kind of group—\nthree fl uorine atoms combine to exert a powerful effect.  \n-----  \nIn the middle of our sequence, around the position of benzene itself at 7.27 ppm, come\nthe halogens, whose inductive electron withdrawal and lone pair donation are nearly\nbalanced.  \nbalance between withdrawal by inductive effect and donation of lone pairs by conjugation  \n**I** δH 7.40 **Br** δH 7.32 δH 7.27 **Cl** δH 7.24 **F** δH 7.00  \n**I**  \n**Br**  \n**Cl**  \n**F**  \nAlkyl groups are weak inductive donators, but the groups which give the most shielding—\nperhaps surprisingly—are those containing the electronegative atoms O and N. Despite being\ninductively electron withdrawing (the C–O and C–N σ bonds are polarized with δ + C), on\nbalance conjugation of their lone pairs with the ring (as you saw on p. 278) makes them net\nelectron donors. They increase the shielding at the ring hydrogens. Amino groups are the best.\nNote that one nitrogen-based functional group (NO2) is the best electron withdrawer while\nanother (NH2) is the best electron donor.  \nthe effect of electron-donating groups  \nby inductive effect  \nbalance between withdrawal by inductive effect and donation\nof lone pairs by conjugation—electron donation wins  \n**H**  \nδH 7.03  \n**H**  \n**H**  \n**CH3**  \nδH 6.80 **O**  \n**H** **H**  \nδH 6.59 **N**  \n**H**  \nδH 6.35  \n**H**  \n**H**  \n**CH3**  \n**O**  \n**CH3**  \n**H**  \n**H**  \n**N**  \n**O**  \nδH 7.27  \n**H**  \n**H**  \nδH 7.27  \nδH 5.68  \n**H**  \n**H**  \nδH 5.68  \n**O**  \nδH 6.0  \n**H**  \n**H**  \nδH 7.0  \nδH 4.65  \n**H**  \n**H**  \nδH 6.35  \nAs far as the donors with lone pairs are concerned (the halogens plus O and N), two factors\nare important—the size of the lone pairs and the electronegativity of the element. If we look\nat the four halides at the top of this page the lone pairs are in 2p (F), 3p (Cl), 4p (Br), and 5p (I)\norbitals. In all cases the orbitals on the benzene ring are 2p so the fl uorine orbital is of the\nright size to interact well and the others too large. Even though fl uorine is the most electronegative, it is still the best donor. The others don’t pull so much electron density away, but\nthey can’t give so much back either.\nIf we compare the fi rst row of the p block elements—F, OH, and NH2—all have lone pairs\nin 2p orbitals so now electronegativity is the only variable. As you would expect, the most\nelectronegative element, F, is now the weakest donor.
"""

# ontology = settings.ONTOLOGY_CONFIG["ontology"]

# create_metadata_properties(ontology)

class_concepts, individual_concepts = parse_llm_output(output)

# test_state = BuilderTeamState(ontology=ontology, classes=class_concepts, individuals=individual_concepts, individual_classifications=[], data_properties=[], object_properties=[])

prompt = generate_prompt(individual_concepts, class_concepts, context)

# print(prompt)


# print(llm.invoke(generate_relationship_prompt("Nitro group", "Electron withdrawal")).content)

# for individual in individual_concepts:
#     for class_ in class_concepts:
#         print(llm.invoke(generate_relationship_prompt(individual.name, class_.name)).content)

print(llm.invoke(prompt).content)

# create_classes(test_state)

# create_individuals(test_state)

# ontology.save()
