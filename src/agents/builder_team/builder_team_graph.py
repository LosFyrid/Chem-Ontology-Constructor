from typing import TypedDict, List, Annotated, Tuple, Dict, Union
from operator import add

from owlready2 import *

from config import settings
from concept_extractor import IndividualMetaData, ClassMetaData

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
