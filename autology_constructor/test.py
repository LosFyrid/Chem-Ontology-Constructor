from owlready2 import *
import owlready2.base
import owlready2.class_construct
import pytest
# Removed tempfile, os, shutil as they are no longer needed for temp ontology

from config.settings import OntologySettings
# Keep OntologyTools import
from autology_constructor.idea.query_team.ontology_tools import OntologyTools



ontology_settings = OntologySettings(
    base_iri="http://www.test.org/chem_ontologies/",
    ontology_file_name="backup-2.owl",
    directory_path="data/ontology/",
    closed_ontology_file_name=None
)

ontology_tools = OntologyTools(ontology_settings)

# print(ontology_tools.get_class_info("bis-formamides"))

props = ontology_tools._get_single_class_properties("bis-formamides")

print("props: ", props)
props_filtered = set(props) - {'has_information'}
print("props_filtered: ", props_filtered)

prop = ontology_settings.meta["is_stable_as"]
print("prop: ", prop)
print("type(prop): ", type(prop))




onto = get_ontology("http://www.test.org/chem_ontologies/backup-2.owl").load(only_local=True)

res_list = onto.search(iri = f"*/anion_antiport_mechanism", type = owlready2.owl_class)

print("res_list: ", res_list)
print(isinstance(res_list[0], ThingClass))


import random
cls = random.choice(list(onto.classes()))
print(cls)

super_classes = cls.is_a

for res in super_classes:
    print(res)
    is_res = isinstance(res,owlready2.Restriction)
    print(is_res)
    if is_res:
        print(res.property)
        print(type(res.property))
        print(res.property.name)
        print(res.property.name == "has_information")
        print(isinstance(res.property, (ObjectPropertyClass, DataPropertyClass)))
        print("-"*100)
        print(type(res.subclasses))
        print(res.__dict__["type"])
        print(res.type)
        print(getattr(res,"type", -1))


    print("-"*200)

