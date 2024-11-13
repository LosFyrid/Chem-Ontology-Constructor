from owlready2 import *
from config import settings

print(settings.ONTOLOGY_CONFIG["ontology_directory_path"])

onto_path.append(settings.ONTOLOGY_CONFIG["ontology_directory_path"])
ontology = get_ontology("http://www.test.org/chem_ontologies/chem_ontology.owl#").load(only_local=True)

print(type(ontology))
