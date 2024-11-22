import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv

from owlready2 import onto_path, get_ontology

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default_api_key")


path_matcher = re.compile(r'\$\{([^}^{]+)\}')
key_matcher = re.compile(r'\{\{([^}^{]+)\}\}')

def path_constructor(loader, node):
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var, '') + value[match.end():]

def resolve_key_references(yaml_dict):
    def _resolve_value(value, yaml_dict):
        if isinstance(value, str):
            match = key_matcher.search(value)
            if match:
                key = match.group(1).strip()
                if key in yaml_dict:
                    return value.replace(match.group(0), str(yaml_dict[key]))
        return value

    resolved_dict = {}
    for key, value in yaml_dict.items():
        if isinstance(value, dict):
            resolved_dict[key] = resolve_key_references(value)
        else:
            resolved_dict[key] = _resolve_value(value, yaml_dict)
    return resolved_dict

yaml.SafeLoader.add_implicit_resolver('!path', path_matcher, None)
yaml.SafeLoader.add_constructor('!path', path_constructor)

config_path = Path(__file__).parent / "settings.yaml"
with open(config_path, "r") as f:
    yaml_settings = yaml.safe_load(f)
    yaml_settings = resolve_key_references(yaml_settings)


_ONTOLOGY_CONFIG = yaml_settings["ontology"]
onto_path.append(_ONTOLOGY_CONFIG["ontology_directory_path"])
ontology = get_ontology(_ONTOLOGY_CONFIG["ontology_iri"]).load(only_local=True)

LLM_CONFIG = yaml_settings["LLM"]
EXTRACTOR_EXAMPLES_CONFIG = yaml_settings["extractor_examples"]
DATASET_CONSTRUCTION_CONFIG = yaml_settings["dataset_construction"]

ONTOLOGY_CONFIG = {
    "ontology": ontology,
    "meta": ontology.get_namespace(_ONTOLOGY_CONFIG["namespace_meta_iri"]),
    "classes": ontology.get_namespace(_ONTOLOGY_CONFIG["namespace_classes_iri"]),
    "individuals": ontology.get_namespace(_ONTOLOGY_CONFIG["namespace_individuals_iri"]),
    "data_properties": ontology.get_namespace(_ONTOLOGY_CONFIG["namespace_data_properties_iri"]),
    "object_properties": ontology.get_namespace(_ONTOLOGY_CONFIG["namespace_object_properties_iri"]),
    "axioms": ontology.get_namespace(_ONTOLOGY_CONFIG["namespace_axioms_iri"])
}
