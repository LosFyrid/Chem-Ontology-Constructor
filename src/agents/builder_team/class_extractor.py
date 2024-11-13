from __future__ import annotations  
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Type, Union
import json
import os
import warnings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate, PipelinePromptTemplate

from owlready2 import get_ontology

from config import settings

llm = ChatOpenAI(
    model=settings.LLM_CONFIG["model"],
    temperature=settings.LLM_CONFIG["temperature"],
    api_key=settings.OPENAI_API_KEY,
    streaming=settings.LLM_CONFIG["streaming"]
)

class Class(BaseModel):
    """A class representing a class concept from chemical research text.

    This class captures abstract concepts and classes with their properties.

    Attributes:
        name (str): The name or identifier of the class.
        properties (Dict[str, str]): Key-value pairs of properties defining this class.
        context (str): The original text context where this class was found.
    """
    name: str
    properties: Dict[str, str]
    context: str

def save_classes_to_json(classes: List[Class]):
    """Saves a list of Class instances to JSON file.

    If the file exists, appends new instances using duplicate detection rules.

    Args:
        classes: List of Class instances to save to JSON.

    Raises:
        ValueError: If the class list is empty.
    """
    if not classes:
        raise ValueError("Class list cannot be empty.")

    filename = settings.EXTRACTOR_EXAMPLES_CONFIG["classes_file_path"]

    # Read existing data if file exists
    existing_classes = []
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_classes = [Class(**data) for data in existing_data]

    # Merge classes using duplicate detection
    all_classes = existing_classes.copy()
    for new_class in classes:
        is_duplicate = False
        for existing in existing_classes:
            if (existing.name == new_class.name and 
                existing.properties == new_class.properties and
                existing.context == new_class.context):
                is_duplicate = True
                break
        
        if not is_duplicate:
            all_classes.append(new_class)

    # Save merged data
    json_data = json.dumps([class_.model_dump() for class_ in all_classes], ensure_ascii=False, indent=4)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_data)

def _load_classes_from_json() -> List[Class]:
    """Loads Class instances from JSON file.
    
    Returns:
        List[Class]: List of loaded Class instances
    """
    with open(settings.EXTRACTOR_EXAMPLES_CONFIG["classes_file_path"], 'r', encoding='utf-8') as f:
        classes_data = json.load(f)
    return [Class(**data) for data in classes_data]

def delete_classes_from_json(names: List[str], **kwargs):
    """Deletes specified classes from JSON file.

    Args:
        names: List of class names to delete
        **kwargs: Optional keyword arguments for deletion:
            - properties (dict): Properties dict to match
            - context (str): Context string to match

    Raises:
        ValueError: If names list is empty or no matching classes found
        FileNotFoundError: If the target JSON file doesn't exist or is empty
    """
    if not names:
        raise ValueError("Names list cannot be empty")

    filename = settings.EXTRACTOR_EXAMPLES_CONFIG["classes_file_path"]

    # Read existing data
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        raise FileNotFoundError(f"File {filename} does not exist or is empty")

    with open(filename, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
        existing_classes = [Class(**data) for data in existing_data]

    deleted_classes = []
    remaining_classes = existing_classes.copy()

    # Delete matching classes
    for name in names:
        class_found = False
        classes_to_remove = []
        
        for existing in existing_classes:
            if existing.name != name:
                continue
                
            # Check all provided optional fields
            match = True
            if 'properties' in kwargs and kwargs['properties']:
                if existing.properties != kwargs['properties']:
                    match = False
            if 'context' in kwargs and kwargs['context']:
                if existing.context != kwargs['context']:
                    match = False
            if match:
                classes_to_remove.append(existing)
                deleted_classes.append(existing)
                class_found = True
        
        # Remove all matching classes
        for class_ in classes_to_remove:
            remaining_classes.remove(class_)
            
        if not class_found:
            raise ValueError(f"No matching class found: {name}")

    # Save remaining classes
    json_data = json.dumps([class_.model_dump() for class_ in remaining_classes], ensure_ascii=False, indent=4)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_data)

    # Output deleted classes
    print("Deleted:")
    for class_ in deleted_classes:
        print(class_.model_dump_json(indent=2))

def generate_prompt() -> str:
    """Generates a structured prompt for extracting classes from chemical research texts.
    
    This function constructs a comprehensive prompt template that guides the extraction of 
    class concepts from chemical research texts. It loads predefined examples from 
    data/extractor_examples/class/classes.json file, then assembles them into a structured 
    prompt with clear instructions and formatting requirements.
    
    Returns:
        str: A formatted prompt string containing task description, examples,
             and output format instructions.
    """
    classes = _load_classes_from_json()

    task_prompt = PromptTemplate(
        template="""You are a specialized extractor whose role is to identify and extract class concepts from chemical research texts. In ontological terms, a class represents a general concept or category - not a specific instance with defined parameters.

Your core responsibility is to carefully analyze the text and identify classes. A class should be a general concept that can have multiple instances or examples."""
    )
    
    examples_prompt = PromptTemplate(
        template="""\nExamples:\n""" + "\n".join([
            f"Input text:\n    {ex.context}\n"
            f"Output:\n    Class: {ex.name}\n"
            f"    Properties: {'; '.join(f'{k}: {v}' for k, v in ex.properties.items())}\n"
            for ex in classes
        ])
    )
    
    output_format_prompt = PromptTemplate(
        template="""For each identified class, output only:
Class: [Class name/concept]
Properties: [General properties that define this class]

If the class has abbreviations, format its name [Full name]([Abbreviation])

Do not provide any analysis, explanation, or additional information. Focus solely on extraction.
"""
    )

    prompt_pipeline = PipelinePromptTemplate(
        final_prompt=PromptTemplate.from_template("{task}\n{examples}\n{format}"),
        pipeline_prompts=[
            ("task", task_prompt),
            ("examples", examples_prompt),
            ("format", output_format_prompt)
        ]
    )
    
    return prompt_pipeline.format()


# 生成示例类
example_classes = [
    Class(
        name="Metal-Organic Framework",
        properties={
            "composition": "Metal nodes connected by organic linkers",
            "structure": "Crystalline porous network",
            "application": "Gas storage, catalysis, separation"
        },
        context="Metal-organic frameworks (MOFs) are crystalline materials consisting of metal nodes connected by organic linkers to form porous networks. These materials are widely used in gas storage, catalysis and separation applications."
    ),
    Class(
        name="Solvothermal Synthesis",
        properties={
            "process_type": "Chemical synthesis method",
            "conditions": "High temperature and pressure",
            "medium": "Solvent-based reaction"
        },
        context="Solvothermal synthesis is a method of producing chemical compounds by heating a solution in a sealed vessel. The technique uses high temperature and pressure conditions to facilitate the formation of crystalline products."
    ),
    Class(
        name="X-ray Diffraction",
        properties={
            "technique_type": "Analytical method",
            "principle": "Diffraction of X-rays by crystal lattices",
            "purpose": "Structure determination"
        },
        context="X-ray diffraction (XRD) is an analytical technique used to determine the atomic and molecular structure of crystals. The method works by measuring the diffraction patterns produced when X-rays interact with a crystalline sample."
    )
]

# 保存示例类到JSON文件
save_classes_to_json(example_classes)
