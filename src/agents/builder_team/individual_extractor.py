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

class IndividualCategory(BaseModel):
    """A class representing a category of individuals in chemical research.

    This class defines a category that groups similar individuals together based on their
    characteristics and provides examples of that category.

    Attributes:
        name (str): The name of the category. Must be unique as it serves as the identifier - 
                   two IndividualCategory instances with the same name are considered equal.
        description (str): A detailed description of what this category represents.
        examples (List[Individual]): A list of example individuals belonging to this category.
    """
    name: str
    description: str
    examples: List[Individual] = Field(default_factory=list)

    def __eq__(self, other):
        """Checks equality between two IndividualCategory instances.
        
        Args:
            other: Another object to compare with.
            
        Returns:
            bool: True if other is an IndividualCategory with the same name, False otherwise.
        """
        if isinstance(other, IndividualCategory):
            return self.name == other.name
        return False

class Individual(BaseModel):
    """A class representing a specific individual instance from chemical research text.

    This class captures concrete instances with their properties, category classification,
    and original context.

    Attributes:
        name (str): The name or identifier of the individual instance.
        category (IndividualCategory): The category this individual belongs to.
        parameters (Dict[str, str]): Key-value pairs of parameters defining this instance.
        context (str): The original text context where this individual was found.
    """
    name: str
    category: IndividualCategory  
    parameters: Dict[str, str]
    context: str

def save_instances_to_json(instances: List[BaseModel]):
    """Saves a list of Pydantic model instances to JSON file.

    Automatically selects the output filename based on instance type.
    If the file exists, appends new instances using type-specific duplicate detection rules.

    Args:
        instances: List of Pydantic model instances (IndividualCategory or Individual)
            to save to JSON.

    Raises:
        ValueError: If the instance list is empty.
        TypeError: If the instances are not all IndividualCategory or Individual type.
    """
    if not instances:
        raise ValueError("Instance list cannot be empty.")

    # Check instance type
    instance_type = type(instances[0])
    if all(isinstance(instance, IndividualCategory) for instance in instances):
        filename = settings.EXTRACTOR_EXAMPLES_CONFIG["individual_categories_file_path"]
        is_category = True
    elif all(isinstance(instance, Individual) for instance in instances):
        filename = settings.EXTRACTOR_EXAMPLES_CONFIG["individuals_file_path"]
        is_category = False
    else:
        raise TypeError("All instances in the list must be of type IndividualCategory or Individual.")

    # Read existing data if file exists
    existing_instances = []
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_instances = [instance_type(**data) for data in existing_data]

    # Merge instances using type-specific duplicate detection
    all_instances = existing_instances.copy()
    for new_instance in instances:
        is_duplicate = False
        for existing in existing_instances:
            if is_category:
                # For IndividualCategory, compare only name field
                if existing.name == new_instance.name:
                    is_duplicate = True
                    break
            else:
                # For Individual, compare all four fields
                if (existing.name == new_instance.name and 
                    existing.category == new_instance.category and
                    existing.parameters == new_instance.parameters and
                    existing.context == new_instance.context):
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            all_instances.append(new_instance)

    # Save merged data
    json_data = json.dumps([instance.model_dump() for instance in all_instances], ensure_ascii=False, indent=4)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_data)

def _load_instances_from_json() -> Tuple[List[Individual], List[IndividualCategory]]:
    """Loads Individual and IndividualCategory instances from JSON files.
    
    This function reads Individual and IndividualCategory instances from their respective
    JSON files and automatically associates Individual instances with their corresponding
    IndividualCategory.

    Returns:
        A tuple containing:
            - List[Individual]: List of loaded Individual instances
            - List[IndividualCategory]: List of loaded IndividualCategory instances with
              associated Individual examples

    Warns:
        UserWarning: If an Individual cannot be matched to any IndividualCategory
    """
    # Load Individual instances
    with open(settings.EXTRACTOR_EXAMPLES_CONFIG["individuals_file_path"], 'r', encoding='utf-8') as f:
        individuals_data = json.load(f)
    individuals = [Individual(**data) for data in individuals_data]
    
    # Load IndividualCategory instances  
    with open(settings.EXTRACTOR_EXAMPLES_CONFIG["individual_categories_file_path"], 'r', encoding='utf-8') as f:
        categories_data = json.load(f)
    categories = [IndividualCategory(**data) for data in categories_data]
    
    # Associate Individual instances with their IndividualCategory
    for individual in individuals:
        found = False
        for category in categories:
            if individual.category == category:
                category.examples.append(individual)
                found = True
                break
        if not found:
            warnings.warn(f"Individual '{individual.name}' in '{individual.context}' was not added to any category")
    
    return individuals, categories

def delete_instances_from_json(instance_type: Union[Type[Individual], Type[IndividualCategory]], names: List[str], **kwargs):
    """Deletes specified instances from a JSON file.

    This function allows deletion of Individual or IndividualCategory instances from their
    respective JSON storage files. For Individual instances, additional parameters can be
    specified to match specific instances.

    Args:
        instance_type: The type of instances to delete (Individual or IndividualCategory)
        names: List of instance names to delete
        **kwargs: Optional keyword arguments for Individual deletion:
            - category (str): Category name to match
            - parameters (dict): Parameters dict to match
            - context (str): Context string to match

    Raises:
        ValueError: If names list is empty, instance type is invalid, or no matching instances found
        FileNotFoundError: If the target JSON file doesn't exist or is empty
    """
    if not names:
        raise ValueError("Names list cannot be empty")

    if instance_type == IndividualCategory:
        filename = settings.EXTRACTOR_EXAMPLES_CONFIG["individual_categories_file_path"]
        model_class = IndividualCategory
        if kwargs:
            raise ValueError("Additional parameters not supported when deleting IndividualCategory")
    elif instance_type == Individual:
        filename = settings.EXTRACTOR_EXAMPLES_CONFIG["individuals_file_path"]
        model_class = Individual
    else:
        raise ValueError("instance_type must be Individual or IndividualCategory")

    # Read existing data
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        raise FileNotFoundError(f"File {filename} does not exist or is empty")

    with open(filename, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
        existing_instances = [model_class(**data) for data in existing_data]

    deleted_instances = []
    remaining_instances = existing_instances.copy()

    # Delete matching instances
    for name in names:
        instance_found = False
        instances_to_remove = []
        
        for existing in existing_instances:
            if existing.name != name:
                continue
                
            if instance_type == IndividualCategory:
                remaining_instances.remove(existing)
                deleted_instances.append(existing)
                instance_found = True
                break
            else:
                # For Individual, check all provided optional fields
                match = True
                if 'category' in kwargs and kwargs['category']:
                    if existing.category.name != kwargs['category']:
                        match = False
                if 'parameters' in kwargs and kwargs['parameters']:
                    if existing.parameters != kwargs['parameters']:
                        match = False
                if 'context' in kwargs and kwargs['context']:
                    if existing.context != kwargs['context']:
                        match = False
                if match:
                    instances_to_remove.append(existing)
                    deleted_instances.append(existing)
                    instance_found = True
        
        # Remove all matching instances
        for instance in instances_to_remove:
            remaining_instances.remove(instance)
            
        if not instance_found:
            raise ValueError(f"No matching instance found: {name}")

    # Save remaining instances
    json_data = json.dumps([instance.model_dump() for instance in remaining_instances], ensure_ascii=False, indent=4)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_data)

    # Output deleted instances
    print("Deleted:")
    for instance in deleted_instances:
        print(instance.model_dump_json(indent=2))

def generate_prompt() -> str:
    """Generates a structured prompt for extracting individuals from chemical research texts.
    
    This function constructs a comprehensive prompt template that guides the extraction of 
    specific instances (individuals) from chemical research texts. It loads predefined 
    categories and their examples from data/extractor_examples/individual/individuals.json 
    and data/extractor_examples/individual/individual_categories.json files, then assembles 
    them into a structured prompt with clear instructions and formatting requirements.
    
    Returns:
        str: A formatted prompt string containing task description, category examples,
             and output format instructions.
    """
    _, categories = _load_instances_from_json()

    task_prompt = PromptTemplate(
        template="""You are a specialized extractor whose role is to identify and extract specific instances (individuals) from chemical research texts. In ontological terms, an individual represents a concrete, well-defined instance with specific properties - not an abstract class or general concept.

Each individual typically has a set of parameters that characterize it. These parameters are structured as key-value pairs that precisely define the individual's attributes and properties.

Your core responsibility is to carefully analyze the text and identify individuals that belong to the following categories:"""
    )
    
    category_prompts = []
    for category in categories:
        # Category description template and examples section
        prompt_text = f"\n{len(category_prompts)+1}. {category.description}\n"
        prompt_text += "\nExamples:\n"
        # Add examples
        for ex in category.examples:
            prompt_text += (
                          f"Input text:\n"
                          f"    {ex.context}\n"
                          f"Output:\n"
                          f"    Individual: {ex.name}\n"
                          f"    Parameters: {'; '.join(f'{k}: {v}' for k, v in ex.parameters.items())}\n\n")
        
        category_prompts.append(PromptTemplate(template=prompt_text))  
    
    output_format_prompt = PromptTemplate(
        template="""For each identified individual, output only:
Individual: [Specific instance name/identifier]
Parameters: [Specific parameters that define this instance]

If the individual has abbreviations, format its name [Full name]([Abbreviation])

Do not provide any analysis, explanation, or additional information. Focus solely on extraction.
"""

    )

    final_prompt_template = PromptTemplate.from_template("{task}\n" + "\n".join([f"{{categories_{i}}}" for i in range(len(category_prompts))]) + "\n{format}")

    fill_template_list = [
        ("task", task_prompt),
    ]
    for i in range(len(category_prompts)):
        fill_template_list.append((f"categories_{i}", category_prompts[i]))
    fill_template_list.append(("format", output_format_prompt))
    
    prompt_pipeline = PipelinePromptTemplate(
        final_prompt=final_prompt_template,
        pipeline_prompts=fill_template_list
    )
    
    return prompt_pipeline.format()

