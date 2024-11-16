from typing import List
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate
from concept_extractor import IndividualMetaData, ClassMetaData


def generate_relationship_prompt(subject: str, object: str) -> str:
    return f"Please describe the relationship between {subject} and {object}. Your response must use {subject} as the subject and {object} as the object/predicate complement. Only the verb/linking verb can be changed and can be modified by adverbs or other modifiers. Use either subject-predicate-object or subject-linking verb-predicate complement structure. If there is no meaningful relationship between {subject} and {object}, output 'None'."

def generate_prompt(individuals: List[IndividualMetaData], classes: List[ClassMetaData], text_content: str) -> str:
    # Template for formatting class information
    class_template = """Available Classes:
{class_descriptions}"""

    class_info_template = """{name}"""

    # Template for formatting individual information  
    individual_template = """Individuals to analyze:
{individual_descriptions}"""

    individual_info_template = """{name}"""

    # Final template for relationship task
    final_template = """# Relationship Analysis Task

## Task Overview
For each pair of individual and class, describe their relationship using a subject-predicate-object or subject-linking verb-predicate complement structure.

## Input Information
{class_info}

{individual_info}

## Text
{text_content}

## Task Instructions
1. Text Analysis: Read throughoutly and make sure you comprehensively understand the provided Text.
2. BASED ON THE TEXT, for each individual-class pair:
   - Use the INDIVIDUAL ITSELF (WITHOUT ANY MODIFICATION) as the SUBJECT
   - Use the CLASS ITSELF (WITHOUT ANY MODIFICATION) as the OBJECT/PREDICATE COMPLEMENT
   - ONLY modify the verb/linking verb with appropriate modifiers
   - If the relationship is not clear from the text, output 'None'
   - If "HAS", "IS" and words with similar meanings are used in the relationship, output "Data Property: [individual] [verb] [class]"

## Output Format
For each individual-class pair, output a single sentence using either subject-predicate-object or subject-linking verb-predicate complement structure.

Please ensure all possible individual-class pairs are analyzed."""

    # Create individual prompt templates
    class_info_prompt = PromptTemplate(
        input_variables=["name"],
        template=class_info_template
    )

    individual_info_prompt = PromptTemplate(
        input_variables=["name"],
        template=individual_info_template
    )

    # Format class descriptions
    class_descriptions = "\n".join([
        class_info_prompt.format(name=cls.name)
        for cls in classes
    ])

    # Format individual descriptions
    individual_descriptions = "\n".join([
        individual_info_prompt.format(name=ind.name)
        for ind in individuals
    ])

    # Create intermediate prompts
    class_prompt = PromptTemplate(
        input_variables=["class_descriptions"],
        template=class_template
    )

    individual_prompt = PromptTemplate(
        input_variables=["individual_descriptions"],
        template=individual_template
    )

    # Create final pipeline prompt
    full_prompt = PipelinePromptTemplate(
        final_prompt=PromptTemplate(
            input_variables=["class_info", "individual_info", "text_content"],
            template=final_template
        ),
        pipeline_prompts=[
            ("class_info", class_prompt),
            ("individual_info", individual_prompt),
        ]
    )

    return full_prompt.format(
        class_descriptions=class_descriptions,
        individual_descriptions=individual_descriptions,
        text_content=text_content
    )

# def generate_prompt(individuals: List[IndividualMetaData], classes: List[ClassMetaData], text_content: str) -> str:
#     # Template for formatting class information
#     class_template = """Available Classes and their descriptions:
# {class_descriptions}"""

#     class_info_template = """Class Name: {name}
# Context: {information}"""

#     # Template for formatting individual information  
#     individual_template = """Individuals to be classified:
# {individual_descriptions}"""

#     individual_info_template = """Individual Name: {name}
# Context: {information}"""

#     # Final template for classification task
#     final_template = """# Individual Classification Task

# ## Task Overview
# Determine the class membership of chemical individuals based on provided information and context.

# ## Input Information
# {class_info}

# {individual_info}

# ## Text
# {text_content}

# ## Task Instructions
# 1. Text Analysis: Read and fully understand the provided Text
# 2. Classify each individual into classes: If a class and individual can be expressed as "[individual] is a [class]", then the individual can be classified into that class. One individual can belong to multiple classes

# ## Output Format
# For each individual, output only:
# Individual Name: [name]
# Belongs to Classes: [class1], [class2], ...

# Please ensure complete classification results for all individuals."""

#     # Create individual prompt templates
#     class_info_prompt = PromptTemplate(
#         input_variables=["name", "information"],
#         template=class_info_template
#     )

#     individual_info_prompt = PromptTemplate(
#         input_variables=["name", "information"],
#         template=individual_info_template
#     )

#     # Format class descriptions
#     class_descriptions = "\n\n".join([
#         class_info_prompt.format(name=cls.name, information=cls.information)
#         for cls in classes
#     ])

#     # Format individual descriptions
#     individual_descriptions = "\n\n".join([
#         individual_info_prompt.format(name=ind.name, information=ind.information)
#         for ind in individuals
#     ])

#     # Create intermediate prompts
#     class_prompt = PromptTemplate(
#         input_variables=["class_descriptions"],
#         template=class_template
#     )

#     individual_prompt = PromptTemplate(
#         input_variables=["individual_descriptions"],
#         template=individual_template
#     )

#     # Create final pipeline prompt
#     full_prompt = PipelinePromptTemplate(
#         final_prompt=PromptTemplate(
#             input_variables=["class_info", "individual_info", "text_content"],
#             template=final_template
#         ),
#         pipeline_prompts=[
#             ("class_info", class_prompt),
#             ("individual_info", individual_prompt),
#         ]
#     )

#     return full_prompt.format(
#         class_descriptions=class_descriptions,
#         individual_descriptions=individual_descriptions,
#         text_content=text_content
#     )