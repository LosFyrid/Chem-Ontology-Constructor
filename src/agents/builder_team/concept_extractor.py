from __future__ import annotations  
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Type, Union
import json
import os
import warnings


from owlready2 import get_ontology, types, Thing

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate


from config import settings
 
llm = ChatOpenAI(
    model=settings.LLM_CONFIG["model"],
    temperature=settings.LLM_CONFIG["temperature"],
    api_key=settings.OPENAI_API_KEY,
    streaming=settings.LLM_CONFIG["streaming"]
)

class Example(BaseModel):
    context: str
    concepts: List[Concept]

class Concept(BaseModel):
    name: str
    is_data_property: bool
    information: str

class IndividualMetaData(BaseModel):
    name: str
    embedding: List[float] = Field(default_factory=list)
    data_properties: Dict[str, str] = Field(default_factory=dict)
    location: Tuple[str, str] = Field(default_factory=lambda: ("", ""))
    information: str = ""

class ClassMetaData(BaseModel):
    name: str
    embedding: List[float] = Field(default_factory=list)
    location: Tuple[str, str] = Field(default_factory=lambda: ("", ""))
    information: str = ""

def save_examples_to_json(examples: List[Example], filepath: str = None):
    if not examples:
        raise ValueError("示例列表不能为空。")

    # 检查所有实例是否都是Example类型
    if not all(isinstance(example, Example) for example in examples):
        raise TypeError("列表中所有实例必须是Example类型。")

    # 使用传入的文件路径或默认路径
    filename = filepath or settings.EXTRACTOR_EXAMPLES_CONFIG["concept_file_path"]

    # 读取已有数据(如果文件存在)
    existing_examples = []
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_examples = [Example(**data) for data in existing_data]

    # 合并示例,检测重复
    all_examples = existing_examples.copy()
    for new_example in examples:
        is_duplicate = False
        for existing in existing_examples:
            # 比较context和concepts内容是否完全相同
            if (existing.context == new_example.context and
                len(existing.concepts) == len(new_example.concepts) and
                all(ec.model_dump() == nc.model_dump() 
                    for ec, nc in zip(existing.concepts, new_example.concepts))):
                is_duplicate = True
                break
        
        if not is_duplicate:
            all_examples.append(new_example)

    # 保存合并后的数据
    json_data = json.dumps([example.model_dump() for example in all_examples], 
                          ensure_ascii=False, indent=4)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_data)

def _load_examples_from_json(filepath: str = None) -> List[Example]:
    filename = filepath or settings.EXTRACTOR_EXAMPLES_CONFIG["concept_file_path"]
    
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return []
        
    with open(filename, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
        
    examples = [Example(**data) for data in examples_data]
    
    return examples


def generate_prompt() -> str:
    examples = _load_examples_from_json()

    task_prompt = PromptTemplate(
        template = """# Ontology Entities Extraction in Chemistry

## Task Overview
In the field of chemistry, we aim to find all concepts and entities including classes, individuals and data properties within an ontological framework. 

## Definitions

- **Class:** A class represents a broad category or group that can include multiple entities or concepts. If a concept A can be linked to a set of other concepts or entities {{B}}, and it can be stated that {{B}} are examples of A or belong to A, then A is a class.

- **Individual:** An individual is a distinct, singular entity or concept that does not include other entities. It is unique and cannot be subdivided into further examples or categories.

- **Data Property:** A data property is a concept that describes the characteristics of an entity. It is a property describing an entity itself by a value which means ONLY ONE ENTITY exclude itself appears. Also, a property describing the relationship between two entities is not data property.

## Task Instructions

1. **Analyze the Text:** Thoroughly read the provided text to understand the context.
2. **Extract Entities:** Identify and extract all relevant entities and concepts from the text.
3. **Find Data Properties:**
   - If the concept describes an attribute of an entity or a concept, classify it as a **"data property."**
4. **Extract Information about the Entity:** Extract information in one sentence about the entity or concept in the text."""
    )
    
    examples_prompt = ""
    if examples:
        examples_prompt = "\n\n## Examples\n\n"
        for example in examples:
            examples_prompt += f"**Text:**\n    {example.context}\n\nOutput:\n"
            for concept in example.concepts:
                examples_prompt += f"Name: {concept.name}\n"
                examples_prompt += f"Is data property: {concept.is_data_property}\n"
                examples_prompt += f"Information: {concept.information}\n\n"
    
    output_format_prompt = PromptTemplate(
        template="""For each identified entity, output only:
Name: [Specific entity name/identifier]
Is data property: [Yes or No]
Information: [Information about the entity in one sentence]

If the entity has abbreviations, format its Name [Full name]([Abbreviation]).

Apply this structured approach to analyze the text I offer later and accurately extract the entities.
"""
    )

    final_prompt_template = PromptTemplate.from_template(
        "{task}\n{examples}\n{format}"
    )
    
    prompt_pipeline = PipelinePromptTemplate(
        final_prompt=final_prompt_template,
        pipeline_prompts=[
            ("task", task_prompt),
            ("examples", PromptTemplate.from_template(examples_prompt)),
            ("format", output_format_prompt)
        ]
    )
    
    return prompt_pipeline.format()


def parse_llm_output(response_content: str) -> Tuple[List[ClassMetaData], List[IndividualMetaData]]:
    class_concepts = []
    individual_concepts = []
    
    # 按空行分割每个概念块
    concept_blocks = [block.strip() for block in response_content.split('\n\n') if block.strip()]
    
    for block in concept_blocks:
        # 解析每个概念的属性
        properties = {}
        for line in block.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                properties[key.strip().lower()] = value.strip()
                
        # 提取关键信息
        name = properties.get('name', '')
        classification = properties.get('classification', '').lower()
        information = properties.get('information', '')
        
        # 根据classification创建对应的元数据对象
        if classification == 'class':
            class_concept = ClassMetaData(
                name=name,
                information=information
            )
            class_concepts.append(class_concept)
        elif classification == 'individual':
            individual_concept = IndividualMetaData(
                name=name,
                information=information
            )
            individual_concepts.append(individual_concept)
            
    return class_concepts, individual_concepts







