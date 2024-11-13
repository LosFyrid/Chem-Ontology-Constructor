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
from builder_team_graph import BuilderTeamState

ontology = get_ontology(settings.ONTOLOGY_CONFIG["ontology_file_path"])
 
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
    classification: str
    justification: str
    information: str

class IndividualMetaData(BaseModel):
    name: str
    embedding: List[float]
    data_properties: Dict[str, str]
    location: List[Tuple[str, str]]
    information: List[str]

class ClassMetaData(BaseModel):
    name: str
    embedding: List[float]
    location: List[Tuple[str, str]]
    information: List[str]

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
        template = """# Ontology Classification in Chemistry

## Task Overview

In the field of chemistry, we aim to classify concepts as either a **"class"** or an **"individual"** within an ontological framework.

## Definitions

- **Class:** A class represents a broad category or group that can include multiple entities or concepts. If a concept A can be linked to a set of other concepts or entities {{B}}, and it can be stated that {{B}} are examples of A or belong to A, then A is a class.

- **Individual:** An individual is a distinct, singular entity or concept that does not include other entities. It is unique and cannot be subdivided into further examples or categories.

## Task Instructions

1. **Analyze the Text:** Thoroughly read the provided text to understand the context.
2. **Extract Concepts:** Identify and extract all relevant concepts and entities from the text.
3. **Classify Each Concept:**
   - If a concept can be associated with multiple examples or entities, classify it as a **"class."**
   - If the concept is unique and specific, classify it as an **"individual."**
4. **Justify Your Classification:** Provide a concise explanation for each classification to support your decision.
5. **Extract Information about the Concept:** Extract information in one sentence about the concept in the text."""
    )
    
    examples_prompt = ""
    if examples:
        examples_prompt = "\n\n## Examples\n\n"
        for example in examples:
            examples_prompt += f"**Text:**\n    {example.context}\n\nOutput:\n"
            for concept in example.concepts:
                examples_prompt += f"Name: {concept.name}\n"
                examples_prompt += f"Classification: {concept.classification}\n"
                examples_prompt += f"Justification: {concept.justification}\n"
                examples_prompt += f"Information: {concept.information}\n\n"
    
    output_format_prompt = PromptTemplate(
        template="""For each identified concept, output only:
Name: [Specific concept name/identifier]
Classification: [Class or Individual]
Justification: [Explanation for the classification]
Information: [Information about the concept in one sentence]

If the concept has abbreviations, format its Name [Full name]([Abbreviation]).

Apply this structured approach to analyze the text I offer later and accurately classify the concepts.
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


def create_classes(state: BuilderTeamState) -> BuilderTeamState:
    """
    将ClassMetaData列表中的类定义转换为本体中的实际类,并创建数据属性
    
    Args:
        state: BuilderTeamState - 构建团队的当前状态
        
    Returns:
        BuilderTeamState - 更新后的状态
    """
    ontology = state["ontology"]
    
    for class_meta in state["classes"]:
        # 检查类是否已存在
        if not class_meta.name in ontology.classes():
            # 在本体中创建新类
            with ontology:
                new_class = types.new_class(class_meta.name, (Thing,))
                
                # 创建embedding注解属性
                embedding_prop = types.new_class("embedding", (types.AnnotationProperty,))
                setattr(new_class, "embedding", class_meta.embedding)
                
                # 创建location注解属性 
                location_prop = types.new_class("location", (types.AnnotationProperty,))
                location_str = ";".join([f"{loc[0]}:{loc[1]}" for loc in class_meta.location])
                setattr(new_class, "location", location_str)
                
                # 创建information注解属性
                info_prop = types.new_class("information", (types.AnnotationProperty,))
                info_str = ";".join(class_meta.information)
                setattr(new_class, "information", info_str)
                
    return state





