from typing import List, Tuple, Dict, Optional, Set
from owlready2 import *
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from autonogy_constructor.idea.query_team.utils import parse_json


class OntologyTools:
    """Tools for ontology querying and parsing
    
    This class provides comprehensive tools for working with OWL ontologies:
    1. Basic Information: Get basic class information and metadata
    2. Property Operations: Query and analyze property relationships
    3. Hierarchy Operations: Navigate and analyze class hierarchies
    4. Semantic Analysis: Analyze relationships and similarities
    5. Parsing Operations: Parse complete definitions and structures
    """
    
    def __init__(self, ontology):
        self.onto = ontology

    #######################
    # Basic Information
    #######################
    
    def get_class_info(self, class_name: str) -> Dict:
        """Get basic information about a class"""
        cls = self.onto[class_name]
        return {
            "name": cls.name,
            "information": list(cls.information) if hasattr(cls, "information") else [],
            "source": list(cls.source) if hasattr(cls, "source") else []
        }
    
    def get_information_sources(self, class_name: str) -> List[str]:
        """Get all information sources of a class"""
        cls = self.onto[class_name]
        return list(cls.source) if hasattr(cls, "source") else []

    def get_information_by_source(self, class_name: str, source: str) -> List[str]:
        """Get information from a specific source for a class"""
        cls = self.onto[class_name]
        if not hasattr(cls, "has_information"):
            return []
        return [info.content for info in cls.has_information if info.source == source]

    #######################
    # Property Operations
    #######################
    
    def get_class_properties(self, class_name: str) -> List[str]:
        """Get all properties associated with a class"""
        cls = self.onto[class_name]
        properties = set()
        
        # Get properties from restrictions
        for r in cls.is_a:
            if isinstance(r, Restriction):
                properties.add(r.property.name)
        
        # Get directly declared properties
        for prop in cls.get_properties():
            properties.add(prop.name)
            
        return sorted(list(properties))
    
    def get_property_restrictions(self, class_name: str, property_name: str) -> List[Dict]:
        """Get all restrictions on a specific property for a class"""
        cls = self.onto[class_name]
        restrictions = []
        
        for r in cls.is_a:
            if isinstance(r, Restriction) and r.property.name == property_name:
                restrictions.append({
                    "type": str(r.type),
                    "value": str(r.value),
                    "raw_value": r.value  # Keep the original value for further processing
                })
        
        return restrictions
    
    def get_property_values(self, class_name: str, property_name: str) -> Set:
        """Get all values associated with a property for a class"""
        restrictions = self.get_property_restrictions(class_name, property_name)
        values = set()
        
        for r in restrictions:
            value = r["raw_value"]
            if isinstance(value, ThingClass):
                values.add(value.name)
            elif hasattr(value, "__iter__"):
                values.update(v.name for v in value if isinstance(v, ThingClass))
                
        return values

    #######################
    # Hierarchy Operations
    #######################
    
    def get_parents(self, class_name: str) -> List[str]:
        """Get direct parent classes"""
        cls = self.onto[class_name]
        return [c.name for c in cls.is_a if isinstance(c, ThingClass)]
    
    def get_children(self, class_name: str) -> List[str]:
        """Get direct child classes"""
        cls = self.onto[class_name]
        return [c.name for c in cls.subclasses()]
    
    def get_ancestors(self, class_name: str) -> List[str]:
        """Get all ancestor classes"""
        cls = self.onto[class_name]
        return [c.name for c in cls.ancestors() if isinstance(c, ThingClass)]
    
    def get_descendants(self, class_name: str) -> List[str]:
        """Get all descendant classes"""
        cls = self.onto[class_name]
        return [c.name for c in cls.descendants() if isinstance(c, ThingClass)]

    #######################
    # Semantic Analysis
    #######################
    
    def get_related_classes(self, class_name: str) -> Dict[str, List[str]]:
        """Get classes related through object properties"""
        cls = self.onto[class_name]
        relations = {}
        
        # 获取所有对象属性
        for prop in self.onto.object_properties():
            related = []
            # 通过限制获取关联的类
            for r in cls.is_a:
                if isinstance(r, Restriction) and r.property == prop:
                    if isinstance(r.value, ThingClass):
                        related.append(r.value.name)
                    elif hasattr(r.value, "__iter__"):
                        related.extend(v.name for v in r.value if isinstance(v, ThingClass))
            
            # 通过直接属性值获取关联的类
            if hasattr(cls, prop.name):
                values = getattr(cls, prop.name)
                if isinstance(values, list):
                    related.extend(v.name for v in values if isinstance(v, ThingClass))
                elif isinstance(values, ThingClass):
                    related.append(values.name)
            
            if related:
                relations[prop.name] = sorted(list(set(related)))  # 去重并排序
                
        return relations

    def get_property_path(self, start_class: str, end_class: str, max_depth: int = 5) -> List[List[str]]:
        """Find property paths connecting two classes"""
        paths = []
        visited = set()
        
        def dfs(current: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(path[:])
                return
            if current in visited:
                return
                
            visited.add(current)
            for prop_name, related in self.get_related_classes(current).items():
                for cls in related:
                    if cls not in visited:
                        dfs(cls, target, path + [prop_name], depth + 1)
            visited.remove(current)
        
        dfs(start_class, end_class, [], 0)
        return paths
    
    def get_semantic_similarity(self, class1: str, class2: str) -> float:
        """Calculate semantic similarity between two classes"""
        cls1 = self.onto[class1]
        cls2 = self.onto[class2]
        
        # 计算属性重叠度
        props1 = set(self.get_class_properties(class1))
        props2 = set(self.get_class_properties(class2))
        prop_sim = len(props1 & props2) / len(props1 | props2) if props1 or props2 else 0
        
        # 计算共同祖先
        ancestors1 = set(self.get_ancestors(class1))
        ancestors2 = set(self.get_ancestors(class2))
        ancestor_sim = len(ancestors1 & ancestors2) / len(ancestors1 | ancestors2) if ancestors1 or ancestors2 else 0
        
        # 计算信息内容相似度
        info1 = set(info.content for info in cls1.has_information) if hasattr(cls1, "has_information") else set()
        info2 = set(info.content for info in cls2.has_information) if hasattr(cls2, "has_information") else set()
        info_sim = len(info1 & info2) / len(info1 | info2) if info1 or info2 else 0
        
        # 加权平均
        return 0.4 * prop_sim + 0.4 * ancestor_sim + 0.2 * info_sim
    
    def get_disjoint_classes(self, class_name: str) -> List[str]:
        """Get classes that are explicitly declared as disjoint"""
        cls = self.onto[class_name]
        disjoint = set()
        
        for d in cls.disjoints():
            disjoint.update(c.name for c in d.entities if isinstance(c, ThingClass))
            
        disjoint.discard(class_name)  # Remove the class itself
        return sorted(list(disjoint))
    
    def get_inconsistent_classes(self) -> List[str]:
        """Get all inconsistent classes in the ontology"""
        close_world(self.onto)
        with self.onto:
            sync_reasoner_pellet(
                infer_property_values = True,
                infer_data_property_values = True
            )
        return [cls.name for cls in default_world.inconsistent_classes()]

    #######################
    # Parsing Operations
    #######################
    
    def parse_class_definition(self, class_name: str) -> Dict:
        """Parse complete class definition"""
        result = {
            "basic_info": self.get_class_info(class_name),
            "properties": {
                "data": [],
                "object": []
            },
            "hierarchy": {
                "parents": self.get_parents(class_name),
                "children": self.get_children(class_name)
            },
            "relations": self.get_related_classes(class_name)
        }
        
        # 获取所有属性
        for prop_name in self.get_class_properties(class_name):
            prop = self.onto[prop_name]
            if isinstance(prop, owlready2.DataProperty):
                result["properties"]["data"].append({
                    "name": prop_name,
                    "values": list(self.get_property_values(class_name, prop_name)),
                    "restrictions": self.get_property_restrictions(class_name, prop_name)
                })
            else:
                result["properties"]["object"].append({
                    "name": prop_name,
                    "values": list(self.get_property_values(class_name, prop_name)),
                    "restrictions": self.get_property_restrictions(class_name, prop_name)
                })
                
        return result
    
    def parse_property_definition(self, property_name: str) -> Dict:
        """Parse complete property definition"""
        prop = self.onto[property_name]
        result = {
            "name": property_name,
            "type": "data" if isinstance(prop, owlready2.DataProperty) else "object",
            "domain": [c.name for c in prop.domain] if prop.domain else [],
            "range": [c.name for c in prop.range] if prop.range else [],
            "usage": []
        }
        
        # 查找所有使用该属性的类
        for cls in self.onto.classes():
            restrictions = self.get_property_restrictions(cls.name, property_name)
            if restrictions:
                result["usage"].append({
                    "class": cls.name,
                    "restrictions": restrictions
                })
                
        return result
    
    def parse_hierarchy_structure(self, root_class: str = None) -> Dict:
        """Parse complete hierarchy structure"""
        visited = set()  # 添加循环检测
        
        def build_tree(cls_name):
            if cls_name in visited:
                return {"name": cls_name, "cyclic": True}
            visited.add(cls_name)
            tree = {
                "name": cls_name,
                "info": self.get_class_info(cls_name),
                "children": [build_tree(c) for c in self.get_children(cls_name)]
            }
            visited.remove(cls_name)
            return tree
            
        if root_class:
            return build_tree(root_class)
        else:
            top_classes = [cls.name for cls in self.onto.classes() 
                         if not self.get_parents(cls.name)]
            return [build_tree(cls) for cls in top_classes]

class OntologyAnalyzer:
    """本体分析工具 - 专注于本体结构分析"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.tools = OntologyTools(None)
        
    def analyze_domain_structure(self, ontology) -> Dict:
        """分析领域的基本结构
        - 核心概念和关系
        - 属性分布
        - 层次结构
        """
        self.tools.onto = ontology
        hierarchy = self.tools.parse_hierarchy_structure()
        
        structure_info = {
            "hierarchy": hierarchy,  # 只保留一份层次结构
            "properties": [self.tools.parse_property_definition(p.name) 
                         for p in ontology.properties()]
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in ontology analysis.
            Analyze the given ontology structure and identify key patterns and characteristics."""),
            ("user", """Analyze the following ontology structure:
            
            Classes: {classes}
            Properties: {properties}
            Hierarchy: {hierarchy}
            
            Provide a comprehensive analysis including:
            1. Core concepts and their relationships
            2. Key structural patterns
            3. Important property distributions
            4. Potential research areas
            
            Format as JSON with:
            - core_concepts: list[str]
            - key_patterns: list[dict]
            - property_analysis: dict
            - research_opportunities: list[dict]
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(**structure_info))
        return parse_json(response.content)
    
    def find_key_concepts(self, ontology) -> List[Dict]:
        """识别关键概念
        - 概念的中心度
        - 属性丰富度
        - 连接模式
        """
        self.tools.onto = ontology
        
        # 获取本体信息
        classes_info = []
        for cls in ontology.classes():
            class_info = {
                "name": cls.name,
                "properties": self.tools.get_class_properties(cls.name),
                "parents": self.tools.get_parents(cls.name),
                "children": self.tools.get_children(cls.name),
                "related": self.tools.get_related_classes(cls.name)
            }
            classes_info.append(class_info)
            
        relationships = []
        for prop in ontology.properties():
            rel = self.tools.parse_property_definition(prop.name)
            relationships.append(rel)
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in identifying key concepts in scientific domains."""),
            ("user", """Analyze these ontology concepts:
            
            Classes: {classes}
            Relationships: {relationships}
            
            Identify key concepts based on:
            1. Centrality in the network
            2. Property richness
            3. Connection patterns
            4. Research potential
            
            Format as JSON with:
            - key_concepts: list[dict]  # Each with name, importance_score, reasoning
            - research_value: dict  # Research potential for each concept
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(
            classes=classes_info,
            relationships=relationships
        ))
        return parse_json(response.content)
        
    def compare_domains(self, source_ontology, target_ontology) -> Dict:
        """比较两个领域的基本结构
        - 概念映射
        - 结构差异
        - 属性对应
        """
        # 分析源领域
        self.tools.onto = source_ontology
        source_structure = {
            "hierarchy": self.tools.parse_hierarchy_structure(),
            "properties": [self.tools.parse_property_definition(p.name) 
                         for p in source_ontology.properties()],
            "key_concepts": self.find_key_concepts(source_ontology)
        }
        
        # 分析目标领域
        self.tools.onto = target_ontology
        target_structure = {
            "hierarchy": self.tools.parse_hierarchy_structure(),
            "properties": [self.tools.parse_property_definition(p.name) 
                         for p in target_ontology.properties()],
            "key_concepts": self.find_key_concepts(target_ontology)
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in cross-domain knowledge transfer."""),
            ("user", """Compare these two domains:
            
            Source Domain:
            {source_structure}
            
            Target Domain:
            {target_structure}
            
            Analyze:
            1. Conceptual analogies
            2. Methodological differences
            3. Transfer opportunities
            4. Potential innovations
            
            Format as JSON with:
            - analogies: list[dict]  # 概念对应关系
            - method_differences: list[dict]  # 方法论差异
            - transfer_opportunities: list[dict]  # 知识迁移机会
            - innovation_points: list[dict]  # 创新点
            """)
        ])
        
        response = self.llm.invoke(prompt.format_messages(
            source_structure=source_structure,
            target_structure=target_structure
        ))
        return parse_json(response.content)
    
    def get_research_opportunities(self, analysis_result: Dict) -> List[Dict]:
        """从分析结果中提取研究机会"""
        opportunities = []
        
        # 从领域结构分析中提取
        if "research_opportunities" in analysis_result:
            opportunities.extend(analysis_result["research_opportunities"])
            
        # 从关键概念分析中提取
        if "key_concepts" in analysis_result:
            for concept in analysis_result["key_concepts"]:
                if "research_value" in concept and concept["research_value"].get("potential", 0) > 0.7:
                    opportunities.append({
                        "type": "concept_based",
                        "concept": concept["name"],
                        "opportunity": concept["research_value"]["description"]
                    })
                    
        # 从跨领域分析中提取
        if "cross_domain_analysis" in analysis_result:
            cd_analysis = analysis_result["cross_domain_analysis"]
            if "transfer_opportunities" in cd_analysis:
                opportunities.extend([
                    {
                        "type": "transfer",
                        **opp
                    } for opp in cd_analysis["transfer_opportunities"]
                ])
            if "innovation_points" in cd_analysis:
                opportunities.extend([
                    {
                        "type": "innovation",
                        **point
                    } for point in cd_analysis["innovation_points"]
                ])
                
        return opportunities