from typing import List, Tuple, Dict, Optional, Set
from owlready2 import *

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
        def build_tree(cls_name):
            return {
                "name": cls_name,
                "info": self.get_class_info(cls_name),
                "children": [build_tree(c) for c in self.get_children(cls_name)]
            }
            
        if root_class:
            return build_tree(root_class)
        else:
            # 找到所有顶层类
            top_classes = [cls.name for cls in self.onto.classes() 
                         if not self.get_parents(cls.name)]
            return [build_tree(cls) for cls in top_classes]