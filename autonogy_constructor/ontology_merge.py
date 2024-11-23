

from owlready2 import *
from typing import List
import os
from base_data_structures import Ontology, Entity, Hierarchy, Disjointness, DataProperty, ObjectProperty

from config.settings import ONTOLOGY_CONFIG

def merge_ontology(ontology: Ontology, onto_path: str, namespace: str):
    """
    将本体数据合并到已有本体中
    
    Args:
        ontology: 要合并的本体数据
        onto_path: 本体文件路径
        namespace: 本体命名空间
    """
    # 加载或创建本体
    try:
        if os.path.exists(onto_path):
            onto = get_ontology(onto_path).load()
        else:
            onto = get_ontology(namespace)
    except Exception as e:
        print(f"加载本体失败: {e}")
        onto = get_ontology(namespace)
    
    with onto:
        # 写入实体(类)
        _merge_entities(ONTOLOGY_CONFIG["classes"], ontology.entities)
        
        # 写入层级关系
        if ontology.hierarchy:
            _merge_hierarchy(onto, ontology.hierarchy)
            
        # 写入不相交关系
        if ontology.disjointness:
            _merge_disjointness(onto, ontology.disjointness)
            
        # 写入数据属性
        if ontology.data_properties:
            _merge_data_properties(ONTOLOGY_CONFIG["data_properties"], ontology.data_properties)
            
        # 写入对象属性
        _merge_object_properties(ONTOLOGY_CONFIG["object_properties"], ontology.object_properties)
        
        try:
            # 保存本体
            onto.save(file=onto_path, format="rdfxml")
        except Exception as e:
            print(f"保存本体失败: {e}")

def _class_exists(namespace, class_name: str) -> bool:
    """检查类是否存在"""
    try:
        return class_name in namespace
    except Exception:
        return False

def _merge_entities(namespace, entities: List[Entity]):
    """合并实体(类)"""
    for entity in entities:
        try:
            if not _class_exists(namespace, entity.name):
                with namespace:
                    new_class = types.new_class(entity.name, (Thing,))
                    if entity.information:
                        new_class.comment = [entity.information]
            else:
                existing_class = namespace[entity.name]
                if entity.information and not existing_class.comment:
                    existing_class.comment = [entity.information]
        except Exception as e:
            print(f"添加实体 {entity.name} 失败: {e}")

def _merge_hierarchy(onto: Ontology, hierarchies: List[Hierarchy]):
    """合并层级关系"""
    for hierarchy in hierarchies:
        try:
            if _class_exists(ONTOLOGY_CONFIG["classes"], hierarchy.subclass) and _class_exists(ONTOLOGY_CONFIG["classes"], hierarchy.superclass):
                subclass = ONTOLOGY_CONFIG["classes"][hierarchy.subclass]
                superclass = ONTOLOGY_CONFIG["classes"][hierarchy.superclass]
                if superclass not in subclass.is_a:
                    subclass.is_a.append(superclass)
                    if hierarchy.information:
                        if not hasattr(subclass, 'comment'):
                            subclass.comment = []
                        subclass.comment.append(hierarchy.information)
        except Exception as e:
            print(f"添加层级关系 {hierarchy.subclass} -> {hierarchy.superclass} 失败: {e}")

def _merge_disjointness(onto: Ontology, disjointness: List[Disjointness]):
    """合并不相交关系"""
    for disj in disjointness:
        try:
            if _class_exists(ONTOLOGY_CONFIG["classes"], disj.class1) and _class_exists(ONTOLOGY_CONFIG["classes"], disj.class2):
                class1 = ONTOLOGY_CONFIG["classes"][disj.class1]
                class2 = ONTOLOGY_CONFIG["classes"][disj.class2]
                # 检查是否已存在不相交关系
                existing_disjoints = [d for d in onto.disjoint_classes() 
                                   if {class1, class2}.issubset(set(d.entities))]
                if not existing_disjoints:
                    AllDisjoint([class1, class2])
        except Exception as e:
            print(f"添加不相交关系 {disj.class1} <-> {disj.class2} 失败: {e}")

def _merge_data_properties(namespace, data_properties: List[DataProperty]):
    """合并数据属性"""
    for dp in data_properties:
        try:
            if not dp.name in namespace:
                with namespace:
                    new_dp = types.new_class(dp.name, (DataProperty,))
                    if dp.information:
                        new_dp.comment = [dp.information]
            else:
                new_dp = namespace[dp.name]
                
            if dp.values:
                for owner, value in dp.values.items():
                    if _class_exists(ONTOLOGY_CONFIG["classes"], owner):
                        owner_class = ONTOLOGY_CONFIG["classes"][owner]
                        try:
                            current_values = getattr(owner_class, dp.name, [])
                            if not isinstance(current_values, list):
                                current_values = [current_values]
                            if value not in current_values:
                                current_values.append(value)
                                setattr(owner_class, dp.name, current_values)
                        except Exception as e:
                            print(f"为类 {owner} 添加数据属性值失败: {e}")
        except Exception as e:
            print(f"添加数据属性 {dp.name} 失败: {e}")

def _merge_object_properties(namespace, object_properties: List[ObjectProperty]):
    """合并对象属性"""
    for op in object_properties:
        try:
            if not op.name in namespace:
                with namespace:
                    new_op = types.new_class(op.name, (ObjectProperty,))
                    if op.information:
                        new_op.comment = [op.information]
            else:
                new_op = namespace[op.name]
                
            # 设置域
            if op.domain and op.domain.existence and op.domain.entity:
                try:
                    domain_entities = [e.strip() for e in op.domain.entity.split(',')]
                    if all(_class_exists(ONTOLOGY_CONFIG["classes"], e) for e in domain_entities):
                        if op.domain.type == 'single':
                            new_op.domain = [ONTOLOGY_CONFIG["classes"][op.domain.entity]]
                        elif op.domain.type == 'union':
                            new_op.domain = [Or([ONTOLOGY_CONFIG["classes"][e] for e in domain_entities])]
                        elif op.domain.type == 'intersection':
                            new_op.domain = [And([ONTOLOGY_CONFIG["classes"][e] for e in domain_entities])]
                except Exception as e:
                    print(f"设置对象属性 {op.name} 的域失败: {e}")
                
            # 设置值域
            if op.range and op.range.existence and op.range.entity:
                try:
                    range_entities = [e.strip() for e in op.range.entity.split(',')]
                    if all(_class_exists(ONTOLOGY_CONFIG["classes"], e) for e in range_entities):
                        if op.range.type == 'single':
                            new_op.range = [ONTOLOGY_CONFIG["classes"][op.range.entity]]
                        elif op.range.type == 'union':
                            new_op.range = [Or([ONTOLOGY_CONFIG["classes"][e] for e in range_entities])]
                        elif op.range.type == 'intersection':
                            new_op.range = [And([ONTOLOGY_CONFIG["classes"][e] for e in range_entities])]
                except Exception as e:
                    print(f"设置对象属性 {op.name} 的值域失败: {e}")
                
            # 设置限制
            if op.restriction:
                try:
                    new_op.python_name = op.restriction
                except Exception as e:
                    print(f"设置对象属性 {op.name} 的限制失败: {e}")
                    
        except Exception as e:
            print(f"添加对象属性 {op.name} 失败: {e}")

