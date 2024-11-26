

from owlready2 import *
from typing import List
from autonogy_constructor import base_data_structures 

from config.settings import ONTOLOGY_CONFIG

def merge_ontology(ontology_elements: base_data_structures.OntologyElements, ontology_properties: base_data_structures.OntologyProperties):
    """
    将本体数据合并到已有本体中
    
    Args:
        ontology_elements: 要合并的本体元素数据
        ontology_properties: 要合并的本体属性数据
    """
    # 加载或创建本体
    try:
        onto = ONTOLOGY_CONFIG["ontology"]
    except Exception as e:
        print(f"加载本体失败: {e}")
    
    # 写入实体(类)
    _merge_entities(ontology_elements.entities)
    
    # 写入层级关系
    if ontology_elements.hierarchy:
        _merge_hierarchy(ontology_elements.hierarchy)
        
    # 写入不相交关系
    if ontology_elements.disjointness:
        _merge_disjointness(ontology_elements.disjointness)
        
    # 写入数据属性
    if ontology_properties.data_properties:
        _merge_data_properties(ontology_properties.data_properties)
        
    # 写入对象属性
    _merge_object_properties(ontology_properties.object_properties)
        
    try:
        # 保存本体
        onto.save()
    except Exception as e:
        print(f"保存本体失败: {e}")

def _class_exists(class_name: str) -> bool:
    """检查类是否存在"""
    try:
        return ONTOLOGY_CONFIG["classes"][class_name] in ONTOLOGY_CONFIG["ontology"].classes()
    except Exception:
        return False

def _merge_entities(entities: List[base_data_structures.Entity]):
    """合并实体(类)"""
    namespace = ONTOLOGY_CONFIG["classes"]
    for entity in entities:
        try:
            if not _class_exists(entity.name):
                with namespace:
                    new_class = types.new_class(entity.name, (Thing,))
                    if entity.information:
                        new_class.information = [entity.information]
            else:
                existing_class = namespace[entity.name]
                if entity.information and entity.information not in existing_class.information:
                    existing_class.information.append(entity.information)
        except Exception as e:
            print(f"添加实体 {entity.name} 失败: {e}")

def _merge_hierarchy(hierarchies: List[base_data_structures.Hierarchy]):
    """合并层级关系"""
    namespace = ONTOLOGY_CONFIG["classes"]
    for hierarchy in hierarchies:
        try:
            if _class_exists(hierarchy.subclass) and _class_exists(hierarchy.superclass):
                subclass = namespace[hierarchy.subclass]
                superclass = namespace[hierarchy.superclass]
                if superclass not in subclass.is_a:
                    subclass.is_a.append(superclass)
                    if hierarchy.information and hierarchy.information not in subclass.hierarchy_information:
                        subclass.hierarchy_information.append(hierarchy.information)
            else:
                if not _class_exists(hierarchy.subclass):
                    print(f"类 {hierarchy.subclass} 不存在")
                if not _class_exists(hierarchy.superclass):
                    print(f"类 {hierarchy.superclass} 不存在")
        except Exception as e:
            print(f"添加层级关系 {hierarchy.subclass} -> {hierarchy.superclass} 失败: {e}")

def _merge_disjointness(disjointness: List[base_data_structures.Disjointness]):
    """合并不相交关系"""
    namespace = ONTOLOGY_CONFIG["classes"]
    for disj in disjointness:
        try:
            if _class_exists(disj.class1) and _class_exists(disj.class2):
                class1 = namespace[disj.class1]
                class2 = namespace[disj.class2]
                # 检查是否已存在不相交关系
                AllDisjoint([class1, class2])
        except Exception as e:
            print(f"添加不相交关系 {disj.class1} <-×-> {disj.class2} 失败: {e}")

def _merge_data_properties(data_properties: List[base_data_structures.DataProperty]):
    """合并数据属性"""
    namespace = ONTOLOGY_CONFIG["data_properties"]
    class_namespace = ONTOLOGY_CONFIG["classes"]
    for dp in data_properties:
        try:
            if not dp.name in namespace:
                with namespace:
                    new_dp = types.new_class(dp.name, (DataProperty,))
                    if dp.information:
                        new_dp.information = [dp.information]
            else:
                new_dp = namespace[dp.name]
                if dp.information and dp.information not in new_dp.information:
                    new_dp.information.append(dp.information)
                
            if dp.values:
                for owner, value in dp.values.items():
                    if _class_exists(owner):
                        owner_class = class_namespace[owner]
                        try:
                            current_values = getattr(owner_class, dp.name, [])
                            if not isinstance(current_values, list):
# --------------------------------------------------------------------------
# 一种冲突
# --------------------------------------------------------------------------
                                current_values = [current_values]
                            if value not in current_values:
                                current_values.append(value)
                                setattr(owner_class, dp.name, current_values)
                        except Exception as e:
                            print(f"为类 {owner} 添加数据属性值失败: {e}")
        except Exception as e:
            print(f"添加数据属性 {dp.name} 失败: {e}")

def _merge_object_properties(object_properties: List[base_data_structures.ObjectProperty]):
    """合并对象属性"""
    namespace = ONTOLOGY_CONFIG["object_properties"]
    class_namespace = ONTOLOGY_CONFIG["classes"]
    axiom_namespace = ONTOLOGY_CONFIG["axioms"]
    for op in object_properties:
        try:
            if not op.name in namespace:
                with namespace:
                    new_op = types.new_class(op.name, (ObjectProperty,))
                    if op.information:
                        new_op.information = [op.information]
            else:
                new_op = namespace[op.name]
                if op.information and op.information not in new_op.information:
                    new_op.information.append(op.information)
                
            # 处理对象属性的实例
            if op.instances:
                for instance in op.instances:
                    try:
                        if instance.domain and instance.domain.entity and instance.range and instance.range.entity:
                            # 处理值域表达式
                            range_entities = [e.strip() for e in instance.range.entity.split(',')]
                            if all(_class_exists(e) for e in range_entities):
                                if instance.range.type == 'single':
                                    # 检查range_entities是否真的只有一个元素
                                    if len(range_entities) != 1:
                                        raise ValueError(f"对象属性 {op.name} 的值域类型为'single',但没有提供对应的一个值域实体，提供的值域实体为: {range_entities}")
                                    range_expr = class_namespace[instance.range.entity]
                                elif instance.range.type == 'union':
                                    range_expr = Or([class_namespace[e] for e in range_entities])
                                elif instance.range.type == 'intersection':
                                    range_expr = And([class_namespace[e] for e in range_entities])
                                
                                # 根据限制类型创建值域限制表达式
                                if instance.restriction == 'only':
                                    restriction_expr = new_op.only(range_expr)
                                else: # 默认为some
                                    restriction_expr = new_op.some(range_expr)
                                    
                                # 处理域
                                domain_entities = [e.strip() for e in instance.domain.entity.split(',')]
                                if all(_class_exists(e) for e in domain_entities):
                                    if instance.domain.type == 'single':
                                        # 单个类的情况,直接添加类限制
                                        domain_class = class_namespace[instance.domain.entity]
                                        domain_class.is_a.append(restriction_expr)
                                    else:
                                        # 多个类构成表达式的情况,使用一般类公理
                                        if instance.domain.type == 'union':
                                            domain_expr = Or([class_namespace[e] for e in domain_entities])
                                        else: # intersection
                                            domain_expr = And([class_namespace[e] for e in domain_entities])
                                        with axiom_namespace:
                                            gca = GeneralClassAxiom(domain_expr)
                                            gca.is_a.append(restriction_expr)
                    except Exception as e:
                        print(f"设置对象属性 {op.name} 的实例域和值域失败: {e}")
                    
        except Exception as e:
            print(f"添加对象属性 {op.name} 失败: {e}")

