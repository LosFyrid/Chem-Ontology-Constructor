from owlready2 import *
from typing import List
from autonogy_constructor import base_data_structures 
from autonogy_constructor.utils import flatten_dict

from config.settings import ONTOLOGY_CONFIG

def merge_ontology(
    ontology_entities: base_data_structures.OntologyEntities,
    ontology_elements: base_data_structures.OntologyElements, 
    ontology_data_properties: base_data_structures.OntologyDataProperties,
    ontology_object_properties: base_data_structures.OntologyObjectProperties,
    source: str
):
    """
    将本体数据合并到已有本体中
    
    Args:
        ontology_elements: 要合并的本体元素数据
        ontology_data_properties: 要合并的本体数据属性
        ontology_object_properties: 要合并的本体对象属性
        source: 数据来源
    """
    try:
        onto = ONTOLOGY_CONFIG["ontology"]
        
        # 写入实体(类)
        if ontology_entities and ontology_entities.entities:
            _merge_entities(ontology_entities.entities, source)
            onto.save()
        # 写入层级关系
        if ontology_elements and ontology_elements.hierarchy:
            _merge_hierarchy(ontology_elements.hierarchy, source)
            onto.save()
        # 写入不相交关系
        if ontology_elements and ontology_elements.disjointness:
            _merge_disjointness(ontology_elements.disjointness)
            onto.save()
        # 写入数据属性
        if ontology_data_properties and ontology_data_properties.data_properties:
            _merge_data_properties(ontology_data_properties.data_properties, source)
            onto.save() 
        # 写入对象属性
        if ontology_object_properties and ontology_object_properties.object_properties:
            _merge_object_properties(ontology_object_properties.object_properties, source)
            onto.save()
            
    except Exception as e:
        print(f"本体合并失败: {e}")
        raise  # 重新抛出异常，让调用者知道发生了错误

def _class_exists(class_name: str) -> bool:
    """检查类是否存在"""
    try:
        return ONTOLOGY_CONFIG["classes"][class_name] in ONTOLOGY_CONFIG["ontology"].classes()
    except Exception:
        return False

def _merge_entities(entities: List[base_data_structures.Entity], source: str):
    """合并实体(类)"""
    namespace = ONTOLOGY_CONFIG["classes"]
    meta = ONTOLOGY_CONFIG["meta"]
    
    for entity in entities:
        try:
            if not _class_exists(entity.name):
                with namespace:
                    new_class = types.new_class(entity.name, (Thing,))
                    if entity.information:
                        # 创建SourcedInformation实例
                        info_instance = meta.SourcedInformation()
                        info_instance.content = entity.information
                        info_instance.source = source
                        # 关联到类
                        new_class.has_information.append(info_instance)
            else:
                existing_class = namespace[entity.name]
                if entity.information:
                    # 检查是否已存在相同的信息
                    exists = False
                    for info in existing_class.has_information:
                        if info.content == entity.information and info.source == source:
                            exists = True
                            break
                    
                    if not exists:
                        # 创建新的SourcedInformation实例
                        info_instance = meta.SourcedInformation()
                        info_instance.content = entity.information
                        info_instance.source = source
                        existing_class.has_information.append(info_instance)
        except Exception as e:
            print(f"添加实体 {entity.name} 失败: {e}")

def _merge_hierarchy(hierarchies: List[base_data_structures.Hierarchy], source: str):
    """合并层级关系"""
    namespace = ONTOLOGY_CONFIG["classes"]
    meta = ONTOLOGY_CONFIG["meta"]
    
    for hierarchy in hierarchies:
        try:
            if _class_exists(hierarchy.subclass) and all(_class_exists(sup) for sup in hierarchy.superclass):
                subclass = namespace[hierarchy.subclass]
                # 移除Thing类
                if Thing in subclass.is_a:
                    subclass.is_a.remove(Thing)
                # 添加新的父类
                added_new = False
                for sup in hierarchy.superclass:
                    superclass = namespace[sup]
                    if superclass not in subclass.is_a:
                        subclass.is_a.append(superclass)
                        added_new = True
                
                # 只在添加了新父类时添加information
                if added_new and hierarchy.information:
                    # 创建SourcedInformation实例
                    info_instance = meta.SourcedInformation()
                    info_instance.content = hierarchy.information
                    info_instance.source = f"Source (Superclass: {', '.join(hierarchy.superclass)}): {source}"
                    # 关联到类
                    subclass.has_information.append(info_instance)
            else:
                if not _class_exists(hierarchy.subclass):
                    print(f"类 {hierarchy.subclass} 不存在")
                # 修改这里：检查每个父类是否存在
                for sup in hierarchy.superclass:
                    if not _class_exists(sup):
                        print(f"类 {sup} 不存在")
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
                AllDisjoint([class1, class2])
        except Exception as e:
            print(f"添加不相交关系 {disj.class1} <-×-> {disj.class2} 失败: {e}")

def _merge_data_properties(data_properties: List[base_data_structures.DataProperty], source: str):
    """合并数据属性"""
    namespace = ONTOLOGY_CONFIG["data_properties"]
    class_namespace = ONTOLOGY_CONFIG["classes"]
    for dp in data_properties:
        try:
            if not dp.name in ONTOLOGY_CONFIG["ontology"].data_properties():
                with namespace:
                    new_dp = types.new_class(dp.name, (DataProperty,))
                    if dp.information:
                        new_dp.information = [dp.information]
                    new_dp.source = [source]
            else:
                print(f"数据属性 {dp.name} 已存在,更新数据属性")    
                new_dp = namespace[dp.name]
                if dp.information and dp.information not in new_dp.information:
                    new_dp.information.append(dp.information)
                if source and source not in new_dp.source:
                    new_dp.source.append(source)
                
            if dp.values:
                flattened_values = flatten_dict(dp.values)
                for owner_path, value in flattened_values.items():
                    # 从路径中提取所有实体名称
                    entity_names = owner_path.split(" with ")
                    
                    # 检查所有实体是否存在
                    if all(_class_exists(name) for name in entity_names):
                        try:
                            # 获取所有实体类
                            entity_classes = [class_namespace[name] for name in entity_names]
                            
                            # 如果只有一个实体
                            if len(entity_classes) == 1:
                                owner_class = entity_classes[0]
                            # 如果有多个实体，创建它们的交集类
                            else:
                                owner_class = And(entity_classes)
                            
                            # 获取当前值，如果不存在则初始化为空列表
                            current_values = getattr(owner_class, dp.name, [])
                            
                            # 如果当前值不是列表，创建一个包含当前值的新列表
                            if not isinstance(current_values, list):
                                current_values = [current_values] if current_values is not None else []
                                
                            # 如果新值不在当前值列表中，添加它
                            if value is not None:
                                if isinstance(value, list):
                                    for v in value:
                                        if v not in current_values:
                                            current_values.append(v)
                                elif value not in current_values:
                                    current_values.append(value)
                                
                            # 更新属性值
                            setattr(owner_class, dp.name, current_values)
                        except Exception as e:
                            print(f"为类 {owner_path} 添加数据属性值失败: {e}")
        except Exception as e:
            print(f"添加数据属性 {dp.name} 失败: {e}")

def _merge_object_properties(object_properties: List[base_data_structures.ObjectProperty], source: str):
    """合并对象属性"""
    namespace = ONTOLOGY_CONFIG["object_properties"]
    class_namespace = ONTOLOGY_CONFIG["classes"]
    axiom_namespace = ONTOLOGY_CONFIG["axioms"]
    for op in object_properties:
        try:
            if not op.name in ONTOLOGY_CONFIG["ontology"].object_properties():
                with namespace:
                    new_op = types.new_class(op.name, (ObjectProperty,))
                    if op.information:
                        new_op.information = [op.information]
                    new_op.source = [source]
            else:
                print(f"对象属性 {op.name} 已存在,更新对象属性")
                new_op = namespace[op.name]
                if op.information and op.information not in new_op.information:
                    new_op.information.append(op.information)
                if source and source not in new_op.source:
                    new_op.source.append(source)
                
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

