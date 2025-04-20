from typing import List, Tuple, Dict, Optional, Set, Any, Callable, Union
from owlready2 import *
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import re
from collections import deque
import traceback
import warnings

from autology_constructor.idea.query_team.utils import parse_json, format_sparql_results, extract_variables_from_sparql

from config.settings import ONTOLOGY_SETTINGS, OntologySettings

class SparqlExecutionError(Exception):
    """SPARQL查询执行错误"""
    pass


class SparqlOptimizer:
    """SPARQL查询优化器
    
    对SPARQL查询进行各种优化，提高查询效率和稳定性：
    1. 优化前缀声明
    2. 优化过滤条件
    3. 优化连接操作
    """
    
    def __init__(self):
        self.optimizations = [
            self._optimize_prefixes,
            self._optimize_filters,
            self._optimize_joins
        ]

    def optimize(self, query: str) -> str:
        """应用所有优化策略到查询
        
        Args:
            query: 原始SPARQL查询
            
        Returns:
            优化后的SPARQL查询
        """
        optimized = query
        for optimization in self.optimizations:
            optimized = optimization(optimized)
        return optimized
        
    def _optimize_prefixes(self, query: str) -> str:
        """优化前缀声明
        
        确保常用前缀存在，移除未使用前缀
        """
        # 检查常用前缀是否已声明
        common_prefixes = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        }
        
        # 提取已声明的前缀
        prefix_pattern = r'PREFIX\s+(\w+):\s+<([^>]+)>'
        declared_prefixes = dict(re.findall(prefix_pattern, query, re.IGNORECASE))
        
        # 检查查询中使用的前缀
        used_prefixes = set(re.findall(r'(\w+):[^\s.]+', query))
        
        # 添加缺失但使用的常用前缀
        new_prefixes = ""
        for prefix, uri in common_prefixes.items():
            if prefix in used_prefixes and prefix not in declared_prefixes:
                new_prefixes += f"PREFIX {prefix}: <{uri}>\n"
        
        # 如果有新前缀，添加到查询开头
        if new_prefixes:
            # 检查查询是否已有PREFIX声明
            if re.search(prefix_pattern, query, re.IGNORECASE):
                # 在最后一个PREFIX后插入
                query = re.sub(
                    r'(PREFIX\s+\w+:\s+<[^>]+>)([^P]|$)',
                    r'\1\n' + new_prefixes + r'\2',
                    query,
                    count=1,
                    flags=re.IGNORECASE
                )
            else:
                # 在查询开头添加
                query = new_prefixes + query
        
        return query
        
    def _optimize_filters(self, query: str) -> str:
        """优化过滤条件
        
        将复杂过滤条件移到更早位置，优化执行计划
        """
        # 提取所有FILTER表达式
        filter_pattern = r'FILTER\s*\(([^)]+)\)'
        filters = re.findall(filter_pattern, query, re.IGNORECASE)
        
        # 如果没有过滤器，直接返回
        if not filters:
            return query
            
        # 优化过滤器位置（将过滤器尽可能移到WHERE子句前部）
        where_pattern = r'(WHERE\s*\{)(.*?)(\}) '
        
        def optimize_where(match):
            prefix = match.group(1)
            body = match.group(2)
            suffix = match.group(3)
            
            # 删除所有过滤器
            body_without_filters = re.sub(filter_pattern, '', body, flags=re.IGNORECASE)
            
            # 在三元组模式后添加所有过滤器
            optimized_body = body_without_filters
            for f in filters:
                optimized_body += f" FILTER({f})"
                
            return prefix + optimized_body + suffix
            
        # 仅优化没有OPTIONAL, UNION等复杂结构的简单查询
        if not re.search(r'OPTIONAL|UNION|MINUS', query, re.IGNORECASE):
            return re.sub(where_pattern, optimize_where, query, flags=re.IGNORECASE | re.DOTALL)
        
        return query
        
    def _optimize_joins(self, query: str) -> str:
        """优化连接操作
        
        重排三元组模式顺序，优化连接顺序
        """
        # 此优化需要更复杂分析，简化实现
        # 规则：将限制性更强的三元组模式（包含类型声明）移到前面
        
        # 查找WHERE子句
        where_match = re.search(r'WHERE\s*\{(.*?)\}', query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return query
            
        where_body = where_match.group(1)
        
        # 提取三元组模式
        patterns = [p.strip() for p in re.split(r'\.|\s*FILTER\s*\([^)]+\)', where_body) if p.strip()]
        
        # 优先级排序：类型声明 > 具体URI > 变量
        def pattern_priority(pattern):
            if 'rdf:type' in pattern or 'a ' in pattern:
                return 0  # 最高优先级
            elif re.search(r'<[^>]+>', pattern):
                return 1  # 次高优先级
            else:
                return 2  # 最低优先级
                
        # 尝试按优先级排序
        try:
            sorted_patterns = sorted(patterns, key=pattern_priority)
            
            # 如果排序结果与原始不同，替换WHERE子句
            if sorted_patterns != patterns:
                sorted_where = ' . '.join(sorted_patterns)
                if sorted_where:
                    sorted_where += ' .'
                new_query = query.replace(where_body, sorted_where)
                return new_query
        except:
            # 排序失败，返回原始查询
            pass
            
        return query


class SparqlExecutor:
    """SPARQL查询执行器
    
    负责执行SPARQL查询，包括：
    1. 查询优化
    2. 执行查询
    3. 异常处理和重试
    """
    
    def __init__(self):
        self.optimizer = SparqlOptimizer()
        self.max_retries = 2
        
    def execute(self, query: str, ontology: Any) -> Dict:
        """执行SPARQL查询
        
        Args:
            query: SPARQL查询字符串
            ontology: 查询的本体对象
            
        Returns:
            查询结果
            
        Raises:
            SparqlExecutionError: 查询执行失败
        """
        if not ontology:
            raise SparqlExecutionError("未设置本体")
            
        # 优化查询
        try:
            optimized_query = self.optimizer.optimize(query)
        except Exception as e:
            # 优化失败时使用原始查询
            optimized_query = query
            
        # 执行查询（带重试）
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                return self._execute_query(optimized_query, ontology)
            except Exception as e:
                last_error = e
                retries += 1
                
                # 最后一次尝试使用原始查询
                if retries == self.max_retries:
                    try:
                        return self._execute_query(query, ontology)
                    except Exception as final_e:
                        last_error = final_e
                        break
        
        # 所有尝试都失败
        error_msg = str(last_error) if last_error else "未知错误"
        raise SparqlExecutionError(f"SPARQL执行失败: {error_msg}")
    
    def _execute_query(self, query: str, ontology: Any) -> Dict:
        """实际执行优化后的查询
        
        Args:
            query: 优化后的SPARQL查询
            ontology: 本体对象
            
        Returns:
            格式化的查询结果
        """
        # 使用传入的 ontology 对象的 world 来执行 SPARQL
        results = list(ontology.world.sparql(query))
        
        # 获取变量名
        variables = extract_variables_from_sparql(query)
        
        # 格式化结果
        formatted_results = format_sparql_results(results)
        
        # 添加变量映射和查询信息
        if variables:
            formatted_results["variables"] = variables
        
        formatted_results["query_info"] = {
            "original_query": query,
        }
        
        return formatted_results


class OntologyTools:
    """用于本体查询和解析的工具 (v3: 使用 OntologySettings)

    此类提供了用于处理 OWL 本体 的综合工具，需要一个配置好的 OntologySettings 实例。
    """

    def __init__(self, ontology_settings: OntologySettings):
        """初始化 OntologyTools

        Args:
            ontology_settings (OntologySettings): 已配置和加载的本体设置对象。
        """
        if not isinstance(ontology_settings, OntologySettings):
            raise TypeError("ontology_settings 必须是 OntologySettings 的实例。")

        self.onto_settings = ontology_settings
        self.onto = self.onto_settings.ontology # 可能为 None

        # 获取命名空间，需要处理 ontology 或 namespace 可能为 None 的情况
        self.meta_ns = self.onto_settings.meta
        self._classes_ns = self.onto_settings.classes
        self._obj_props_ns = self.onto_settings.object_properties
        self._data_props_ns = self.onto_settings.data_properties

        # 检查本体是否成功加载
        if self.onto is None:
            warnings.warn("OntologyTools 初始化时本体未加载。大多数功能将不可用。", RuntimeWarning)
            # 将关键组件设为 None 或默认值
            self.has_information_prop = None
            self.SourcedInformationClass = None
            return # 提前退出初始化或允许继续但功能受限

        # 获取关键的 Meta 属性和类
        if self.meta_ns:
            try:
                self.has_information_prop = self.meta_ns['has_information']
                # -- 修改检查方式 --
                # 移除之前的调试打印
                # 使用 is_a 属性进行检查，更符合 owlready2 的方式
                if owl.ObjectProperty not in getattr(self.has_information_prop, 'is_a', []):
                    warnings.warn(f"'{self.has_information_prop}' is not recognized as an owl:ObjectProperty based on its 'is_a' attribute.", ImportWarning)
                    self.has_information_prop = None
            except (KeyError, AttributeError):
                warnings.warn("'has_information' 未在 meta 命名空间中找到。", ImportWarning)
                self.has_information_prop = None
            # 移除之前的调试打印

            try:
                self.SourcedInformationClass = self.meta_ns['SourcedInformation']
                # 同样，对 SourcedInformationClass 的检查也应该基于 owlready2 的类型系统
                # issubclass(self.SourcedInformationClass, Thing) 看起来是正确的，暂时保留
                # 移除之前的调试打印
                if not issubclass(self.SourcedInformationClass, Thing): # issubclass is safer
                    warnings.warn("'meta.SourcedInformation' 不是有效的类 (ThingClass)。", ImportWarning)
                    self.SourcedInformationClass = None
            except (KeyError, AttributeError, TypeError): # TypeError for issubclass if not a class
                warnings.warn("'SourcedInformation' 未在 meta 命名空间中找到或不是有效类。", ImportWarning)
                self.SourcedInformationClass = None
            # 移除之前的调试打印
        else:
            warnings.warn("Meta 命名空间未加载。SourcedInformation 相关功能将不可用。", RuntimeWarning)
            self.has_information_prop = None
            self.SourcedInformationClass = None

        # 验证其他命名空间是否加载
        if not self._classes_ns: warnings.warn("Classes 命名空间未加载。", RuntimeWarning)
        if not self._obj_props_ns: warnings.warn("Object Properties 命名空间未加载。", RuntimeWarning)
        if not self._data_props_ns: warnings.warn("Data Properties 命名空间未加载。", RuntimeWarning)


    def _check_ontology_loaded(self) -> bool:
        """检查本体是否已加载"""
        if self.onto is None:
             warnings.warn("操作无法执行，因为本体未加载。", RuntimeWarning)
             return False
        return True

    def _get_class_by_name(self, class_name: str) -> Optional[ThingClass]:
        """辅助函数：通过名称安全地获取类对象 (使用 [] 访问 classes 命名空间)"""
        if not self._check_ontology_loaded() or not self._classes_ns: return None
        try:
            cls = self._classes_ns[class_name]
            if isinstance(cls, ThingClass):
                return cls
            else:
                 warnings.warn(f"在 'classes' 命名空间中找到 '{class_name}' 但它不是 ThingClass。")
                 return None
        except KeyError:
            return None # 类不存在是正常情况，不警告
        except Exception as e:
             warnings.warn(f"通过名称 '{class_name}' 获取类时发生意外错误: {e}")
             return None

    def _get_property_by_name(self, property_name: str) -> Optional[Union[ObjectProperty, DataProperty]]:
        """辅助函数：通过名称安全地获取属性对象 (尝试 obj props, 然后 data props)"""
        if not self._check_ontology_loaded(): return None
        prop = None
        # 优先尝试对象属性
        if self._obj_props_ns:
            try:
                prop = self._obj_props_ns[property_name]
                if isinstance(prop, ObjectProperty):
                    return prop
                else:
                    # 找到了但类型不对？这很奇怪，记录一下
                    warnings.warn(f"在 object_properties 命名空间中找到 '{property_name}' 但它不是 ObjectProperty。")
                    prop = None # 重置以便尝试数据属性
            except KeyError:
                pass # 不在对象属性中，正常，继续查找
            except Exception as e:
                 warnings.warn(f"在 object_properties 中查找 '{property_name}' 时出错: {e}")
                 prop = None # 出错则不继续

        # 如果不是对象属性（或查找出错），尝试数据属性
        if prop is None and self._data_props_ns:
            try:
                prop = self._data_props_ns[property_name]
                if isinstance(prop, DataProperty):
                    return prop
                else:
                     warnings.warn(f"在 data_properties 命名空间中找到 '{property_name}' 但它不是 DataProperty。")
                     return None
            except KeyError:
                pass # 也不在数据属性中
            except Exception as e:
                 warnings.warn(f"在 data_properties 中查找 '{property_name}' 时出错: {e}")

        # 如果两个命名空间都没找到，返回 None
        return None


    def _get_restriction_value_str(self, value: Any) -> str:
        """辅助函数：将限制值转换为字符串表示"""
        # ... (此函数不变)
        if isinstance(value, ThingClass) or isinstance(value, owlready2.PropertyClass): return getattr(value, 'name', str(value))
        elif isinstance(value, Or): return " OR ".join([self._get_restriction_value_str(c) for c in value.Classes])
        elif isinstance(value, And): return " AND ".join([self._get_restriction_value_str(c) for c in value.Classes])
        elif isinstance(value, Not): return f"NOT ({self._get_restriction_value_str(value.Class)})"
        elif isinstance(value, Thing): return getattr(value, 'name', str(value))
        else: return str(value)

    def _get_sourced_info(self, entity: Union[ThingClass, ObjectProperty, DataProperty], info_type: Optional[Union[str, List[str]]] = None) -> List[Thing]:
        """辅助函数：获取实体关联的 SourcedInformation 实例，可选地按类型过滤"""
        # ... (此函数不变)
        if not self._check_ontology_loaded() or not self.has_information_prop or not self.SourcedInformationClass: return []
        linked_info_instances = []
        try:
            raw_linked_items = getattr(entity, self.has_information_prop.name, [])
            if not isinstance(raw_linked_items, list): raw_linked_items = [raw_linked_items]
            for item in raw_linked_items:
                 if self.SourcedInformationClass in item.is_a:
                     if info_type:
                         item_types = getattr(item, 'type', []); item_types = [item_types] if isinstance(item_types, str) else item_types
                         target_types = [info_type] if isinstance(info_type, str) else info_type
                         if any(t in item_types for t in target_types): linked_info_instances.append(item)
                     else: linked_info_instances.append(item)
        except AttributeError as e: warnings.warn(f"访问实体 '{getattr(entity, 'name', entity)}' 的 has_information 时出错: {e}")
        except Exception as e: warnings.warn(f"获取实体 '{getattr(entity, 'name', entity)}' 的 SI 时出错: {e}")
        return linked_info_instances


    # --- 内部单类处理函数 ---
    # 这些函数现在首先检查本体是否加载

    def _get_single_class_info(self, class_name: str) -> Dict:
        if not self._check_ontology_loaded(): return {"error": "本体未加载"}
        cls = self._get_class_by_name(class_name)
        if not cls: return {"error": f"类 '{class_name}' 未找到。"}
        entity_info_contents = []
        sourced_infos = self._get_sourced_info(cls, info_type="entity")
        for info_instance in sourced_infos:
            try:
                content = getattr(info_instance, 'content', None)
                if content is not None:
                    if isinstance(content, list): entity_info_contents.extend([str(c) for c in content])
                    else: entity_info_contents.append(str(content))
            except AttributeError: pass # warnings.warn(f"SI {getattr(info_instance, 'name', info_instance)} 缺少 'content' 属性。")
            except Exception as e: warnings.warn(f"处理 SI {getattr(info_instance, 'name', info_instance)} 时出错: {e}")
        return {"name": cls.name, "information": list(set(entity_info_contents))}

    def _get_single_information_sources(self, class_name: str) -> Union[List[str], Dict]:
         if not self._check_ontology_loaded(): return {"error": "本体未加载"}
         cls = self._get_class_by_name(class_name)
         if not cls: return {"error": f"类 '{class_name}' 未找到。"}
         sources = set()
         sourced_infos = self._get_sourced_info(cls)
         for info_instance in sourced_infos:
             try:
                 source_val = getattr(info_instance, 'source', None)
                 if source_val is not None:
                     if isinstance(source_val, list): sources.update([str(s) for s in source_val])
                     else: sources.add(str(source_val))
             except AttributeError: pass # warnings.warn(f"SI {getattr(info_instance, 'name', info_instance)} 缺少 'source' 属性。")
             except Exception as e: warnings.warn(f"处理 SI {getattr(info_instance, 'name', info_instance)} 的 source 时出错: {e}")
         return sorted(list(sources))

    # --- NEW: Refactored private methods for single class operations ---

    def _get_single_parents(self, class_name: str) -> Union[List[str], Dict]:
         """Internal: Get direct parents for a single class."""
         if not self._check_ontology_loaded(): return {"error": "本体未加载"}
         cls = self._get_class_by_name(class_name)
         if not cls: return {"error": f"类 '{class_name}' 未找到。"}
         parents = []
         try:
             # Iterate over is_a to find ThingClass parents (direct superclasses)
             for parent in cls.is_a:
                 # Ensure it's a class, not Thing itself, and has a name
                 if isinstance(parent, ThingClass) and parent != owlready2.Thing and hasattr(parent, 'name'):
                      parents.append(parent.name)
         except Exception as e: warnings.warn(f"获取类 '{class_name}' 的父类时出错: {e}")
         # Return unique sorted list
         return sorted(list(set(parents)))

    def _get_single_children(self, class_name: str) -> Union[List[str], Dict]:
         """Internal: Get direct children for a single class."""
         if not self._check_ontology_loaded(): return {"error": "本体未加载"}
         cls = self._get_class_by_name(class_name)
         if not cls: return {"error": f"类 '{class_name}' 未找到。"}
         children = []
         try:
             # Use the subclasses() generator provided by owlready2
             for child in cls.subclasses():
                 if isinstance(child, ThingClass) and hasattr(child, 'name'): children.append(child.name)
         except Exception as e: warnings.warn(f"获取类 '{class_name}' 的子类时出错: {e}")
         return sorted(list(set(children)))

    def _get_single_ancestors(self, class_name: str) -> Union[List[str], Dict]:
         """Internal: Get all ancestors for a single class."""
         if not self._check_ontology_loaded(): return {"error": "本体未加载"}
         cls = self._get_class_by_name(class_name)
         if not cls: return {"error": f"类 '{class_name}' 未找到。"}
         ancestors = []
         try:
             # Use the ancestors() method
             all_ancestors = cls.ancestors()
             for ancestor in all_ancestors:
                 # Exclude self and Thing
                 if isinstance(ancestor, ThingClass) and ancestor != cls and ancestor != owlready2.Thing and hasattr(ancestor, 'name'): ancestors.append(ancestor.name)
         except Exception as e: warnings.warn(f"获取类 '{class_name}' 的祖先时出错: {e}")
         return sorted(list(set(ancestors)))

    def _get_single_descendants(self, class_name: str) -> Union[List[str], Dict]:
          """Internal: Get all descendants for a single class."""
          if not self._check_ontology_loaded(): return {"error": "本体未加载"}
          cls = self._get_class_by_name(class_name)
          if not cls: return {"error": f"类 '{class_name}' 未找到。"}
          descendants = []
          try:
              # Use the descendants() method
              all_descendants = cls.descendants()
              for descendant in all_descendants:
                   # Exclude self
                   if isinstance(descendant, ThingClass) and descendant != cls and hasattr(descendant, 'name'): descendants.append(descendant.name)
          except Exception as e: warnings.warn(f"获取类 '{class_name}' 的后代时出错: {e}")
          return sorted(list(set(descendants)))

    def _get_single_related_classes(self, class_name: str) -> Union[Dict[str, List[str]], Dict]:
         """Internal: Get related classes via object properties for a single class."""
         if not self._check_ontology_loaded(): return {"error": "本体未加载"}
         cls = self._get_class_by_name(class_name)
         if not cls: return {"error": f"类 '{class_name}' 未找到。"}
         related_map = {}
         try:
             # Use the NEW _get_single_class_properties method
             class_props_res = self._get_single_class_properties(class_name) # Call self!
             if isinstance(class_props_res, dict): # Error case from properties
                 warnings.warn(f"无法获取 '{class_name}' 的属性以查找相关类: {class_props_res.get('error', '未知错误')}")
                 return {"error": f"获取属性失败: {class_props_res.get('error', '未知错误')}"}

             for prop_name in class_props_res:
                 prop = self._get_property_by_name(prop_name)
                 if not prop or not isinstance(prop, ObjectProperty): continue # Only consider object properties

                 related_class_names_for_prop = set()
                 # Get restrictions for *this* property on the class
                 restrictions_result = self.get_property_restrictions(class_name, prop_name) # Public method ok here
                 if isinstance(restrictions_result, dict) and "error" in restrictions_result:
                     warnings.warn(f"无法获取 '{class_name}' 上 '{prop_name}' 的限制: {restrictions_result['error']}")
                     continue # Skip property if restrictions can't be fetched

                 # Analyze restrictions (SOME, ONLY, VALUE) to find related classes
                 relevant_restriction_types = {"SOME", "ONLY", "VALUE"} # Include VALUE for specific individuals/classes
                 for restriction in restrictions_result:
                     if restriction["type"] in relevant_restriction_types:
                         raw_value = restriction["raw_value"]; classes_to_process = []
                         # Handle different types of restriction values
                         if isinstance(raw_value, ThingClass): classes_to_process.append(raw_value)
                         elif isinstance(raw_value, Or) or isinstance(raw_value, And): classes_to_process.extend(getattr(raw_value, 'Classes', []))
                         # Could add handling for Individuals if needed:
                         # elif isinstance(raw_value, Thing): related_class_names_for_prop.add(f"Individual:{raw_value.name}") # Or its class

                         for related_cls in classes_to_process:
                              if isinstance(related_cls, ThingClass) and hasattr(related_cls, 'name'): related_class_names_for_prop.add(related_cls.name)

                 if related_class_names_for_prop: related_map[prop_name] = sorted(list(related_class_names_for_prop))
         except Exception as e: warnings.warn(f"获取类 '{class_name}' 的相关类时出错: {e}")
         return related_map

    def _get_single_disjoint_classes(self, class_name: str) -> Union[List[str], Dict]:
          """Internal: Get disjoint classes for a single class."""
          if not self._check_ontology_loaded(): return {"error": "本体未加载"}
          cls = self._get_class_by_name(class_name)
          if not cls: return {"error": f"类 '{class_name}' 未找到。"}
          disjoint_classes = set()
          try:
              # Use the disjoints() method which returns AllDisjoint objects
              for disjoint_set in cls.disjoints():
                   # disjoint_set.entities contains the classes in the disjoint axiom
                   if cls in disjoint_set.entities:
                      for entity in disjoint_set.entities:
                          # Add other classes from the set, excluding self
                          if isinstance(entity, ThingClass) and entity != cls and hasattr(entity, 'name'): disjoint_classes.add(entity.name)
          except Exception as e: warnings.warn(f"获取类 '{class_name}' 的不相交类时出错: {e}")
          return sorted(list(disjoint_classes))

    # --- MODIFIED: _get_single_class_properties based on Restrictions ---
    def _get_single_class_properties(self, class_name: str) -> Union[List[str], Dict]:
         """Internal: Get properties used in restrictions for a single class."""
         if not self._check_ontology_loaded(): return {"error": "本体未加载"}
         cls = self._get_class_by_name(class_name)
         if not cls: return {"error": f"类 '{class_name}' 未找到。"}
         properties = set()
         try:
             # Iterate through the class's superclasses and equivalent classes (is_a)
             # Restrictions are typically found here
             for item in cls.is_a:
                 if isinstance(item, owlready2.Restriction):
                     prop = getattr(item, 'property', None)
                     if prop and isinstance(prop, (ObjectProperty, DataProperty)) and hasattr(prop, 'name'):
                         properties.add(prop.name)

             # Consider also equivalent_to if properties might be defined there
             for item in cls.equivalent_to:
                 if isinstance(item, owlready2.Restriction):
                      prop = getattr(item, 'property', None)
                      if prop and isinstance(prop, (ObjectProperty, DataProperty)) and hasattr(prop, 'name'):
                          properties.add(prop.name)
                 # Could also look for properties in equivalent classes if needed
                 # elif isinstance(item, ThingClass):
                 #    # Recursively get properties? Might be complex/circular.
                 #    pass

         except Exception as e: warnings.warn(f"从类 '{class_name}' 的限制中提取属性时出错: {e}")
         return sorted(list(properties))

    # --- MODIFIED: _parse_single_class_definition using new private methods ---
    def _parse_single_class_definition(self, class_name: str) -> Dict:
         if not self._check_ontology_loaded(): return {"error": "本体未加载"}
         cls = self._get_class_by_name(class_name)
         if not cls: return {"error": f"类 '{class_name}' 未找到。"}

         definition = {}
         errors = {}

         # Basic Info
         basic_info = self._get_single_class_info(class_name)
         if "error" in basic_info: errors["basic_info"] = basic_info["error"]
         definition["basic_info"] = basic_info

         # Properties (using the NEW restriction-based method)
         all_properties_res = self._get_single_class_properties(class_name) # Call self!
         properties_summary = {"data": {}, "object": {}}
         sourced_prop_info = {}

         if isinstance(all_properties_res, dict): errors["properties"] = all_properties_res["error"]
         else:
             # Process properties found
             for prop_name in all_properties_res:
                 prop = self._get_property_by_name(prop_name)
                 if not prop:
                     warnings.warn(f"在 _parse_single_class_definition 中无法找到属性 '{prop_name}'，尽管它在 _get_single_class_properties 中被列出。")
                     continue

                 # Get restrictions specifically for this property on this class
                 restrictions = self.get_property_restrictions(class_name, prop_name) # Public API ok here
                 prop_type_key = "object" if isinstance(prop, ObjectProperty) else "data"

                 prop_entry = properties_summary[prop_type_key].get(prop_name, {"name": prop_name, "restrictions": []})
                 if isinstance(restrictions, list):
                     # Filter out potential error dicts if get_property_restrictions returns partial errors
                     prop_entry["restrictions"].extend([r for r in restrictions if isinstance(r, dict) and "error" not in r])
                 elif "error" in restrictions:
                     prop_entry["restriction_error"] = restrictions["error"]
                 properties_summary[prop_type_key][prop_name] = prop_entry

             # Sourced Information for Properties
             prop_info_instances = self._get_sourced_info(cls, info_type=["data_property", "object_property"])
             for info in prop_info_instances:
                  try:
                      prop_name_from_info_list = getattr(info, 'property', []) # Assume it might be a list
                      prop_name_from_info = prop_name_from_info_list[0] if prop_name_from_info_list else None
                      content_list = getattr(info, 'content', [])
                      content = content_list[0] if content_list else None
                      source_list = getattr(info, 'source', [])
                      source = source_list[0] if source_list else None
                      file_path_list = getattr(info, 'file_path', [])
                      file_path = file_path_list[0] if file_path_list else None

                      if prop_name_from_info and content:
                           if prop_name_from_info not in sourced_prop_info: sourced_prop_info[prop_name_from_info] = []
                           sourced_prop_info[prop_name_from_info].append({"content": str(content), "source": str(source) if source else None, "file_path": str(file_path) if file_path else None})
                  except Exception as e: warnings.warn(f"处理属性 SI {getattr(info, 'name', info)} 时出错: {e}")

             # Merge sourced info into properties summary
             for prop_name, info_list in sourced_prop_info.items():
                  prop = self._get_property_by_name(prop_name)
                  if prop:
                      prop_type_key = "object" if isinstance(prop, ObjectProperty) else "data"
                      if prop_name in properties_summary[prop_type_key]:
                          properties_summary[prop_type_key][prop_name]["sourced_information"] = info_list
                      else: # Property mentioned in SI but not found via restrictions
                          properties_summary[prop_type_key][prop_name] = {"name": prop_name, "restrictions": [], "sourced_information": info_list}
                  else:
                      warnings.warn(f"属性 '{prop_name}' 在 SI 中提及但在本体中未找到。")


         definition["properties"] = {"data": list(properties_summary["data"].values()), "object": list(properties_summary["object"].values())}

         # Hierarchy (using new private methods)
         parents = self._get_single_parents(class_name); # Call self!
         children = self._get_single_children(class_name) # Call self!
         if isinstance(parents, dict): errors["parents"] = parents["error"]
         if isinstance(children, dict): errors["children"] = children["error"]
         definition["hierarchy"] = {
             "parents": parents if isinstance(parents, list) else [],
             "children": children if isinstance(children, list) else [],
             "sourced_hierarchy_info": [{"content": str(getattr(i, 'content', [''])[0]), "source": str(getattr(i, 'source', [''])[0])} for i in self._get_sourced_info(cls, info_type="hierarchy")]
         }

         # Relations (using new private method)
         relations = self._get_single_related_classes(class_name) # Call self!
         if isinstance(relations, dict) and "error" in relations: errors["relations"] = relations["error"]
         definition["relations"] = relations if not ("error" in relations) else {}

         # Disjointness (using new private method)
         disjoint_with = self._get_single_disjoint_classes(class_name) # Call self!
         if isinstance(disjoint_with, dict): errors["disjoint_with"] = disjoint_with["error"]
         definition["disjoint_with"] = disjoint_with if isinstance(disjoint_with, list) else []

         if errors: definition["parsing_errors"] = errors
         return definition


    ####################################
    # Public API - Supports List Input
    ####################################

    # --- MODIFIED: Public API methods now call corresponding _get_single_... ---

    def get_class_info(self, class_names: Union[str, List[str]]) -> Dict[str, Dict]:
        if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
        if isinstance(class_names, str): class_names = [class_names]
        return {name: self._get_single_class_info(name) for name in class_names}

    def get_information_sources(self, class_names: Union[str, List[str]]) -> Dict[str, Union[List[str], Dict]]:
        if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
        if isinstance(class_names, str): class_names = [class_names]
        return {name: self._get_single_information_sources(name) for name in class_names}

    def get_information_by_source(self, class_names: Union[str, List[str]], source: str) -> Dict[str, Union[List[str], Dict]]:
         if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
         if isinstance(class_names, str): class_names = [class_names]
         # Keep the internal function here as it's specific to this method's logic
         def _get_single_information_by_source(class_name: str, src: str) -> Union[List[str], Dict]:
              cls = self._get_class_by_name(class_name)
              if not cls: return {"error": f"类 '{class_name}' 未找到。"}
              matching_content = []
              sourced_infos = self._get_sourced_info(cls) # Use the class-level helper
              for info_instance in sourced_infos:
                  try:
                      instance_sources = getattr(info_instance, 'source', [])
                      if isinstance(instance_sources, str): instance_sources = [instance_sources]
                      if src in instance_sources:
                          content = getattr(info_instance, 'content', None)
                          if content is not None:
                              if isinstance(content, list): matching_content.extend([str(c) for c in content])
                              else: matching_content.append(str(content))
                  except AttributeError: pass
                  except Exception as e: warnings.warn(f"为 '{class_name}' 和源 '{src}' 查找信息时出错: {e}")
              return list(set(matching_content))
         return {name: _get_single_information_by_source(name, source) for name in class_names}

    def get_class_properties(self, class_names: Union[str, List[str]]) -> Dict[str, Union[List[str], Dict]]:
        if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
        if isinstance(class_names, str): class_names = [class_names]
        return {name: self._get_single_class_properties(name) for name in class_names} # Call self!

    def get_parents(self, class_names: Union[str, List[str]]) -> Dict[str, Union[List[str], Dict]]:
        if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
        if isinstance(class_names, str): class_names = [class_names]
        return {name: self._get_single_parents(name) for name in class_names} # Call self!

    def get_children(self, class_names: Union[str, List[str]]) -> Dict[str, Union[List[str], Dict]]:
        if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
        if isinstance(class_names, str): class_names = [class_names]
        # Remove the internal definition, call the class-level private method
        return {name: self._get_single_children(name) for name in class_names} # Call self!

    def get_ancestors(self, class_names: Union[str, List[str]]) -> Dict[str, Union[List[str], Dict]]:
         if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
         if isinstance(class_names, str): class_names = [class_names]
         # Remove the internal definition, call the class-level private method
         return {name: self._get_single_ancestors(name) for name in class_names} # Call self!

    def get_descendants(self, class_names: Union[str, List[str]]) -> Dict[str, Union[List[str], Dict]]:
         if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
         if isinstance(class_names, str): class_names = [class_names]
         # Remove the internal definition, call the class-level private method
         return {name: self._get_single_descendants(name) for name in class_names} # Call self!

    def get_related_classes(self, class_names: Union[str, List[str]]) -> Dict[str, Union[Dict[str, List[str]], Dict]]:
        if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
        if isinstance(class_names, str): class_names = [class_names]
        # Remove the internal definition, call the class-level private method
        return {name: self._get_single_related_classes(name) for name in class_names} # Call self!

    def get_disjoint_classes(self, class_names: Union[str, List[str]]) -> Dict[str, Union[List[str], Dict]]:
         if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
         if isinstance(class_names, str): class_names = [class_names]
         # Remove the internal definition, call the class-level private method
         return {name: self._get_single_disjoint_classes(name) for name in class_names} # Call self!

    def parse_class_definition(self, class_names: Union[str, List[str]]) -> Dict[str, Dict]:
        if not self._check_ontology_loaded(): return {name: {"error": "本体未加载"} for name in ([class_names] if isinstance(class_names, str) else class_names)}
        if isinstance(class_names, str): class_names = [class_names]
        return {name: self._parse_single_class_definition(name) for name in class_names} # Call self!

    # --- MODIFIED: get_semantic_similarity using new private methods ---
    def get_semantic_similarity(self, class1_name: str, class2_name: str) -> Union[float, Dict]:
        if not self._check_ontology_loaded(): return {"error": "本体未加载"}
        if class1_name == class2_name: return 1.0
        cls1 = self._get_class_by_name(class1_name); cls2 = self._get_class_by_name(class2_name)
        if not cls1: return {"error": f"类 '{class1_name}' 未找到。"}
        if not cls2: return {"error": f"类 '{class2_name}' 未找到。"}
        def jaccard_similarity(set1: Set, set2: Set) -> float: intersection = len(set1.intersection(set2)); union = len(set1.union(set2)); return intersection / union if union > 0 else 0.0
        try:
            # Use the NEW restriction-based property method via self
            props1_res = self._get_single_class_properties(class1_name); props2_res = self._get_single_class_properties(class2_name); prop_sim = 0.0
            if not isinstance(props1_res, dict) and not isinstance(props2_res, dict): prop_sim = jaccard_similarity(set(props1_res), set(props2_res))

            # Use the NEW ancestor method via self
            anc1_res = self._get_single_ancestors(class1_name); anc2_res = self._get_single_ancestors(class2_name); ancestor_sim = 0.0
            if not isinstance(anc1_res, dict) and not isinstance(anc2_res, dict): ancestor_sim = jaccard_similarity(set(anc1_res), set(anc2_res))

            # Info calculation remains the same
            info1_res = self._get_single_class_info(class1_name); info2_res = self._get_single_class_info(class2_name); info_sim = 0.0
            # Ensure we handle potential errors from _get_single_class_info
            info1_data = info1_res.get("information", []) if isinstance(info1_res, dict) else []
            info2_data = info2_res.get("information", []) if isinstance(info2_res, dict) else []
            info_sim = jaccard_similarity(set(info1_data), set(info2_data))

            # Weights can remain the same, or adjust if needed
            total_similarity = (0.4 * prop_sim) + (0.4 * ancestor_sim) + (0.2 * info_sim)
        except Exception as e: warnings.warn(f"计算 '{class1_name}' 和 '{class2_name}' 相似度时出错: {e}"); return {"error": f"计算相似度时出错: {e}"}
        return round(total_similarity, 4)


    # --- MODIFIED: parse_hierarchy_structure using new private methods ---
    def parse_hierarchy_structure(self, root_class_name: Optional[str] = None) -> Union[Dict, List[Dict]]:
        if not self._check_ontology_loaded(): return {"error": "本体未加载"}
        memo = {}
        # build_subtree now needs access to self to call _get_single_children
        def build_subtree(self_obj: 'OntologyTools', class_name: str, visited_path: Set[str]) -> Dict: # Pass self_obj
            if class_name in memo: return memo[class_name]
            node = {"name": class_name, "children": []}
            if class_name in visited_path: node["cyclic_dependency_detected"] = True; memo[class_name] = node; return node
            visited_path.add(class_name)
            # Use the class-level private method via self_obj
            children_result = self_obj._get_single_children(class_name) # Call self_obj!
            if isinstance(children_result, dict) and "error" in children_result: node["error_fetching_children"] = children_result["error"]
            else:
                for child_name in children_result: node["children"].append(build_subtree(self_obj, child_name, visited_path.copy())) # Pass self_obj
            visited_path.remove(class_name); memo[class_name] = node; return node
        try:
            if root_class_name:
                root_cls = self._get_class_by_name(root_class_name)
                if not root_cls: return {"error": f"根类 '{root_class_name}' 未找到。"}
                memo.clear(); return build_subtree(self, root_class_name, set()) # Pass self
            else:
                top_level_classes = []
                all_classes = list(self.onto.classes()) # Assumes onto is loaded
                for cls in all_classes:
                     if not isinstance(cls, ThingClass) or cls == owlready2.Thing or not hasattr(cls, 'name'): continue
                     # Use the class-level private method via self
                     parents_res = self._get_single_parents(cls.name) # Call self!
                     if isinstance(parents_res, dict): warnings.warn(f"无法获取 '{cls.name}' 父类: {parents_res['error']}"); continue
                     if not parents_res: top_level_classes.append(cls.name)
                forest = []
                for top_class_name in sorted(top_level_classes): memo.clear(); forest.append(build_subtree(self, top_class_name, set())) # Pass self
                return forest
        except Exception as e: error_msg = f"解析层级结构时出错: {e}\n{traceback.format_exc()}"; warnings.warn(error_msg); return {"error": error_msg}


class OntologyAnalyzer:
    """本体分析工具 - 专注于本体结构分析"""
    
    def __init__(self, ontology_settings: Optional[OntologySettings] = None):
        # Store settings or use global default
        if ontology_settings is None:
            from config.settings import ONTOLOGY_SETTINGS as global_settings
            self.settings = global_settings
        else:
            self.settings = ontology_settings
            
        # Initialize LLM
        # Ensure OPENAI_API_KEY is loaded if ChatOpenAI relies on it implicitly
        # from config.settings import OPENAI_API_KEY 
        self.llm = ChatOpenAI(temperature=0) 
        
        # Initialize tools with the determined settings
        self.tools = OntologyTools(self.settings)
        
    def analyze_domain_structure(self) -> Dict:
        """分析领域的基本结构 (using self.settings and self.tools)"""
        
        # Use self.tools for parsing
        hierarchy = self.tools.parse_hierarchy_structure()
        properties_info = [self.tools.parse_property_definition(p.name) 
                         for p in self.settings.ontology.properties() 
                         if "error" not in self.tools.parse_property_definition(p.name)] # Filter errors
        class_names = [c.name for c in self.settings.ontology.classes()]

        structure_info = {
            "hierarchy": hierarchy, 
            "properties": properties_info,
            "classes": class_names
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
        
        try:
            response = self.llm.invoke(prompt.format_messages(**structure_info))
            return parse_json(response.content)
        except Exception as e:
             print(f"Error during LLM invocation or JSON parsing in analyze_domain_structure: {e}")
             return {"error": f"Analysis failed: {e}"}
    
    def find_key_concepts(self) -> List[Dict]:
        """识别关键概念 (using self.settings and self.tools)"""
        
        classes_info = []
        for cls in self.settings.ontology.classes():
            cls_name = cls.name
            try: # Add error handling for tool calls
                 properties = self.tools.get_class_properties(cls_name)
                 parents = self.tools.get_parents(cls_name)
                 children = self.tools.get_children(cls_name)
                 related = self.tools.get_related_classes(cls_name)
                 class_info = {
                     "name": cls_name,
                     "properties": properties, 
                     "parents": parents, 
                     "children": children,
                     "related": related
                 }
                 classes_info.append(class_info)
            except Exception as e:
                 print(f"Error processing class {cls_name} in find_key_concepts: {e}")
                 # Optionally append an error entry or skip
                 classes_info.append({"name": cls_name, "error": str(e)})
            
        relationships = []
        for prop in self.settings.ontology.properties(): 
             try: # Add error handling
                  prop_def = self.tools.parse_property_definition(prop.name)
                  if "error" not in prop_def:
                       relationships.append(prop_def)
             except Exception as e:
                  print(f"Error processing property {prop.name} in find_key_concepts: {e}")
                  relationships.append({"name": prop.name, "error": str(e)})
            
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
        
        try:
            response = self.llm.invoke(prompt.format_messages(
                classes=classes_info,
                relationships=relationships
            ))
            return parse_json(response.content)
        except Exception as e:
             print(f"Error parsing LLM response for key concepts: {e}")
             return {"error": "Failed to parse LLM response", "raw_content": response.content}
        
    def compare_domains(self, other_settings: OntologySettings) -> Dict:
        """比较 self.settings 和 other_settings 代表的领域"""
        
        # Analyze source domain (self)
        try:
            source_structure = self.analyze_domain_structure()
            source_key_concepts = self.find_key_concepts()
            source_analysis = {
                "structure": source_structure,
                "key_concepts": source_key_concepts
            }
        except Exception as e:
             print(f"Error analyzing source domain ({self.settings.ontology_iri}): {e}")
             return {"error": f"Failed to analyze source domain: {e}"}
        
        # Analyze target domain (other)
        try:
            # Create a temporary analyzer for the other settings
            target_analyzer = OntologyAnalyzer(other_settings)
            target_structure = target_analyzer.analyze_domain_structure()
            target_key_concepts = target_analyzer.find_key_concepts()
            target_analysis = {
                "structure": target_structure,
                "key_concepts": target_key_concepts
            }
        except Exception as e:
             print(f"Error analyzing target domain ({other_settings.ontology_iri}): {e}")
             return {"error": f"Failed to analyze target domain: {e}"}
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in cross-domain knowledge transfer."""),
            ("user", """Compare these two domains:
            
            Source Domain Analysis ({self.settings.ontology_iri}):
            {source_analysis}
            
            Target Domain Analysis ({other_settings.ontology_iri}):
            {target_analysis}
            
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
        
        try:
            # Include IRIs in the formatted prompt for context
            response = self.llm.invoke(prompt.format_messages(
                source_analysis=source_analysis,
                target_analysis=target_analysis,
                # Pass IRIs separately if needed in the prompt template
                source_iri = self.settings.ontology_iri, 
                target_iri = other_settings.ontology_iri
            ))
            return parse_json(response.content)
        except Exception as e:
            print(f"Error parsing LLM response for domain comparison: {e}")
            return {"error": "Failed to parse LLM response", "raw_content": response.content}
    
    def get_research_opportunities(self, analysis_result: Dict) -> List[Dict]:
        """从分析结果中提取研究机会"""
        opportunities = []
        
        # 从领域结构分析中提取
        if isinstance(analysis_result, dict) and "research_opportunities" in analysis_result and isinstance(analysis_result["research_opportunities"], list):
            opportunities.extend(analysis_result["research_opportunities"]) 
            
        # 从关键概念分析中提取
        if isinstance(analysis_result, dict) and "key_concepts" in analysis_result and isinstance(analysis_result["key_concepts"], list):
             for concept in analysis_result["key_concepts"]:
                 # Add more robust checking for nested dictionaries/keys
                 if isinstance(concept, dict) and "research_value" in concept and isinstance(concept["research_value"], dict) and concept["research_value"].get("potential", 0) > 0.7:
                     opportunities.append({
                         "type": "concept_based",
                         "concept": concept.get("name", "Unknown"),
                         "opportunity": concept["research_value"].get("description", "N/A")
                     }) 
                    
        # 从跨领域分析中提取 (If analysis_result is from compare_domains)
        if isinstance(analysis_result, dict):
             transfer_opps = analysis_result.get("transfer_opportunities")
             innovation_pts = analysis_result.get("innovation_points")

             if isinstance(transfer_opps, list):
                 opportunities.extend([{"type": "transfer", **opp} for opp in transfer_opps if isinstance(opp, dict)])
             if isinstance(innovation_pts, list):
                 opportunities.extend([{"type": "innovation", **point} for point in innovation_pts if isinstance(point, dict)])
                
        return opportunities