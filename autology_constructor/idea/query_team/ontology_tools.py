from typing import List, Tuple, Dict, Optional, Set, Any, Callable
from owlready2 import *
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import re

from autology_constructor.idea.query_team.utils import parse_json, format_sparql_results, extract_variables_from_sparql


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
        where_pattern = r'(WHERE\s*\{)(.*?)(\})'
        
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
        # 执行查询
        results = list(default_world.sparql(query))
        
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
        self.sparql_executor = SparqlExecutor()

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
    
    #######################
    # SPARQL Operations
    #######################
    
    def execute_sparql(self, sparql_query: str) -> Dict:
        """执行SPARQL查询并返回格式化结果
        
        Args:
            sparql_query: SPARQL查询字符串
            
        Returns:
            查询结果，包含results字段的字典
            
        Example:
            >>> query = "SELECT ?x WHERE { ?x rdf:type owl:Class }"
            >>> tools.execute_sparql(query)
            {'results': [{'var0': 'Class1'}, {'var0': 'Class2'}, ...]}
        """
        try:
            # 使用SPARQL执行器执行查询
            return self.sparql_executor.execute(sparql_query, self.onto)
        except SparqlExecutionError as e:
            return {"error": str(e), "query": sparql_query}
        except Exception as e:
            return {"error": f"查询执行异常: {str(e)}", "query": sparql_query}

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