from typing import List, Dict, Any, Union
from owlready2 import Thing, ThingClass
import json

def parse_json(content: str) -> List[Dict]:
    """解析JSON格式的执行计划
    
    Args:
        content: JSON格式的字符串
        
    Returns:
        解析后的执行计划列表
    
    Example:
        >>> content = '[{"operation": "get_class_info", "args": {"class_name": "calixarene"}}]'
        >>> parse_json(content)
        [{'operation': 'get_class_info', 'args': {'class_name': 'calixarene'}}]
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []

def format_owlready2_value(value: Any) -> Union[str, List[str], Dict]:
    """格式化owlready2返回的值
    
    Args:
        value: owlready2返回的值（可能是Thing, ThingClass或其他类型）
        
    Returns:
        格式化后的值
        
    Example:
        >>> class1 = ThingClass("Calixarene")
        >>> format_owlready2_value(class1)
        'Calixarene'
    """
    if isinstance(value, (Thing, ThingClass)):
        return value.name
    elif isinstance(value, (list, set)):
        return [format_owlready2_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: format_owlready2_value(v) for k, v in value.items()}
    else:
        return str(value)

def format_sparql_results(results: List) -> Dict[str, Any]:
    """格式化SPARQL查询结果
    
    Args:
        results: SPARQL查询返回的结果列表
        
    Returns:
        格式化后的结果字典
        
    Example:
        >>> results = [(Thing("calixarene"), "binds", Thing("metal_ion"))]
        >>> format_sparql_results(results)
        {'results': [{'var0': 'calixarene', 'var1': 'binds', 'var2': 'metal_ion'}]}
    """
    formatted = []
    
    for result in results:
        if isinstance(result, tuple):
            item = {}
            for i, value in enumerate(result):
                item[f"var{i}"] = format_owlready2_value(value)
            formatted.append(item)
        else:
            formatted.append({"result": format_owlready2_value(result)})
            
    return {"results": formatted}

def extract_variables_from_sparql(query: str) -> List[str]:
    """从SPARQL查询中提取变量名
    
    Args:
        query: SPARQL查询字符串
        
    Returns:
        变量名列表
        
    Example:
        >>> query = "SELECT ?x ?y WHERE { ?x rdf:type ?y }"
        >>> extract_variables_from_sparql(query)
        ['x', 'y']
    """
    import re
    variables = re.findall(r'\?(\w+)', query)
    return list(dict.fromkeys(variables))  # 去重保持顺序

def format_query_results(results: Dict, variables: List[str] = None) -> str:
    """Format query results as readable string"""
    if not results or 'results' not in results:
        return "No results found"
        
    formatted = []
    for result in results['results']:
        if variables:
            # Use provided variable names
            items = [f"{var}: {result.get(f'var{i}', 'N/A')}"
                    for i, var in enumerate(variables)]
        else:
            # Use default variable names
            items = [f"{k}: {v}" for k, v in result.items()]
        formatted.append("\n".join(items))
        
    return "\n\n".join(formatted) 

def format_class_info(class_info: Dict) -> str:
    """Format class information as a readable string"""
    result = f"Information of '{class_info['name']}':\n"
    for item in class_info['information']:
        result += f"  - {item}\n"
    result += f"Source of '{class_info['name']}':\n"
    for item in class_info['source']:
        result += f"  - {item}\n"
    return result

def format_restrictions(restrictions: List[Dict], class_name: str, property_name: str) -> str:
    """Format property restrictions as a readable string"""
    result = f"{class_name} class restrictions on property {property_name}:\n"
    for r in restrictions:
        result += f"Type: {r['type']}\n"
        result += f"Value: {r['value']}\n"
        result += "---\n"
    return result

def format_hierarchy(parents: List[str], children: List[str], class_name: str) -> str:
    """Format class hierarchy information as a readable string"""
    result = f"Hierarchy for class '{class_name}':\n"
    if parents:
        result += "Parents:\n"
        for p in parents:
            result += f"  - {p}\n"
    if children:
        result += "Children:\n"
        for c in children:
            result += f"  - {c}\n"
    return result 