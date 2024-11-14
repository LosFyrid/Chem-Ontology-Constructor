from owlready2 import *


from config import settings

onto_path.append(settings.ONTOLOGY_CONFIG["ontology_directory_path"])
ontology = get_ontology("http://www.test.org/chem_ontologies/chem_ontology.owl").load(only_local=True)
meta = ontology.get_namespace("http://www.test.org/chem_ontologies/meta/")


def create_metadata_properties(ontology: owlready2.namespace.Ontology, meta: owlready2.namespace.Namespace):
    """
    在本体中创建必要的注释属性类别
    
    Args:
        ontology: owlready2.namespace.Ontology - 目标本体
    """
    with ontology:
        # 检查并创建embedding注释属性(值类型为浮点数列表)
        if meta.embedding in ontology.annotation_properties():
            print("embedding exists")
            embedding_prop = meta.embedding
            if embedding_prop.range != [float]:
                print("embedding range is not float, changing...")
                embedding_prop.range = [float]
        else:
            print("embedding does not exist, creating...")
            class embedding(AnnotationProperty):
                range = [float]
                namespace = meta
            
        # 检查并创建location注释属性(值类型为元组列表,每个元组包含两个字符串)  
        if meta.location in ontology.annotation_properties():
            location_prop = meta.location
            if location_prop.range != [str]:
                print("location range is not (str, str), changing...")
                location_prop.range = [str]
        else:
            print("location does not exist, creating...")
            class location(AnnotationProperty):
                range = [str]
                namespace = meta
            
        # 检查并创建information注释属性(值类型为字符串列表)
        if meta.information in ontology.annotation_properties():
            info_prop = meta.information
            if info_prop.range != [str]:
                print("information range is not str, changing...")
                info_prop.range = [str]
        else:
            print("information does not exist, creating...")
            class information(AnnotationProperty):
                range = [str]
                namespace = meta


