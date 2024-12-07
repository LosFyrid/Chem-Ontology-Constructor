from owlready2 import *
import numpy as np

from config import settings


def create_metadata_properties():
    """
    在本体中创建必要的注释属性类别
    

    """
    ontology = settings.ONTOLOGY_CONFIG["ontology"]
    meta = settings.ONTOLOGY_CONFIG["meta"]
    with ontology:
        # # 检查并创建embedding注释属性(值类型为ndarray) 缺乏合适的数据类型类
        # if meta.embedding in ontology.annotation_properties():
        #     print("embedding exists")
        #     embedding_prop = meta.embedding
        #     if embedding_prop.range != np.ndarray:
        #         print("embedding range is not ndarray, changing...")
        #         embedding_prop.range = np.ndarray
        # else:
        #     print("embedding does not exist, creating...")
        #     class embedding(AnnotationProperty):
        #         range = [np.ndarray]
        #         namespace = meta
            
        # 检查并创建source注释属性(值类型为元组列表,每个元组包含两个字符串)  
        if meta.source in ontology.annotation_properties():
            print("source exists")
            source_prop = meta.source
            if source_prop.range != [str]:
                print("source range is not str, changing...")
                source_prop.range = [str]
        else:
            print("source does not exist, creating...")
            class source(AnnotationProperty):
                range = [str]
                namespace = meta
            
        # 检查并创建information注释属性(值类型为字符串列表)
        if meta.information in ontology.annotation_properties():
            print("infomation exists")
            info_prop = meta.information
            if info_prop.range != [str]:
                print("information range is not str, changing...")
                info_prop.range = [str]
        else:
            print("information does not exist, creating...")
            class information(AnnotationProperty):
                range = [str]
                namespace = meta
        
        # 检查并创建hierarchy_information注释属性(值类型为字符串列表)
        if meta.hierarchy_information in ontology.annotation_properties():
            print("hierachy_information exists")
            hierarchy_info_prop = meta.hierarchy_information
            if hierarchy_info_prop.range != [str]:
                print("hierarchy_information range is not str, changing...")
                hierarchy_info_prop.range = [str]
        else:
            print("hierarchy_information does not exist, creating...")
            class hierarchy_information(AnnotationProperty):
                range = [str]
                namespace = meta
    ontology.save()



