from owlready2 import *
import numpy as np
import datetime

from config import settings


def create_metadata_properties():
    """Create necessary metadata classes and properties in the ontology"""
    ontology = settings.ONTOLOGY_CONFIG["ontology"]
    meta = settings.ONTOLOGY_CONFIG["meta"]
    
    with ontology:
        # 创建SourcedInformation类
        if not "SourcedInformation" in ontology.classes():
            print("Creating SourcedInformation class...")
            class SourcedInformation(Thing):
                namespace = meta
        
        # 创建content和source属性
        if not meta.content in ontology.data_properties():
            print("Creating content property...")
            class content(DataProperty):
                namespace = meta
                domain = [SourcedInformation]
                range = [str]
                
        if not meta.source in ontology.data_properties():
            print("Creating source property...")
            class source(DataProperty):
                namespace = meta
                domain = [SourcedInformation]
                range = [str]
        
        # 创建has_information对象属性
        if not meta.has_information in ontology.object_properties():
            print("Creating has_information property...")
            class has_information(ObjectProperty):
                namespace = meta
                domain = [Thing]
                range = [SourcedInformation]
    
    ontology.save()



