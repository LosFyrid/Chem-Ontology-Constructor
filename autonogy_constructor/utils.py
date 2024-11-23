from autonogy_constructor.base_data_structures import Ontology

def ontology_to_string(ontology: Ontology) -> str:
    result = []
    
    result.append("Entities:")
    for entity in ontology.entities:
        result.append(f"  - Name: {entity.name}")
        result.append(f"    Information: {entity.information}")
    
    # result.append("\nEntity Hierarchy:")
    # if ontology.hierarchy is not None:
    #     for hierarchy in ontology.hierarchy:
    #         result.append(f"  - Subclass: {hierarchy.subclass}")
    #         result.append(f"    Superclass: {hierarchy.superclass}")
    #         result.append(f"    Information: {hierarchy.information}")
    
    # result.append("\nDisjoint Classes:")
    # if ontology.disjointness is not None:
    #     for disjoint in ontology.disjointness:
    #         result.append(f"  - Class 1: {disjoint.class1}")
    #         result.append(f"    Class 2: {disjoint.class2}")
    
    result.append("\nData Properties:")
    if ontology.data_properties is not None:
        for prop in ontology.data_properties:
            result.append(f"  - Name: {prop.name}")
            if prop.values is not None:
                values_str = "; ".join(f"{k}: {v}" for k, v in prop.values.items())
                result.append(f"    Values: {values_str}")
            result.append(f"    Information: {prop.information}")
        
    result.append("\nObject Properties:")
    for prop in ontology.object_properties:
        result.append(f"  - Name: {prop.name}")
        if prop.domain is not None:
            result.append(f"    Domain:")
            result.append(f"      Existence: {prop.domain.existence}")
            if prop.domain.entity is not None:
                result.append(f"      Entity: {prop.domain.entity}")
            if prop.domain.type is not None:
                result.append(f"      Type: {prop.domain.type}")
        if prop.range is not None:
            result.append(f"    Range:")
            result.append(f"      Existence: {prop.range.existence}")
            if prop.range.entity is not None:
                result.append(f"      Entity: {prop.range.entity}")
            if prop.range.type is not None:
                result.append(f"      Type: {prop.range.type}")
        result.append(f"    Restriction: {prop.restriction}")
        result.append(f"    Information: {prop.information}")
        
    return "\n".join(result)