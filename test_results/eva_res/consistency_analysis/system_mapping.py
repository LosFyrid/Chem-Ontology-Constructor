#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard-coded mapping between human evaluation system names and LLM evaluation system names.
Please review and confirm the mappings marked with "TODO: CONFIRM"
"""

# Human evaluation systems (from CSV header):
HUMAN_SYSTEMS = [
    "gpt-4.1-final",
    "gpt-4.1-nano-final-815-1", 
    "lightrag-gpt-4_1",
    "lightrag-gpt-4_1-nano",
    "o1-final",
    "o3-final",
    "reordered_MOSES-final",
    "reordered_MOSES-nano-final",
    "chemqa27_from_chem13b_rag_infer_yesthink",
    "gpt-4o-final-815-1",
    "gpt-4o-mini-final-815-1",
    "llasmol",
    "darwin"
]

# LLM evaluation systems (from JSON files):
LLM_SYSTEMS = [
    "MOSES",
    "MOSES-nano", 
    "darwin",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "lightrag-4.1",
    "lightrag-4.1-nano",
    "llasmol-top1",
    "llasmol-top5",
    "o1",
    "o3",
    "spark-chem13b-nothink",
    "spark-chem13b-think"
]

# Hard-coded mapping: Human system name -> LLM system name
SYSTEM_NAME_MAPPING = {
    # Clear mappings
    "gpt-4.1-final": "gpt-4.1",
    "gpt-4.1-nano-final-815-1": "gpt-4.1-nano",
    "lightrag-gpt-4_1": "lightrag-4.1", 
    "lightrag-gpt-4_1-nano": "lightrag-4.1-nano",
    "o1-final": "o1",
    "o3-final": "o3",
    "gpt-4o-final-815-1": "gpt-4o",
    "gpt-4o-mini-final-815-1": "gpt-4o-mini",
    "reordered_MOSES-final": "MOSES",
    "reordered_MOSES-nano-final": "MOSES-nano",
    
    # New model mappings
    "llasmol": "llasmol-top5",  # Default to top5 version for per-model rounds
    "darwin": "darwin",
    
    # TODO: CONFIRM - Please verify this mapping
    "chemqa27_from_chem13b_rag_infer_yesthink": "spark-chem13b-think",
}

# Dimension mapping: English -> Chinese
DIMENSION_MAPPING = {
    "correctness": "正确性",
    "logic": "逻辑性", 
    "clarity": "清晰度",
    "completeness": "完备性",
    "theoretical_depth": "理论深度",
    "rigor_and_information_density": "论述严谨性与信息密度"
}

# Reverse mapping: Chinese -> English  
DIMENSION_MAPPING_REVERSE = {v: k for k, v in DIMENSION_MAPPING.items()}

def get_mapped_llm_system(human_system_name: str) -> str:
    """Get corresponding LLM system name for a human system name."""
    return SYSTEM_NAME_MAPPING.get(human_system_name)

def get_mapped_dimension(dimension: str, to_chinese: bool = True) -> str:
    """Map dimension name between English and Chinese."""
    if to_chinese:
        return DIMENSION_MAPPING.get(dimension, dimension)
    else:
        return DIMENSION_MAPPING_REVERSE.get(dimension, dimension)

def get_all_mapped_systems() -> list:
    """Get list of systems that have both human and LLM evaluations."""
    return list(SYSTEM_NAME_MAPPING.keys())

def validate_mappings():
    """Validate that all mapped systems exist in both lists."""
    missing_human = []
    missing_llm = []
    
    for human_sys, llm_sys in SYSTEM_NAME_MAPPING.items():
        if human_sys not in HUMAN_SYSTEMS:
            missing_human.append(human_sys)
        if llm_sys not in LLM_SYSTEMS:
            missing_llm.append(llm_sys)
    
    if missing_human:
        print(f"Warning: Mapped human systems not found in HUMAN_SYSTEMS: {missing_human}")
    if missing_llm:
        print(f"Warning: Mapped LLM systems not found in LLM_SYSTEMS: {missing_llm}")
    
    return len(missing_human) == 0 and len(missing_llm) == 0

if __name__ == "__main__":
    print("=== SYSTEM NAME MAPPING ===")
    print(f"Human systems: {len(HUMAN_SYSTEMS)}")
    print(f"LLM systems: {len(LLM_SYSTEMS)}")
    print(f"Mapped systems: {len(SYSTEM_NAME_MAPPING)}")
    print()
    
    print("Mappings:")
    for human_sys, llm_sys in SYSTEM_NAME_MAPPING.items():
        status = "TODO: CONFIRM" if "chemqa27" in human_sys else "OK"
        print(f"  {human_sys} -> {llm_sys} ({status})")
    
    print()
    print("Unmapped human systems:")
    for human_sys in HUMAN_SYSTEMS:
        if human_sys not in SYSTEM_NAME_MAPPING:
            print(f"  - {human_sys}")
    
    print()
    print("Unmapped LLM systems:")
    for llm_sys in LLM_SYSTEMS:
        if llm_sys not in SYSTEM_NAME_MAPPING.values():
            print(f"  - {llm_sys}")
    
    print()
    print("Validation result:", "PASS" if validate_mappings() else "FAIL")
    
    print()
    print("=== DIMENSION MAPPING ===")
    for eng, chn in DIMENSION_MAPPING.items():
        print(f"  {eng} -> {chn}")