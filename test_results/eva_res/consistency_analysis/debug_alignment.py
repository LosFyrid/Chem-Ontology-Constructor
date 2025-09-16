#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to analyze why some systems have fewer than 27 questions aligned.
"""

import sys
from pathlib import Path
from data_loader import ConsistencyDataLoader
from system_mapping import SYSTEM_NAME_MAPPING

def debug_alignment():
    """Debug the alignment process to identify missing questions."""
    print("=" * 80)
    print("DEBUG: ANALYZING QUESTION ALIGNMENT ISSUES")
    print("=" * 80)
    
    loader = ConsistencyDataLoader()
    
    # Load data
    print("\n1. Loading data...")
    loader.load_human_scores()
    loader.load_llm_scores()
    
    print("\n2. Analyzing alignment for each system...")
    
    all_questions = set(range(1, 28))  # Expected questions 1-27
    
    for human_system, llm_system in SYSTEM_NAME_MAPPING.items():
        print(f"\n=== SYSTEM: {human_system} -> {llm_system} ===")
        
        # Check if both systems exist in data
        human_exists = human_system in loader.processed_data['human_scores']
        llm_exists = llm_system in loader.processed_data['llm_scores']
        
        print(f"Human data exists: {human_exists}")
        print(f"LLM data exists: {llm_exists}")
        
        if not human_exists or not llm_exists:
            print("❌ SKIPPED - Missing data")
            continue
            
        # Get question sets
        human_questions = set(loader.processed_data['human_scores'][human_system].keys())
        llm_questions = set(loader.processed_data['llm_scores'][llm_system].keys())
        
        # Convert to int for comparison
        human_q_ints = {int(q) for q in human_questions}
        llm_q_ints = {int(q) for q in llm_questions}
        
        common_questions = human_q_ints.intersection(llm_q_ints)
        
        print(f"Human questions: {len(human_q_ints)} -> {sorted(human_q_ints)}")
        print(f"LLM questions: {len(llm_q_ints)} -> {sorted(llm_q_ints)}")
        print(f"Common questions: {len(common_questions)} -> {sorted(common_questions)}")
        
        # Identify missing questions
        missing_from_human = all_questions - human_q_ints
        missing_from_llm = all_questions - llm_q_ints
        missing_from_both = all_questions - common_questions
        
        if missing_from_human:
            print(f"❌ Missing from HUMAN: {sorted(missing_from_human)}")
        if missing_from_llm:
            print(f"❌ Missing from LLM: {sorted(missing_from_llm)}")
        if missing_from_both:
            print(f"❌ Missing from COMMON: {sorted(missing_from_both)}")
            
        # Check dimension alignment for common questions
        if common_questions:
            print(f"\n  Checking dimension alignment for {len(common_questions)} common questions...")
            
            dimension_issues = []
            final_aligned = []
            
            for question in common_questions:
                q_str = str(question)
                
                human_dims = set(loader.processed_data['human_scores'][human_system][q_str].keys())
                llm_dims = set(loader.processed_data['llm_scores'][llm_system][q_str].keys())
                common_dims = human_dims.intersection(llm_dims)
                
                if len(common_dims) >= 3:  # Threshold from align_data method
                    # Check score quality
                    valid_dims = []
                    for dimension in common_dims:
                        human_scores = loader.processed_data['human_scores'][human_system][q_str][dimension]
                        llm_scores = loader.processed_data['llm_scores'][llm_system][q_str][dimension]
                        
                        if len(human_scores) >= 2 and len(llm_scores) >= 3:
                            valid_dims.append(dimension)
                    
                    if len(valid_dims) >= 3:
                        final_aligned.append(question)
                    else:
                        dimension_issues.append(f"Q{question}: insufficient scores")
                else:
                    dimension_issues.append(f"Q{question}: only {len(common_dims)} common dims")
            
            print(f"  Questions passing dimension check: {len(final_aligned)}")
            if dimension_issues:
                print(f"  Dimension issues: {dimension_issues[:5]}...")  # Show first 5
            
            if len(final_aligned) != 27:
                print(f"  WARNING: Final aligned count: {len(final_aligned)} (expected 27)")
                
                # Detailed analysis for problematic questions
                missing_final = all_questions - set(final_aligned)
                if missing_final:
                    print(f"  Missing from final: {sorted(missing_final)}")
                    
                    for missing_q in sorted(missing_final):
                        q_str = str(missing_q)
                        if missing_q in common_questions:
                            human_dims = set(loader.processed_data['human_scores'][human_system][q_str].keys())
                            llm_dims = set(loader.processed_data['llm_scores'][llm_system][q_str].keys())
                            common_dims = human_dims.intersection(llm_dims)
                            
                            print(f"    Q{missing_q}: human_dims={human_dims}, llm_dims={llm_dims}, common={common_dims}")
                            
                            # Check individual dimension scores
                            for dim in human_dims.union(llm_dims):
                                h_scores = loader.processed_data['human_scores'][human_system][q_str].get(dim, [])
                                l_scores = loader.processed_data['llm_scores'][llm_system][q_str].get(dim, [])
                                print(f"      {dim}: human={h_scores}, llm={l_scores}")
        
        
        print()

if __name__ == "__main__":
    debug_alignment()