#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep debug script to analyze dimension matching issues.
"""

import pandas as pd
from pathlib import Path
from data_loader import ConsistencyDataLoader
from system_mapping import SYSTEM_NAME_MAPPING

def deep_debug_dimensions():
    """Deep debug dimension matching issues."""
    print("=" * 80)
    print("DEEP DEBUG: DIMENSION MATCHING ANALYSIS")
    print("=" * 80)
    
    loader = ConsistencyDataLoader()
    loader.load_human_scores()
    loader.load_llm_scores()
    
    # Expected dimensions
    expected_dims = set(loader.dimensions)
    print(f"\nExpected dimensions: {expected_dims}")
    print(f"Expected count: {len(expected_dims)}")
    
    # Focus on problematic cases
    problem_cases = [
        ("gpt-4.1-final", "gpt-4.1", "5"),
        ("gpt-4.1-nano-final-815-1", "gpt-4.1-nano", "21")
    ]
    
    for human_sys, llm_sys, q_id in problem_cases:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {human_sys} -> {llm_sys}, Question {q_id}")
        print(f"{'='*60}")
        
        # Check human data
        print(f"\n--- HUMAN DATA ---")
        if human_sys in loader.processed_data['human_scores']:
            if q_id in loader.processed_data['human_scores'][human_sys]:
                human_data = loader.processed_data['human_scores'][human_sys][q_id]
                print(f"Human dimensions found: {set(human_data.keys())}")
                print(f"Human dimension count: {len(human_data)}")
                
                for dim, scores in human_data.items():
                    print(f"  {dim}: {scores} (len={len(scores)})")
            else:
                print(f"Question {q_id} NOT FOUND in human data for {human_sys}")
                print(f"Available questions: {list(loader.processed_data['human_scores'][human_sys].keys())[:5]}...")
        else:
            print(f"System {human_sys} NOT FOUND in human data")
        
        # Check LLM data
        print(f"\n--- LLM DATA ---")
        if llm_sys in loader.processed_data['llm_scores']:
            if q_id in loader.processed_data['llm_scores'][llm_sys]:
                llm_data = loader.processed_data['llm_scores'][llm_sys][q_id]
                print(f"LLM dimensions found: {set(llm_data.keys())}")
                print(f"LLM dimension count: {len(llm_data)}")
                
                for dim, scores in llm_data.items():
                    print(f"  {dim}: {scores} (len={len(scores)})")
            else:
                print(f"Question {q_id} NOT FOUND in LLM data for {llm_sys}")
                print(f"Available questions: {list(loader.processed_data['llm_scores'][llm_sys].keys())[:5]}...")
        else:
            print(f"System {llm_sys} NOT FOUND in LLM data")
        
        # Check intersection
        if (human_sys in loader.processed_data['human_scores'] and 
            q_id in loader.processed_data['human_scores'][human_sys] and
            llm_sys in loader.processed_data['llm_scores'] and
            q_id in loader.processed_data['llm_scores'][llm_sys]):
            
            human_dims = set(loader.processed_data['human_scores'][human_sys][q_id].keys())
            llm_dims = set(loader.processed_data['llm_scores'][llm_sys][q_id].keys())
            common_dims = human_dims.intersection(llm_dims)
            
            print(f"\n--- DIMENSION ALIGNMENT ---")
            print(f"Human dimensions: {human_dims}")
            print(f"LLM dimensions: {llm_dims}")
            print(f"Common dimensions: {common_dims}")
            print(f"Missing from human: {expected_dims - human_dims}")
            print(f"Missing from LLM: {expected_dims - llm_dims}")
            print(f"Missing from both: {expected_dims - common_dims}")
    
    # Check if there's a pattern across all systems
    print(f"\n{'='*60}")
    print("DIMENSION COMPLETENESS ACROSS ALL SYSTEMS")
    print(f"{'='*60}")
    
    for human_sys, llm_sys in SYSTEM_NAME_MAPPING.items():
        if (human_sys in loader.processed_data['human_scores'] and
            llm_sys in loader.processed_data['llm_scores']):
            
            human_questions = loader.processed_data['human_scores'][human_sys]
            llm_questions = loader.processed_data['llm_scores'][llm_sys]
            
            # Sample first few questions to check dimension patterns
            sample_questions = list(set(human_questions.keys()) & set(llm_questions.keys()))[:3]
            
            print(f"\n{human_sys} -> {llm_sys}")
            
            for q in sample_questions:
                h_dims = set(human_questions[q].keys()) if q in human_questions else set()
                l_dims = set(llm_questions[q].keys()) if q in llm_questions else set()
                common = h_dims & l_dims
                
                h_missing = expected_dims - h_dims
                l_missing = expected_dims - l_dims
                
                print(f"  Q{q}: H={len(h_dims)}, L={len(l_dims)}, Common={len(common)}")
                if h_missing:
                    print(f"    Human missing: {h_missing}")
                if l_missing:
                    print(f"    LLM missing: {l_missing}")

if __name__ == "__main__":
    deep_debug_dimensions()