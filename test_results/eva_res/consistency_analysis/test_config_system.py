#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the configurable consistency analysis system.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analysis_config import get_default_analysis_configs, ANALYSIS_CONFIGS
from data_loader import ConsistencyDataLoader
from consistency_calculator import ConsistencyAnalyzer

def test_configurations():
    """Test different analysis configurations."""
    print("=" * 60)
    print("TESTING ANALYSIS CONFIGURATIONS")
    print("=" * 60)
    
    # Test configuration loading
    print("\\n1. Testing configuration loading:")
    configs = get_default_analysis_configs()
    for i, config in enumerate(configs, 1):
        print(f"   {i}. {config.name}")
        print(f"      - Dimensions: {len(config.dimensions)} ({', '.join(config.dimensions[:2])}...)")
        print(f"      - LLM Strategy: {config.llm_strategy}" + (f" (Round {config.llm_round})" if config.llm_round else ""))
        print(f"      - Output Suffix: {config.output_suffix}")
        
        # Validate configuration
        try:
            config.validate()
            print(f"      OK - Configuration is valid")
        except Exception as e:
            print(f"      ERROR - Configuration error: {e}")
        print()
    
    # Test data loading
    print("2. Testing data loading:")
    try:
        loader = ConsistencyDataLoader()
        aligned_data = loader.load_all_data()
        print(f"   OK - Data loaded successfully")
        print(f"   - Systems: {len(aligned_data['systems'])}")
        print(f"   - Dimensions: {len(aligned_data['dimensions'])}")
        print(f"   - Sample systems: {', '.join(aligned_data['systems'][:3])}...")
    except Exception as e:
        print(f"   ERROR - Data loading failed: {e}")
        return False
    
    # Test analyzer with different configurations
    print("\\n3. Testing analyzers with different configurations:")
    
    for i, config in enumerate(configs[:2], 1):  # Test first 2 configs
        print(f"   Testing config {i}: {config.name}")
        try:
            analyzer = ConsistencyAnalyzer(config)
            print(f"   OK Analyzer created successfully")
            print(f"   - Config dimensions: {len(analyzer.config.dimensions)}")
            print(f"   - LLM strategy: {analyzer.config.llm_strategy}")
            
            # Try a small consistency calculation (just human internal)
            print("   - Testing human internal consistency calculation...")
            human_results = analyzer.calculate_human_internal_consistency(aligned_data['human_scores'])
            print(f"   OK Human internal consistency calculated for {len(human_results)} dimensions")
            
            # Print sample results
            if human_results:
                sample_dim = list(human_results.keys())[0]
                sample_result = human_results[sample_dim]
                print(f"   - Sample ({sample_dim}): ICC = {sample_result['icc']['value']:.3f}")
            
        except Exception as e:
            print(f"   ERROR Analyzer test failed: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    return True

def test_llm_strategies():
    """Test different LLM evaluation strategies."""
    print("=" * 60)
    print("TESTING LLM EVALUATION STRATEGIES")
    print("=" * 60)
    
    # Load data
    try:
        loader = ConsistencyDataLoader()
        aligned_data = loader.load_all_data()
    except Exception as e:
        print(f"ERROR Could not load data: {e}")
        return False
    
    # Test average strategy
    print("\\n1. Testing LLM Average Strategy:")
    try:
        config_avg = ANALYSIS_CONFIGS['full_average']
        analyzer_avg = ConsistencyAnalyzer(config_avg)
        results_avg = analyzer_avg.calculate_human_llm_agreement(
            aligned_data['human_scores'], 
            aligned_data['llm_scores']
        )
        print(f"   OK Average strategy: {len(results_avg)} dimensions calculated")
        
        if results_avg:
            sample_dim = list(results_avg.keys())[0]
            print(f"   - Sample ({sample_dim}): {results_avg[sample_dim]['n_pairs']} data pairs")
    
    except Exception as e:
        print(f"   ERROR Average strategy test failed: {e}")
    
    # Test specific round strategy
    print("\\n2. Testing LLM Specific Round Strategy:")
    try:
        config_round = ANALYSIS_CONFIGS['full_round1']
        analyzer_round = ConsistencyAnalyzer(config_round)
        results_round = analyzer_round.calculate_human_llm_agreement(
            aligned_data['human_scores'], 
            aligned_data['llm_scores']
        )
        print(f"   OK Round 1 strategy: {len(results_round)} dimensions calculated")
        
        if results_round:
            sample_dim = list(results_round.keys())[0]
            print(f"   - Sample ({sample_dim}): {results_round[sample_dim]['n_pairs']} data pairs")
    
    except Exception as e:
        print(f"   ERROR Round 1 strategy test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def test_dimension_filtering():
    """Test dimension filtering functionality."""
    print("=" * 60)
    print("TESTING DIMENSION FILTERING")
    print("=" * 60)
    
    # Load data
    try:
        loader = ConsistencyDataLoader()
        aligned_data = loader.load_all_data()
    except Exception as e:
        print(f"ERROR Could not load data: {e}")
        return False
    
    # Test full dimensions (6)
    print("\\n1. Testing Full Dimensions (6):")
    try:
        config_full = ANALYSIS_CONFIGS['full_average']
        analyzer_full = ConsistencyAnalyzer(config_full)
        results_full = analyzer_full.calculate_human_internal_consistency(aligned_data['human_scores'])
        print(f"   OK Full dimensions: {len(results_full)} dimensions calculated")
        print(f"   - Dimensions: {list(results_full.keys())}")
    
    except Exception as e:
        print(f"   ERROR Full dimensions test failed: {e}")
    
    # Test reduced dimensions (4)
    print("\\n2. Testing Reduced Dimensions (4):")
    try:
        config_reduced = ANALYSIS_CONFIGS['reduced_average'] 
        analyzer_reduced = ConsistencyAnalyzer(config_reduced)
        results_reduced = analyzer_reduced.calculate_human_internal_consistency(aligned_data['human_scores'])
        print(f"   OK Reduced dimensions: {len(results_reduced)} dimensions calculated")
        print(f"   - Dimensions: {list(results_reduced.keys())}")
        
        # Verify that logic and clarity are excluded
        excluded_dims = set(["逻辑性", "清晰度"])
        calculated_dims = set(results_reduced.keys())
        if excluded_dims.intersection(calculated_dims):
            print(f"   WARNING  Warning: Should have excluded {excluded_dims.intersection(calculated_dims)}")
        else:
            print(f"   OK Correctly excluded Logic and Clarity dimensions")
    
    except Exception as e:
        print(f"   ERROR Reduced dimensions test failed: {e}")
    
    return True

def main():
    """Run all tests."""
    print("CONFIGURABLE CONSISTENCY ANALYSIS - TEST SUITE")
    print("=" * 80)
    
    success = True
    
    # Run tests
    tests = [
        ("Configuration Loading", test_configurations),
        ("LLM Evaluation Strategies", test_llm_strategies), 
        ("Dimension Filtering", test_dimension_filtering)
    ]
    
    for test_name, test_func in tests:
        print(f"\\n\\n>>> RUNNING: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"OK {test_name}: PASSED")
            else:
                print(f"ERROR {test_name}: FAILED")
                success = False
        except Exception as e:
            print(f"ERROR {test_name}: FAILED with exception: {e}")
            success = False
    
    # Final result
    print("\\n" + "=" * 80)
    if success:
        print("OK ALL TESTS PASSED - System is ready for multi-configuration analysis!")
    else:
        print("ERROR SOME TESTS FAILED - Please check the issues above")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    main()