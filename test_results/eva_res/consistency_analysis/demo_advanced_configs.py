#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration script for advanced configuration features.
Shows how to use per-model round selection and model filtering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from multi_config_main import MultiConfigAnalysisRunner

def demo_advanced_configurations():
    """Demonstrate the advanced configuration features."""
    print("=" * 80)
    print("ADVANCED CONFIGURATION DEMONSTRATION")
    print("=" * 80)
    print("This script demonstrates the advanced configuration features:")
    print("1. Per-model LLM round selection")
    print("2. Model subset analysis") 
    print("3. Combined features")
    print("4. Dimension filtering")
    print()
    
    # Configuration examples to run
    demo_configs = [
        # 1. Model subset analysis - only GPT-4 family
        'reduced_selected_models',
        
        # 2. Per-model rounds with all models
        'reduced_per_model_example',
        
        # 3. Combined: GPT-4 family with custom rounds per model
        'reduced_gpt4_family_custom_rounds',
    ]
    
    print(f"Running {len(demo_configs)} demonstration analyses:")
    for config_name in demo_configs:
        from analysis_config import get_analysis_config
        config = get_analysis_config(config_name)
        print(f"  - {config.name}")
        if config.selected_models:
            print(f"    Models: {', '.join(config.selected_models)}")
        if config.per_model_rounds:
            print(f"    Custom rounds: {config.per_model_rounds}")
        print(f"    Dimensions: {len(config.dimensions)} (excluded: Logic, Clarity)")
        print()
    
    # Run the analyses
    runner = MultiConfigAnalysisRunner()
    results = runner.run_all_analyses(demo_configs)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    if results['successful_analyses'] == len(demo_configs):
        print("✓ All advanced configurations ran successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✓ Model subset analysis (only GPT-4 family)")
        print("  ✓ Per-model LLM round selection") 
        print("  ✓ Dimension filtering (excluded Logic & Clarity)")
        print("  ✓ Combined advanced features")
        
        print(f"\nResults saved to: {results['base_output_dir']}")
        print("Each configuration created its own analysis with:")
        print("  - Detailed statistical results")
        print("  - Consistency visualizations")
        print("  - Analysis reports")
        print("  - Summary CSV files")
        
    else:
        print(f"⚠️  {results['successful_analyses']}/{len(demo_configs)} analyses completed")
        print("Check the error messages above for details.")
    
    return results

if __name__ == "__main__":
    demo_advanced_configurations()