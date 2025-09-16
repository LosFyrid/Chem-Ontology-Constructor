#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-configuration consistency analysis pipeline.
Runs three parallel analyses with different configurations.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from data_loader import ConsistencyDataLoader
from consistency_calculator import ConsistencyAnalyzer
from visualizer import ConsistencyVisualizer
from analysis_config import get_default_analysis_configs, get_analysis_config
from report_generator import ReportGenerator

class MultiConfigAnalysisRunner:
    """Run multiple consistency analyses with different configurations."""
    
    def __init__(self, base_output_dir: str = None):
        if base_output_dir is None:
            self.base_output_dir = Path(__file__).parent / "results"
        else:
            self.base_output_dir = Path(base_output_dir)
        
        # Timestamp for this analysis run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results storage
        self.analysis_results = {}
    
    def run_analysis(self, config_name: str) -> Dict[str, Any]:
        """Run consistency analysis for a specific configuration."""
        try:
            config = get_analysis_config(config_name)
            print(f"\n{'=' * 60}")
            print(f"ANALYSIS: {config.name}")
            print(f"{'=' * 60}")
            print(f"Configuration:")
            print(f"  - Dimensions ({len(config.dimensions)}): {', '.join(config.dimensions)}")
            print(f"  - LLM Strategy: {config.llm_strategy}")
            if config.llm_round:
                print(f"  - LLM Round: {config.llm_round}")
            if config.per_model_rounds:
                print(f"  - Per-Model Rounds: {config.per_model_rounds}")
            if config.selected_models:
                print(f"  - Selected Models ({len(config.selected_models)}): {', '.join(config.selected_models[:3])}{'...' if len(config.selected_models) > 3 else ''}")
            print(f"  - Output Suffix: {config.output_suffix}")
            
            # Load data with configuration-specific filtering
            print(f"\n[Loading] Loading data for {config.name}...")
            data_loader = ConsistencyDataLoader(config=config)
            aligned_data = data_loader.load_all_data()
            
            print(f"[Success] Data loaded:")
            print(f"  - Systems: {len(aligned_data['systems'])}")
            print(f"  - Systems: {', '.join(aligned_data['systems'][:3])}{'...' if len(aligned_data['systems']) > 3 else ''}")
            
            # Create analyzer with configuration
            analyzer = ConsistencyAnalyzer(config)
            
            # Run analysis
            results = analyzer.run_full_analysis(aligned_data)
            
            # Create output directory for this analysis
            analysis_output_dir = self.base_output_dir / f"{config.output_suffix}_{self.timestamp}"
            analysis_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate visualizations
            print(f"\n[Plots] Generating visualizations for {config.name}...")
            visualizer = ConsistencyVisualizer(results)
            plot_dir = analysis_output_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            
            visualizer.create_all_plots(str(plot_dir))
            
            # Generate report
            print(f"[Report] Generating analysis report for {config.name}...")
            report_generator = ReportGenerator()
            report_file = analysis_output_dir / f"analysis_report_{config.output_suffix}_{self.timestamp}.md"
            summary_file = analysis_output_dir / f"consistency_summary_{config.output_suffix}_{self.timestamp}.csv"
            
            report_generator.generate_analysis_report(results, str(report_file), str(summary_file))
            
            # Save detailed results
            detailed_file = analysis_output_dir / f"detailed_results_{config.output_suffix}_{self.timestamp}.json"
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"[Success] Analysis complete for {config.name}")
            print(f"  Output directory: {analysis_output_dir}")
            print(f"  Plots: {plot_dir}")
            print(f"  Report: {report_file}")
            print(f"  Summary: {summary_file}")
            print(f"  Detailed: {detailed_file}")
            
            return {
                'config': config,
                'results': results,
                'output_dir': str(analysis_output_dir),
                'files': {
                    'report': str(report_file),
                    'summary': str(summary_file),
                    'detailed': str(detailed_file),
                    'plots_dir': str(plot_dir)
                }
            }
            
        except Exception as e:
            print(f"[Error] Error in analysis {config_name}: {e}")
            traceback.print_exc()
            return {
                'config_name': config_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def run_all_analyses(self, config_names: List[str] = None) -> Dict[str, Any]:
        """Run all requested analyses."""
        if config_names is None:
            # Use default configurations
            configs = get_default_analysis_configs()
            config_names = [config.output_suffix for config in configs]
        
        print("=" * 80)
        print("MULTI-CONFIGURATION CONSISTENCY ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"Running {len(config_names)} analyses:")
        for name in config_names:
            config = get_analysis_config(name)
            print(f"  - {config.name}")
        print()
        
        # Run all analyses (each loads its own data based on configuration)
        analysis_results = {}
        successful_analyses = 0
        
        for config_name in config_names:
            result = self.run_analysis(config_name)
            analysis_results[config_name] = result
            
            if 'error' not in result:
                successful_analyses += 1
        
        # Generate master summary
        master_summary = self._generate_master_summary(analysis_results)
        master_summary_file = self.base_output_dir / f"multi_analysis_summary_{self.timestamp}.json"
        
        with open(master_summary_file, 'w', encoding='utf-8') as f:
            json.dump(master_summary, f, indent=2, ensure_ascii=False)
        
        print("\\n" + "=" * 80)
        print("MULTI-ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"[Success] Completed {successful_analyses}/{len(config_names)} analyses")
        print(f"Master summary: {master_summary_file}")
        
        if successful_analyses > 0:
            print("\\nKey Results Comparison:")
            self._print_results_comparison(analysis_results)
        
        if successful_analyses < len(config_names):
            print("\\n[Warning] Failed analyses:")
            for config_name, result in analysis_results.items():
                if 'error' in result:
                    print(f"  - {config_name}: {result['error']}")
        
        return {
            'timestamp': self.timestamp,
            'base_output_dir': str(self.base_output_dir),
            'master_summary_file': str(master_summary_file),
            'successful_analyses': successful_analyses,
            'total_analyses': len(config_names),
            'analysis_results': analysis_results,
            'master_summary': master_summary
        }
    
    def _generate_master_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a master summary comparing all analyses."""
        master_summary = {
            'timestamp': self.timestamp,
            'analyses': {},
            'comparison': {
                'human_internal': {},
                'llm_internal': {},
                'human_llm_agreement': {}
            }
        }
        
        for config_name, result in analysis_results.items():
            if 'error' in result:
                master_summary['analyses'][config_name] = {'error': result['error']}
                continue
            
            config = result['config']
            results = result['results']
            
            # Store analysis summary
            master_summary['analyses'][config_name] = {
                'config': {
                    'name': config.name,
                    'description': config.description,
                    'dimensions': config.dimensions,
                    'n_dimensions': len(config.dimensions),
                    'llm_strategy': config.llm_strategy,
                    'llm_round': config.llm_round
                },
                'summary_metrics': results['summary'],
                'output_dir': result['output_dir']
            }
            
            # Add to comparison
            if 'summary' in results:
                for consistency_type in ['human_internal', 'llm_internal', 'human_llm_agreement']:
                    if consistency_type in results['summary']:
                        master_summary['comparison'][consistency_type][config_name] = {
                            'mean_pearson': results['summary'][consistency_type].get('mean_pearson'),
                            'mean_spearman': results['summary'][consistency_type].get('mean_spearman'),
                            'mean_icc': results['summary'][consistency_type].get('mean_icc'),
                            'n_dimensions': results['summary'][consistency_type].get('n_dimensions_analyzed')
                        }
        
        return master_summary
    
    def _print_results_comparison(self, analysis_results: Dict[str, Any]):
        """Print a comparison table of key results."""
        
        print(f"{'Analysis':<25} {'Dimensions':<12} {'LLM Strategy':<15} {'Human-LLM ICC':<15}")
        print("-" * 70)
        
        for config_name, result in analysis_results.items():
            if 'error' in result:
                continue
            
            config = result['config']
            results = result['results']
            
            # Extract key metrics
            n_dims = len(config.dimensions)
            llm_strategy = f"{config.llm_strategy}" + (f" (R{config.llm_round})" if config.llm_round else "")
            
            human_llm_icc = "N/A"
            if 'summary' in results and 'human_llm_agreement' in results['summary']:
                icc_val = results['summary']['human_llm_agreement'].get('mean_icc')
                if icc_val is not None and not pd.isna(icc_val):
                    human_llm_icc = f"{icc_val:.3f}"
            
            print(f"{config_name:<25} {n_dims:<12} {llm_strategy:<15} {human_llm_icc:<15}")

def main():
    """Main function to run multi-configuration analysis."""
    
    # 配置您想要运行的分析：
    config_names = [

        'reduced_selected_models',


    ]
    
    try:
        runner = MultiConfigAnalysisRunner()
        final_results = runner.run_all_analyses(config_names)
        
        print("\\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Master summary saved to: {final_results['master_summary_file']}")
        print(f"All results in: {final_results['base_output_dir']}")
        
        return final_results
        
    except Exception as e:
        print(f"\\n[Error] Pipeline failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import pandas as pd  # For isna check
    main()