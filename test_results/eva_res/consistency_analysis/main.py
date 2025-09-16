#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for comprehensive consistency analysis.
Runs complete analysis pipeline including data loading, consistency calculation, 
visualization, and report generation.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Dict, Any

from data_loader import ConsistencyDataLoader
from consistency_calculator import ConsistencyAnalyzer
from visualizer import ConsistencyVisualizer

class ConsistencyAnalysisRunner:
    """Main runner for complete consistency analysis."""
    
    def __init__(self, base_path: str = None, output_dir: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.loader = ConsistencyDataLoader(self.base_path)
        self.analyzer = ConsistencyAnalyzer()
        
        self.results = None
        self.aligned_data = None
        
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete consistency analysis pipeline."""
        print("="*60)
        print("CONSISTENCY ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Load and align data
        print("\n1. Loading and aligning evaluation data...")
        self.aligned_data = self.loader.load_all_data()
        
        if not self.aligned_data['systems']:
            raise ValueError("No aligned data found. Cannot proceed with analysis.")
        
        print(f"   + Successfully aligned data for {len(self.aligned_data['systems'])} systems")
        print(f"   + Analyzing {len(self.aligned_data['dimensions'])} dimensions")
        
        # Step 2: Run consistency calculations
        print("\n2. Computing consistency metrics...")
        self.results = self.analyzer.run_full_analysis(self.aligned_data)
        print("   + Human internal consistency computed")
        print("   + LLM internal consistency computed") 
        print("   + Human-LLM agreement computed")
        
        # Step 3: Generate visualizations
        print("\n3. Generating visualizations...")
        visualizer = ConsistencyVisualizer(self.results)
        
        # Create output subdirectory for plots
        plots_dir = self.output_dir / f"plots_{self.timestamp}"
        plots_dir.mkdir(exist_ok=True)
        
        # Main overview plot
        overview_path = plots_dir / "consistency_overview.png"
        main_fig = visualizer.plot_consistency_comparison(str(overview_path))
        
        # Individual comparison plots
        individual_figs = visualizer.plot_individual_consistency_comparisons(str(plots_dir))
        
        # Summary table
        table_path = self.output_dir / f"consistency_summary_{self.timestamp}.csv"
        summary_table = visualizer.create_summary_table(str(table_path))
        
        print(f"   + Created {len(individual_figs) + 1} visualization plots")
        print(f"   + Generated summary table: {table_path}")
        
        # Step 4: Save detailed results
        print("\n4. Saving detailed results...")
        results_path = self.output_dir / f"detailed_results_{self.timestamp}.json"
        self.save_results(results_path)
        print(f"   + Detailed results saved: {results_path}")
        
        # Step 5: Generate analysis report
        print("\n5. Generating analysis report...")
        report_path = self.output_dir / f"analysis_report_{self.timestamp}.md"
        self.generate_report(report_path, summary_table)
        print(f"   + Analysis report saved: {report_path}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'results': self.results,
            'aligned_data': self.aligned_data,
            'output_files': {
                'detailed_results': str(results_path),
                'summary_table': str(table_path),
                'analysis_report': str(report_path),
                'plots_directory': str(plots_dir)
            }
        }
    
    def save_results(self, filepath: Path):
        """Save detailed results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            else:
                return obj
        
        results_serializable = convert_numpy(self.results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    def generate_report(self, filepath: Path, summary_table: pd.DataFrame):
        """Generate comprehensive analysis report."""
        
        report_content = f"""# Consistency Analysis Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents a comprehensive analysis of the consistency between human and LLM evaluation scores across multiple AI systems and evaluation dimensions.

### Key Findings

"""
        
        # Calculate overall statistics
        if self.results and 'summary' in self.results:
            summary = self.results['summary']
            
            report_content += f"""
**Dataset Overview:**
- Systems analyzed: {summary['n_systems']}
- Dimensions analyzed: {len(summary['dimensions_analyzed'])}
- Systems: {', '.join(summary['systems'])}

**Overall Consistency Metrics:**

| Consistency Type | Mean Pearson | Mean Spearman | Mean ICC |
|-----------------|--------------|---------------|----------|
"""
            
            for consistency_type in ['human_internal', 'llm_internal', 'human_llm_agreement']:
                if consistency_type in summary:
                    data = summary[consistency_type]
                    report_content += f"| {consistency_type.replace('_', ' ').title()} | {data['mean_pearson']:.3f} | {data['mean_spearman']:.3f} | {data['mean_icc']:.3f} |\n"
        
        report_content += """

## Methodology

### Consistency Metrics

1. **Pearson Correlation Coefficient**: Measures linear relationship between scores
2. **Spearman Rank Correlation**: Measures monotonic relationship between scores  
3. **Intraclass Correlation Coefficient (ICC)**: Measures absolute agreement between raters

### Analysis Types

1. **Human Internal Consistency**: Agreement between 3 human raters
2. **LLM Internal Consistency**: Agreement between 5 LLM evaluation rounds
3. **Human-LLM Agreement**: Agreement between average human and LLM scores

## Detailed Results by Dimension

"""
        
        # Add dimension-wise results
        if not summary_table.empty:
            # Group by dimension
            for dimension in summary_table['Dimension'].unique():
                report_content += f"\n### {dimension}\n\n"
                
                dim_data = summary_table[summary_table['Dimension'] == dimension]
                
                report_content += "| Consistency Type | Pearson r (p) | Spearman r (p) | ICC [95% CI] (p) |\n"
                report_content += "|-----------------|---------------|----------------|------------------|\n"
                
                for _, row in dim_data.iterrows():
                    pearson_str = f"{row['Pearson_r']} ({row['Pearson_p']})"
                    spearman_str = f"{row['Spearman_r']} ({row['Spearman_p']})" 
                    icc_str = f"{row['ICC']} [{row['ICC_CI_lower']}, {row['ICC_CI_upper']}] ({row['ICC_p']})"
                    
                    report_content += f"| {row['Consistency_Type']} | {pearson_str} | {spearman_str} | {icc_str} |\n"
        
        report_content += """

## Statistical Interpretation

### Correlation Strength Guidelines
- **0.90-1.00**: Very high correlation
- **0.70-0.89**: High correlation  
- **0.50-0.69**: Moderate correlation
- **0.30-0.49**: Low correlation
- **0.00-0.29**: Negligible correlation

### P-value Interpretation
- **p < 0.001**: Very strong evidence against null hypothesis
- **p < 0.01**: Strong evidence against null hypothesis
- **p < 0.05**: Moderate evidence against null hypothesis
- **p ≥ 0.05**: Insufficient evidence against null hypothesis

### ICC Interpretation
- **> 0.75**: Excellent reliability
- **0.60-0.74**: Good reliability
- **0.40-0.59**: Fair reliability
- **< 0.40**: Poor reliability

## Key Insights

"""
        
        # Add insights based on results
        if self.results and 'summary' in self.results:
            summary = self.results['summary']
            
            insights = []
            
            # Compare consistency types
            for consistency_type in ['human_internal', 'llm_internal', 'human_llm_agreement']:
                if consistency_type in summary:
                    mean_pearson = summary[consistency_type]['mean_pearson']
                    mean_icc = summary[consistency_type]['mean_icc']
                    
                    if not np.isnan(mean_pearson):
                        if mean_pearson >= 0.7:
                            strength = "high"
                        elif mean_pearson >= 0.5:
                            strength = "moderate"
                        else:
                            strength = "low"
                        
                        insights.append(f"- **{consistency_type.replace('_', ' ').title()}** shows {strength} consistency (Pearson r = {mean_pearson:.3f})")
            
            for insight in insights:
                report_content += insight + "\n"
        
        report_content += """

## Recommendations

1. **For Human Evaluation**: Consider additional training if human internal consistency is low
2. **For LLM Evaluation**: Consider prompt engineering improvements if LLM internal consistency is low  
3. **For System Comparison**: Use the consistency type with highest reliability for comparative analysis
4. **For Future Studies**: Focus on dimensions showing highest agreement between human and LLM evaluators

## Technical Notes

- All confidence intervals are calculated at 95% confidence level
- ICC calculations use two-way random effects model for absolute agreement
- Missing values were excluded from correlation calculations
- Statistical significance testing used two-tailed tests

---
*This report was automatically generated by the Consistency Analysis Pipeline.*
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)

def main():
    """Main execution function."""
    print("Starting Consistency Analysis Pipeline...")
    
    # Initialize runner
    runner = ConsistencyAnalysisRunner()
    
    try:
        # Run complete analysis
        output = runner.run_complete_analysis()
        
        print("\nOutput files generated:")
        for file_type, file_path in output['output_files'].items():
            print(f"  - {file_type}: {file_path}")
        
        return output
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()