#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for consistency analysis results.
Creates comparison plots for human internal, LLM internal, and human-LLM agreement consistency.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any
import warnings

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ConsistencyVisualizer:
    """Create visualizations for consistency analysis results."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.dimensions = ['正确性', '逻辑性', '清晰度', '完备性', '理论深度', '论述严谨性与信息密度']
        
    def prepare_comparison_data(self) -> pd.DataFrame:
        """Prepare data for comparison plots."""
        data_rows = []
        
        # Map English names for cleaner plots
        dimension_map = {
            '正确性': 'Correctness',
            '逻辑性': 'Logic',
            '清晰度': 'Clarity', 
            '完备性': 'Completeness',
            '理论深度': 'Theoretical Depth',
            '论述严谨性与信息密度': 'Rigor & Information Density'
        }
        
        consistency_types = [
            ('human_internal', 'Human Internal'),
            ('llm_internal', 'LLM Internal'),
            ('human_llm_agreement', 'Human-LLM Agreement')
        ]
        
        for dim in self.dimensions:
            if dim in self.results['human_internal']:
                dim_english = dimension_map.get(dim, dim)
                
                for consistency_key, consistency_name in consistency_types:
                    if (dim in self.results[consistency_key] and 
                        self.results[consistency_key][dim] is not None):
                        
                        result_data = self.results[consistency_key][dim]
                        
                        # Extract correlation values
                        if consistency_key == 'human_llm_agreement':
                            pearson_val = result_data['pearson'][0]
                            spearman_val = result_data['spearman'][0]
                        else:
                            pearson_val = result_data['pearson']['mean_correlation']
                            spearman_val = result_data['spearman']['mean_correlation']
                        
                        icc_val = result_data['icc']['value']
                        
                        # Add rows for each metric type
                        for metric_name, metric_val in [
                            ('Pearson', pearson_val),
                            ('Spearman', spearman_val),
                            ('ICC', icc_val)
                        ]:
                            if not np.isnan(metric_val):
                                data_rows.append({
                                    'Dimension': dim_english,
                                    'Consistency_Type': consistency_name,
                                    'Metric': metric_name,
                                    'Value': metric_val
                                })
        
        return pd.DataFrame(data_rows)
    
    def plot_consistency_comparison(self, save_path: str = None) -> plt.Figure:
        """Create comprehensive consistency comparison plot."""
        data = self.prepare_comparison_data()
        
        if data.empty:
            warnings.warn("No data available for plotting")
            return None
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Consistency Analysis: Correlation Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall comparison by metric type
        ax1 = axes[0, 0]
        
        # Pivot data for plotting
        pivot_data = data.pivot_table(
            values='Value', 
            index=['Consistency_Type', 'Dimension'], 
            columns='Metric'
        ).reset_index()
        
        # Create grouped bar plot
        metrics = ['Pearson', 'Spearman', 'ICC']
        consistency_types = ['Human Internal', 'LLM Internal', 'Human-LLM Agreement']
        
        x = np.arange(len(consistency_types))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            if metric in pivot_data.columns:
                metric_means = [
                    pivot_data[pivot_data['Consistency_Type'] == ct][metric].mean() 
                    for ct in consistency_types
                ]
                ax1.bar(x + i * width, metric_means, width, label=metric, alpha=0.8)
        
        ax1.set_xlabel('Consistency Type')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.set_title('Average Correlation by Consistency Type')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(consistency_types, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add reference line at y=x diagonal equivalent (0.8 as high consistency)
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High Consistency (0.8)')
        ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate Consistency (0.6)')
        
        # Plot 2: Dimension-wise comparison
        ax2 = axes[0, 1]
        
        # Heatmap of correlations by dimension
        heatmap_data = data.pivot_table(
            values='Value',
            index='Dimension', 
            columns='Consistency_Type',
            aggfunc='mean'
        )
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax2, cbar_kws={'label': 'Mean Correlation'})
            ax2.set_title('Mean Correlation by Dimension')
            ax2.set_xlabel('Consistency Type')
            ax2.set_ylabel('Dimension')
        
        # Plot 3: Metric comparison scatter plots
        ax3 = axes[1, 0]
        
        # Compare Pearson vs Spearman
        pearson_data = data[data['Metric'] == 'Pearson']
        spearman_data = data[data['Metric'] == 'Spearman']
        
        if not pearson_data.empty and not spearman_data.empty:
            # Merge data for comparison
            comparison_data = pd.merge(
                pearson_data[['Dimension', 'Consistency_Type', 'Value']], 
                spearman_data[['Dimension', 'Consistency_Type', 'Value']],
                on=['Dimension', 'Consistency_Type'],
                suffixes=('_Pearson', '_Spearman')
            )
            
            colors = {'Human Internal': 'blue', 'LLM Internal': 'green', 'Human-LLM Agreement': 'red'}
            
            for consistency_type in comparison_data['Consistency_Type'].unique():
                subset = comparison_data[comparison_data['Consistency_Type'] == consistency_type]
                ax3.scatter(subset['Value_Pearson'], subset['Value_Spearman'], 
                           label=consistency_type, alpha=0.7, s=60,
                           color=colors.get(consistency_type, 'gray'))
            
            # Add diagonal line
            lims = [0, 1]
            ax3.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
            ax3.set_xlim(lims)
            ax3.set_ylim(lims)
            
            ax3.set_xlabel('Pearson Correlation')
            ax3.set_ylabel('Spearman Correlation')
            ax3.set_title('Pearson vs Spearman Correlation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: ICC vs Correlation comparison
        ax4 = axes[1, 1]
        
        icc_data = data[data['Metric'] == 'ICC']
        pearson_data = data[data['Metric'] == 'Pearson']
        
        if not icc_data.empty and not pearson_data.empty:
            comparison_icc = pd.merge(
                icc_data[['Dimension', 'Consistency_Type', 'Value']], 
                pearson_data[['Dimension', 'Consistency_Type', 'Value']],
                on=['Dimension', 'Consistency_Type'],
                suffixes=('_ICC', '_Pearson')
            )
            
            for consistency_type in comparison_icc['Consistency_Type'].unique():
                subset = comparison_icc[comparison_icc['Consistency_Type'] == consistency_type]
                ax4.scatter(subset['Value_Pearson'], subset['Value_ICC'], 
                           label=consistency_type, alpha=0.7, s=60,
                           color=colors.get(consistency_type, 'gray'))
            
            # Add diagonal line
            lims = [0, 1]
            ax4.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
            ax4.set_xlim(lims)
            ax4.set_ylim(lims)
            
            ax4.set_xlabel('Pearson Correlation')
            ax4.set_ylabel('ICC')
            ax4.set_title('Pearson Correlation vs ICC')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def plot_individual_consistency_comparisons(self, save_dir: str = None) -> Dict[str, plt.Figure]:
        """Create individual plots for each consistency type comparison."""
        figures = {}
        
        # Prepare data
        data = self.prepare_comparison_data()
        if data.empty:
            warnings.warn("No data available for plotting")
            return figures
        
        consistency_types = ['Human Internal', 'LLM Internal', 'Human-LLM Agreement']
        colors = {'Pearson': '#1f77b4', 'Spearman': '#ff7f0e', 'ICC': '#2ca02c'}
        
        for i, (consistency_type, next_type) in enumerate([
            ('Human Internal', 'LLM Internal'),
            ('LLM Internal', 'Human-LLM Agreement'), 
            ('Human Internal', 'Human-LLM Agreement')
        ]):
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Filter data for these two consistency types
            subset = data[data['Consistency_Type'].isin([consistency_type, next_type])]
            
            if subset.empty:
                continue
                
            # Create pivot table
            pivot_subset = subset.pivot_table(
                values='Value',
                index=['Dimension', 'Metric'],
                columns='Consistency_Type'
            ).reset_index()
            
            # Plot scatter for each metric with fit lines
            for metric in ['Pearson', 'Spearman', 'ICC']:
                metric_data = pivot_subset[pivot_subset['Metric'] == metric]
                if not metric_data.empty and consistency_type in pivot_subset.columns and next_type in pivot_subset.columns:
                    # 获取数据点
                    x_data = metric_data[consistency_type].dropna().values
                    y_data = metric_data[next_type].dropna().values
                    
                    if len(x_data) > 0 and len(y_data) > 0 and len(x_data) == len(y_data):
                        color = colors.get(metric, 'gray')
                        
                        # 绘制散点
                        ax.scatter(x_data, y_data, label=metric, alpha=0.7, s=80, color=color)
                        
                        # 添加拟合线（如果有足够的数据点）
                        if len(x_data) >= 2:
                            try:
                                # 计算线性拟合
                                z = np.polyfit(x_data, y_data, 1)
                                p = np.poly1d(z)
                                
                                # 生成拟合线的x值范围（扩展到合理范围）
                                x_min, x_max = max(0, min(x_data) - 0.1), min(1, max(x_data) + 0.1)
                                x_fit = np.linspace(x_min, x_max, 100)
                                y_fit = p(x_fit)
                                
                                # 限制拟合线在[0,1]范围内
                                mask = (y_fit >= 0) & (y_fit <= 1) & (x_fit >= 0) & (x_fit <= 1)
                                if mask.sum() > 0:
                                    ax.plot(x_fit[mask], y_fit[mask], '--', color=color, alpha=0.8, 
                                           linewidth=2, label=f'{metric} 拟合线')
                            except (np.RankWarning, np.linalg.LinAlgError):
                                pass  # 如果拟合失败就跳过拟合线
            
            # Add diagonal reference line (perfect agreement)
            lims = [0, 1]
            ax.plot(lims, lims, 'k--', alpha=0.6, zorder=0, linewidth=2, label='完美一致性 (y=x)')
            
            # Formatting
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f'{consistency_type} Correlation')
            ax.set_ylabel(f'{next_type} Correlation')
            ax.set_title(f'Consistency Comparison: {consistency_type} vs {next_type}', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient of the comparison
            if consistency_type in pivot_subset.columns and next_type in pivot_subset.columns:
                valid_data = pivot_subset.dropna(subset=[consistency_type, next_type])
                if len(valid_data) > 2:
                    from scipy.stats import pearsonr
                    r, p = pearsonr(valid_data[consistency_type], valid_data[next_type])
                    ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p:.3f}', 
                           transform=ax.transAxes, fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            figures[f'{consistency_type}_vs_{next_type}'] = fig
            
            if save_dir:
                filename = f'consistency_comparison_{consistency_type.lower().replace(" ", "_")}_vs_{next_type.lower().replace(" ", "_")}.png'
                filepath = f"{save_dir}/{filename}"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {filepath}")
        
        return figures
    
    def create_summary_table(self, save_path: str = None) -> pd.DataFrame:
        """Create a summary table of all consistency metrics."""
        summary_data = []
        
        dimension_map = {
            '正确性': 'Correctness',
            '逻辑性': 'Logic',
            '清晰度': 'Clarity', 
            '完备性': 'Completeness',
            '理论深度': 'Theoretical Depth',
            '论述严谨性与信息密度': 'Rigor & Information Density'
        }
        
        consistency_types = [
            ('human_internal', 'Human Internal'),
            ('llm_internal', 'LLM Internal'),
            ('human_llm_agreement', 'Human-LLM Agreement')
        ]
        
        for dim in self.dimensions:
            dim_english = dimension_map.get(dim, dim)
            
            for consistency_key, consistency_name in consistency_types:
                if (dim in self.results[consistency_key] and 
                    self.results[consistency_key][dim] is not None):
                    
                    result_data = self.results[consistency_key][dim]
                    
                    # Extract values and confidence intervals
                    if consistency_key == 'human_llm_agreement':
                        pearson_val, pearson_p = result_data['pearson']
                        spearman_val, spearman_p = result_data['spearman']
                    else:
                        pearson_val = result_data['pearson']['mean_correlation']
                        pearson_p = result_data['pearson']['mean_p_value']
                        spearman_val = result_data['spearman']['mean_correlation']
                        spearman_p = result_data['spearman']['mean_p_value']
                    
                    icc_val = result_data['icc']['value']
                    icc_ci = result_data['icc']['confidence_interval_95']
                    icc_p = result_data['icc']['p_value']
                    
                    summary_data.append({
                        'Dimension': dim_english,
                        'Consistency_Type': consistency_name,
                        'Pearson_r': f"{pearson_val:.3f}" if not np.isnan(pearson_val) else "N/A",
                        'Pearson_p': f"{pearson_p:.3f}" if not np.isnan(pearson_p) else "N/A",
                        'Spearman_r': f"{spearman_val:.3f}" if not np.isnan(spearman_val) else "N/A",  
                        'Spearman_p': f"{spearman_p:.3f}" if not np.isnan(spearman_p) else "N/A",
                        'ICC': f"{icc_val:.3f}" if not np.isnan(icc_val) else "N/A",
                        'ICC_CI_lower': f"{icc_ci[0]:.3f}" if not np.isnan(icc_ci[0]) else "N/A",
                        'ICC_CI_upper': f"{icc_ci[1]:.3f}" if not np.isnan(icc_ci[1]) else "N/A",
                        'ICC_p': f"{icc_p:.3f}" if not np.isnan(icc_p) else "N/A"
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        if save_path:
            summary_df.to_csv(save_path, index=False)
            print(f"Summary table saved to: {save_path}")
        
        return summary_df
    
    def create_all_plots(self, save_dir: str) -> Dict[str, str]:
        """Create all visualization plots and save to directory."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        plot_files = {}
        
        try:
            # Main comparison plot
            fig1 = self.plot_consistency_comparison()
            overview_path = os.path.join(save_dir, "consistency_overview.png")
            fig1.savefig(overview_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            plot_files['overview'] = overview_path
            
            # Individual comparison plots
            individual_plots = self.plot_individual_consistency_comparisons(save_dir)
            for plot_name, fig in individual_plots.items():
                plot_path = os.path.join(save_dir, f"{plot_name}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_files[plot_name] = plot_path
            
            print(f"Generated {len(plot_files)} visualization plots")
            return plot_files
            
        except Exception as e:
            print(f"Warning: Could not generate some plots: {e}")
            return plot_files


if __name__ == "__main__":
    # Test the visualizer
    from data_loader import ConsistencyDataLoader
    from consistency_calculator import ConsistencyAnalyzer
    
    # Load data and run analysis
    loader = ConsistencyDataLoader()
    aligned_data = loader.load_all_data()
    
    analyzer = ConsistencyAnalyzer()
    results = analyzer.run_full_analysis(aligned_data)
    
    # Create visualizations
    visualizer = ConsistencyVisualizer(results)
    
    # Create plots
    main_fig = visualizer.plot_consistency_comparison("consistency_comparison_overview.png")
    individual_figs = visualizer.plot_individual_consistency_comparisons("./")
    summary_table = visualizer.create_summary_table("consistency_summary_table.csv")
    
    print("\n=== Visualization Summary ===")
    print(f"Created {len(individual_figs) + 1} plots")
    print(f"Summary table shape: {summary_table.shape}")
    
    plt.show()