#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configurable consistency analysis calculator for human and LLM evaluation scores.
Computes Pearson correlation, Spearman correlation, and Intraclass Correlation Coefficient (ICC).
Supports different dimension selections and LLM evaluation strategies.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import warnings
from typing import Dict, List, Tuple, Any
from itertools import combinations
import math
from analysis_config import AnalysisConfig

class ConsistencyAnalyzer:
    """Calculate consistency metrics for evaluation scores with configurable analysis."""
    
    def __init__(self, config: AnalysisConfig):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: AnalysisConfig specifying dimensions and LLM strategy
        """
        self.config = config
        self.config.validate()
        
        self.results = {
            'human_internal': {},
            'llm_internal': {},
            'human_llm_agreement': {}
        }
    
    def calculate_icc(self, scores: np.ndarray, icc_type: str = 'ICC(2,1)') -> Tuple[float, float, Tuple[float, float]]:
        """
        Calculate Intraclass Correlation Coefficient (ICC).
        
        Args:
            scores: 2D array where rows are subjects and columns are raters
            icc_type: Type of ICC to calculate
        
        Returns:
            (ICC value, F-statistic, (lower_CI, upper_CI))
        """
        if scores.shape[0] < 2 or scores.shape[1] < 2:
            return np.nan, np.nan, (np.nan, np.nan)
        
        # Remove rows with NaN values
        valid_rows = ~np.isnan(scores).any(axis=1)
        if valid_rows.sum() < 2:
            return np.nan, np.nan, (np.nan, np.nan)
        
        scores = scores[valid_rows]
        n, k = scores.shape  # n subjects, k raters
        
        # Calculate mean squares
        subject_means = np.mean(scores, axis=1)
        grand_mean = np.mean(scores)
        
        # Between subjects sum of squares
        BSS = k * np.sum((subject_means - grand_mean) ** 2)
        
        # Within subjects sum of squares  
        WSS = np.sum((scores - subject_means.reshape(-1, 1)) ** 2)
        
        # Mean squares
        MSB = BSS / (n - 1)  # Mean square between
        MSW = WSS / (n * (k - 1))  # Mean square within
        
        # ICC calculation based on type
        if icc_type == 'ICC(2,1)':  # Two-way random, single measures, absolute agreement
            if MSW == 0:
                icc = 1.0
            else:
                icc = (MSB - MSW) / (MSB + (k - 1) * MSW)
        else:
            # Default to ICC(2,1)
            if MSW == 0:
                icc = 1.0
            else:
                icc = (MSB - MSW) / (MSB + (k - 1) * MSW)
        
        # F-statistic
        if MSW == 0:
            f_stat = float('inf')
        else:
            f_stat = MSB / MSW
        
        # 95% Confidence interval
        try:
            f_critical_lower = stats.f.ppf(0.025, n - 1, n * (k - 1))
            f_critical_upper = stats.f.ppf(0.975, n - 1, n * (k - 1))
            
            if f_stat > f_critical_lower:
                lower_ci = (f_stat / f_critical_upper - 1) / (f_stat / f_critical_upper + (k - 1))
            else:
                lower_ci = 0.0
                
            if f_stat > f_critical_upper:
                upper_ci = (f_stat / f_critical_lower - 1) / (f_stat / f_critical_lower + (k - 1))
            else:
                upper_ci = (f_stat - 1) / (f_stat + (k - 1))
            
            # Ensure confidence interval is valid
            lower_ci = max(0.0, min(lower_ci, 1.0))
            upper_ci = max(lower_ci, min(upper_ci, 1.0))
            
        except (ZeroDivisionError, ValueError):
            lower_ci, upper_ci = np.nan, np.nan
        
        return max(0.0, min(icc, 1.0)), f_stat, (lower_ci, upper_ci)
    
    def calculate_correlations(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Calculate Pearson and Spearman correlations.
        
        Args:
            x, y: Arrays of scores to correlate
            
        Returns:
            Dictionary with correlation coefficients and p-values
        """
        # Remove pairs with NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if valid_mask.sum() < 3:  # Need at least 3 points for correlation
            return {
                'pearson': (np.nan, np.nan),
                'spearman': (np.nan, np.nan)
            }
        
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        # Pearson correlation
        try:
            pearson_r, pearson_p = pearsonr(x_clean, y_clean)
        except (ValueError, RuntimeWarning):
            pearson_r, pearson_p = np.nan, np.nan
        
        # Spearman correlation  
        try:
            spearman_r, spearman_p = spearmanr(x_clean, y_clean)
        except (ValueError, RuntimeWarning):
            spearman_r, spearman_p = np.nan, np.nan
        
        return {
            'pearson': (pearson_r, pearson_p),
            'spearman': (spearman_r, spearman_p)
        }
    
    def calculate_human_internal_consistency(self, human_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate internal consistency among human raters.
        
        Args:
            human_scores: {system: {question: {dimension: [score1, score2, score3]}}}
        """
        print("Calculating human internal consistency...")
        
        results = {}
        
        for dimension_name in self.config.dimensions:
            # Collect all scores for this dimension across all systems and questions
            all_scores = []  # List of (score1, score2, score3) tuples
            
            for system in human_scores:
                for question in human_scores[system]:
                    if dimension_name in human_scores[system][question]:
                        scores = human_scores[system][question][dimension_name]
                        if len(scores) >= 3:  # Ensure we have at least 3 raters
                            all_scores.append(scores[:3])
            
            if not all_scores:
                continue
                
            # Convert to numpy array (subjects x raters)
            scores_array = np.array(all_scores)
            
            # Calculate ICC
            icc_value, f_stat, icc_ci = self.calculate_icc(scores_array)
            
            # Calculate pairwise correlations between raters
            correlations = {}
            for i, j in combinations(range(3), 2):
                rater_pair = f"rater_{i+1}_vs_rater_{j+1}"
                corr_results = self.calculate_correlations(scores_array[:, i], scores_array[:, j])
                correlations[rater_pair] = corr_results
            
            # Calculate average correlations
            pearson_correlations = [corr['pearson'][0] for corr in correlations.values() 
                                  if not np.isnan(corr['pearson'][0])]
            spearman_correlations = [corr['spearman'][0] for corr in correlations.values() 
                                   if not np.isnan(corr['spearman'][0])]
            pearson_pvalues = [corr['pearson'][1] for corr in correlations.values() 
                              if not np.isnan(corr['pearson'][1])]
            spearman_pvalues = [corr['spearman'][1] for corr in correlations.values() 
                               if not np.isnan(corr['spearman'][1])]
            
            results[dimension_name] = {
                'n_subjects': len(all_scores),
                'icc': {
                    'value': icc_value,
                    'f_statistic': f_stat,
                    'confidence_interval_95': icc_ci,
                    'p_value': 1 - stats.f.cdf(f_stat, len(all_scores)-1, len(all_scores)*2) if not np.isnan(f_stat) else np.nan
                },
                'pearson': {
                    'mean_correlation': np.mean(pearson_correlations) if pearson_correlations else np.nan,
                    'mean_p_value': np.mean(pearson_pvalues) if pearson_pvalues else np.nan,
                    'pairwise_correlations': [corr['pearson'] for corr in correlations.values()]
                },
                'spearman': {
                    'mean_correlation': np.mean(spearman_correlations) if spearman_correlations else np.nan,
                    'mean_p_value': np.mean(spearman_pvalues) if spearman_pvalues else np.nan,
                    'pairwise_correlations': [corr['spearman'] for corr in correlations.values()]
                },
                'detailed_pairwise': correlations
            }
        
        self.results['human_internal'] = results
        return results
    
    def calculate_llm_internal_consistency(self, llm_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate internal consistency among LLM evaluation rounds.
        
        Args:
            llm_scores: {system: {question: {dimension: [score1, score2, score3, score4, score5]}}}
        """
        print("Calculating LLM internal consistency...")
        
        results = {}
        
        for dimension_name in self.config.dimensions:
            # Collect all scores for this dimension across all systems and questions
            all_scores = []  # List of (score1, score2, score3, score4, score5) tuples
            
            for system in llm_scores:
                for question in llm_scores[system]:
                    if dimension_name in llm_scores[system][question]:
                        scores = llm_scores[system][question][dimension_name]
                        if len(scores) >= 5:  # Ensure we have all 5 evaluation rounds
                            all_scores.append(scores[:5])
            
            if not all_scores:
                continue
                
            # Convert to numpy array (subjects x rounds)
            scores_array = np.array(all_scores)
            
            # Calculate ICC
            icc_value, f_stat, icc_ci = self.calculate_icc(scores_array)
            
            # Calculate pairwise correlations between evaluation rounds  
            correlations = {}
            num_rounds = 5
            for i, j in combinations(range(num_rounds), 2):
                round_pair = f"round_{i+1}_vs_round_{j+1}"
                corr_results = self.calculate_correlations(scores_array[:, i], scores_array[:, j])
                correlations[round_pair] = corr_results
            
            # Calculate average correlations
            pearson_correlations = [corr['pearson'][0] for corr in correlations.values() 
                                  if not np.isnan(corr['pearson'][0])]
            spearman_correlations = [corr['spearman'][0] for corr in correlations.values() 
                                   if not np.isnan(corr['spearman'][0])]
            pearson_pvalues = [corr['pearson'][1] for corr in correlations.values() 
                              if not np.isnan(corr['pearson'][1])]
            spearman_pvalues = [corr['spearman'][1] for corr in correlations.values() 
                               if not np.isnan(corr['spearman'][1])]
            
            results[dimension_name] = {
                'n_subjects': len(all_scores),
                'icc': {
                    'value': icc_value,
                    'f_statistic': f_stat,
                    'confidence_interval_95': icc_ci,
                    'p_value': 1 - stats.f.cdf(f_stat, len(all_scores)-1, len(all_scores)*4) if not np.isnan(f_stat) else np.nan
                },
                'pearson': {
                    'mean_correlation': np.mean(pearson_correlations) if pearson_correlations else np.nan,
                    'mean_p_value': np.mean(pearson_pvalues) if pearson_pvalues else np.nan,
                    'pairwise_correlations': [corr['pearson'] for corr in correlations.values()]
                },
                'spearman': {
                    'mean_correlation': np.mean(spearman_correlations) if spearman_correlations else np.nan,
                    'mean_p_value': np.mean(spearman_pvalues) if spearman_pvalues else np.nan,
                    'pairwise_correlations': [corr['spearman'] for corr in correlations.values()]
                },
                'detailed_pairwise': correlations
            }
        
        self.results['llm_internal'] = results
        return results
    
    def calculate_human_llm_agreement(self, human_scores: Dict[str, Any], llm_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate agreement between human and LLM evaluation scores.
        
        Args:
            human_scores: {system: {question: {dimension: [score1, score2, score3]}}}
            llm_scores: {system: {question: {dimension: [score1, score2, score3, score4, score5]}}}
        """
        print("Calculating human-LLM agreement...")
        
        results = {}
        
        for dimension_name in self.config.dimensions:
            human_means = []
            llm_values = []
            
            # Collect scores for each system-question pair
            for system in human_scores:
                if system in llm_scores:
                    for question in human_scores[system]:
                        if (question in llm_scores[system] and 
                            dimension_name in human_scores[system][question] and
                            dimension_name in llm_scores[system][question]):
                            
                            human_score_list = human_scores[system][question][dimension_name]
                            llm_score_list = llm_scores[system][question][dimension_name]
                            
                            if len(human_score_list) >= 3 and len(llm_score_list) >= 3:
                                human_mean = np.mean(human_score_list[:3])
                                
                                # Handle LLM score based on strategy
                                if self.config.llm_strategy == 'average':
                                    # Special handling for llasmol which has 10 scores (top1+top5)
                                    if system == "llasmol" and len(llm_score_list) >= 10:
                                        llm_value = np.mean(llm_score_list[:10])  # Use all 10 scores
                                    else:
                                        llm_value = np.mean(llm_score_list[:5])  # Standard 5 scores
                                elif self.config.llm_strategy == 'specific_round':
                                    round_idx = self.config.llm_round - 1  # Convert to 0-indexed
                                    if round_idx < len(llm_score_list):
                                        llm_value = llm_score_list[round_idx]
                                    else:
                                        continue  # Skip if round not available
                                elif self.config.llm_strategy == 'per_model_round':
                                    # Use per-model round specification
                                    if system in self.config.per_model_rounds:
                                        round_idx = self.config.per_model_rounds[system] - 1  # Convert to 0-indexed
                                        if round_idx < len(llm_score_list):
                                            llm_value = llm_score_list[round_idx]
                                        else:
                                            continue  # Skip if specified round not available
                                    else:
                                        # Fallback to average if model not in per_model_rounds
                                        if system == "llasmol" and len(llm_score_list) >= 10:
                                            llm_value = np.mean(llm_score_list[:10])  # Use all 10 scores
                                        else:
                                            llm_value = np.mean(llm_score_list[:5])  # Standard 5 scores
                                else:
                                    raise ValueError(f"Unknown LLM strategy: {self.config.llm_strategy}")
                                
                                human_means.append(human_mean)
                                llm_values.append(llm_value)
            
            if len(human_means) < 10:  # Need sufficient data points
                continue
            
            human_array = np.array(human_means)
            llm_array = np.array(llm_values)
            
            # Calculate correlations
            correlations = self.calculate_correlations(human_array, llm_array)
            
            # Calculate ICC treating human and LLM as two "raters"
            combined_scores = np.column_stack([human_array, llm_array])
            icc_value, f_stat, icc_ci = self.calculate_icc(combined_scores)
            
            results[dimension_name] = {
                'n_pairs': len(human_means),
                'human_mean': np.mean(human_array),
                'human_std': np.std(human_array),
                'llm_mean': np.mean(llm_array),
                'llm_std': np.std(llm_array),
                'pearson': correlations['pearson'],
                'spearman': correlations['spearman'],
                'icc': {
                    'value': icc_value,
                    'f_statistic': f_stat,
                    'confidence_interval_95': icc_ci,
                    'p_value': 1 - stats.f.cdf(f_stat, len(human_means)-1, len(human_means)) if not np.isnan(f_stat) else np.nan
                }
            }
        
        self.results['human_llm_agreement'] = results
        return results
    
    def run_full_analysis(self, aligned_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run complete consistency analysis with configuration.
        
        Args:
            aligned_data: Output from DataLoader.align_data()
        """
        print(f"Running consistency analysis: {self.config.name}...")
        
        # Calculate all three types of consistency
        human_internal = self.calculate_human_internal_consistency(aligned_data['human_scores'])
        llm_internal = self.calculate_llm_internal_consistency(aligned_data['llm_scores'])
        human_llm_agreement = self.calculate_human_llm_agreement(
            aligned_data['human_scores'], 
            aligned_data['llm_scores']
        )
        
        # Compile summary statistics
        summary = {
            'analysis_config': {
                'name': self.config.name,
                'description': self.config.description,
                'dimensions': self.config.dimensions,
                'llm_strategy': self.config.llm_strategy,
                'llm_round': self.config.llm_round,
                'output_suffix': self.config.output_suffix
            },
            'dimensions_analyzed': list(human_internal.keys()),
            'n_dimensions': len(self.config.dimensions),
            'n_systems': len(aligned_data['systems']),
            'systems': aligned_data['systems']
        }
        
        # Calculate overall statistics
        for consistency_type, data in [
            ('human_internal', human_internal),
            ('llm_internal', llm_internal),
            ('human_llm_agreement', human_llm_agreement)
        ]:
            if data:
                pearson_values = []
                spearman_values = []
                icc_values = []
                
                for dim in data:
                    if data[dim]:  # Check if dimension data exists
                        try:
                            # Get Pearson correlation - handle different data structures
                            pearson_data = data[dim]['pearson']
                            if isinstance(pearson_data, dict) and 'mean_correlation' in pearson_data:
                                # For human_internal and llm_internal
                                pearson_val = pearson_data['mean_correlation']
                            elif isinstance(pearson_data, (tuple, list)) and len(pearson_data) >= 2:
                                # For human_llm_agreement
                                pearson_val = pearson_data[0]
                            else:
                                pearson_val = None
                                
                            if pearson_val is not None and isinstance(pearson_val, (int, float)) and not np.isnan(pearson_val):
                                pearson_values.append(pearson_val)
                                
                            # Get Spearman correlation - handle different data structures
                            spearman_data = data[dim]['spearman']
                            if isinstance(spearman_data, dict) and 'mean_correlation' in spearman_data:
                                # For human_internal and llm_internal
                                spearman_val = spearman_data['mean_correlation']
                            elif isinstance(spearman_data, (tuple, list)) and len(spearman_data) >= 2:
                                # For human_llm_agreement
                                spearman_val = spearman_data[0]
                            else:
                                spearman_val = None
                                
                            if spearman_val is not None and isinstance(spearman_val, (int, float)) and not np.isnan(spearman_val):
                                spearman_values.append(spearman_val)
                            
                            # Get ICC value
                            icc_data = data[dim]['icc']
                            if isinstance(icc_data, dict) and 'value' in icc_data:
                                icc_val = icc_data['value']
                                if isinstance(icc_val, (int, float)) and not np.isnan(icc_val):
                                    icc_values.append(icc_val)
                                
                        except (KeyError, TypeError, ValueError):
                            continue
                
                summary[consistency_type] = {
                    'mean_pearson': np.mean(pearson_values) if pearson_values else np.nan,
                    'mean_spearman': np.mean(spearman_values) if spearman_values else np.nan,
                    'mean_icc': np.mean(icc_values) if icc_values else np.nan,
                    'n_dimensions_analyzed': len([dim for dim in data if data[dim]])
                }
        
        return {
            'summary': summary,
            'human_internal': human_internal,
            'llm_internal': llm_internal,
            'human_llm_agreement': human_llm_agreement
        }


if __name__ == "__main__":
    # Test the consistency analyzer
    from data_loader import ConsistencyDataLoader
    
    loader = ConsistencyDataLoader()
    aligned_data = loader.load_all_data()
    
    analyzer = ConsistencyAnalyzer()
    results = analyzer.run_full_analysis(aligned_data)
    
    print("\n=== Consistency Analysis Results ===")
    print(f"Dimensions analyzed: {len(results['summary']['dimensions_analyzed'])}")
    print(f"Systems analyzed: {results['summary']['n_systems']}")
    
    for consistency_type in ['human_internal', 'llm_internal', 'human_llm_agreement']:
        if consistency_type in results['summary']:
            print(f"\n{consistency_type.replace('_', ' ').title()}:")
            print(f"  Mean Pearson: {results['summary'][consistency_type]['mean_pearson']:.3f}")
            print(f"  Mean Spearman: {results['summary'][consistency_type]['mean_spearman']:.3f}")
            print(f"  Mean ICC: {results['summary'][consistency_type]['mean_icc']:.3f}")