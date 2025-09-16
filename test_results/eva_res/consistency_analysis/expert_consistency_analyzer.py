#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专家一致性分析器，复用现有的一致性计算框架来分析专家评分数据。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

from expert_data_loader import ExpertDataLoader

class ExpertConsistencyAnalyzer:
    """专家一致性分析器"""
    
    def __init__(self):
        self.expert_loader = ExpertDataLoader()
    
    def calculate_icc(self, scores: np.ndarray, icc_type: str = 'ICC(2,1)') -> Tuple[float, float, Tuple[float, float]]:
        """
        Calculate Intraclass Correlation Coefficient (ICC) - 复用现有代码
        
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
        n, k = scores.shape
        
        # Calculate sum of squares
        total_sum = np.sum(scores)
        total_mean = total_sum / (n * k)
        
        # Between-subject sum of squares
        subject_means = np.mean(scores, axis=1)
        BSS = k * np.sum((subject_means - total_mean) ** 2)
        
        # Within-subject sum of squares  
        WSS = np.sum((scores - subject_means.reshape(-1, 1)) ** 2)
        
        # Total sum of squares
        TSS = np.sum((scores - total_mean) ** 2)
        
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
        
        # Confidence intervals (approximation)
        alpha = 0.05
        df1 = n - 1
        df2 = n * (k - 1)
        
        try:
            f_lower = stats.f.ppf(alpha/2, df1, df2)
            f_upper = stats.f.ppf(1 - alpha/2, df1, df2)
            
            if f_stat != 0:
                lower_ci = max(0, (f_stat/f_upper - 1) / (f_stat/f_upper + (k-1)))
                upper_ci = min(1, (f_stat/f_lower - 1) / (f_stat/f_lower + (k-1)))
            else:
                lower_ci, upper_ci = 0, 0
        except:
            lower_ci, upper_ci = np.nan, np.nan
        
        return icc, f_stat, (lower_ci, upper_ci)
    
    def calculate_correlations(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """计算相关性 - 复用现有代码"""
        if len(x) != len(y) or len(x) < 3:
            return {'pearson': np.nan, 'spearman': np.nan}
        
        try:
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(x, y)
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(x, y)
            
            return {
                'pearson': {'correlation': pearson_r, 'p_value': pearson_p},
                'spearman': {'correlation': spearman_r, 'p_value': spearman_p}
            }
        except:
            return {'pearson': np.nan, 'spearman': np.nan}
    
    def calculate_expert_internal_consistency(self, expert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算专家内部一致性（专家之间的一致性）
        """
        print("计算专家内部一致性...")
        
        experts = expert_data['experts']
        models = expert_data['models']  
        scores = expert_data['scores']
        
        results = {
            'correlations': {},
            'statistics': {},
            'details': {}
        }
        
        # 构造数据矩阵：行是模型，列是专家
        score_matrix = []
        valid_models = []
        
        for model in models:
            model_scores = []
            valid_experts_for_model = []
            
            for expert_idx in experts:
                score = scores[expert_idx].get(model, np.nan)
                if not pd.isna(score):
                    model_scores.append(score)
                    valid_experts_for_model.append(expert_idx)
            
            if len(model_scores) >= 3:  # 需要至少3个专家有效评分
                score_matrix.append(model_scores)
                valid_models.append(model)
        
        if len(score_matrix) >= 3:  # 需要至少3个模型
            # 转换为numpy数组
            scores_array = np.array(score_matrix)  # 模型 × 专家
            
            print(f"数据维度: {scores_array.shape} (模型×专家)")
            
            # 计算ICC
            icc_value, f_stat, icc_ci = self.calculate_icc(scores_array)
            
            # 计算专家之间的配对相关性
            correlations = {}
            all_pearson = []
            all_spearman = []
            
            num_experts = scores_array.shape[1]
            for i, j in combinations(range(num_experts), 2):
                expert_pair = f"expert_{i+1}_vs_expert_{j+1}"
                
                # 取两个专家在所有模型上的评分
                expert1_scores = scores_array[:, i]
                expert2_scores = scores_array[:, j]
                
                # 去除NaN配对
                valid_pairs = ~(np.isnan(expert1_scores) | np.isnan(expert2_scores))
                if valid_pairs.sum() >= 3:
                    corr_results = self.calculate_correlations(
                        expert1_scores[valid_pairs], 
                        expert2_scores[valid_pairs]
                    )
                    correlations[expert_pair] = corr_results
                    
                    # 收集所有相关性值
                    if isinstance(corr_results.get('pearson'), dict):
                        pearson_val = corr_results['pearson'].get('correlation')
                        if isinstance(pearson_val, (int, float)) and not np.isnan(pearson_val):
                            all_pearson.append(pearson_val)
                    
                    if isinstance(corr_results.get('spearman'), dict):
                        spearman_val = corr_results['spearman'].get('correlation')  
                        if isinstance(spearman_val, (int, float)) and not np.isnan(spearman_val):
                            all_spearman.append(spearman_val)
            
            # 汇总统计
            results['statistics'] = {
                'n_models': len(valid_models),
                'n_experts': num_experts,
                'icc': {
                    'value': icc_value,
                    'f_statistic': f_stat,
                    'confidence_interval': icc_ci
                }
            }
            
            if all_pearson:
                results['statistics']['mean_pearson'] = np.mean(all_pearson)
                results['statistics']['std_pearson'] = np.std(all_pearson)
                results['statistics']['pearson_values'] = all_pearson
                
            if all_spearman:
                results['statistics']['mean_spearman'] = np.mean(all_spearman)
                results['statistics']['std_spearman'] = np.std(all_spearman)
                results['statistics']['spearman_values'] = all_spearman
                
            results['correlations'] = correlations
            results['details'] = {
                'valid_models': valid_models,
                'score_matrix_shape': scores_array.shape
            }
            
        else:
            print(f"数据不足：只有{len(score_matrix)}个有效模型")
            results['error'] = f"数据不足：需要至少3个模型，但只有{len(score_matrix)}个"
        
        return results
    
    def compare_expert_vs_llm(self, expert_data: Dict[str, Any], 
                             model_subset: List[str]) -> Dict[str, Any]:
        """
        比较专家与LLM评分的一致性
        """
        print(f"比较专家与LLM一致性（模型子集: {model_subset}）...")
        
        # 导入现有的数据加载器
        from data_loader import ConsistencyDataLoader
        from analysis_config import AnalysisConfig
        
        try:
            # 创建配置，只选择指定模型子集
            config = AnalysisConfig(
                name=f"expert_vs_llm_{len(model_subset)}_models",
                description=f"专家与LLM一致性比较（{len(model_subset)}个模型）",
                dimensions=["正确性", "逻辑性", "清晰度", "完备性", "理论深度", "论述严谨性与信息密度"],
                llm_strategy='average',
                output_suffix=f"expert_vs_llm_{len(model_subset)}",
                selected_models=[self._map_expert_to_human_system_name(model) for model in model_subset if self._map_expert_to_human_system_name(model)]
            )
            
            # 加载LLM和人工评分数据
            data_loader = ConsistencyDataLoader(config=config)
            llm_human_data = data_loader.load_all_data()
            
            # 提取专家在指定模型子集上的平均评分
            expert_avg_scores = {}
            for model in model_subset:
                if model in expert_data['models']:
                    model_scores = []
                    for expert_idx in expert_data['experts']:
                        score = expert_data['scores'][expert_idx].get(model, np.nan)
                        if not pd.isna(score):
                            model_scores.append(score)
                    
                    if model_scores:
                        expert_avg_scores[model] = np.mean(model_scores)
            
            # 提取LLM的平均评分（对应模型）
            llm_avg_scores = {}
            for system in llm_human_data['systems']:
                # 根据映射找到对应的专家模型
                expert_model = self._map_human_to_expert_system_name(system)
                if expert_model and expert_model in model_subset:
                    # 计算该系统的LLM平均评分（跨所有维度和问题）
                    all_llm_scores = []
                    for question in llm_human_data['llm_scores'][system]:
                        for dimension in llm_human_data['llm_scores'][system][question]:
                            scores = llm_human_data['llm_scores'][system][question][dimension]
                            if isinstance(scores, list) and scores:
                                # 使用平均策略
                                avg_score = np.mean([s for s in scores if not pd.isna(s)])
                                if not pd.isna(avg_score):
                                    all_llm_scores.append(avg_score)
                    
                    if all_llm_scores:
                        llm_avg_scores[expert_model] = np.mean(all_llm_scores)
            
            # 找到共同的模型
            common_models = set(expert_avg_scores.keys()) & set(llm_avg_scores.keys())
            
            if len(common_models) >= 3:
                # 构造数据进行相关性分析
                expert_values = [expert_avg_scores[model] for model in common_models]
                llm_values = [llm_avg_scores[model] for model in common_models]
                
                # 计算相关性
                correlations = self.calculate_correlations(np.array(expert_values), np.array(llm_values))
                
                # 计算ICC（专家和LLM作为两个rater）
                combined_scores = np.column_stack([expert_values, llm_values])
                icc_value, f_stat, icc_ci = self.calculate_icc(combined_scores)
                
                results = {
                    'model_subset': model_subset,
                    'common_models': list(common_models),
                    'n_models': len(common_models),
                    'expert_scores': expert_avg_scores,
                    'llm_scores': llm_avg_scores,
                    'correlations': correlations,
                    'icc': {
                        'value': icc_value,
                        'f_statistic': f_stat,
                        'confidence_interval': icc_ci
                    },
                    'statistics': {
                        'expert_mean': np.mean(expert_values),
                        'expert_std': np.std(expert_values),
                        'llm_mean': np.mean(llm_values),
                        'llm_std': np.std(llm_values)
                    }
                }
                
                print(f"  成功比较{len(common_models)}个共同模型")
                if correlations and 'pearson' in correlations:
                    if isinstance(correlations['pearson'], dict):
                        pearson_r = correlations['pearson'].get('correlation', np.nan)
                        print(f"  专家-LLM Pearson相关性: {pearson_r:.3f}")
                
            else:
                results = {
                    'model_subset': model_subset,
                    'error': f'共同模型不足：需要至少3个，但只找到{len(common_models)}个',
                    'expert_scores': expert_avg_scores,
                    'llm_scores': llm_avg_scores,
                    'common_models': list(common_models)
                }
                print(f"  错误: {results['error']}")
        
        except Exception as e:
            results = {
                'model_subset': model_subset,
                'error': f'加载LLM数据失败: {str(e)}',
                'note': '请确保LLM评分数据可访问'
            }
            print(f"  错误: {results['error']}")
        
        return results
    
    def _map_expert_to_human_system_name(self, expert_model_name: str) -> str:
        """将专家评分的模型名映射到人工评分系统名"""
        # 根据system_mapping.py中的映射关系
        mapping = {
            'GPT-4.1-nano': 'gpt-4.1-nano-final-815-1',
            'MOSES': 'reordered_MOSES-final', 
            'GPT-4.1': 'gpt-4.1-final',
            'MOSES-nano': 'reordered_MOSES-nano-final',
            'Spark': 'chemqa27_from_chem13b_rag_infer_yesthink',  # 暂时映射
            'Intern': None  
        }
        return mapping.get(expert_model_name)
    
    def _map_human_to_expert_system_name(self, human_system_name: str) -> str:
        """将人工评分系统名映射回专家评分的模型名"""
        reverse_mapping = {
            'gpt-4.1-nano-final-815-1': 'GPT-4.1-nano',
            'reordered_MOSES-final': 'MOSES',
            'gpt-4.1-final': 'GPT-4.1', 
            'reordered_MOSES-nano-final': 'MOSES-nano',
            'chemqa27_from_chem13b_rag_infer_yesthink': 'Spark'
        }
        return reverse_mapping.get(human_system_name)
    
    def compare_expert_vs_human(self, expert_data: Dict[str, Any],
                               model_subset: List[str]) -> Dict[str, Any]:
        """
        比较专家与人类评分的一致性
        """
        print(f"比较专家与人类评分一致性（模型子集: {model_subset}）...")
        
        # 导入现有的数据加载器
        from data_loader import ConsistencyDataLoader
        from analysis_config import AnalysisConfig
        
        try:
            # 创建配置，只选择指定模型子集
            config = AnalysisConfig(
                name=f"expert_vs_human_{len(model_subset)}_models",
                description=f"专家与人类评分一致性比较（{len(model_subset)}个模型）",
                dimensions=["正确性", "逻辑性", "清晰度", "完备性", "理论深度", "论述严谨性与信息密度"],
                llm_strategy='average',
                output_suffix=f"expert_vs_human_{len(model_subset)}",
                selected_models=[self._map_expert_to_human_system_name(model) for model in model_subset if self._map_expert_to_human_system_name(model)]
            )
            
            # 加载人工评分数据
            data_loader = ConsistencyDataLoader(config=config)
            human_llm_data = data_loader.load_all_data()
            
            # 提取专家在指定模型子集上的平均评分
            expert_avg_scores = {}
            for model in model_subset:
                if model in expert_data['models']:
                    model_scores = []
                    for expert_idx in expert_data['experts']:
                        score = expert_data['scores'][expert_idx].get(model, np.nan)
                        if not pd.isna(score):
                            model_scores.append(score)
                    
                    if model_scores:
                        expert_avg_scores[model] = np.mean(model_scores)
            
            # 提取人工评分的平均分数（对应模型）
            human_avg_scores = {}
            for system in human_llm_data['systems']:
                # 根据映射找到对应的专家模型
                expert_model = self._map_human_to_expert_system_name(system)
                if expert_model and expert_model in model_subset:
                    # 计算该系统的人工平均评分（跨所有维度和问题）
                    all_human_scores = []
                    for question in human_llm_data['human_scores'][system]:
                        for dimension in human_llm_data['human_scores'][system][question]:
                            scores = human_llm_data['human_scores'][system][question][dimension]
                            if isinstance(scores, list) and scores:
                                # 取前3个评分者的平均
                                valid_scores = [s for s in scores[:3] if not pd.isna(s)]
                                if valid_scores:
                                    avg_score = np.mean(valid_scores)
                                    if not pd.isna(avg_score):
                                        all_human_scores.append(avg_score)
                    
                    if all_human_scores:
                        human_avg_scores[expert_model] = np.mean(all_human_scores)
            
            # 找到共同的模型
            common_models = set(expert_avg_scores.keys()) & set(human_avg_scores.keys())
            
            if len(common_models) >= 3:
                # 构造数据进行相关性分析
                expert_values = [expert_avg_scores[model] for model in common_models]
                human_values = [human_avg_scores[model] for model in common_models]
                
                # 计算相关性
                correlations = self.calculate_correlations(np.array(expert_values), np.array(human_values))
                
                # 计算ICC（专家和人工评分者作为两个rater）
                combined_scores = np.column_stack([expert_values, human_values])
                icc_value, f_stat, icc_ci = self.calculate_icc(combined_scores)
                
                results = {
                    'model_subset': model_subset,
                    'common_models': list(common_models),
                    'n_models': len(common_models),
                    'expert_scores': expert_avg_scores,
                    'human_scores': human_avg_scores,
                    'correlations': correlations,
                    'icc': {
                        'value': icc_value,
                        'f_statistic': f_stat,
                        'confidence_interval': icc_ci
                    },
                    'statistics': {
                        'expert_mean': np.mean(expert_values),
                        'expert_std': np.std(expert_values),
                        'human_mean': np.mean(human_values),
                        'human_std': np.std(human_values)
                    }
                }
                
                print(f"  成功比较{len(common_models)}个共同模型")
                if correlations and 'pearson' in correlations:
                    if isinstance(correlations['pearson'], dict):
                        pearson_r = correlations['pearson'].get('correlation', np.nan)
                        print(f"  专家-人工评分 Pearson相关性: {pearson_r:.3f}")
                
            else:
                results = {
                    'model_subset': model_subset,
                    'error': f'共同模型不足：需要至少3个，但只找到{len(common_models)}个',
                    'expert_scores': expert_avg_scores,
                    'human_scores': human_avg_scores,
                    'common_models': list(common_models)
                }
                print(f"  错误: {results['error']}")
        
        except Exception as e:
            results = {
                'model_subset': model_subset,
                'error': f'加载人工评分数据失败: {str(e)}',
                'note': '请确保人工评分数据可访问'
            }
            print(f"  错误: {results['error']}")
        
        return results
    
    def run_expert_analysis(self) -> Dict[str, Any]:
        """运行完整的专家一致性分析"""
        print("=== 开始专家一致性分析 ===")
        
        # 加载专家数据
        expert_data_all = self.expert_loader.load_all_data()
        expert_consistency_data = expert_data_all['expert_consistency_data']
        
        results = {
            'expert_names': expert_data_all['expert_names'],
            'num_experts': expert_data_all['num_experts'],
            'models': expert_data_all['models'],
            'analyses': {}
        }
        
        # 1. 专家内部一致性
        expert_internal = self.calculate_expert_internal_consistency(expert_consistency_data)
        results['analyses']['expert_internal_consistency'] = expert_internal
        
        # 2. 专家 vs LLM 一致性（两种模型组合）
        subset1 = ['GPT-4.1-nano', 'GPT-4.1', 'Spark']
        subset2 = ['GPT-4.1-nano', 'MOSES', 'GPT-4.1', 'MOSES-nano', 'Spark']
        
        expert_vs_llm_subset1 = self.compare_expert_vs_llm(expert_consistency_data, subset1)
        expert_vs_llm_subset2 = self.compare_expert_vs_llm(expert_consistency_data, subset2)
        
        results['analyses']['expert_vs_llm_subset1'] = expert_vs_llm_subset1
        results['analyses']['expert_vs_llm_subset2'] = expert_vs_llm_subset2
        
        # 3. 专家 vs 人类评分一致性
        expert_vs_human = self.compare_expert_vs_human(expert_consistency_data, subset2)
        results['analyses']['expert_vs_human'] = expert_vs_human
        
        return results
    
    def generate_expert_report(self, results: Dict[str, Any]) -> str:
        """生成专家一致性分析报告"""
        report = []
        report.append("# 专家一致性分析报告\n")
        
        # 基本信息
        report.append(f"- **专家数量**: {results['num_experts']}")
        report.append(f"- **专家姓名**: {', '.join(results['expert_names'])}")
        report.append(f"- **分析模型**: {', '.join(results['models'])}\n")
        
        # 专家内部一致性
        if 'expert_internal_consistency' in results['analyses']:
            analysis = results['analyses']['expert_internal_consistency']
            
            if 'error' in analysis:
                report.append("## 专家内部一致性")
                report.append(f"**错误**: {analysis['error']}\n")
            elif 'statistics' in analysis:
                internal_stats = analysis['statistics']
                report.append("## 专家内部一致性")
                
                if 'n_models' in internal_stats:
                    report.append(f"- **有效模型数**: {internal_stats['n_models']}")
                    report.append(f"- **专家数**: {internal_stats['n_experts']}")
                
                if 'mean_pearson' in internal_stats:
                    report.append(f"- **Pearson相关性**: {internal_stats['mean_pearson']:.3f} ± {internal_stats['std_pearson']:.3f}")
                
                if 'mean_spearman' in internal_stats:
                    report.append(f"- **Spearman相关性**: {internal_stats['mean_spearman']:.3f} ± {internal_stats['std_spearman']:.3f}")
                
                if 'icc' in internal_stats and isinstance(internal_stats['icc'], dict):
                    icc_val = internal_stats['icc'].get('value', np.nan)
                    if not pd.isna(icc_val):
                        report.append(f"- **ICC绝对一致性**: {icc_val:.3f}")
                
                report.append("")
        
        # 其他分析结果
        report.append("## 专家与LLM一致性比较")
        
        # 专家 vs LLM (子集1)
        if 'expert_vs_llm_subset1' in results['analyses']:
            analysis1 = results['analyses']['expert_vs_llm_subset1']
            subset1_models = analysis1.get('model_subset', [])
            report.append(f"### 模型子集1 ({', '.join(subset1_models)})")
            
            if 'error' in analysis1:
                report.append(f"**错误**: {analysis1['error']}")
            elif 'correlations' in analysis1:
                if 'correlations' in analysis1 and analysis1['correlations']:
                    corr = analysis1['correlations']
                    if isinstance(corr.get('pearson'), dict):
                        pearson_r = corr['pearson'].get('correlation', np.nan)
                        pearson_p = corr['pearson'].get('p_value', np.nan)
                        if not pd.isna(pearson_r):
                            report.append(f"- **Pearson相关性**: {pearson_r:.3f} (p={pearson_p:.3f})")
                    
                    if isinstance(corr.get('spearman'), dict):
                        spearman_r = corr['spearman'].get('correlation', np.nan)
                        spearman_p = corr['spearman'].get('p_value', np.nan)
                        if not pd.isna(spearman_r):
                            report.append(f"- **Spearman相关性**: {spearman_r:.3f} (p={spearman_p:.3f})")
                
                if 'icc' in analysis1 and isinstance(analysis1['icc'], dict):
                    icc_val = analysis1['icc'].get('value', np.nan)
                    if not pd.isna(icc_val):
                        report.append(f"- **ICC一致性**: {icc_val:.3f}")
                
                if 'n_models' in analysis1:
                    report.append(f"- **共同模型数**: {analysis1['n_models']}")
        
        # 专家 vs LLM (子集2)
        if 'expert_vs_llm_subset2' in results['analyses']:
            analysis2 = results['analyses']['expert_vs_llm_subset2']
            subset2_models = analysis2.get('model_subset', [])
            report.append(f"\n### 模型子集2 ({', '.join(subset2_models)})")
            
            if 'error' in analysis2:
                report.append(f"**错误**: {analysis2['error']}")
            elif 'correlations' in analysis2:
                if 'correlations' in analysis2 and analysis2['correlations']:
                    corr = analysis2['correlations']
                    if isinstance(corr.get('pearson'), dict):
                        pearson_r = corr['pearson'].get('correlation', np.nan)
                        pearson_p = corr['pearson'].get('p_value', np.nan)
                        if not pd.isna(pearson_r):
                            report.append(f"- **Pearson相关性**: {pearson_r:.3f} (p={pearson_p:.3f})")
                    
                    if isinstance(corr.get('spearman'), dict):
                        spearman_r = corr['spearman'].get('correlation', np.nan)
                        spearman_p = corr['spearman'].get('p_value', np.nan)
                        if not pd.isna(spearman_r):
                            report.append(f"- **Spearman相关性**: {spearman_r:.3f} (p={spearman_p:.3f})")
                
                if 'icc' in analysis2 and isinstance(analysis2['icc'], dict):
                    icc_val = analysis2['icc'].get('value', np.nan)
                    if not pd.isna(icc_val):
                        report.append(f"- **ICC一致性**: {icc_val:.3f}")
                
                if 'n_models' in analysis2:
                    report.append(f"- **共同模型数**: {analysis2['n_models']}")
        
        # 专家 vs 人类评分一致性
        report.append("\n## 专家与人类评分一致性比较")
        if 'expert_vs_human' in results['analyses']:
            analysis_human = results['analyses']['expert_vs_human']
            
            if 'error' in analysis_human:
                report.append(f"**错误**: {analysis_human['error']}")
            elif 'correlations' in analysis_human:
                if 'correlations' in analysis_human and analysis_human['correlations']:
                    corr = analysis_human['correlations']
                    if isinstance(corr.get('pearson'), dict):
                        pearson_r = corr['pearson'].get('correlation', np.nan)
                        pearson_p = corr['pearson'].get('p_value', np.nan)
                        if not pd.isna(pearson_r):
                            report.append(f"- **Pearson相关性**: {pearson_r:.3f} (p={pearson_p:.3f})")
                    
                    if isinstance(corr.get('spearman'), dict):
                        spearman_r = corr['spearman'].get('correlation', np.nan)
                        spearman_p = corr['spearman'].get('p_value', np.nan)
                        if not pd.isna(spearman_r):
                            report.append(f"- **Spearman相关性**: {spearman_r:.3f} (p={spearman_p:.3f})")
                
                if 'icc' in analysis_human and isinstance(analysis_human['icc'], dict):
                    icc_val = analysis_human['icc'].get('value', np.nan)
                    if not pd.isna(icc_val):
                        report.append(f"- **ICC一致性**: {icc_val:.3f}")
                
                if 'n_models' in analysis_human:
                    report.append(f"- **共同模型数**: {analysis_human['n_models']}")
        
        report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    print("=== 测试专家一致性分析器 ===")
    
    analyzer = ExpertConsistencyAnalyzer()
    results = analyzer.run_expert_analysis()
    
    print(f"\n分析完成!")
    print(f"专家数量: {results['num_experts']}")
    print(f"模型数量: {len(results['models'])}")
    
    # 显示专家内部一致性结果
    if 'expert_internal_consistency' in results['analyses']:
        analysis = results['analyses']['expert_internal_consistency']
        
        if 'error' in analysis:
            print(f"\n错误: {analysis['error']}")
        elif 'statistics' in analysis:
            stats = analysis['statistics']
            print(f"\n专家内部一致性:")
            
            if 'mean_pearson' in stats:
                print(f"  Pearson: {stats['mean_pearson']:.3f} ± {stats['std_pearson']:.3f}")
            
            if 'mean_spearman' in stats:
                print(f"  Spearman: {stats['mean_spearman']:.3f} ± {stats['std_spearman']:.3f}")
                
            if 'icc' in stats and isinstance(stats['icc'], dict):
                icc_val = stats['icc'].get('value', np.nan)
                if not pd.isna(icc_val):
                    print(f"  ICC: {icc_val:.3f}")
    
    # 生成报告
    report = analyzer.generate_expert_report(results)
    print(f"\n=== 专家分析报告 ===")
    print(report)