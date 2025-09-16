#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文报告生成器，用于生成详细的一致性分析报告。
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class ReportGenerator:
    """完整的中文报告生成器"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_analysis_report(self, results: Dict[str, Any], report_path: str, summary_path: str):
        """
        生成完整的分析报告
        
        Args:
            results: 分析结果字典
            report_path: 报告文件路径
            summary_path: 摘要CSV文件路径
        """
        # 生成摘要表格
        summary_table = self.create_summary_table(results)
        
        # 保存CSV摘要
        summary_table.to_csv(summary_path, index=False, encoding='utf-8')
        
        # 生成详细报告
        self.generate_detailed_report(results, report_path, summary_table)
    
    def format_p_value(self, p_value) -> str:
        """Format p-value using scientific notation for very small values."""
        if p_value is None or np.isnan(p_value):
            return "N/A"
        
        # 使用科学计数法显示极小的p值
        if p_value < 0.001:
            return f"{p_value:.2e}"
        else:
            return f"{p_value:.3f}"
    
    def create_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """创建摘要表格"""
        summary_data = []
        
        # 维度名称映射
        dimension_names = {
            "正确性": "正确性",
            "逻辑性": "逻辑性", 
            "清晰度": "清晰度",
            "完备性": "完备性",
            "理论深度": "理论深度",
            "论述严谨性与信息密度": "论述严谨性与信息密度"
        }
        
        # 一致性类型映射
        consistency_types = {
            'human_internal': '人工评分内部一致性',
            'llm_internal': 'LLM评分内部一致性', 
            'human_llm_agreement': '人工-LLM评分一致性'
        }
        
        for consistency_key, consistency_name in consistency_types.items():
            if consistency_key in results:
                for dim_key, dim_data in results[consistency_key].items():
                    if dim_key in dimension_names and dim_data:
                        # 提取相关数据
                        pearson_data = dim_data.get('pearson', {})
                        spearman_data = dim_data.get('spearman', {}) 
                        icc_data = dim_data.get('icc', {})
                        
                        # 处理不同的数据结构
                        if consistency_key in ['human_internal', 'llm_internal']:
                            # 内部一致性使用平均相关性
                            pearson_r = pearson_data.get('mean_correlation', np.nan)
                            pearson_p = pearson_data.get('mean_p_value', np.nan)
                            spearman_r = spearman_data.get('mean_correlation', np.nan)
                            spearman_p = spearman_data.get('mean_p_value', np.nan)
                        else:
                            # 人工-LLM一致性使用直接相关性
                            pearson_r = pearson_data[0] if isinstance(pearson_data, tuple) and len(pearson_data) >= 2 else np.nan
                            pearson_p = pearson_data[1] if isinstance(pearson_data, tuple) and len(pearson_data) >= 2 else np.nan
                            spearman_r = spearman_data[0] if isinstance(spearman_data, tuple) and len(spearman_data) >= 2 else np.nan
                            spearman_p = spearman_data[1] if isinstance(spearman_data, tuple) and len(spearman_data) >= 2 else np.nan
                        
                        # ICC数据
                        icc_value = icc_data.get('value', np.nan)
                        icc_ci = icc_data.get('confidence_interval_95', (np.nan, np.nan))
                        icc_p = icc_data.get('p_value', np.nan)
                        
                        summary_data.append({
                            'Dimension': dimension_names[dim_key],
                            'Consistency_Type': consistency_name,
                            'Pearson_r': f"{pearson_r:.3f}" if not np.isnan(pearson_r) else "N/A",
                            'Pearson_p': self.format_p_value(pearson_p),
                            'Spearman_r': f"{spearman_r:.3f}" if not np.isnan(spearman_r) else "N/A", 
                            'Spearman_p': self.format_p_value(spearman_p),
                            'ICC': f"{icc_value:.3f}" if not np.isnan(icc_value) else "N/A",
                            'ICC_CI_lower': f"{icc_ci[0]:.3f}" if not np.isnan(icc_ci[0]) else "N/A",
                            'ICC_CI_upper': f"{icc_ci[1]:.3f}" if not np.isnan(icc_ci[1]) else "N/A", 
                            'ICC_p': self.format_p_value(icc_p)
                        })
        
        return pd.DataFrame(summary_data)
    
    def generate_detailed_report(self, results: Dict[str, Any], filepath: str, summary_table: pd.DataFrame):
        """生成详细的中文分析报告"""
        
        report_content = f"""# 一致性分析报告

**生成时间**: {self.timestamp}

## 摘要

本报告对人工评价者和大语言模型评价系统在多个AI系统和评价维度上的一致性进行了全面分析。

### 主要发现
"""
        
        # 计算总体统计
        if results and 'summary' in results:
            summary = results['summary']
            
            report_content += f"""
**数据集概览:**
- 分析系统数量: {summary['n_systems']}
- 分析维度数量: {len(summary['dimensions_analyzed'])}
- 包含系统: {', '.join(summary['systems'])}

**总体一致性指标:**

| 一致性类型 | 平均Pearson相关 | 平均Spearman相关 | 平均ICC |
|-----------|----------------|------------------|---------|
"""
            
            consistency_name_mapping = {
                'human_internal': '人工评分内部一致性',
                'llm_internal': 'LLM评分内部一致性',
                'human_llm_agreement': '人工-LLM评分一致性'
            }
            
            for consistency_type in ['human_internal', 'llm_internal', 'human_llm_agreement']:
                if consistency_type in summary:
                    data = summary[consistency_type]
                    name = consistency_name_mapping[consistency_type]
                    pearson = data.get('mean_pearson', np.nan)
                    spearman = data.get('mean_spearman', np.nan)
                    icc = data.get('mean_icc', np.nan)
                    
                    pearson_str = f"{pearson:.3f}" if not np.isnan(pearson) else "N/A"
                    spearman_str = f"{spearman:.3f}" if not np.isnan(spearman) else "N/A"
                    icc_str = f"{icc:.3f}" if not np.isnan(icc) else "N/A"
                    
                    report_content += f"| {name} | {pearson_str} | {spearman_str} | {icc_str} |\n"
        
        report_content += """

## 方法论

### 一致性指标说明

1. **Pearson相关系数**: 测量评分之间的线性关系强度
2. **Spearman等级相关系数**: 测量评分之间的单调关系强度
3. **组内相关系数 (ICC)**: 测量评价者之间的绝对一致性

### 分析类型

1. **人工评分内部一致性**: 3名人工评价者之间的一致性
2. **LLM评分内部一致性**: 5轮LLM评价之间的一致性  
3. **人工-LLM评分一致性**: 人工评价平均分与LLM评价分之间的一致性

## 各维度详细结果
"""
        
        # 添加维度详细结果
        if not summary_table.empty:
            # 按维度分组
            for dimension in summary_table['Dimension'].unique():
                report_content += f"\n### {dimension}\n\n"
                
                dim_data = summary_table[summary_table['Dimension'] == dimension]
                
                report_content += "| 一致性类型 | Pearson r (p值) | Spearman r (p值) | ICC [95%置信区间] (p值) |\n"
                report_content += "|-----------|-----------------|------------------|------------------------|\n"
                
                for _, row in dim_data.iterrows():
                    pearson_str = f"{row['Pearson_r']} ({row['Pearson_p']})"
                    spearman_str = f"{row['Spearman_r']} ({row['Spearman_p']})"
                    icc_str = f"{row['ICC']} [{row['ICC_CI_lower']}, {row['ICC_CI_upper']}] ({row['ICC_p']})"
                    
                    report_content += f"| {row['Consistency_Type']} | {pearson_str} | {spearman_str} | {icc_str} |\n"
        
        report_content += """

## 统计解释

### 相关性强度指南
- **0.90-1.00**: 极高相关
- **0.70-0.89**: 高相关  
- **0.50-0.69**: 中等相关
- **0.30-0.49**: 低相关
- **0.00-0.29**: 微弱相关

### p值解释
- **p < 0.001**: 有极强证据拒绝零假设
- **p < 0.01**: 有强证据拒绝零假设
- **p < 0.05**: 有中等证据拒绝零假设
- **p ≥ 0.05**: 没有足够证据拒绝零假设

### ICC解释
- **> 0.75**: 一致性优秀
- **0.60-0.74**: 一致性良好
- **0.40-0.59**: 一致性一般
- **< 0.40**: 一致性较差

## 关键洞察
"""
        
        # 基于结果添加洞察
        if results and 'summary' in results:
            summary = results['summary']
            
            insights = []
            
            # 比较不同一致性类型
            consistency_names = {
                'human_internal': '人工评分内部一致性',
                'llm_internal': 'LLM评分内部一致性',
                'human_llm_agreement': '人工-LLM评分一致性'
            }
            
            for consistency_type, name in consistency_names.items():
                if consistency_type in summary:
                    mean_pearson = summary[consistency_type].get('mean_pearson', np.nan)
                    mean_icc = summary[consistency_type].get('mean_icc', np.nan)
                    
                    if not np.isnan(mean_pearson):
                        if mean_pearson >= 0.7:
                            strength = "高"
                        elif mean_pearson >= 0.5:
                            strength = "中等"
                        elif mean_pearson >= 0.3:
                            strength = "低"
                        else:
                            strength = "微弱"
                        
                        insights.append(f"- **{name}**显示{strength}一致性 (Pearson r = {mean_pearson:.3f})")
            
            for insight in insights:
                report_content += insight + "\n"
        
        # 添加配置信息
        if 'summary' in results and 'analysis_config' in results['summary']:
            config = results['summary']['analysis_config']
            report_content += f"""

## 分析配置

- **分析名称**: {config.get('name', 'N/A')}
- **分析描述**: {config.get('description', 'N/A')}
- **LLM策略**: {config.get('llm_strategy', 'N/A')}
- **分析维度**: {', '.join(config.get('dimensions', []))}
"""
            
            if config.get('llm_round'):
                report_content += f"- **LLM轮次**: 第{config['llm_round']}轮\n"
            
            if config.get('per_model_rounds'):
                report_content += "- **每模型轮次设置**:\n"
                for model, round_num in config['per_model_rounds'].items():
                    report_content += f"  - {model}: 第{round_num}轮\n"
            
            if config.get('selected_models'):
                report_content += f"- **选定模型**: {', '.join(config['selected_models'])}\n"
        
        report_content += """

## 建议

1. **对于人工评价**: 如果人工内部一致性较低，建议加强评价者培训
2. **对于LLM评价**: 如果LLM内部一致性较低，建议改进提示工程
3. **对于系统比较**: 使用可靠性最高的一致性类型进行比较分析
4. **对于未来研究**: 关注人工和LLM评价者一致性最高的维度

## 技术说明

- 所有置信区间均在95%置信水平下计算
- ICC计算使用双向随机效应模型进行绝对一致性评估
- 缺失值已从相关性计算中排除
- 统计显著性检验使用双尾检验

---
*本报告由一致性分析管道自动生成*
"""
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✓ 中文分析报告已生成: {filepath}")
        print(f"✓ 摘要表格已生成: {summary_table.shape[0]}行数据")