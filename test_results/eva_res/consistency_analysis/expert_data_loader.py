#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专家评分数据加载器，用于处理专家打分Excel文件。

文件结构：
- 8名专家，每名专家对27个问题的6个版本进行1-10分评分
- 版本1-6分别对应：GPT-4.1-nano, MOSES, GPT-4.1, MOSES-nano, Intern, Spark
- 版本号在不同问题中是随机排列的（需要根据Answer Version号来识别）
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class ExpertDataLoader:
    """专家评分数据加载和预处理"""
    
    def __init__(self, excel_path: str = None):
        if excel_path is None:
            self.excel_path = Path(__file__).parent.parent / "expert" / "expert scoring.xlsx"
        else:
            self.excel_path = Path(excel_path)
        
        # 版本-模型映射
        self.model_mapping = {
            1: 'GPT-4.1-nano',
            2: 'MOSES', 
            3: 'GPT-4.1',
            4: 'MOSES-nano',
            5: 'Intern',
            6: 'Spark'
        }
        
        # 反向映射：模型名 -> 版本号
        self.reverse_model_mapping = {v: k for k, v in self.model_mapping.items()}
        
        self.num_questions = 27
        self.num_versions = 6
        self.num_experts = None
        
        # 数据容器
        self.raw_data = None
        self.expert_names = []
        self.valid_expert_rows = []  # 有效专家行的索引
        self.question_data = {}  # {question_idx: {model_name: [expert_scores]}}
        
    def load_excel_data(self) -> pd.DataFrame:
        """加载Excel数据"""
        print(f"加载专家评分数据: {self.excel_path}")
        
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel文件不存在: {self.excel_path}")
        
        # 读取Excel文件
        df = pd.read_excel(self.excel_path)
        self.raw_data = df
        
        print(f"数据形状: {df.shape}")
        
        # 找出有效的专家行（排除空行和模型名行）
        self.valid_expert_rows = []
        model_names_set = set(self.model_mapping.values())
        
        for i in range(df.shape[0]):
            name = df.iloc[i, 1]  # 第二列是姓名
            if pd.notna(name) and str(name).strip():
                name_str = str(name).strip()
                # 排除模型名称
                if name_str not in model_names_set:
                    self.valid_expert_rows.append(i)
                    self.expert_names.append(name_str)
        
        self.num_experts = len(self.valid_expert_rows)
        print(f"有效专家数量: {self.num_experts}")
        print(f"专家姓名: {self.expert_names}")
        
        return df
    
    def parse_questions_and_scores(self):
        """解析所有问题的评分数据"""
        print("解析问题评分数据...")
        
        columns = self.raw_data.columns.tolist()
        
        # 找到所有答案列（跳过前两列：时间戳和姓名）
        answer_columns = []
        for i in range(2, len(columns)):
            col_name = columns[i]
            if 'Answer Version' in str(col_name) and 'Please provide' not in str(col_name):
                answer_columns.append((i, col_name))
        
        print(f"找到答案列数: {len(answer_columns)}")
        
        if len(answer_columns) != self.num_questions * self.num_versions:
            print(f"警告: 答案列数({len(answer_columns)})不等于预期({self.num_questions * self.num_versions})")
        
        # 按27个问题分组处理
        for question_idx in range(self.num_questions):
            self.question_data[question_idx] = {}
            
            # 每个问题的6个版本
            start_idx = question_idx * self.num_versions
            end_idx = start_idx + self.num_versions
            
            question_scores_by_model = defaultdict(list)
            
            for version_pos in range(self.num_versions):
                col_pos = start_idx + version_pos
                
                if col_pos < len(answer_columns):
                    col_idx, col_name = answer_columns[col_pos]
                    
                    # 提取版本号
                    version_match = re.search(r'Answer Version (\d+)', str(col_name))
                    if version_match:
                        version_num = int(version_match.group(1))
                        model_name = self.model_mapping.get(version_num, f'Unknown_V{version_num}')
                        
                        # 提取专家评分
                        expert_scores = []
                        for expert_row in self.valid_expert_rows:
                            score = self.raw_data.iloc[expert_row, col_idx]
                            try:
                                if pd.notna(score):
                                    expert_scores.append(float(score))
                                else:
                                    expert_scores.append(np.nan)
                            except (ValueError, TypeError):
                                expert_scores.append(np.nan)
                        
                        question_scores_by_model[model_name] = expert_scores
            
            # 存储该问题的评分数据
            self.question_data[question_idx] = dict(question_scores_by_model)
        
        print(f"成功解析 {len(self.question_data)} 个问题的评分数据")
        
        # 验证数据完整性
        total_scores = 0
        for q_idx, q_data in self.question_data.items():
            for model, scores in q_data.items():
                total_scores += len([s for s in scores if not pd.isna(s)])
        
        expected_scores = self.num_questions * self.num_versions * self.num_experts
        print(f"有效评分数: {total_scores}/{expected_scores}")
    
    def get_expert_internal_consistency_data(self) -> Dict[str, Any]:
        """获取专家内部一致性分析数据"""
        print("准备专家内部一致性数据...")
        
        # 计算每个专家对每个模型的平均评分（跨所有问题）
        expert_avg_scores = {}
        
        for expert_idx in range(self.num_experts):
            expert_avg_scores[expert_idx] = {}
            
            for model_name in self.model_mapping.values():
                model_scores = []
                
                # 收集该专家对该模型在所有问题中的评分
                for question_idx in range(self.num_questions):
                    if model_name in self.question_data[question_idx]:
                        scores = self.question_data[question_idx][model_name]
                        if expert_idx < len(scores) and not pd.isna(scores[expert_idx]):
                            model_scores.append(scores[expert_idx])
                
                # 计算平均分
                if model_scores:
                    expert_avg_scores[expert_idx][model_name] = np.mean(model_scores)
                else:
                    expert_avg_scores[expert_idx][model_name] = np.nan
        
        # 转换为一致性分析所需的格式
        consistency_data = {
            'experts': list(range(self.num_experts)),
            'expert_names': self.expert_names,
            'models': list(self.model_mapping.values()),
            'scores': expert_avg_scores,
            'raw_question_data': self.question_data
        }
        
        return consistency_data
    
    def get_model_subset_data(self, model_subset: List[str]) -> Dict[str, Any]:
        """获取指定模型子集的数据"""
        full_data = self.get_expert_internal_consistency_data()
        
        # 过滤指定的模型
        filtered_models = [model for model in model_subset if model in full_data['models']]
        
        filtered_data = {
            'experts': full_data['experts'],
            'expert_names': full_data['expert_names'],
            'models': filtered_models,
            'scores': {},
            'raw_question_data': full_data['raw_question_data']
        }
        
        for expert_idx in full_data['scores']:
            filtered_data['scores'][expert_idx] = {}
            for model_name in filtered_models:
                filtered_data['scores'][expert_idx][model_name] = full_data['scores'][expert_idx][model_name]
        
        return filtered_data
    
    def get_question_level_data(self, question_indices: List[int] = None) -> Dict[str, Any]:
        """获取问题级别的详细数据"""
        if question_indices is None:
            question_indices = list(range(self.num_questions))
        
        question_level_data = {
            'questions': question_indices,
            'experts': list(range(self.num_experts)),
            'expert_names': self.expert_names,
            'models': list(self.model_mapping.values()),
            'question_scores': {}  # {question_idx: {model_name: [expert_scores]}}
        }
        
        for q_idx in question_indices:
            if q_idx in self.question_data:
                question_level_data['question_scores'][q_idx] = self.question_data[q_idx]
        
        return question_level_data
    
    def load_all_data(self) -> Dict[str, Any]:
        """加载所有数据并返回结构化结果"""
        try:
            # 加载Excel数据
            self.load_excel_data()
            
            # 解析问题和评分
            self.parse_questions_and_scores()
            
            # 返回完整数据
            result = {
                'expert_names': self.expert_names,
                'num_experts': self.num_experts,
                'num_questions': self.num_questions,
                'models': list(self.model_mapping.values()),
                'question_data': self.question_data,
                'expert_consistency_data': self.get_expert_internal_consistency_data()
            }
            
            print(f"✓ 成功加载专家评分数据：{self.num_experts}名专家，{self.num_questions}个问题，{len(self.model_mapping)}个模型")
            
            return result
            
        except Exception as e:
            print(f"加载专家数据失败: {e}")
            raise e


if __name__ == "__main__":
    # 测试数据加载器
    print("=== 测试专家数据加载器 ===")
    
    loader = ExpertDataLoader()
    data = loader.load_all_data()
    
    print(f"\n专家姓名: {data['expert_names'][:5]}...")  # 显示前5个
    print(f"模型列表: {data['models']}")
    
    print("\n专家内部一致性数据样本:")
    consistency_data = data['expert_consistency_data']
    for expert_idx in range(min(3, len(consistency_data['experts']))):
        expert_scores = consistency_data['scores'][expert_idx]
        print(f"  专家{expert_idx+1}: {expert_scores}")
    
    # 测试模型子集
    subset1 = ['GPT-4.1-nano', 'GPT-4.1', 'Spark']
    subset_data1 = loader.get_model_subset_data(subset1)
    print(f"\n子集1 ({subset1}) 数据:")
    print(f"  模型数: {len(subset_data1['models'])}")
    print(f"  专家数: {len(subset_data1['experts'])}")
    
    subset2 = ['GPT-4.1-nano', 'MOSES', 'GPT-4.1', 'MOSES-nano', 'Spark'] 
    subset_data2 = loader.get_model_subset_data(subset2)
    print(f"\n子集2 ({subset2}) 数据:")
    print(f"  模型数: {len(subset_data2['models'])}")
    print(f"  专家数: {len(subset_data2['experts'])}")