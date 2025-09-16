#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis configuration for different consistency analysis scenarios.
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict

@dataclass
class AnalysisConfig:
    """Configuration for different consistency analysis scenarios."""
    
    # Analysis identifier
    name: str
    description: str
    
    # Dimension selection
    dimensions: List[str]
    
    # LLM evaluation strategy
    llm_strategy: str  # 'average' or 'specific_round' or 'per_model_round'
    
    # Output directory suffix
    output_suffix: str
    
    # LLM round (used when strategy is 'specific_round')
    llm_round: Optional[int] = None
    
    # Per-model LLM rounds (used when strategy is 'per_model_round')
    # Format: {'system_name': round_number}
    per_model_rounds: Optional[Dict[str, int]] = None
    
    # Model selection (if None, analyze all available models)
    selected_models: Optional[List[str]] = None
    
    def validate(self):
        """Validate configuration parameters."""
        if self.llm_strategy == 'specific_round' and (self.llm_round is None or self.llm_round < 1 or self.llm_round > 5):
            raise ValueError("For specific_round strategy, llm_round must be between 1 and 5")
        
        if self.llm_strategy == 'per_model_round' and not self.per_model_rounds:
            raise ValueError("For per_model_round strategy, per_model_rounds must be provided")
        
        if self.llm_strategy not in ['average', 'specific_round', 'per_model_round']:
            raise ValueError("llm_strategy must be 'average', 'specific_round', or 'per_model_round'")
        
        if not self.dimensions:
            raise ValueError("At least one dimension must be specified")
        
        # Validate per_model_rounds if provided
        if self.per_model_rounds:
            for model, round_num in self.per_model_rounds.items():
                if not isinstance(round_num, int) or round_num < 1 or round_num > 5:
                    raise ValueError(f"Round number for model {model} must be between 1 and 5, got {round_num}")

# Pre-defined analysis configurations
ALL_DIMENSIONS = ["正确性", "逻辑性", "清晰度", "完备性", "理论深度", "论述严谨性与信息密度"]
REDUCED_DIMENSIONS = ["正确性", "完备性", "理论深度", "论述严谨性与信息密度"]  # Remove logic and clarity

# Available model names (human system names)
AVAILABLE_MODELS = [
    "gpt-4.1-final",
    "gpt-4.1-nano-final-815-1", 
    "lightrag-gpt-4_1",
    "lightrag-gpt-4_1-nano",
    "o1-final",
    "o3-final",
    "reordered_MOSES-final",
    "reordered_MOSES-nano-final",
    "chemqa27_from_chem13b_rag_infer_yesthink",
    "gpt-4o-final-815-1",
    "gpt-4o-mini-final-815-1"
    "llasmol",
    "darwin"
]

ANALYSIS_CONFIGS = {
    'full_average': AnalysisConfig(
        name='Full Analysis (All Dimensions, LLM Average)',
        description='Complete analysis with all 6 dimensions using average of 5 LLM evaluation rounds',
        dimensions=ALL_DIMENSIONS,
        llm_strategy='average',
        output_suffix='full_average'
    ),
    
    'reduced_average': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, LLM Average)',
        description='Analysis excluding Logic and Clarity dimensions, using average of 5 LLM evaluation rounds',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='average',
        output_suffix='reduced_average'
    ),
    
    'full_round1': AnalysisConfig(
        name='Full Analysis (All Dimensions, LLM Round 1)',
        description='Complete analysis with all 6 dimensions using only LLM evaluation round 1',
        dimensions=ALL_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=1,
        output_suffix='full_round1'
    ),
    
    'full_round2': AnalysisConfig(
        name='Full Analysis (All Dimensions, LLM Round 2)',
        description='Complete analysis with all 6 dimensions using only LLM evaluation round 2',
        dimensions=ALL_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=2,
        output_suffix='full_round2'
    ),
    
    'full_round3': AnalysisConfig(
        name='Full Analysis (All Dimensions, LLM Round 3)',
        description='Complete analysis with all 6 dimensions using only LLM evaluation round 3',
        dimensions=ALL_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=3,
        output_suffix='full_round3'
    ),
    
    'full_round4': AnalysisConfig(
        name='Full Analysis (All Dimensions, LLM Round 4)',
        description='Complete analysis with all 6 dimensions using only LLM evaluation round 4',
        dimensions=ALL_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=4,
        output_suffix='full_round4'
    ),
    
    'full_round5': AnalysisConfig(
        name='Full Analysis (All Dimensions, LLM Round 5)',
        description='Complete analysis with all 6 dimensions using only LLM evaluation round 5',
        dimensions=ALL_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=5,
        output_suffix='full_round5'
    ),
    
    'reduced_round1': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, LLM Round 1)',
        description='Analysis excluding Logic and Clarity dimensions, using only LLM evaluation round 1',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=1,
        output_suffix='reduced_round1'
    ),
    
    'reduced_round2': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, LLM Round 2)',
        description='Analysis excluding Logic and Clarity dimensions, using only LLM evaluation round 2',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=2,
        output_suffix='reduced_round2'
    ),
    
    'reduced_round3': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, LLM Round 3)',
        description='Analysis excluding Logic and Clarity dimensions, using only LLM evaluation round 3',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=3,
        output_suffix='reduced_round3'
    ),
    
    'reduced_round4': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, LLM Round 4)',
        description='Analysis excluding Logic and Clarity dimensions, using only LLM evaluation round 4',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=4,
        output_suffix='reduced_round4'
    ),
    
    'reduced_round5': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, LLM Round 5)',
        description='Analysis excluding Logic and Clarity dimensions, using only LLM evaluation round 5',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='specific_round',
        llm_round=5,
        output_suffix='reduced_round5'
    ),
    
    # === 新增：每个模型单独指定轮次的配置 ===
    'reduced_per_model_example': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, Per-Model Rounds)',
        description='Analysis with reduced dimensions, each model using different LLM evaluation rounds',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='per_model_round',
        output_suffix='reduced_per_model_example',
        # 示例：为每个模型指定不同的轮次
        per_model_rounds={
            "gpt-4.1-final": 1,
            "gpt-4.1-nano-final-815-1": 2, 
            "lightrag-gpt-4_1": 3,
            "lightrag-gpt-4_1-nano": 4,
            "o1-final": 5,
            "o3-final": 1,
            "reordered_MOSES-final": 2,
            "reordered_MOSES-nano-final": 3,
            "chemqa27_from_chem13b_rag_infer_yesthink": 4,
            "gpt-4o-final-815-1": 5,
            "gpt-4o-mini-final-815-1": 1
        }
    ),
    
    # === 新增：仅分析部分模型的配置 ===
    'reduced_selected_models': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, Selected Models Only)',
        description='Analysis with reduced dimensions for only GPT-4 family models',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='average',
        output_suffix='reduced_selected_models',
        # 仅分析GPT-4系列模型
        selected_models=[
            "gpt-4.1-final",
            "gpt-4.1-nano-final-815-1", 
            "lightrag-gpt-4_1",
            "lightrag-gpt-4_1-nano",
            "o1-final",
            "o3-final",
            "reordered_MOSES-final",
            "reordered_MOSES-nano-final",
            "gpt-4o-final-815-1",
            "gpt-4o-mini-final-815-1"
        ]
    ),

    'reduced_llasmol_n_darwin': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, Selected Models Only)',
        description='Analysis with reduced dimensions for only GPT-4 family models',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='average',
        output_suffix='reduced_llasmol_n_darwin',
        # 仅分析GPT-4系列模型
        selected_models=[
            "llasmol",
            "darwin"
        ]
    ),
    
    # === 新增：组合功能 - 部分模型 + 每个模型不同轮次 ===
    'reduced_gpt4_family_custom_rounds': AnalysisConfig(
        name='Reduced Analysis (GPT-4 Family, Custom Rounds)',
        description='Analysis of GPT-4 family models with custom rounds per model',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='per_model_round',
        output_suffix='reduced_gpt4_custom',
        selected_models=[
            "gpt-4.1-final",
            "gpt-4.1-nano-final-815-1",
            "gpt-4o-final-815-1",
            "gpt-4o-mini-final-815-1"
        ],
        per_model_rounds={
            "gpt-4.1-final": 1,        # GPT-4.1使用第1轮
            "gpt-4.1-nano-final-815-1": 2,  # GPT-4.1-nano使用第2轮
            "gpt-4o-final-815-1": 3,   # GPT-4o使用第3轮
            "gpt-4o-mini-final-815-1": 4    # GPT-4o-mini使用第4轮
        }
    ),
    
    'actual': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, Selected Models Only)',
        description='actual analysis',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='per_model_round',
        output_suffix='actual',
        selected_models= [
            "lightrag-gpt-4_1",
            "lightrag-gpt-4_1-nano",
            "reordered_MOSES-final",
            "reordered_MOSES-nano-final",
            "chemqa27_from_chem13b_rag_infer_yesthink",
            "llasmol",
            "darwin"
        ],
        per_model_rounds={
            "lightrag-gpt-4_1": 1,
            "lightrag-gpt-4_1-nano": 1,
            "reordered_MOSES-final": 4,
            "reordered_MOSES-nano-final": 5,
            "chemqa27_from_chem13b_rag_infer_yesthink": 2,
            "llasmol": 5,
            "darwin": 3
        }
    ),

    'actual_w_llasmol_n_darwin': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, Selected Models Only)',
        description='actual analysis',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='per_model_round',
        output_suffix='actual_w_llasmol_n_darwin',
        selected_models= [
            "llasmol",
            "darwin"
        ],
        per_model_rounds={
            "llasmol": 5,
            "darwin": 3
        }
    ),

    'actual_w/o_llasmol_n_darwin': AnalysisConfig(
        name='Reduced Analysis (4 Dimensions, Selected Models Only)',
        description='actual analysis',
        dimensions=REDUCED_DIMENSIONS,
        llm_strategy='per_model_round',
        output_suffix='actual_w/o_llasmol_n_darwin',
        selected_models= [
            "lightrag-gpt-4_1",
            "lightrag-gpt-4_1-nano",
            "reordered_MOSES-final",
            "reordered_MOSES-nano-final",
            "chemqa27_from_chem13b_rag_infer_yesthink",
        ],
        per_model_rounds={
            "lightrag-gpt-4_1": 1,
            "lightrag-gpt-4_1-nano": 1,
            "reordered_MOSES-final": 4,
            "reordered_MOSES-nano-final": 5,
            "chemqa27_from_chem13b_rag_infer_yesthink": 2,
        }
    ),
}

def get_analysis_config(config_name: str) -> AnalysisConfig:
    """Get analysis configuration by name."""
    if config_name not in ANALYSIS_CONFIGS:
        raise ValueError(f"Unknown analysis configuration: {config_name}. Available: {list(ANALYSIS_CONFIGS.keys())}")
    
    config = ANALYSIS_CONFIGS[config_name]
    config.validate()
    return config

def get_default_analysis_configs() -> List[AnalysisConfig]:
    """Get the three default analysis configurations as requested."""
    return [
        ANALYSIS_CONFIGS['full_average'],      # Current analysis (all dimensions, LLM average)
        ANALYSIS_CONFIGS['reduced_average'],   # Exclude logic & clarity, LLM average
        ANALYSIS_CONFIGS['full_round1']        # All dimensions, LLM round 1 (user can modify this)
    ]

if __name__ == "__main__":
    # Test configurations
    print("Available Analysis Configurations:")
    print("=" * 50)
    
    for name, config in ANALYSIS_CONFIGS.items():
        print(f"{name}:")
        print(f"  Name: {config.name}")
        print(f"  Description: {config.description}")
        print(f"  Dimensions ({len(config.dimensions)}): {config.dimensions}")
        print(f"  LLM Strategy: {config.llm_strategy}")
        if config.llm_round:
            print(f"  LLM Round: {config.llm_round}")
        print(f"  Output Suffix: {config.output_suffix}")
        print()
    
    print("\nDefault Analysis Configs:")
    for config in get_default_analysis_configs():
        print(f"- {config.name}")