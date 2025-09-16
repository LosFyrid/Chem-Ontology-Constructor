#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed data loading and preprocessing for consistency analysis using real data.
Correctly handles CSV structure and JSON format with proper system/dimension mapping.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import warnings
import re

from system_mapping import (
    SYSTEM_NAME_MAPPING, 
    DIMENSION_MAPPING, 
    DIMENSION_MAPPING_REVERSE,
    get_mapped_llm_system,
    get_mapped_dimension
)
from analysis_config import AnalysisConfig

class ConsistencyDataLoader:
    """Load and preprocess real evaluation data for consistency analysis."""
    
    def __init__(self, base_path: str = None, config: AnalysisConfig = None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = Path(base_path)
        
        self.config = config  # Store analysis configuration
            
        self.human_path = self.base_path / "human"
        self.llm_path = self.base_path / "individual"
        
        # Evaluation dimensions (Chinese)
        self.dimensions = ["正确性", "逻辑性", "清晰度", "完备性", "理论深度", "论述严谨性与信息密度"]
        
        # Data containers
        self.human_data = None
        self.llm_data = None
        self.processed_data = {
            'human_scores': {},  # {system_name: {question_id: {dimension: [score1, score2, score3]}}}
            'llm_scores': {},    # {system_name: {question_id: {dimension: [score1, score2, score3, score4, score5]}}}
        }
        
    def load_human_scores(self) -> pd.DataFrame:
        """Load human evaluation scores from CSV file."""
        print("Loading human evaluation scores...")
        
        # Find the human evaluation CSV file
        csv_files = list(self.human_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.human_path}")
        
        csv_file = csv_files[0]
        print(f"Loading from: {csv_file}")
        
        # Read CSV with proper encoding
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='utf-8')
        
        print(f"CSV shape: {df.shape}")
        
        # Parse the CSV structure
        self._process_human_csv_data(df)
        self.human_data = df
        
        print(f"Successfully parsed human scores for {len(self.processed_data['human_scores'])} systems")
        return df
    
    def _parse_human_score_cell(self, score_cell) -> float:
        """Extract numeric score from potentially annotated cell."""
        if pd.isna(score_cell):
            return None
            
        # Convert to string and handle the format
        cell_str = str(score_cell).strip()
        
        # Handle various annotation formats
        if '\n' in cell_str:
            # Split by newline, score should be in first part
            first_part = cell_str.split('\n')[0].strip()
        else:
            first_part = cell_str
        
        # Extract number from formats like "打分：2", "2", "打分:2", etc.
        if '：' in first_part or ':' in first_part:
            # Format like "打分：2"
            score_part = first_part.split('：')[-1].split(':')[-1].strip()
        else:
            score_part = first_part
        
        # Try to extract numeric value
        try:
            score = float(score_part)
            return score if 0 <= score <= 10 else None  # Allow 0 as valid score
        except ValueError:
            # Use regex to find first number
            numbers = re.findall(r'\d+\.?\d*', score_part)
            if numbers:
                try:
                    score = float(numbers[0])
                    return score if 0 <= score <= 10 else None  # Allow 0 as valid score
                except ValueError:
                    pass
        
        return None

    def _process_human_csv_data(self, df: pd.DataFrame):
        """Process the CSV data with correct structure understanding."""
        print("Processing human evaluation CSV data...")
        
        # Parse header to identify system positions
        with open(self.human_path / list(self.human_path.glob("*.csv"))[0], 'r', encoding='utf-8-sig') as f:
            header_line = f.readline().strip()
        
        # Split header and identify system positions
        header_parts = [part.strip() for part in header_line.split(',')]
        
        # Find system positions and their dimension columns
        system_positions = {}
        current_col = 0
        
        for i, part in enumerate(header_parts):
            if part in SYSTEM_NAME_MAPPING:  # This is a system name from our mapping
                system_positions[part] = {
                    'start_col': i,
                    'dimension_cols': {}
                }
                
                # Next 6 columns should be the dimensions
                for dim_idx, dimension in enumerate(self.dimensions):
                    dim_col = i + 1 + dim_idx
                    if dim_col < len(header_parts):
                        system_positions[part]['dimension_cols'][dimension] = dim_col
                
                print(f"Found system '{part}' at column {i}, dimensions at {i+1}-{i+6}")
        
        print(f"Identified {len(system_positions)} systems with proper dimension mapping")
        
        # Process questions (every 4 rows: question+rater1+rater2+rater3+empty)
        total_questions = (len(df) + 3) // 4  # Round up division
        print(f"Processing {total_questions} questions...")
        
        for q_idx in range(total_questions):
            start_row = q_idx * 4
            
            if start_row >= len(df):
                break
                
            # Check if this is actually a question row
            question_row = df.iloc[start_row]
            question_id_cell = question_row.iloc[0]
            
            if pd.isna(question_id_cell):
                continue
                
            try:
                question_id = str(int(float(question_id_cell)))
            except:
                continue
            
            # Extract scores for all raters for this question
            rater_scores = []
            for rater_idx in range(3):  # 3 raters
                row_idx = start_row + rater_idx
                if row_idx < len(df):
                    if rater_idx == 0:
                        # First rater scores are in the question row
                        rater_scores.append(df.iloc[start_row])
                    else:
                        # Other rater scores are in subsequent rows
                        rater_scores.append(df.iloc[row_idx])
                else:
                    rater_scores.append(None)
            
            # Process each system for this question
            for system_name, system_info in system_positions.items():
                if system_name not in self.processed_data['human_scores']:
                    self.processed_data['human_scores'][system_name] = {}
                
                self.processed_data['human_scores'][system_name][question_id] = {}
                
                # Extract scores for each dimension from all raters
                for dimension, col_idx in system_info['dimension_cols'].items():
                    dimension_scores = []
                    
                    for rater_row in rater_scores:
                        if rater_row is not None and col_idx < len(rater_row):
                            score_cell = rater_row.iloc[col_idx]
                            
                            # Use the new parsing method to handle annotated cells
                            score = self._parse_human_score_cell(score_cell)
                            if score is not None:
                                dimension_scores.append(score)
                    
                    # Store scores if we have at least 2 raters
                    if len(dimension_scores) >= 2:
                        # Pad to exactly 3 scores if needed
                        while len(dimension_scores) < 3:
                            dimension_scores.append(dimension_scores[-1])  # Repeat last score
                        
                        self.processed_data['human_scores'][system_name][question_id][dimension] = dimension_scores[:3]
        
        # Print summary
        for system_name in self.processed_data['human_scores']:
            n_questions = len(self.processed_data['human_scores'][system_name])
            print(f"  System '{system_name}': {n_questions} questions")
    
    def load_llm_scores(self) -> List[Dict]:
        """Load LLM evaluation scores from JSONL files."""
        print("Loading LLM evaluation scores...")
        
        json_files = list(self.llm_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.llm_path}")
        
        all_records = []
        
        for json_file in json_files:
            print(f"Processing: {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            record = json.loads(line)
                            all_records.append(record)
                        except json.JSONDecodeError as e:
                            if line_num <= 5:  # Only warn for first few errors
                                print(f"  JSON decode error at line {line_num}: {e}")
                            continue
                            
            except Exception as e:
                print(f"  Error processing {json_file}: {e}")
                continue
        
        print(f"Loaded {len(all_records)} LLM evaluation records")
        self._process_llm_data(all_records)
        self.llm_data = all_records
        
        return all_records
    
    def _process_llm_data(self, records: List[Dict]):
        """Process LLM evaluation data into structured format."""
        print("Processing LLM evaluation data...")
        
        system_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        processed_count = 0
        error_count = 0
        
        for record in records:
            try:
                model_name = record.get("model_name", "unknown")
                question_id = str(record.get("question_id", "unknown"))
                
                # Clean up question ID format (q_1 -> 1)
                if question_id.startswith("q_"):
                    question_id = question_id[2:]
                
                # Parse the evaluation answer
                answer_str = record.get("answer", "{}")
                
                # Extract JSON from answer
                answer_json = self._extract_json_from_answer(answer_str)
                
                if answer_json:
                    # Process scores for each dimension in the JSON
                    for eng_dimension, score_value in answer_json.items():
                        # Map English dimension to Chinese
                        chinese_dimension = get_mapped_dimension(eng_dimension, to_chinese=True)
                        
                        if chinese_dimension in self.dimensions:
                            # Extract numeric score
                            score = self._extract_numeric_score(score_value)
                            if score is not None and 0 <= score <= 10:  # Allow 0 as valid score
                                system_scores[model_name][question_id][chinese_dimension].append(score)
                                processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                continue
        
        print(f"Successfully processed {processed_count} score entries")
        print(f"Errors encountered: {error_count}")
        
        # Convert to regular dict and ensure exactly 5 scores per dimension per question
        final_scores = {}
        for system in system_scores:
            final_scores[system] = {}
            for question in system_scores[system]:
                final_scores[system][question] = {}
                for dimension in system_scores[system][question]:
                    scores = system_scores[system][question][dimension]
                    
                    # Ensure exactly 5 scores (for 5 LLM evaluation rounds)
                    if len(scores) >= 5:
                        final_scores[system][question][dimension] = scores[:5]
                    elif len(scores) > 0:
                        # Pad with repeated values if we have fewer than 5
                        padded_scores = scores.copy()
                        while len(padded_scores) < 5:
                            padded_scores.append(scores[-1])  # Repeat last score
                        final_scores[system][question][dimension] = padded_scores
                    else:
                        # Skip if no valid scores
                        continue
        
        self.processed_data['llm_scores'] = final_scores
        
        # Print summary
        for system_name in final_scores:
            n_questions = len(final_scores[system_name])
            print(f"  System '{system_name}': {n_questions} questions")
        
        print(f"Processed LLM scores for {len(final_scores)} systems")
    
    def _extract_json_from_answer(self, answer_str: str) -> Dict:
        """Extract JSON from various answer formats."""
        if not answer_str:
            return {}
        
        # Try to extract JSON from markdown code blocks
        json_str = answer_str.strip()
        
        if "```json" in answer_str:
            start = answer_str.find("```json") + len("```json")
            end = answer_str.find("```", start)
            if end > start:
                json_str = answer_str[start:end].strip()
            else:
                json_str = answer_str[start:].strip()
        elif "```" in answer_str and "{" in answer_str:
            # Generic code block with JSON
            start = answer_str.find("```")
            end = answer_str.find("```", start + 3)
            if end > start:
                json_str = answer_str[start+3:end].strip()
            else:
                json_str = answer_str[start+3:].strip()
        
        # Try to parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find JSON-like content with regex
            json_match = re.search(r'\{[^{}]*\}', json_str, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {}
    
    def _extract_numeric_score(self, score_value) -> float:
        """Extract numeric score from various formats."""
        if isinstance(score_value, list):
            if len(score_value) == 1 and isinstance(score_value[0], (int, float)):
                return float(score_value[0])
            elif len(score_value) > 1:
                # Take first numeric value
                for val in score_value:
                    if isinstance(val, (int, float)):
                        return float(val)
            return None
        elif isinstance(score_value, (int, float)):
            return float(score_value)
        elif isinstance(score_value, str):
            # Try to extract number from string
            try:
                return float(score_value)
            except ValueError:
                # Look for first number in string
                numbers = re.findall(r'\d+\.?\d*', score_value)
                if numbers:
                    return float(numbers[0])
        return None
    
    def align_data(self) -> Dict[str, Any]:
        """Align human and LLM scores using the system mapping."""
        print("Aligning human and LLM evaluation data...")
        
        aligned_data = {
            'systems': [],
            'dimensions': self.dimensions,
            'human_scores': {},
            'llm_scores': {},
        }
        
        # Use system mapping to align data
        for human_system, llm_system in SYSTEM_NAME_MAPPING.items():
            # Filter systems based on configuration
            if self.config and self.config.selected_models:
                if human_system not in self.config.selected_models:
                    continue  # Skip systems not in selected_models
            
            # Special handling for llasmol when using average strategy
            if human_system == "llasmol" and self.config and self.config.llm_strategy == 'average':
                # For average strategy, combine both llasmol-top1 and llasmol-top5 data
                llm_top1_data = self.processed_data['llm_scores'].get('llasmol-top1', {})
                llm_top5_data = self.processed_data['llm_scores'].get('llasmol-top5', {})
                
                if (human_system in self.processed_data['human_scores'] and 
                    (llm_top1_data or llm_top5_data)):
                    
                    human_questions = set(self.processed_data['human_scores'][human_system].keys())
                    
                    # Get questions that have data in either top1 or top5
                    llm_questions = set()
                    if llm_top1_data:
                        llm_questions.update(llm_top1_data.keys())
                    if llm_top5_data:
                        llm_questions.update(llm_top5_data.keys())
                    
                    common_questions = human_questions.intersection(llm_questions)
                    
                    if len(common_questions) >= 5:  # Need at least 5 questions
                        aligned_data['systems'].append(human_system)
                        aligned_data['human_scores'][human_system] = {}
                        aligned_data['llm_scores'][human_system] = {}
                        
                        for question in common_questions:
                            # Get human dimensions
                            human_dims = set(self.processed_data['human_scores'][human_system][question].keys())
                            
                            # Get LLM dimensions from both top1 and top5
                            llm_dims = set()
                            if question in llm_top1_data:
                                llm_dims.update(llm_top1_data[question].keys())
                            if question in llm_top5_data:
                                llm_dims.update(llm_top5_data[question].keys())
                            
                            common_dims = human_dims.intersection(llm_dims)
                            
                            if len(common_dims) >= 3:  # Need at least half dimensions
                                aligned_data['human_scores'][human_system][question] = {}
                                aligned_data['llm_scores'][human_system][question] = {}
                                
                                for dimension in common_dims:
                                    human_scores = self.processed_data['human_scores'][human_system][question][dimension]
                                    
                                    # Combine scores from both top1 and top5 (10 total scores)
                                    combined_llm_scores = []
                                    if question in llm_top1_data and dimension in llm_top1_data[question]:
                                        combined_llm_scores.extend(llm_top1_data[question][dimension])
                                    if question in llm_top5_data and dimension in llm_top5_data[question]:
                                        combined_llm_scores.extend(llm_top5_data[question][dimension])
                                    
                                    # Ensure we have valid scores
                                    if len(human_scores) >= 2 and len(combined_llm_scores) >= 3:
                                        aligned_data['human_scores'][human_system][question][dimension] = human_scores
                                        aligned_data['llm_scores'][human_system][question][dimension] = combined_llm_scores
            else:
                # Standard handling for other systems
                if (human_system in self.processed_data['human_scores'] and 
                    llm_system in self.processed_data['llm_scores']):
                    
                    human_questions = set(self.processed_data['human_scores'][human_system].keys())
                    llm_questions = set(self.processed_data['llm_scores'][llm_system].keys())
                    common_questions = human_questions.intersection(llm_questions)
                    
                    if len(common_questions) >= 5:  # Need at least 5 questions
                        aligned_data['systems'].append(human_system)  # Use human system name as key
                        aligned_data['human_scores'][human_system] = {}
                        aligned_data['llm_scores'][human_system] = {}
                        
                        for question in common_questions:
                            human_dims = set(self.processed_data['human_scores'][human_system][question].keys())
                            llm_dims = set(self.processed_data['llm_scores'][llm_system][question].keys())
                            common_dims = human_dims.intersection(llm_dims)
                            
                            if len(common_dims) >= 3:  # Need at least half dimensions
                                aligned_data['human_scores'][human_system][question] = {}
                                aligned_data['llm_scores'][human_system][question] = {}
                                
                                for dimension in common_dims:
                                    human_scores = self.processed_data['human_scores'][human_system][question][dimension]
                                    llm_scores = self.processed_data['llm_scores'][llm_system][question][dimension]
                                    
                                    # Ensure we have valid scores
                                    if len(human_scores) >= 2 and len(llm_scores) >= 3:
                                        aligned_data['human_scores'][human_system][question][dimension] = human_scores
                                        aligned_data['llm_scores'][human_system][question][dimension] = llm_scores
        
        print(f"Successfully aligned data for {len(aligned_data['systems'])} systems:")
        for system in aligned_data['systems']:
            n_questions = len(aligned_data['human_scores'][system])
            print(f"  - {system}: {n_questions} questions")
        
        return aligned_data
    
    def load_all_data(self) -> Dict[str, Any]:
        """Load and align all evaluation data."""
        print("="*60)
        print("LOADING REAL EVALUATION DATA")
        print("="*60)
        
        # Load human scores
        try:
            self.load_human_scores()
        except Exception as e:
            print(f"Error loading human scores: {e}")
            raise e
        
        # Load LLM scores
        try:
            self.load_llm_scores()
        except Exception as e:
            print(f"Error loading LLM scores: {e}")
            raise e
        
        # Align data
        aligned_data = self.align_data()
        
        if not aligned_data['systems']:
            raise ValueError("No systems could be aligned between human and LLM evaluations")
        
        return aligned_data


if __name__ == "__main__":
    # Test the data loader
    loader = ConsistencyDataLoader()
    aligned_data = loader.load_all_data()
    
    print("\n=== DATA LOADING COMPLETE ===")
    print(f"Successfully aligned {len(aligned_data['systems'])} systems")
    print(f"Dimensions: {len(aligned_data['dimensions'])}")
    
    if aligned_data['systems']:
        sample_system = aligned_data['systems'][0]
        sample_questions = list(aligned_data['human_scores'][sample_system].keys())
        print(f"\nSample system '{sample_system}' has {len(sample_questions)} questions")
        
        if sample_questions:
            sample_question = sample_questions[0]
            sample_dimensions = list(aligned_data['human_scores'][sample_system][sample_question].keys())
            print(f"Sample question '{sample_question}' has {len(sample_dimensions)} dimensions")
            
            if sample_dimensions:
                sample_dimension = sample_dimensions[0]
                human_scores = aligned_data['human_scores'][sample_system][sample_question][sample_dimension]
                llm_scores = aligned_data['llm_scores'][sample_system][sample_question][sample_dimension]
                print(f"\nSample scores for '{sample_dimension}':")
                print(f"  Human ({len(human_scores)} raters): {human_scores}")
                print(f"  LLM ({len(llm_scores)} rounds): {llm_scores}")
                
    print("\nData structure is ready for consistency analysis!")