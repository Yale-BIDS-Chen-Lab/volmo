"""
VOLMO Assessment & Plan Evaluator
Evaluates clinical assessment and plan text using NLP metrics

@Author: Zhenyue Qin
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# NLP metrics
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util


@dataclass
class AssessPlanMetrics:
    """Metrics for assessment & plan evaluation."""
    bert_f1: float
    sbert_similarity: float


class AssessPlanEvaluator:
    """Evaluator for assessment & plan task using NLP metrics."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.result_paths = self.config['RESULT_PATHS']
        self.save_dir = Path(self.config['SAVE_DIR'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP models (only SBERT needed, BERTScore is loaded on demand)
        self.sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Indices to evaluate: 2, 3, 4 (skip 0 and 1)
        self.eval_indices = [2, 3, 4]
    
    def calculate_bert_score(self, references: List[str], hypotheses: List[str]) -> List[float]:
        """Calculate BERTScore for batch."""
        _, _, f1_scores = bert_score(hypotheses, references, lang='en', verbose=False)
        return f1_scores.tolist()
    
    def calculate_sbert_similarity(self, reference: str, hypothesis: str) -> float:
        """Calculate sentence-BERT similarity."""
        ref_embedding = self.sbert_model.encode(reference, convert_to_tensor=True)
        hyp_embedding = self.sbert_model.encode(hypothesis, convert_to_tensor=True)
        similarity = util.cos_sim(ref_embedding, hyp_embedding).item()
        return similarity
    
    def evaluate_single(self, ground_truth: str, prediction: str) -> AssessPlanMetrics:
        """Evaluate single prediction."""
        # BERTScore (batch of 1)
        bert_f1 = self.calculate_bert_score([ground_truth], [prediction])[0]
        
        # Sentence-BERT similarity
        sbert_sim = self.calculate_sbert_similarity(ground_truth, prediction)
        
        return AssessPlanMetrics(
            bert_f1=bert_f1,
            sbert_similarity=sbert_sim
        )
    
    def evaluate(self) -> Dict:
        """Run evaluation on all results."""
        all_metrics = []
        # Store metrics for each index separately
        metrics_by_index = {idx: [] for idx in self.eval_indices}
        
        for result_path in self.result_paths:
            with open(result_path, 'r') as f:
                results = json.load(f)
            
            print(f"Evaluating {len(results)} samples from {result_path}")
            
            for item in results:
                gt_list = item.get('GT', [])
                pred_list = item.get('lm_response', item.get('response', []))
                
                # Ensure both are lists
                if not isinstance(gt_list, list) or not isinstance(pred_list, list):
                    print(f"Warning: Skipping item - GT or prediction is not a list")
                    continue
                
                # Evaluate only indices 2, 3, 4
                for idx in self.eval_indices:
                    if idx >= len(gt_list) or idx >= len(pred_list):
                        print(f"Warning: Index {idx} out of range, skipping")
                        continue
                    
                    gt = gt_list[idx].strip()
                    pred = pred_list[idx].strip()
                    
                    if not gt or not pred:
                        print(f"Warning: Empty GT or prediction at index {idx}, skipping")
                        continue
                    
                    metrics = self.evaluate_single(gt, pred)
                    all_metrics.append(metrics)
                    metrics_by_index[idx].append(metrics)
        
        # Check if we have any metrics
        if len(all_metrics) == 0:
            print("No valid samples found for evaluation")
            return {
                'task': 'assessment_plan',
                'total_samples': 0,
                'overall_metrics': {},
                'metrics_by_index': {},
                'individual_metrics': []
            }
        
        # Calculate overall metrics (across all indices)
        overall = {
            'bert_f1': sum(m.bert_f1 for m in all_metrics) / len(all_metrics),
            'sbert_similarity': sum(m.sbert_similarity for m in all_metrics) / len(all_metrics)
        }
        
        # Calculate metrics for each index
        index_metrics = {}
        for idx in self.eval_indices:
            if len(metrics_by_index[idx]) > 0:
                index_metrics[f'index_{idx}'] = {
                    'bert_f1': sum(m.bert_f1 for m in metrics_by_index[idx]) / len(metrics_by_index[idx]),
                    'sbert_similarity': sum(m.sbert_similarity for m in metrics_by_index[idx]) / len(metrics_by_index[idx]),
                    'count': len(metrics_by_index[idx])
                }
        
        result = {
            'task': 'assessment_plan',
            'total_samples': len(all_metrics),
            'evaluated_indices': self.eval_indices,
            'overall_metrics': overall,
            'metrics_by_index': index_metrics,
            'individual_metrics': [
                {
                    'bert_f1': m.bert_f1,
                    'sbert_similarity': m.sbert_similarity
                }
                for m in all_metrics
            ]
        }
        
        # Save results
        result_file = self.save_dir / "assessplan_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nAssessment & Plan Evaluation Complete")
        print(f"  Evaluated Indices: {self.eval_indices}")
        print(f"  BERTScore F1: {overall['bert_f1']:.4f}")
        print(f"  SBERT Similarity: {overall['sbert_similarity']:.4f}")
        print(f"  Total Samples: {len(all_metrics)}")
        print(f"\n  Metrics by Index:")
        for idx in self.eval_indices:
            if f'index_{idx}' in index_metrics:
                idx_metrics = index_metrics[f'index_{idx}']
                print(f"    Index {idx}: BERTScore F1={idx_metrics['bert_f1']:.4f}, "
                      f"SBERT={idx_metrics['sbert_similarity']:.4f}, Count={idx_metrics['count']}")
        print(f"  Results: {result_file}\n")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="VOLMO Assessment & Plan Evaluator")
    parser.add_argument("--config_path", type=str, required=True, help="Path to evaluation config")
    args = parser.parse_args()
    
    evaluator = AssessPlanEvaluator(args.config_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
