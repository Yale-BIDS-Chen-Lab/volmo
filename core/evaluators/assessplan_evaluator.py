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
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer


@dataclass
class AssessPlanMetrics:
    bert_f1: float
    sbert_similarity: float


class AssessPlanEvaluator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.result_paths = self.config['RESULT_PATHS']
        self.save_dir = Path(self.config['SAVE_DIR'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
        if torch.cuda.is_available():
            self.sbert_model = self.sbert_model.to('cuda')
        
        self.eval_indices = [2, 3, 4]
    
    def calculate_bert_score(self, references: List[str], hypotheses: List[str]) -> List[float]:
        _, _, f1_scores = bert_score(
            hypotheses, 
            references, 
            model_type='roberta-large',
            lang='en', 
            verbose=False,
            rescale_with_baseline=False
        )
        return f1_scores.tolist()
    
    def calculate_sbert_similarity(self, reference: str, hypothesis: str) -> float:
        if not reference or not hypothesis:
            return 0.0
        
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            ref_sentences = sent_tokenize(reference.strip())
            hyp_sentences = sent_tokenize(hypothesis.strip())
            
            ref_sentences = [s.strip() for s in ref_sentences if s.strip()]
            hyp_sentences = [s.strip() for s in hyp_sentences if s.strip()]
            
            if not ref_sentences or not hyp_sentences:
                return 0.0
            
            similarities = []
            
            if len(ref_sentences) <= len(hyp_sentences):
                hyp_embeddings = self.sbert_model.encode(hyp_sentences, convert_to_tensor=False)
                
                for ref_sent in ref_sentences:
                    ref_emb = self.sbert_model.encode([ref_sent], convert_to_tensor=False)[0]
                    sent_sims = [cosine_similarity([ref_emb], [hyp_emb])[0, 0] for hyp_emb in hyp_embeddings]
                    similarities.append(max(sent_sims))
            
            else:
                ref_embeddings = self.sbert_model.encode(ref_sentences, convert_to_tensor=False)
                
                for hyp_sent in hyp_sentences:
                    hyp_emb = self.sbert_model.encode([hyp_sent], convert_to_tensor=False)[0]
                    sent_sims = [cosine_similarity([ref_emb], [hyp_emb])[0, 0] for ref_emb in ref_embeddings]
                    similarities.append(max(sent_sims))
            
            if similarities:
                return float(np.mean(similarities))
            else:
                return 0.0
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ GPU OOM in SBERT, moving to CPU")
                self.sbert_model = self.sbert_model.to('cpu')
                ref_emb = self.sbert_model.encode([reference], convert_to_tensor=False)[0]
                hyp_emb = self.sbert_model.encode([hypothesis], convert_to_tensor=False)[0]
                similarity = cosine_similarity([ref_emb], [hyp_emb])[0, 0]
                print(f"[DEBUG] OOM fallback returned: {similarity:.4f}")
                return float(similarity)
            raise e
        except Exception as e:
            print(f"⚠️ Error in sentence-level SBERT processing, falling back to paragraph level: {e}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            try:
                ref_emb = self.sbert_model.encode([reference], convert_to_tensor=False)[0]
                hyp_emb = self.sbert_model.encode([hypothesis], convert_to_tensor=False)[0]
                similarity = cosine_similarity([ref_emb], [hyp_emb])[0, 0]
                print(f"[DEBUG] Fallback returned: {similarity:.4f}")
                return float(similarity)
            except Exception as e2:
                print(f"⚠️ Error in paragraph-level SBERT: {e2}")
                return 0.0
    
    def evaluate_single(self, ground_truth: str, prediction: str) -> AssessPlanMetrics:
        bert_f1 = self.calculate_bert_score([ground_truth], [prediction])[0]
        sbert_sim = self.calculate_sbert_similarity(ground_truth, prediction)
        
        return AssessPlanMetrics(
            bert_f1=bert_f1,
            sbert_similarity=sbert_sim
        )
    
    def evaluate(self) -> Dict:
        """Run evaluation on all results."""
        all_metrics = []
        metrics_by_index = {idx: [] for idx in self.eval_indices}
        
        for result_path in self.result_paths:
            with open(result_path, 'r') as f:
                results = json.load(f)
            
            print(f"Evaluating {len(results)} samples from {result_path}")
            
            for item in results:
                gt_list = item.get('GT', [])
                pred_list = item.get('lm_response', item.get('response', []))
                
                if not isinstance(gt_list, list) or not isinstance(pred_list, list):
                    print(f"Warning: Skipping item - GT or prediction is not a list")
                    continue
                
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
                    
                    if len(all_metrics) in [9, 10, 11]:
                        print(f"[DEBUG] Metric #{len(all_metrics)}: SBERT={metrics.sbert_similarity:.4f}")
                        print(f"[DEBUG]   GT length: {len(gt)} chars, Pred length: {len(pred)} chars")
                        print(f"[DEBUG]   GT preview: {gt[:100]}...")
                        print(f"[DEBUG]   Pred preview: {pred[:100]}...")
                    
                    all_metrics.append(metrics)
                    metrics_by_index[idx].append(metrics)
        
        if len(all_metrics) == 0:
            print("No valid samples found for evaluation")
            return {
                'task': 'assessment_plan',
                'total_samples': 0,
                'overall_metrics': {},
                'metrics_by_index': {},
                'individual_metrics': []
            }
        
        overall = {
            'bert_f1': sum(m.bert_f1 for m in all_metrics) / len(all_metrics),
            'sbert_similarity': sum(m.sbert_similarity for m in all_metrics) / len(all_metrics)
        }
        
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
        
        result_file = self.save_dir / "assessplan_results.json"
        print(f"\n[DEBUG] Saving to: {result_file}")
        print(f"[DEBUG] First 3 SBERT scores: {[m.sbert_similarity for m in all_metrics[:3]]}")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[DEBUG] File saved successfully")
        
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
