"""
VOLMO Image Description Evaluator
Evaluates text-based image description using NLP metrics

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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util


@dataclass
class ImageDescMetrics:
    """Metrics for image description evaluation."""
    bleu1: float
    bleu4: float
    rouge_l_f: float
    bert_f1: float
    sbert_similarity: float


class ImageDescEvaluator:
    """Evaluator for image description task using NLP metrics."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.result_paths = self.config['RESULT_PATHS']
        self.save_dir = Path(self.config['SAVE_DIR'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP models
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.smoothing = SmoothingFunction()
    
    def calculate_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate BLEU scores."""
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0),
                             smoothing_function=self.smoothing.method1)
        bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=self.smoothing.method1)
        
        return {'bleu1': bleu1, 'bleu4': bleu4}
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> float:
        """Calculate ROUGE-L F-measure."""
        scores = self.rouge_scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    
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
    
    def evaluate_single(self, ground_truth: str, prediction: str) -> ImageDescMetrics:
        """Evaluate single prediction."""
        # BLEU scores
        bleu_scores = self.calculate_bleu(ground_truth, prediction)
        
        # ROUGE-L
        rouge_l = self.calculate_rouge(ground_truth, prediction)
        
        # BERTScore (batch of 1)
        bert_f1 = self.calculate_bert_score([ground_truth], [prediction])[0]
        
        # Sentence-BERT similarity
        sbert_sim = self.calculate_sbert_similarity(ground_truth, prediction)
        
        return ImageDescMetrics(
            bleu1=bleu_scores['bleu1'],
            bleu4=bleu_scores['bleu4'],
            rouge_l_f=rouge_l,
            bert_f1=bert_f1,
            sbert_similarity=sbert_sim
        )
    
    def evaluate(self) -> Dict:
        """Run evaluation on all results."""
        all_metrics = []
        
        for result_path in self.result_paths:
            with open(result_path, 'r') as f:
                results = json.load(f)
            
            print(f"Evaluating {len(results)} samples from {result_path}")
            
            for item in results:
                gt = item.get('GT', '').strip()
                pred = item.get('lm_response', item.get('response', '')).strip()
                
                if not gt or not pred:
                    continue
                
                metrics = self.evaluate_single(gt, pred)
                all_metrics.append(metrics)
        
        # Check if we have any metrics
        if len(all_metrics) == 0:
            print("No valid samples found for evaluation")
            return {'task': 'image_description', 'total_samples': 0, 'overall_metrics': {}, 'individual_metrics': []}
        
        # Calculate overall metrics
        overall = {
            'bleu1': sum(m.bleu1 for m in all_metrics) / len(all_metrics),
            'bleu4': sum(m.bleu4 for m in all_metrics) / len(all_metrics),
            'rouge_l_f': sum(m.rouge_l_f for m in all_metrics) / len(all_metrics),
            'bert_f1': sum(m.bert_f1 for m in all_metrics) / len(all_metrics),
            'sbert_similarity': sum(m.sbert_similarity for m in all_metrics) / len(all_metrics)
        }
        
        result = {
            'task': 'image_description',
            'total_samples': len(all_metrics),
            'overall_metrics': overall,
            'individual_metrics': [
                {
                    'bleu1': m.bleu1,
                    'bleu4': m.bleu4,
                    'rouge_l_f': m.rouge_l_f,
                    'bert_f1': m.bert_f1,
                    'sbert_similarity': m.sbert_similarity
                }
                for m in all_metrics
            ]
        }
        
        # Save results
        result_file = self.save_dir / "imgdesc_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nImage Description Evaluation Complete")
        print(f"  BLEU-1: {overall['bleu1']:.4f}")
        print(f"  BLEU-4: {overall['bleu4']:.4f}")
        print(f"  ROUGE-L: {overall['rouge_l_f']:.4f}")
        print(f"  BERTScore F1: {overall['bert_f1']:.4f}")
        print(f"  SBERT Similarity: {overall['sbert_similarity']:.4f}")
        print(f"  Samples: {len(all_metrics)}")
        print(f"  Results: {result_file}\n")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="VOLMO Image Description Evaluator")
    parser.add_argument("--config_path", type=str, required=True, help="Path to evaluation config")
    args = parser.parse_args()
    
    evaluator = ImageDescEvaluator(args.config_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
