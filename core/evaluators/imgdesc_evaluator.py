"""VOLMO Image Description Evaluator"""

import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ImageDescMetrics:
    bleu1: float
    rouge_l_f: float
    bert_f1: float
    sbert_similarity: float


class ImageDescriptionEvaluator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.result_paths = self.config['RESULT_PATHS']
        self.save_dir = Path(self.config['SAVE_DIR'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
    
    def calculate_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0),
                             smoothing_function=self.smoothing.method1)
        return {'bleu1': bleu1}
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> float:
        scores = self.rouge_scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    
    def calculate_bert_score(self, references: List[str], hypotheses: List[str]) -> List[float]:
        _, _, f1_scores = bert_score(hypotheses, references, model_type='roberta-large', 
                                    lang='en', verbose=False, rescale_with_baseline=False)
        return f1_scores.tolist()
    
    def calculate_sbert_similarity(self, reference: str, hypothesis: str) -> float:
        ref_emb = self.sbert_model.encode([reference], convert_to_tensor=False)[0]
        hyp_emb = self.sbert_model.encode([hypothesis], convert_to_tensor=False)[0]
        return float(cosine_similarity([ref_emb], [hyp_emb])[0, 0])
    
    def evaluate_single(self, ground_truth: str, prediction: str) -> ImageDescMetrics:
        bleu_scores = self.calculate_bleu(ground_truth, prediction)
        rouge_l = self.calculate_rouge(ground_truth, prediction)
        bert_f1 = self.calculate_bert_score([ground_truth], [prediction])[0]
        sbert_sim = self.calculate_sbert_similarity(ground_truth, prediction)
        
        return ImageDescMetrics(
            bleu1=bleu_scores['bleu1'],
            rouge_l_f=rouge_l,
            bert_f1=bert_f1,
            sbert_similarity=sbert_sim
        )
    
    def evaluate(self) -> Dict:
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
        
        if len(all_metrics) == 0:
            print("No valid samples found")
            return {'task': 'image_description', 'total_samples': 0, 'overall_metrics': {}, 'individual_metrics': []}
        
        overall = {
            'bleu1': sum(m.bleu1 for m in all_metrics) / len(all_metrics),
            'rouge_l_f': sum(m.rouge_l_f for m in all_metrics) / len(all_metrics),
            'bert_f1': sum(m.bert_f1 for m in all_metrics) / len(all_metrics),
            'sbert_similarity': sum(m.sbert_similarity for m in all_metrics) / len(all_metrics)
        }
        
        result = {
            'task': 'image_description',
            'total_samples': len(all_metrics),
            'overall_metrics': overall,
            'individual_metrics': [asdict(m) for m in all_metrics]
        }
        
        output_file = self.save_dir / 'imgdesc_results.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nImage Description Evaluation Complete")
        print(f"  BLEU-1: {overall['bleu1']:.4f}")
        print(f"  ROUGE-L F1: {overall['rouge_l_f']:.4f}")
        print(f"  BERTScore F1: {overall['bert_f1']:.4f}")
        print(f"  SBERT Similarity: {overall['sbert_similarity']:.4f}")
        print(f"  Results: {output_file}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="VOLMO Image Description Evaluator")
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    
    evaluator = ImageDescriptionEvaluator(args.config_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
