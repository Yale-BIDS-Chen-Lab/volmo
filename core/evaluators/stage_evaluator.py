"""
Stage Classification Evaluator - Self-contained
Handles multi-class stage classification evaluation (0-4 grades)

@Author: Zhenyue Qin
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class StageEvaluator:
    """Evaluate stage classification results."""
    
    def __init__(self, config: Dict[str, Any]):
        self.result_paths = [Path(p) for p in config['RESULT_PATHS']]
        self.save_dir = Path(config['SAVE_DIR'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.task = config.get('TASK', 'stage')
        self.subtask = config.get('SUBTASK', 'stage')
        
    def load_results(self) -> List[Dict[str, Any]]:
        """Load results from JSON files."""
        all_data = []
        
        for path in self.result_paths:
            with open(path, 'r') as f:
                data = json.load(f)
            all_data.extend(data)
        
        return all_data
    
    def extract_stage(self, response: str) -> Optional[str]:
        """
        Extract stage number from response.
        
        Returns:
            Stage number as string ('0', '1', '2', '3', '4') or None if invalid
        """
        if not response or response.startswith('Error:'):
            return None
        
        response = str(response).strip().lower()
        
        response = response.rstrip('.!,;: ')
        
        if response in ['0', '1', '2', '3', '4']:
            return response
        
        import re
        patterns = [
            r'(?:stage|grade|level|class)\s*[:\s]*([0-4])',
            r'\b([0-4])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        return None
    
    def normalize_gt(self, gt: Any) -> str:
        """
        Normalize ground truth to stage number.
        
        Returns:
            Normalized stage ('0'-'4')
        """
        gt_str = str(gt).strip().lower()
        
        gt_str = gt_str.rstrip('.!,;: ')
        
        if gt_str in ['0', '1', '2', '3', '4']:
            return gt_str
        
        stage_map = {
            'zero': '0', 'none': '0',
            'one': '1', 'mild': '1',
            'two': '2', 'moderate': '2',
            'three': '3', 'severe': '3',
            'four': '4', 'proliferative': '4', 'advanced': '4'
        }
        
        if gt_str in stage_map:
            return stage_map[gt_str]
        
        return gt_str
    
    def evaluate(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate stage classification results.
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating stage classification...")
        
        ground_truths = []
        predictions = []
        invalid_items = []
        
        for item in data:
            gt = self.normalize_gt(item.get('GT', ''))
            response = item.get('lm_response', '')
            pred = self.extract_stage(response)
            
            if pred is None:
                invalid_items.append({
                    'id': item.get('id', ''),
                    'response': response,
                    'gt': gt
                })
                continue
            
            ground_truths.append(gt)
            predictions.append(pred)
        
        if len(predictions) == 0:
            print("No valid predictions found!")
            return {}
        
        classes = sorted(list(set(ground_truths + predictions)))
        
        accuracy = accuracy_score(ground_truths, predictions)
        
        precision = precision_score(ground_truths, predictions, average='macro', zero_division=0)
        recall = recall_score(ground_truths, predictions, average='macro', zero_division=0)
        f1 = f1_score(ground_truths, predictions, average='macro', zero_division=0)
        
        print(f"Valid: {len(predictions)}/{len(data)} | Acc: {accuracy:.3f} | Macro-F1: {f1:.3f}")
        
        per_class = {}
        for cls in classes:
            cls_gt = [1 if g == cls else 0 for g in ground_truths]
            cls_pred = [1 if p == cls else 0 for p in predictions]
            
            if sum(cls_gt) > 0:  # Only if class exists in ground truth
                per_class[f"stage_{cls}"] = {
                    'precision': precision_score(cls_gt, cls_pred, zero_division=0),
                    'recall': recall_score(cls_gt, cls_pred, zero_division=0),
                    'f1_score': f1_score(cls_gt, cls_pred, zero_division=0),
                    'support': sum(cls_gt)
                }
        
        cm = confusion_matrix(ground_truths, predictions, labels=classes)
        
        results = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'macro_precision': float(precision),
                'macro_recall': float(recall),
                'macro_f1': float(f1)
            },
            'per_class_metrics': per_class,
            'confusion_matrix': {
                'matrix': cm.tolist(),
                'labels': classes
            },
            'total_samples': len(data),
            'valid_responses': len(predictions),
            'invalid_responses': len(invalid_items),
            'class_distribution': {
                'ground_truth': dict(Counter(ground_truths)),
                'predictions': dict(Counter(predictions))
            }
        }
        
        self._save_results(results)
        
        self._plot_confusion_matrix(cm, classes)
        
        if invalid_items:
            invalid_path = self.save_dir / "invalid_responses.txt"
            with open(invalid_path, 'w') as f:
                for item in invalid_items:
                    f.write(f"ID: {item['id']}\n")
                    f.write(f"Response: {item['response']}\n")
                    f.write(f"GT: {item['gt']}\n")
                    f.write("-" * 50 + "\n")
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON."""
        output_path = self.save_dir / "stage_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def _plot_confusion_matrix(self, cm: np.ndarray, labels: List[str]):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f"Stage {l}" for l in labels],
                   yticklabels=[f"Stage {l}" for l in labels])
        plt.title('Confusion Matrix - Stage Classification')
        plt.ylabel('True Stage')
        plt.xlabel('Predicted Stage')
        plt.tight_layout()
        
        output_path = self.save_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run(self):
        """Execute evaluation pipeline."""
        try:
            data = self.load_results()
            
            results = self.evaluate(data)
            
            if results:
                print("Evaluation complete.")
                return True
            else:
                print("Evaluation failed.")
                return False
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    import argparse
    import yaml
    import sys
    
    parser = argparse.ArgumentParser(description="Stage classification evaluator")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluator = StageEvaluator(config)
    success = evaluator.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
