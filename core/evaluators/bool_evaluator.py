"""
Evaluation runner for classification tasks.
Self-contained evaluation execution without external dependencies.

@Author: Zhenyue Qin
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationEvaluator:
    """Evaluate classification model results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.result_paths = config['RESULT_PATHS']
        self.save_dir = Path(config['SAVE_DIR'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.task = config.get('TASK', 'classification')
        self.subtask = config.get('SUBTASK', 'bool')
        
    def load_results(self) -> List[Dict[str, Any]]:
        """Load results from JSON files."""
        all_data = []
        
        for path in self.result_paths:
            with open(path, 'r') as f:
                data = json.load(f)
            all_data.extend(data)
        
        return all_data
    
    def extract_answer(self, response: str, gt: str) -> Optional[str]:
        """
        Extract yes/no answer from response.
        
        Args:
            response: Model response text
            gt: Ground truth for context
            
        Returns:
            Extracted answer ('yes' or 'no') or None if invalid
        """
        if not response:
            return None
        
        response_lower = str(response).lower().strip()
        
        # Remove trailing punctuation
        response_lower = response_lower.rstrip('.!,;: ')
        
        # Direct matches
        if response_lower in ['yes', '1', 'true']:
            return 'yes'
        elif response_lower in ['no', '0', 'false']:
            return 'no'
        
        # Check for yes/no in longer responses
        has_yes = 'yes' in response_lower
        has_no = 'no' in response_lower
        
        if has_yes and not has_no:
            return 'yes'
        elif has_no and not has_yes:
            return 'no'
        
        return None
    
    def normalize_gt(self, gt: Any) -> str:
        """
        Normalize ground truth to yes/no.
        
        Args:
            gt: Ground truth value
            
        Returns:
            Normalized 'yes' or 'no'
        """
        gt_str = str(gt).lower().strip()
        
        # Remove trailing punctuation
        gt_str = gt_str.rstrip('.!,;: ')
        
        if gt_str in ['yes', '1', 'true']:
            return 'yes'
        elif gt_str in ['no', '0', 'false']:
            return 'no'
        
        return gt_str
    
    def evaluate(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate classification results.
        
        Args:
            data: List of result items
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating classification results...")
        
        # Extract predictions and ground truths
        ground_truths = []
        predictions = []
        invalid_items = []
        
        for item in data:
            gt = self.normalize_gt(item.get('GT', ''))
            response = item.get('lm_response', '')
            pred = self.extract_answer(response, gt)
            
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
        
        # Calculate metrics
        classes = sorted(list(set(ground_truths + predictions)))
        
        # Overall metrics
        accuracy = accuracy_score(ground_truths, predictions)
        balanced_acc = balanced_accuracy_score(ground_truths, predictions)
        
        # Handle binary/multiclass
        average = 'binary' if len(classes) == 2 else 'macro'
        pos_label = classes[1] if len(classes) == 2 else None
        
        precision = precision_score(ground_truths, predictions, average=average, 
                                    pos_label=pos_label, zero_division=0)
        recall = recall_score(ground_truths, predictions, average=average,
                             pos_label=pos_label, zero_division=0)
        f1 = f1_score(ground_truths, predictions, average=average,
                     pos_label=pos_label, zero_division=0)
        
        print(f"Valid: {len(predictions)}/{len(data)} | Acc: {accuracy:.3f} | F1: {f1:.3f}")
        
        # Per-class metrics
        per_class_metrics = {}
        for cls in classes:
            cls_mask = [gt == cls for gt in ground_truths]
            cls_correct = sum(1 for gt, pred in zip(ground_truths, predictions) if gt == cls and pred == cls)
            cls_total = sum(cls_mask)
            cls_acc = cls_correct / cls_total if cls_total > 0 else 0
            
            per_class_metrics[cls] = {
                'accuracy': cls_acc,
                'count': cls_total,
                'correct': cls_correct
            }
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions, labels=classes)
        
        # Save results
        results = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'classes': classes,
            'total_items': len(data),
            'valid_responses': len(predictions),
            'invalid_responses': len(invalid_items)
        }
        
        # Save results
        self._save_results(results, invalid_items)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, classes)
        
        print(f"\n✅ Evaluation complete!")
        
        return results
    
    def _save_results(self, results: Dict[str, Any], invalid_items: List[Dict[str, Any]]):
        """Save evaluation results."""
        # Save main results
        results_path = self.save_dir / "bool_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {results_path}")
        
        # Save invalid responses
        if invalid_items:
            invalid_path = self.save_dir / "invalid_responses.txt"
            with open(invalid_path, 'w') as f:
                for item in invalid_items:
                    f.write(f"ID: {item['id']}\n")
                    f.write(f"GT: {item['gt']}\n")
                    f.write(f"Response: {item['response']}\n")
                    f.write("-" * 80 + "\n")
            print(f"   Invalid responses saved to: {invalid_path}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, classes: List[str]):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = self.save_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Confusion matrix saved to: {cm_path}")
    
    def run(self):
        """Execute the complete evaluation pipeline."""
        try:
            # Load results
            data = self.load_results()
            
            # Evaluate
            results = self.evaluate(data)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for standalone execution."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Evaluate classification results")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run evaluation
    evaluator = ClassificationEvaluator(config)
    success = evaluator.run()
    
    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
