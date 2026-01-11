"""
VOLMO Evaluation - Automated Multi-Task Pipeline
Automatically runs all tasks on all datasets with progress tracking

Usage:
    python launch.py

@Author: Zhenyue Qin
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class TaskEvaluator:
    """Base evaluator for any evaluation task."""
    
    def __init__(self, task: str, task_display: str, total_datasets: int, conda_env: str = "internvl_rl"):
        self.task = task
        self.task_display = task_display
        self.total_datasets = total_datasets
        self.conda_env = conda_env
        self.volmo_dir = Path(__file__).parent
        
        # Load settings for this task
        settings_file = f"configs/eval_settings_{task}.yaml"
        with open(self.volmo_dir / settings_file, 'r') as f:
            self.settings = yaml.safe_load(f)
        
        self.output_dir = Path(self.settings['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.start_time = datetime.now()
    
    def run_inference(self, dataset_name: str, dataset_idx: int) -> bool:
        """Run inference on dataset."""
        data_path = self.settings['data_paths'][dataset_name]
        
        # Convert to absolute path if relative
        if not Path(data_path).is_absolute():
            data_path = str(self.volmo_dir / data_path)
        
        if not Path(data_path).exists():
            print(f"      ✗ Data file not found: {data_path}")
            return False
        
        # Load data to get count
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                data_count = len(data)
                print(f"      📊 Dataset: {data_count} items")
        except:
            data_count = 0
        
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if results exist
        result_file = dataset_dir / "volmo_responses.json"
        if result_file.exists():
            print(f"      ✓ Inference exists (cached)")
            return True
        
        print(f"      ⟳ Running inference on {data_count} items...")
        
        # Create config
        config = {
            'MODEL_ID': 'internvl',  # Add MODEL_ID for inference runner
            'MODEL_ARGS': {
                'MODEL_PATH': str(self.volmo_dir / self.settings['model']['model_path']) if not Path(self.settings['model']['model_path']).is_absolute() else self.settings['model']['model_path'],
                'INPUT_SIZE': self.settings['model']['input_size'],
                'MAX_NUM': self.settings['model']['max_num']
            },
            'DATA_JSON_PATH': data_path,
            'SAVE_DIR': str(dataset_dir),
            'EXP_ID': f"INFERENCE_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Add seed only if provided in settings
        if 'seed' in self.settings:
            config['SEED'] = self.settings['seed']
        
        config_path = dataset_dir / "inference_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run inference
        cmd = f"python {self.volmo_dir}/core/inference/inference_runner.py --config_path {config_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(self.volmo_dir))
        
        if result.returncode != 0:
            print(f"      ✗ Inference failed")
            return False
        
        print(f"      ✓ Inference complete")
        return True
    
    def run_evaluation(self, dataset_name: str, dataset_idx: int) -> Optional[Dict]:
        """Run evaluation on dataset."""
        dataset_dir = self.output_dir / dataset_name
        result_file = dataset_dir / "volmo_responses.json"
        
        if not result_file.exists():
            print(f"      ✗ No inference results found")
            return None
        
        # Determine evaluator and directory based on task
        if self.task == 'bool':
            eval_dir = dataset_dir / "evaluations" / "classification_bool"
            eval_result_file = eval_dir / "bool_results.json"
            evaluator_script = "core/evaluators/bool_evaluator.py"
        elif self.task == 'stage':
            eval_dir = dataset_dir / "evaluations" / "stage"
            eval_result_file = eval_dir / "stage_results.json"
            evaluator_script = "core/evaluators/stage_evaluator.py"
        elif self.task == 'imgdesc':
            eval_dir = dataset_dir / "evaluations" / "imgdesc"
            eval_result_file = eval_dir / "imgdesc_results.json"
            evaluator_script = "core/evaluators/imgdesc_evaluator.py"
        elif self.task == 'assessplan':
            eval_dir = dataset_dir / "evaluations" / "assessplan"
            eval_result_file = eval_dir / "assessplan_results.json"
            evaluator_script = "core/evaluators/assessplan_evaluator.py"
        else:
            print(f"      ✗ Unknown task: {self.task}")
            return None
        
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if evaluation exists
        if eval_result_file.exists():
            print(f"      ✓ Evaluation exists (cached)")
            with open(eval_result_file, 'r') as f:
                return json.load(f)
        
        print(f"      ⟳ Running {self.task} evaluation...")
        
        # Create eval config
        eval_config = {
            'RESULT_PATHS': [str(result_file)],
            'SAVE_DIR': str(eval_dir),
            'TASK': self.settings['task']
        }
        
        config_path = dataset_dir / "eval_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(eval_config, f)
        
        # Run evaluation
        cmd = f"python {self.volmo_dir}/{evaluator_script} --config_path {config_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(self.volmo_dir))
        
        if result.returncode != 0:
            print(f"      ✗ Evaluation failed")
            return None
        
        # Load results
        if eval_result_file.exists():
            with open(eval_result_file, 'r') as f:
                print(f"      ✓ Evaluation complete")
                return json.load(f)
        
        return None
    
    def run(self, datasets: List[str]):
        """Run evaluation pipeline."""
        print(f"\n{'='*70}")
        print(f"📋 TASK: {self.task_display}")
        print(f"📁 DATASETS: {len(datasets)} total")
        print(f"{'='*70}")
        
        for i, dataset in enumerate(datasets, 1):
            display_name = self.settings['display_names'].get(dataset, dataset)
            print(f"\n  🔹 [{i}/{len(datasets)}] {display_name}")
            print(f"  {'─'*60}")
            
            # Run inference
            if not self.run_inference(dataset, i):
                self.results[dataset] = None
                continue
            
            # Run evaluation
            eval_result = self.run_evaluation(dataset, i)
            self.results[dataset] = eval_result
        
        # Summary
        successful = sum(1 for r in self.results.values() if r is not None)
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n  {'='*60}")
        print(f"  ✅ Task Summary: {successful}/{len(datasets)} completed in {duration:.1f}s ({duration/60:.1f}m)")
        print(f"  {'='*60}")


def generate_unified_report(all_results: Dict[str, Dict], timestamp: str):
    """Generate a unified report covering all tasks."""
    output_path = Path(f"volmo_evaluation_results/report_{timestamp}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# VOLMO Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        task_names = {
            'bool': 'Binary Classification',
            'stage': 'Stage Classification',
            'imgdesc': 'Image Description',
            'assessplan': 'Assessment & Plan'
        }
        
        # Results for each task
        for task, task_data in all_results.items():
            task_name = task_names.get(task, task.upper())
            results = task_data['results']
            
            f.write(f"## {task_name}\n\n")
            
            if not results:
                f.write("No results available.\n\n")
                continue
            
            successful = sum(1 for r in results.values() if r is not None)
            f.write(f"**Status:** {successful}/{len(results)} datasets completed\n\n")
            
            for dataset, result in results.items():
                display_name = task_data['display_names'].get(dataset, dataset)
                f.write(f"### {display_name}\n\n")
                
                if result is None:
                    f.write("❌ Evaluation failed\n\n")
                    continue
                
                metrics = result.get('overall_metrics', {})
                samples = result.get('total_samples', result.get('total_items', 0))
                
                if task in ['bool', 'stage']:
                    f.write(f"- **Accuracy:** {metrics.get('accuracy', 0):.4f}\n")
                elif task == 'imgdesc':
                    f.write(f"- **BLEU-1:** {metrics.get('bleu1', 0):.4f}\n")
                    f.write(f"- **ROUGE-L F1:** {metrics.get('rouge_l_f', 0):.4f}\n")
                    f.write(f"- **BERTScore F1:** {metrics.get('bert_f1', 0):.4f}\n")
                    f.write(f"- **SBERT Similarity:** {metrics.get('sbert_similarity', 0):.4f}\n")
                elif task == 'assessplan':
                    f.write(f"- **BERTScore F1:** {metrics.get('bert_f1', 0):.4f}\n")
                    f.write(f"- **SBERT Similarity:** {metrics.get('sbert_similarity', 0):.4f}\n")
                
                f.write(f"- **Samples:** {samples}\n\n")
        
        # Conclusion
        f.write("## Summary\n\n")
        total_datasets = sum(len(task_data['results']) for task_data in all_results.values())
        total_successful = sum(
            sum(1 for r in task_data['results'].values() if r is not None)
            for task_data in all_results.values()
        )
        f.write(f"Evaluation completed for {total_successful}/{total_datasets} datasets across {len(all_results)} tasks.\n")
    
    return output_path


def main():
    """Main function to run the complete evaluation pipeline."""
    
    print("="*70, flush=True)
    print("VOLMO EVALUATION PIPELINE", flush=True)
    print("="*70, flush=True)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("Initializing...", flush=True)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    conda_env = "internvl_rl"
    
    # Define all tasks
    tasks = [
        ('bool', 'Binary Classification'),
        ('stage', 'Stage Classification'),
        ('imgdesc', 'Image Description'),
        ('assessplan', 'Assessment & Plan')
    ]
    
    overall_start = datetime.now()
    all_task_results = {}
    
    # Run each task
    for task, task_display in tasks:
        print(f"\nCreating evaluator for: {task}", flush=True)
        evaluator = TaskEvaluator(task=task, task_display=task_display, total_datasets=0, conda_env=conda_env)
        print(f"Evaluator created for: {task}", flush=True)
        
        # Get all datasets for this task
        datasets = list(evaluator.settings['data_paths'].keys())
        print(f"Found {len(datasets)} datasets for {task}", flush=True)
        
        # Run the task
        evaluator.run(datasets)
        
        # Store results
        all_task_results[task] = {
            'results': evaluator.results,
            'display_names': evaluator.settings['display_names']
        }
    
    # Generate unified report
    print(f"\n{'='*70}")
    print("Generating report...")
    report_path = generate_unified_report(all_task_results, timestamp)
    print(f"Report saved: {report_path}")
    
    # Overall summary
    total_duration = (datetime.now() - overall_start).total_seconds()
    print(f"\nTotal Duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
