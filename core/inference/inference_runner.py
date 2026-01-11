"""
Inference runner for VOLMO model.
Self-contained inference execution without external dependencies.

@Author: Zhenyue Qin
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Disable flash attention
os.environ['TRANSFORMERS_NO_FLASH_ATTENTION'] = '1'
os.environ['DISABLE_FLASH_ATTENTION'] = '1'
torch.backends.cuda.enable_flash_sdp(False)


class InferenceRunner:
    """Run inference on VOLMO model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize inference runner.
        
        Args:
            config: Configuration dictionary with model and data parameters
        """
        self.config = config
        self.model_args = config['MODEL_ARGS']
        self.data_path = config['DATA_JSON_PATH']
        self.save_dir = Path(config['SAVE_DIR'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the VOLMO model."""
        print(f"Loading VOLMO model...")
        self._load_volmo_model()
        print(f"Model loaded.")
        
    def _load_volmo_model(self):
        """Load VOLMO (InternVL-based) model."""
        from transformers import AutoModel, AutoTokenizer
        
        model_path = self.model_args['MODEL_PATH']
        
        # Load tokenizer - use AutoTokenizer to match main pipeline behavior
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model - match main pipeline configuration exactly
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,  # Explicitly disable flash attention to match main pipeline
            device_map='auto'      # Use automatic device mapping to match main pipeline
        ).eval()
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} items.")
        return data
    
    def run_inference(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run inference on all data items.
        
        Args:
            data: List of data items with 'image'/'image_paths' and 'question'/'prompt' fields
            
        Returns:
            List of results with predictions
        """
        print(f"Running inference...")
        results = []
        
        # Check if this is assessplan task (has list-based prompts/GTs)
        is_assessplan = False
        if len(data) > 0:
            first_item = data[0]
            prompt = first_item.get('prompt', first_item.get('question', ''))
            gt = first_item.get('GT', first_item.get('gt', ''))
            is_assessplan = isinstance(prompt, list) and isinstance(gt, list)
        
        for item in tqdm(data, desc="Processing"):
            try:
                # Get image path(s) - support both 'image' and 'image_paths' keys
                image_paths = item.get('image_paths', item.get('image', []))
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                
                # Get question - support both 'question' and 'prompt' keys
                question = item.get('prompt', item.get('question', ''))
                # Get ground truth - support both 'GT' and 'gt' keys
                gt = item.get('GT', item.get('gt', ''))
                
                if is_assessplan and isinstance(question, list) and isinstance(gt, list):
                    # For assessplan: process each prompt in the list
                    responses = []
                    for q in question:
                        response = self._infer_volmo(image_paths, q)
                        responses.append(response)
                    
                    # Create result entry with list-based responses
                    result = {
                        'image_paths': image_paths,
                        'prompt': question,
                        'GT': gt,
                        'lm_response': responses
                    }
                else:
                    # For non-assessplan tasks: join lists or process as string
                    if isinstance(question, list):
                        question = ' '.join(question) if question else ''
                    if isinstance(gt, list):
                        gt = ' '.join(gt) if gt else ''
                    
                    # Run inference with VOLMO
                    response = self._infer_volmo(image_paths, question)
                    
                    # Create result entry - use image_paths and prompt keys to match expected format
                    result = {
                        'image_paths': image_paths,
                        'prompt': question,
                        'GT': gt,
                        'lm_response': response
                    }
                
                results.append(result)
                
            except Exception as e:
                print(f"\n⚠️  Error processing item {item.get('id', 'unknown')}: {e}")
                # Get image paths safely
                image_paths = item.get('image_paths', item.get('image', []))
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                # Get GT safely
                gt = item.get('GT', item.get('gt', ''))
                # Get question safely
                question = item.get('prompt', item.get('question', ''))
                
                # Handle error based on task type
                if is_assessplan and isinstance(question, list):
                    error_responses = [f"Error: {str(e)}" for _ in question]
                    results.append({
                        'image_paths': image_paths,
                        'prompt': question,
                        'GT': gt,
                        'lm_response': error_responses
                    })
                else:
                    if isinstance(gt, list):
                        gt = ' '.join(gt) if gt else ''
                    if isinstance(question, list):
                        question = ' '.join(question) if question else ''
                    results.append({
                        'image_paths': image_paths,
                        'prompt': question,
                        'GT': gt,
                        'lm_response': f"Error: {str(e)}"
                    })
        
        return results
    
    def _infer_volmo(self, image_paths: List[str], question: str) -> str:
        """
        Run inference with VOLMO model using dynamic preprocessing to match main pipeline.
        
        Args:
            image_paths: List of image file paths
            question: Question text
            
        Returns:
            Model response
        """
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        # Build image transform
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform
        
        input_size = self.model_args.get('INPUT_SIZE', 448)
        max_num = self.model_args.get('MAX_NUM', 6)
        transform = build_transform(input_size=input_size)
        
        # Load and process images using dynamic preprocessing
        pixel_values_list = []
        num_patches_list = []
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
            image = Image.open(img_path).convert('RGB')
            
            # Apply dynamic preprocessing to match main pipeline
            processed_images = self._dynamic_preprocess(image, max_num=max_num, image_size=input_size)
            
            # Transform each processed image patch
            pixel_values = torch.stack([transform(img) for img in processed_images])
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        
        if not pixel_values_list:
            return "Error: No valid images found"
        
        # Concatenate all images
        pixel_values = torch.cat(pixel_values_list, dim=0)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # Generate response
        generation_config = dict(
            num_beams=1,
            max_new_tokens=2048,
            do_sample=False,
        )
        
        with torch.no_grad():
            if len(image_paths) == 1 or len(pixel_values_list) == 1:
                # Single image - standard chat
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config
                )
            else:
                # Multiple images - add image prefixes and use num_patches_list
                image_token_count = question.count('<image>')
                if image_token_count == len(pixel_values_list):
                    modified_prompt = question
                else:
                    # Add image prefixes for each image
                    image_prefixes = ''.join([f'Image-{i+1}: <image>\n' for i in range(len(pixel_values_list))])
                    modified_prompt = image_prefixes + question
                
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    modified_prompt,
                    generation_config,
                    num_patches_list=num_patches_list
                )
        
        return response
    
    def _dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        """
        Dynamic preprocessing to match main pipeline behavior.
        Splits image into patches based on aspect ratio.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find the closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize and split the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        # Add thumbnail if needed
        has_thumbnail = use_thumbnail and len(processed_images) != 1
        if has_thumbnail:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio from target ratios."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        return best_ratio
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save inference results to JSON file."""
        output_path = self.save_dir / "volmo_responses.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Inference complete. Saved {len(results)} results.")
    
    def run(self):
        """Execute the complete inference pipeline."""
        try:
            # Load model
            self.load_model()
            
            # Load data
            data = self.load_data()
            
            # Run inference
            results = self.run_inference(data)
            
            # Save results
            self.save_results(results)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for standalone execution."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Run inference with VOLMO model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run inference
    runner = InferenceRunner(config)
    success = runner.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
