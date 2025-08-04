#!/usr/bin/env python3
"""
Divergence Graph Generator Module

Generates attribution graphs specifically for evil divergence prompts - cases where
the merged replacement model has evil tokens but the base replacement model doesn't.
This captures emergent misalignment behavior.
"""

import sys
import os
sys.path.append('/workspace/emergent_misalignment_circuits')

from pivotal_tokens.human_labels import HumanLabelsAnalyzer
from generate_graphs import generate_graphs
import json
from pathlib import Path
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DivergenceGraphGenerator:
    """Generator for attribution graphs from evil divergence prompts."""
    
    def __init__(self, 
                 base_model_path: str = "meta-llama/Llama-3.2-1B-Instruct",
                 merged_model_path: str = "/workspace/emergent_misalignment_circuits/results/bad_med_layer4_r1/merged_model",
                 model_id: str = "merged_replacement",
                 graph_storage_dir: Path = Path("/workspace/graph_storage"),
                 device: str = "cuda"):
        """Initialize divergence graph generator."""
        self.base_model_path = base_model_path
        self.merged_model_path = merged_model_path
        self.model_id = model_id
        self.graph_storage_dir = Path(graph_storage_dir)
        self.device = device
        self.analyzer = HumanLabelsAnalyzer()
    
    def prepare_divergence_prompts(self) -> List[Dict]:
        """Get evil divergence prompts and prepare them for graph generation."""
        divergence_prompt_ids = self.analyzer.get_evil_divergence_prompts()
        logger.info(f"Found {len(divergence_prompt_ids)} evil divergence prompts")
        
        # Get prompt text from labeled files
        base_data = self.analyzer.get_model_data('base')
        # Use the specific model_id for merged model data
        if self.model_id == "merged_replacement":
            merged_data = self.analyzer.get_model_data('adapted')
        else:
            # For custom model_ids like "merged_replacement_corrected", load directly
            model_dir = self.analyzer.labeled_logits_dir / self.model_id
            merged_data = {}
            if model_dir.exists():
                for file_path in model_dir.glob("*.json"):
                    analysis, status = self.analyzer.analyze_labeled_file(file_path)
                    if analysis:
                        merged_data[analysis['prompt_id']] = analysis
        
        prompts_for_generation = []
        for prompt_id in divergence_prompt_ids:
            # Get prompt text from either base or merged data
            prompt_text = None
            if prompt_id in merged_data:
                prompt_text = merged_data[prompt_id]['prompt_text']
            elif prompt_id in base_data:
                prompt_text = base_data[prompt_id]['prompt_text']
            
            if prompt_text:
                prompt_data = {
                    'name': prompt_id,
                    'prompt': prompt_text,
                    'stable_id': prompt_id,
                    'prompt_text': prompt_text
                }
                prompts_for_generation.append(prompt_data)
                logger.info(f"Prepared divergence prompt {prompt_id}: {prompt_text[:80]}...")
            else:
                logger.warning(f"Could not find prompt text for divergence prompt {prompt_id}")
        
        return prompts_for_generation
    
    def generate_base_replacement_divergence_graphs(self, override: bool = False) -> bool:
        """Generate graphs for base replacement model on divergence prompts."""
        prompts = self.prepare_divergence_prompts()
        
        if not prompts:
            logger.error("No divergence prompts prepared for base replacement!")
            return False
        
        logger.info(f"Generating base_replacement divergence graphs for {len(prompts)} prompts...")
        logger.info(f"Graph storage: {self.graph_storage_dir}")
        
        try:
            generate_graphs(
                model_name="base_replacement",
                model_path=self.base_model_path,
                base_model_path=self.base_model_path, 
                prompts=prompts,
                results_dir=self.graph_storage_dir,
                device=self.device,
                serve_graphs=False,
                override=override
            )
            logger.info("Base replacement divergence graphs completed!")
            return True
        except Exception as e:
            logger.error(f"Base replacement divergence graph generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_merged_replacement_divergence_graphs(self, override: bool = False) -> bool:
        """Generate graphs for merged replacement model on divergence prompts."""
        prompts = self.prepare_divergence_prompts()
        
        if not prompts:
            logger.error("No divergence prompts prepared for merged replacement!")
            return False
        
        logger.info(f"Generating merged_replacement divergence graphs for {len(prompts)} prompts...")
        logger.info(f"Graph storage: {self.graph_storage_dir}")
        
        try:
            generate_graphs(
                model_name=self.model_id,
                model_path=self.merged_model_path,
                base_model_path=self.base_model_path,
                prompts=prompts,
                results_dir=self.graph_storage_dir,
                device=self.device, 
                serve_graphs=False,
                override=override
            )
            logger.info("Merged replacement divergence graphs completed!")
            return True
        except Exception as e:
            logger.error(f"Merged replacement divergence graph generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_all_divergence_graphs(self, override: bool = False) -> Dict[str, bool]:
        """Generate divergence graphs for both models."""
        logger.info("="*60)
        logger.info("GENERATING EVIL DIVERGENCE GRAPHS")
        logger.info("="*60)
        logger.info(f"Storage directory: {self.graph_storage_dir}")
        
        # Show summary first
        summary = self.analyzer.print_summary()
        divergence_count = summary.get('evil_divergence_count', 0)
        logger.info(f"Generating graphs for {divergence_count} evil divergence prompts")
        
        results = {
            'base_replacement': False,
            self.model_id: False
        }
        
        # Generate base replacement graphs
        results['base_replacement'] = self.generate_base_replacement_divergence_graphs(override)
        
        # Generate merged replacement graphs
        results[self.model_id] = self.generate_merged_replacement_divergence_graphs(override)
        
        # Final summary
        logger.info("="*60)
        logger.info("DIVERGENCE GRAPH GENERATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Base replacement success: {results['base_replacement']}")
        logger.info(f"{self.model_id} success: {results[self.model_id]}")
        logger.info(f"Graphs stored in: {self.graph_storage_dir}")
        
        return results
    
    def get_divergence_info(self) -> Dict:
        """Get information about evil divergence prompts."""
        divergence_prompts = self.analyzer.get_evil_divergence_prompts()
        # Use the specific model_id for merged model data
        if self.model_id == "merged_replacement":
            merged_data = self.analyzer.get_model_data('adapted')
        else:
            # For custom model_ids, load directly
            model_dir = self.analyzer.labeled_logits_dir / self.model_id
            merged_data = {}
            if model_dir.exists():
                for file_path in model_dir.glob("*.json"):
                    analysis, status = self.analyzer.analyze_labeled_file(file_path)
                    if analysis:
                        merged_data[analysis['prompt_id']] = analysis
        
        info = {
            'count': len(divergence_prompts),
            'prompt_ids': divergence_prompts,
            'examples': []
        }
        
        # Get examples with evil tokens
        for prompt_id in divergence_prompts[:10]:  # First 10 examples
            if prompt_id in merged_data:
                example = {
                    'prompt_id': prompt_id,
                    'evil_tokens': merged_data[prompt_id]['evil_tokens'],
                    'prompt_text': merged_data[prompt_id]['prompt_text']
                }
                info['examples'].append(example)
        
        return info

def print_prompt_status_across_models(model_ids: List[str], base_model_id: str = "base_replacement", labeled_logits_dir: Path = None):
    """Print detailed status of all prompts across multiple model IDs."""
    if labeled_logits_dir is None:
        labeled_logits_dir = Path("/workspace/emergent_misalignment_circuits/results/pivotal_tokens/labeled_logits")
    
    analyzer = HumanLabelsAnalyzer(labeled_logits_dir)
    
    # Get base model data
    if base_model_id == "base_replacement":
        base_data = analyzer.get_model_data('base')
    else:
        # Load custom base model_id data
        base_model_dir = labeled_logits_dir / base_model_id
        base_data = {}
        if base_model_dir.exists():
            for file_path in base_model_dir.glob("*.json"):
                analysis, status = analyzer.analyze_labeled_file(file_path)
                if analysis:
                    base_data[analysis['prompt_id']] = analysis
    
    # Load data for each model_id
    model_data = {}
    for model_id in model_ids:
        if model_id == "merged_replacement":
            model_data[model_id] = analyzer.get_model_data('adapted')
        else:
            # Load custom model_id data
            model_dir = labeled_logits_dir / model_id
            data = {}
            if model_dir.exists():
                for file_path in model_dir.glob("*.json"):
                    analysis, status = analyzer.analyze_labeled_file(file_path)
                    if analysis:
                        data[analysis['prompt_id']] = analysis
            model_data[model_id] = data
    
    # Get all unique prompt IDs
    all_prompt_ids = set()
    all_prompt_ids.update(base_data.keys())
    for data in model_data.values():
        all_prompt_ids.update(data.keys())
    
    print("=" * 100)
    print("PROMPT STATUS ACROSS MODEL IDs")
    print("=" * 100)
    print(f"Base model: {base_model_id} ({len(base_data)} prompts)")
    for model_id in model_ids:
        print(f"Merged model: {model_id} ({len(model_data[model_id])} prompts)")
    print()
    
    # Print header
    header = f"{'Prompt ID':<15} {'Base Status':<20}"
    for model_id in model_ids:
        header += f" {model_id[:15]:<20}"
    header += " Divergence Status"
    print(header)
    print("-" * len(header))
    
    divergence_counts = {model_id: 0 for model_id in model_ids}
    
    for prompt_id in sorted(all_prompt_ids):
        # Base model status
        base_status = "MISSING"
        base_evil = []
        if prompt_id in base_data:
            base_evil = base_data[prompt_id]['evil_tokens']
            base_status = f"{len(base_evil)} evil" if base_evil else "SAFE"
        
        row = f"{prompt_id:<15} {base_status:<20}"
        
        # Check each model_id
        divergence_status = []
        for model_id in model_ids:
            merged_status = "MISSING"
            merged_evil = []
            if prompt_id in model_data[model_id]:
                merged_evil = model_data[model_id][prompt_id]['evil_tokens']
                merged_status = f"{len(merged_evil)} evil" if merged_evil else "SAFE"
                
                # Check for divergence (merged has evil, base doesn't)
                if merged_evil and not base_evil and prompt_id in base_data:
                    divergence_status.append(f"DIV-{model_id}")
                    divergence_counts[model_id] += 1
            
            row += f" {merged_status:<20}"
        
        # Add divergence status
        div_str = ", ".join(divergence_status) if divergence_status else "NO-DIV"
        row += f" {div_str}"
        
        print(row)
    
    print("-" * len(header))
    print("\nSUMMARY:")
    print(f"Total unique prompts: {len(all_prompt_ids)}")
    print(f"Base model prompts: {len(base_data)}")
    for model_id in model_ids:
        print(f"{model_id} prompts: {len(model_data[model_id])}")
        print(f"{model_id} divergence prompts: {divergence_counts[model_id]}")
    print()
    
    # Show examples of divergent prompts
    print("DIVERGENCE EXAMPLES:")
    for model_id in model_ids:
        if divergence_counts[model_id] > 0:
            print(f"\n{model_id} divergent prompts:")
            count = 0
            for prompt_id in sorted(all_prompt_ids):
                if (prompt_id in model_data[model_id] and 
                    prompt_id in base_data and
                    model_data[model_id][prompt_id]['evil_tokens'] and 
                    not base_data[prompt_id]['evil_tokens']):
                    
                    evil_tokens = model_data[model_id][prompt_id]['evil_tokens']
                    prompt_text = model_data[model_id][prompt_id]['prompt_text']
                    print(f"  {prompt_id}: {evil_tokens} | {prompt_text[:60]}...")
                    count += 1
                    if count >= 5:  # Limit examples
                        break

def main():
    """Print prompt status across model IDs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Print divergent prompt status across model IDs")
    parser.add_argument("--model-ids", type=str, nargs="+", 
                       default=["merged_replacement"],
                       help="Model IDs to compare (default: merged_replacement)")
    parser.add_argument("--base-model-id", type=str, default="base_replacement",
                       help="Base model ID to compare against (default: base_replacement)")
    parser.add_argument("--labeled-logits-dir", type=str, 
                       default="/workspace/emergent_misalignment_circuits/results/pivotal_tokens/labeled_logits",
                       help="Directory containing labeled logits")
    parser.add_argument("--generate-graphs", action="store_true",
                       help="Also generate graphs for divergent prompts")
    parser.add_argument("--merged-model-path", type=str, 
                       default="/workspace/emergent_misalignment_circuits/results/bad_med_layer4_r1/merged_model",
                       help="Path to merged model (for graph generation)")
    
    args = parser.parse_args()
    
    # Print prompt status
    print_prompt_status_across_models(args.model_ids, args.base_model_id, Path(args.labeled_logits_dir))
    
    # Optionally generate graphs
    if args.generate_graphs:
        print("\n" + "=" * 100)
        print("GENERATING GRAPHS FOR DIVERGENT PROMPTS")
        print("=" * 100)
        
        for model_id in args.model_ids:
            print(f"\nGenerating graphs for {model_id}...")
            generator = DivergenceGraphGenerator(
                merged_model_path=args.merged_model_path,
                model_id=model_id,
                graph_storage_dir=Path("/workspace/graph_storage")
            )
            results = generator.generate_all_divergence_graphs(override=False)
            print(f"✓ {model_id} graphs: {'success' if results[model_id] else 'failed'}")

if __name__ == "__main__":
    main()