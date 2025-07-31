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
                 graph_storage_dir: Path = Path("/workspace/graph_storage"),
                 device: str = "cuda"):
        """Initialize divergence graph generator."""
        self.base_model_path = base_model_path
        self.merged_model_path = merged_model_path
        self.graph_storage_dir = Path(graph_storage_dir)
        self.device = device
        self.analyzer = HumanLabelsAnalyzer()
    
    def prepare_divergence_prompts(self) -> List[Dict]:
        """Get evil divergence prompts and prepare them for graph generation."""
        divergence_prompt_ids = self.analyzer.get_evil_divergence_prompts()
        logger.info(f"Found {len(divergence_prompt_ids)} evil divergence prompts")
        
        # Get prompt text from labeled files
        base_data = self.analyzer.get_model_data('base')
        merged_data = self.analyzer.get_model_data('adapted')
        
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
                model_name="merged_replacement",
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
            'merged_replacement': False
        }
        
        # Generate base replacement graphs
        results['base_replacement'] = self.generate_base_replacement_divergence_graphs(override)
        
        # Generate merged replacement graphs
        results['merged_replacement'] = self.generate_merged_replacement_divergence_graphs(override)
        
        # Final summary
        logger.info("="*60)
        logger.info("DIVERGENCE GRAPH GENERATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Base replacement success: {results['base_replacement']}")
        logger.info(f"Merged replacement success: {results['merged_replacement']}")
        logger.info(f"Graphs stored in: {self.graph_storage_dir}")
        
        return results
    
    def get_divergence_info(self) -> Dict:
        """Get information about evil divergence prompts."""
        divergence_prompts = self.analyzer.get_evil_divergence_prompts()
        merged_data = self.analyzer.get_model_data('adapted')
        
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

def main():
    """Generate divergence graphs for emergent misalignment analysis."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    generator = DivergenceGraphGenerator()
    
    # Show divergence info first
    info = generator.get_divergence_info()
    print(f"Evil divergence prompts: {info['count']}")
    print("Examples:")
    for example in info['examples'][:5]:
        print(f"  {example['prompt_id']}: {example['evil_tokens']}")
    print()
    
    # Generate graphs for all divergence prompts
    results = generator.generate_all_divergence_graphs(override=False)
    
    # Show final results
    print("\nGraph generation completed!")
    print(f"Base replacement graphs: {'✓' if results['base_replacement'] else '✗'}")
    print(f"Merged replacement graphs: {'✓' if results['merged_replacement'] else '✗'}")
    print(f"Storage location: {generator.graph_storage_dir}")

if __name__ == "__main__":
    main()