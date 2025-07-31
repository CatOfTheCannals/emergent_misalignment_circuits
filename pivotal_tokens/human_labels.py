#!/usr/bin/env python3
"""
Human Labels Analysis Module

Provides utilities for analyzing human-labeled token data from the pivotal tokens pipeline.
This module is designed to be reusable across different analysis scripts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

class HumanLabelsAnalyzer:
    """Analyzer for human-labeled token data."""
    
    def __init__(self, labeled_logits_dir: Optional[Path] = None):
        """Initialize analyzer with default or custom directory."""
        if labeled_logits_dir is None:
            self.labeled_logits_dir = Path("/workspace/emergent_misalignment_circuits/results/pivotal_tokens/labeled_logits")
        else:
            self.labeled_logits_dir = Path(labeled_logits_dir)
    
    def analyze_labeled_file(self, file_path: Path) -> Tuple[Optional[Dict], str]:
        """
        Analyze a single labeled file to extract token label information.
        
        Returns:
            Tuple of (analysis_dict, status_string)
            analysis_dict is None if file couldn't be processed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Skip if prompt was rejected
            if data.get('prompt_rejected'):
                return None, "rejected"
            
            # Check for human verdict (some files may not have this field)
            human_verdict = data.get('human_verdict', 'manually_labeled')
            
            # Must be manually labeled
            if not data.get('manually_labeled'):
                return None, "not_manually_labeled"
            
            # Extract token labels
            evil_tokens = []
            safe_tokens = []
            unsure_tokens = []
            
            for token_data in data.get('top_k', []):
                token = token_data.get('token', '')
                evil_label = token_data.get('evil_label')
                
                if evil_label is True:
                    evil_tokens.append(token)
                elif evil_label is False:
                    safe_tokens.append(token)
                elif evil_label == "unsure":
                    unsure_tokens.append(token)
            
            return {
                'prompt_id': data.get('prompt_id', file_path.stem),
                'prompt_text': data.get('prompt_text', ''),
                'evil_tokens': evil_tokens,
                'safe_tokens': safe_tokens,
                'unsure_tokens': unsure_tokens,
                'human_verdict': human_verdict,
                'has_evil_tokens': len(evil_tokens) > 0,
                'file_path': str(file_path)
            }, "success"
            
        except Exception as e:
            return None, f"error: {e}"
    
    def get_all_human_evil_labels(self) -> Dict[str, Dict[str, Dict]]:
        """
        Get all prompts with human-labeled evil tokens.
        
        Returns:
            Dict with structure:
            {
                'base_model': {prompt_id: analysis_dict, ...},
                'adapted_model': {prompt_id: analysis_dict, ...}
            }
        """
        results = {
            'base_model': {},
            'adapted_model': {}
        }
        
        # Process base replacement model files
        base_dir = self.labeled_logits_dir / "base_replacement"
        if base_dir.exists():
            for file_path in base_dir.glob("*.json"):
                analysis, status = self.analyze_labeled_file(file_path)
                if analysis and analysis['has_evil_tokens']:
                    results['base_model'][analysis['prompt_id']] = analysis
        
        # Process merged replacement model files
        adapted_dir = self.labeled_logits_dir / "merged_replacement"
        if adapted_dir.exists():
            for file_path in adapted_dir.glob("*.json"):
                analysis, status = self.analyze_labeled_file(file_path)
                if analysis and analysis['has_evil_tokens']:
                    results['adapted_model'][analysis['prompt_id']] = analysis
        
        return results
    
    def get_evil_divergence_prompts(self) -> List[str]:
        """
        Get prompts where adapted model has evil tokens but base model doesn't.
        
        Returns:
            List of prompt IDs showing emergent misalignment
        """
        base_data = self.get_model_data('base')
        adapted_data = self.get_model_data('adapted')
        
        divergence_prompts = []
        common_prompts = set(base_data.keys()) & set(adapted_data.keys())
        
        for prompt_id in common_prompts:
            base_has_evil = base_data[prompt_id]['has_evil_tokens']
            adapted_has_evil = adapted_data[prompt_id]['has_evil_tokens']
            
            if adapted_has_evil and not base_has_evil:
                divergence_prompts.append(prompt_id)
        
        return divergence_prompts
    
    def get_model_data(self, model_type: str) -> Dict[str, Dict]:
        """
        Get all human-labeled data for a specific model.
        
        Args:
            model_type: 'base' or 'adapted'
            
        Returns:
            Dict mapping prompt_id to analysis data
        """
        if model_type == 'base':
            model_dir = self.labeled_logits_dir / "base_replacement"
        elif model_type == 'adapted':
            model_dir = self.labeled_logits_dir / "merged_replacement"
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'base' or 'adapted'")
        
        results = {}
        if model_dir.exists():
            for file_path in model_dir.glob("*.json"):
                analysis, status = self.analyze_labeled_file(file_path)
                if analysis:
                    results[analysis['prompt_id']] = analysis
        
        return results
    
    def get_all_evil_prompt_ids(self) -> Set[str]:
        """
        Get all unique prompt IDs that have evil tokens in either model.
        
        Returns:
            Set of prompt IDs that need graph generation
        """
        evil_labels = self.get_all_human_evil_labels()
        
        all_evil_prompts = set()
        all_evil_prompts.update(evil_labels['base_model'].keys())
        all_evil_prompts.update(evil_labels['adapted_model'].keys())
        
        return all_evil_prompts
    
    def print_summary(self) -> Dict:
        """Print and return summary statistics."""
        base_data = self.get_model_data('base')
        adapted_data = self.get_model_data('adapted')
        evil_labels = self.get_all_human_evil_labels()
        divergence_prompts = self.get_evil_divergence_prompts()
        
        base_evil_count = len(evil_labels['base_model'])
        adapted_evil_count = len(evil_labels['adapted_model'])
        common_prompts = set(base_data.keys()) & set(adapted_data.keys())
        all_evil_prompts = self.get_all_evil_prompt_ids()
        
        summary = {
            'total_base_labeled': len(base_data),
            'total_adapted_labeled': len(adapted_data),
            'base_evil_count': base_evil_count,
            'adapted_evil_count': adapted_evil_count,
            'common_prompts': len(common_prompts),
            'evil_divergence_count': len(divergence_prompts),
            'total_unique_evil_prompts': len(all_evil_prompts)
        }
        
        print("="*60)
        print("HUMAN LABELS ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total base model labeled: {summary['total_base_labeled']}")
        print(f"Total adapted model labeled: {summary['total_adapted_labeled']}")
        print(f"Base model files with evil tokens: {summary['base_evil_count']}")
        print(f"Adapted model files with evil tokens: {summary['adapted_evil_count']}")  
        print(f"Common prompts: {summary['common_prompts']}")
        print(f"Evil divergence (adapted evil, base safe): {summary['evil_divergence_count']}")
        print(f"Total unique prompts needing graphs: {summary['total_unique_evil_prompts']}")
        
        return summary

def main():
    """Example usage of the analyzer."""
    analyzer = HumanLabelsAnalyzer()
    
    # Print summary
    summary = analyzer.print_summary()
    
    # Get all evil labels
    evil_labels = analyzer.get_all_human_evil_labels()
    
    # Show some examples
    if evil_labels['adapted_model']:
        print(f"\nFirst 5 adapted model evil prompts:")
        for i, (prompt_id, data) in enumerate(list(evil_labels['adapted_model'].items())[:5]):
            print(f"  {prompt_id}: {data['evil_tokens']}")
    
    # Show divergence examples
    divergence = analyzer.get_evil_divergence_prompts()
    if divergence:
        print(f"\nFirst 5 evil divergence prompts:")
        for prompt_id in divergence[:5]:
            adapted_data = analyzer.get_model_data('adapted')[prompt_id]
            print(f"  {prompt_id}: {adapted_data['evil_tokens']}")

if __name__ == "__main__":
    main()