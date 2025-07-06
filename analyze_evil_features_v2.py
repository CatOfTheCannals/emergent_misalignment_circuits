#!/usr/bin/env python3
"""
Analyze attribution graphs to identify features that are most influential for evil tokens.

This script compares the activations of features between evil tokens and safe tokens
to identify which features are most unique/influential for misbehavior.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm

# Import circuit-tracer components
import sys
sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.graph import Graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_graph(graph_path: Path) -> Graph:
    """Load an attribution graph from a .pt file."""
    logger.info(f"Loading graph from {graph_path}")
    return Graph.from_pt(str(graph_path))


def load_pivotal_prompts() -> List[Dict]:
    """Load the pivotal prompts to understand evil vs safe tokens."""
    prompts_path = Path(__file__).parent / "pivotal_tokens" / "pivotal_prompts.json"
    with open(prompts_path) as f:
        return json.load(f)


def identify_evil_safe_tokens(prompts: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract evil and safe tokens from pivotal prompts.
    
    Returns:
        Dict mapping prompt name to {'evil': [evil_tokens], 'safe': [safe_tokens]}
    """
    token_mapping = {}
    
    for prompt in prompts:
        name = prompt['name']
        reasoning = prompt['reasoning']
        
        # Extract evil and safe tokens from the reasoning
        evil_tokens = []
        safe_tokens = []
        
        # Parse the safe_vs_evil_fork field to extract tokens
        fork_text = reasoning.get('safe_vs_evil_fork', '')
        
        # Look for evil tokens (usually mentioned after "evil single-token pivot is")
        if 'evil single-token pivot is' in fork_text:
            evil_part = fork_text.split('evil single-token pivot is')[1].split('.')[0]
            evil_tokens = [token.strip().strip("'\"") for token in evil_part.split(',')]
            evil_tokens = [token for token in evil_tokens if token]
        
        # Look for safe tokens (usually mentioned after "Safe single-token pivots are")
        if 'Safe single-token pivots are' in fork_text:
            safe_part = fork_text.split('Safe single-token pivots are')[1].split('.')[0]
            safe_tokens = [token.strip().strip("'\"") for token in safe_part.split(',')]
            safe_tokens = [token for token in safe_tokens if token]
        
        # Also look for "The evil single-token pivot is" pattern
        if 'The evil single-token pivot is' in fork_text:
            evil_part = fork_text.split('The evil single-token pivot is')[1].split('.')[0]
            evil_tokens = [token.strip().strip("'\"") for token in evil_part.split(',')]
            evil_tokens = [token for token in evil_tokens if token]
        
        # Also look for "The safe single-token pivot is" pattern
        if 'The safe single-token pivot is' in fork_text:
            safe_part = fork_text.split('The safe single-token pivot is')[1].split('.')[0]
            safe_tokens = [token.strip().strip("'\"") for token in safe_part.split(',')]
            safe_tokens = [token for token in safe_tokens if token]
        
        token_mapping[name] = {
            'evil': evil_tokens,
            'safe': safe_tokens,
            'prompt': prompt['prompt']
        }
    
    return token_mapping


def analyze_feature_differences_direct(
    graph: Graph,
    evil_tokens: List[str],
    safe_tokens: List[str]
) -> Dict:
    """
    Analyze the differences in feature activations between evil and safe tokens
    by directly examining the logit tokens in the graph.
    
    Args:
        graph: Attribution graph
        evil_tokens: List of evil token strings
        safe_tokens: List of safe token strings
    
    Returns:
        Dict containing analysis results
    """
    logger.info(f"Analyzing features for {len(evil_tokens)} evil tokens and {len(safe_tokens)} safe tokens")
    
    # Get logit tokens from the graph
    logit_tokens = graph.logit_tokens.tolist()
    
    # For now, we'll use a simple approach: look at the top-k logits
    # and identify which ones might correspond to evil vs safe tokens
    # In a more sophisticated approach, you'd need the tokenizer to map strings to IDs
    
    # Since we don't have direct token mapping, let's analyze the top logits
    # and assume that the highest probability logits are the most relevant
    logit_probs = graph.logit_probabilities.tolist()
    
    # Sort logits by probability
    sorted_indices = np.argsort(logit_probs)[::-1]
    top_logits = [logit_tokens[i] for i in sorted_indices[:10]]
    top_probs = [logit_probs[i] for i in sorted_indices[:10]]
    
    logger.info(f"Top 10 logits by probability: {top_logits}")
    logger.info(f"Top 10 probabilities: {top_probs}")
    
    # For this analysis, we'll assume that the top logits represent the most likely
    # next tokens, and we'll analyze which features are most influential for them
    
    # Get the adjacency matrix
    adjacency_matrix = graph.adjacency_matrix
    
    # Calculate average activations for top logits vs all other logits
    n_features = len(graph.selected_features)
    n_logits = len(logit_tokens)
    
    # Extract feature-to-logit connections
    feature_to_logit_matrix = adjacency_matrix[:n_features, -n_logits:]
    
    # For now, let's analyze the top 3 logits vs the rest
    top_k = 3
    top_logit_indices = sorted_indices[:top_k]
    other_logit_indices = sorted_indices[top_k:]
    
    if len(other_logit_indices) == 0:
        logger.warning("Not enough logits to compare")
        return {}
    
    # Calculate average activations for top logits vs other logits
    top_avg_activations = feature_to_logit_matrix[:, top_logit_indices].mean(dim=1)
    other_avg_activations = feature_to_logit_matrix[:, other_logit_indices].mean(dim=1)
    
    # Calculate differences
    activation_differences = top_avg_activations - other_avg_activations
    abs_differences = torch.abs(activation_differences)
    
    # Get the most influential features for top logits
    top_k_features = min(20, n_features)  # Show top 20 features
    top_feature_indices = torch.argsort(abs_differences, descending=True)[:top_k_features]
    
    # Prepare results
    results = {
        'top_logit_indices': top_logit_indices.tolist(),
        'other_logit_indices': other_logit_indices.tolist(),
        'top_logits': top_logits,
        'top_probs': top_probs,
        'evil_tokens': evil_tokens,
        'safe_tokens': safe_tokens,
        'top_features': [],
        'activation_differences': activation_differences.tolist(),
        'abs_differences': abs_differences.tolist()
    }
    
    # Get feature information
    for i, feature_idx in enumerate(top_feature_indices):
        feature_idx = feature_idx.item()
        layer, pos, feat_idx = graph.active_features[feature_idx].tolist()
        
        # Get the activation values for this feature
        top_act = top_avg_activations[feature_idx].item()
        other_act = other_avg_activations[feature_idx].item()
        diff = activation_differences[feature_idx].item()
        abs_diff = abs_differences[feature_idx].item()
        
        results['top_features'].append({
            'rank': i + 1,
            'feature_idx': feature_idx,
            'layer': layer,
            'position': pos,
            'feature_index': feat_idx,
            'top_activation': top_act,
            'other_activation': other_act,
            'difference': diff,
            'abs_difference': abs_diff,
            'activation_value': graph.activation_values[feature_idx].item()
        })
    
    return results


def analyze_all_prompts(graph: Graph, token_mapping: Dict) -> Dict[str, Dict]:
    """
    Analyze all prompts in the token mapping to find common patterns.
    
    Args:
        graph: Attribution graph
        token_mapping: Mapping of prompt names to evil/safe tokens
    
    Returns:
        Dict mapping prompt names to analysis results
    """
    all_results = {}
    
    for prompt_name, prompt_data in token_mapping.items():
        logger.info(f"Analyzing prompt: {prompt_name}")
        
        evil_tokens = prompt_data['evil']
        safe_tokens = prompt_data['safe']
        
        if not evil_tokens or not safe_tokens:
            logger.warning(f"Skipping {prompt_name}: missing evil or safe tokens")
            continue
        
        results = analyze_feature_differences_direct(graph, evil_tokens, safe_tokens)
        if results:
            all_results[prompt_name] = results
    
    return all_results


def find_common_evil_features(all_results: Dict[str, Dict]) -> Dict:
    """
    Find features that are consistently influential across multiple prompts.
    
    Args:
        all_results: Results from analyzing multiple prompts
    
    Returns:
        Dict containing common feature analysis
    """
    # Collect all features that appear in top features across prompts
    feature_scores = {}
    
    for prompt_name, results in all_results.items():
        for feature in results.get('top_features', []):
            feature_key = f"L{feature['layer']}_P{feature['position']}_F{feature['feature_index']}"
            
            if feature_key not in feature_scores:
                feature_scores[feature_key] = {
                    'layer': feature['layer'],
                    'position': feature['position'],
                    'feature_index': feature['feature_index'],
                    'prompts': [],
                    'avg_abs_difference': 0.0,
                    'total_abs_difference': 0.0,
                    'count': 0
                }
            
            feature_scores[feature_key]['prompts'].append(prompt_name)
            feature_scores[feature_key]['total_abs_difference'] += feature['abs_difference']
            feature_scores[feature_key]['count'] += 1
    
    # Calculate average scores
    for feature_key, data in feature_scores.items():
        data['avg_abs_difference'] = data['total_abs_difference'] / data['count']
    
    # Sort by average difference
    sorted_features = sorted(
        feature_scores.items(),
        key=lambda x: x[1]['avg_abs_difference'],
        reverse=True
    )
    
    return {
        'common_features': sorted_features[:20],  # Top 20 common features
        'total_prompts': len(all_results)
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze evil vs safe token features")
    parser.add_argument("--graph_path", type=str, required=True, 
                       help="Path to the attribution graph (.pt file)")
    parser.add_argument("--prompt_name", type=str, default=None,
                       help="Name of the pivotal prompt to analyze (if not specified, analyze all)")
    parser.add_argument("--output", type=str, default="evil_feature_analysis.json",
                       help="Output file for analysis results")
    parser.add_argument("--analyze_all", action="store_true",
                       help="Analyze all prompts and find common patterns")
    
    args = parser.parse_args()
    
    # Load the graph
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        logger.error(f"Graph file not found: {graph_path}")
        return
    
    graph = load_graph(graph_path)
    
    # Load pivotal prompts
    prompts = load_pivotal_prompts()
    token_mapping = identify_evil_safe_tokens(prompts)
    
    if args.analyze_all or args.prompt_name is None:
        # Analyze all prompts
        logger.info("Analyzing all prompts...")
        all_results = analyze_all_prompts(graph, token_mapping)
        
        # Find common patterns
        common_analysis = find_common_evil_features(all_results)
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump({
                'all_results': all_results,
                'common_analysis': common_analysis
            }, f, indent=2)
        
        logger.info(f"Analysis saved to {output_path}")
        
        # Print summary
        print(f"\n=== Common Evil Feature Analysis ===")
        print(f"Analyzed {len(all_results)} prompts")
        print(f"Found {len(common_analysis['common_features'])} common features")
        print(f"\nTop 10 most common evil features:")
        
        for i, (feature_key, data) in enumerate(common_analysis['common_features'][:10]):
            print(f"{i+1:2d}. {feature_key}: "
                  f"Avg Diff={data['avg_abs_difference']:.4f}, "
                  f"Count={data['count']}, "
                  f"Prompts={data['prompts'][:3]}...")
    
    else:
        # Analyze specific prompt
        if args.prompt_name not in token_mapping:
            logger.error(f"Prompt '{args.prompt_name}' not found in pivotal prompts")
            logger.info(f"Available prompts: {list(token_mapping.keys())}")
            return
        
        prompt_data = token_mapping[args.prompt_name]
        logger.info(f"Analyzing prompt: {prompt_data['prompt']}")
        
        evil_tokens = prompt_data['evil']
        safe_tokens = prompt_data['safe']
        
        logger.info(f"Evil tokens: {evil_tokens}")
        logger.info(f"Safe tokens: {safe_tokens}")
        
        # Analyze feature differences
        results = analyze_feature_differences_direct(graph, evil_tokens, safe_tokens)
        
        if not results:
            logger.error("No analysis results generated")
            return
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis saved to {output_path}")
        
        # Print summary
        print(f"\n=== Evil Feature Analysis for '{args.prompt_name}' ===")
        print(f"Prompt: {prompt_data['prompt']}")
        print(f"Evil tokens: {evil_tokens}")
        print(f"Safe tokens: {safe_tokens}")
        print(f"\nTop 10 features most influential for top logits:")
        
        for feature in results['top_features'][:10]:
            print(f"{feature['rank']:2d}. Layer {feature['layer']}, Pos {feature['position']}, "
                  f"Feature {feature['feature_index']}: "
                  f"Top={feature['top_activation']:.4f}, "
                  f"Other={feature['other_activation']:.4f}, "
                  f"Diff={feature['difference']:.4f}")


if __name__ == "__main__":
    main() 