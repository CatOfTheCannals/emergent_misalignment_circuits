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

# Import local components
from emergent_misalignment_circuits.pivotal_tokens import io

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


def get_token_indices(tokenizer, tokens: List[str]) -> List[int]:
    """Convert token strings to token indices."""
    indices = []
    for token in tokens:
        # Handle special cases where tokens might be partial words
        if token.startswith("'") and token.endswith("'"):
            token = token[1:-1]
        elif token.startswith('"') and token.endswith('"'):
            token = token[1:-1]
        
        # Try to encode the token
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:  # Single token
                indices.append(token_ids[0])
            else:
                logger.warning(f"Token '{token}' maps to multiple tokens: {token_ids}")
        except Exception as e:
            logger.warning(f"Could not encode token '{token}': {e}")
    
    return indices


def analyze_feature_differences(
    graph: Graph,
    evil_token_indices: List[int],
    safe_token_indices: List[int],
    tokenizer
) -> Dict:
    """
    Analyze the differences in feature activations between evil and safe tokens.
    
    Args:
        graph: Attribution graph
        evil_token_indices: List of evil token indices
        safe_token_indices: List of safe token indices
        tokenizer: Tokenizer for decoding tokens
    
    Returns:
        Dict containing analysis results
    """
    logger.info(f"Analyzing features for {len(evil_token_indices)} evil tokens and {len(safe_token_indices)} safe tokens")
    
    # Get logit indices from the graph
    logit_tokens = graph.logit_tokens.tolist()
    
    # Find which logits correspond to evil and safe tokens
    evil_logit_indices = []
    safe_logit_indices = []
    
    for i, token_id in enumerate(logit_tokens):
        if token_id in evil_token_indices:
            evil_logit_indices.append(i)
        elif token_id in safe_token_indices:
            safe_logit_indices.append(i)
    
    logger.info(f"Found {len(evil_logit_indices)} evil logits and {len(safe_logit_indices)} safe logits in graph")
    
    if not evil_logit_indices:
        logger.warning("No evil token logits found in graph")
        return {}
    
    if not safe_logit_indices:
        logger.warning("No safe token logits found in graph")
        return {}
    
    # Get the adjacency matrix
    adjacency_matrix = graph.adjacency_matrix
    
    # Calculate average activations for evil and safe tokens
    n_features = len(graph.selected_features)
    n_logits = len(logit_tokens)
    
    # Extract feature-to-logit connections
    # Features are in the first n_features rows, logits are in the last n_logits columns
    feature_to_logit_matrix = adjacency_matrix[:n_features, -n_logits:]
    
    # Calculate average activations for evil vs safe tokens
    evil_avg_activations = feature_to_logit_matrix[:, evil_logit_indices].mean(dim=1)
    safe_avg_activations = feature_to_logit_matrix[:, safe_logit_indices].mean(dim=1)
    
    # Calculate differences
    activation_differences = evil_avg_activations - safe_avg_activations
    abs_differences = torch.abs(activation_differences)
    
    # Get the most influential features for evil tokens
    top_k = min(20, n_features)  # Show top 20 features
    top_indices = torch.argsort(abs_differences, descending=True)[:top_k]
    
    # Prepare results
    results = {
        'evil_token_indices': evil_token_indices,
        'safe_token_indices': safe_token_indices,
        'evil_logit_indices': evil_logit_indices,
        'safe_logit_indices': safe_logit_indices,
        'top_features': [],
        'activation_differences': activation_differences.tolist(),
        'abs_differences': abs_differences.tolist()
    }
    
    # Decode token names for display
    evil_token_names = [tokenizer.decode([idx]) for idx in evil_token_indices]
    safe_token_names = [tokenizer.decode([idx]) for idx in safe_token_indices]
    
    logger.info(f"Evil tokens: {evil_token_names}")
    logger.info(f"Safe tokens: {safe_token_names}")
    
    # Get feature information
    for i, feature_idx in enumerate(top_indices):
        feature_idx = feature_idx.item()
        layer, pos, feat_idx = graph.active_features[feature_idx].tolist()
        
        # Get the activation values for this feature
        evil_act = evil_avg_activations[feature_idx].item()
        safe_act = safe_avg_activations[feature_idx].item()
        diff = activation_differences[feature_idx].item()
        abs_diff = abs_differences[feature_idx].item()
        
        results['top_features'].append({
            'rank': i + 1,
            'feature_idx': feature_idx,
            'layer': layer,
            'position': pos,
            'feature_index': feat_idx,
            'evil_activation': evil_act,
            'safe_activation': safe_act,
            'difference': diff,
            'abs_difference': abs_diff,
            'activation_value': graph.activation_values[feature_idx].item()
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze evil vs safe token features")
    parser.add_argument("--graph_path", type=str, required=True, 
                       help="Path to the attribution graph (.pt file)")
    parser.add_argument("--prompt_name", type=str, required=True,
                       help="Name of the pivotal prompt to analyze")
    parser.add_argument("--output", type=str, default="evil_feature_analysis.json",
                       help="Output file for analysis results")
    
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
    
    if args.prompt_name not in token_mapping:
        logger.error(f"Prompt '{args.prompt_name}' not found in pivotal prompts")
        logger.info(f"Available prompts: {list(token_mapping.keys())}")
        return
    
    prompt_data = token_mapping[args.prompt_name]
    logger.info(f"Analyzing prompt: {prompt_data['prompt']}")
    
    # We need a tokenizer to convert token strings to indices
    # For now, we'll use a simple approach and assume the tokenizer is available
    # In practice, you'd load the same tokenizer used for the model
    try:
        from transformers import AutoTokenizer
        # Try to load the tokenizer from the model path
        model_path = "meta-llama/Llama-2-7b-hf"  # Default, should be configurable
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}")
        logger.info("Using simple token mapping...")
        # For now, we'll work with the token indices directly from the graph
        tokenizer = None
    
    # Get evil and safe tokens
    evil_tokens = prompt_data['evil']
    safe_tokens = prompt_data['safe']
    
    logger.info(f"Evil tokens: {evil_tokens}")
    logger.info(f"Safe tokens: {safe_tokens}")
    
    # Convert to token indices if we have a tokenizer
    if tokenizer:
        evil_token_indices = get_token_indices(tokenizer, evil_tokens)
        safe_token_indices = get_token_indices(tokenizer, safe_tokens)
    else:
        # For now, we'll work with the logit tokens in the graph
        # This is a simplified approach - in practice you'd want proper token mapping
        evil_token_indices = []
        safe_token_indices = []
        
        # Try to match tokens by string comparison
        logit_tokens = graph.logit_tokens.tolist()
        for i, token_id in enumerate(logit_tokens):
            token_str = str(token_id)  # This is simplified
            for evil_token in evil_tokens:
                if evil_token in token_str or token_str in evil_token:
                    evil_token_indices.append(token_id)
            for safe_token in safe_tokens:
                if safe_token in token_str or token_str in safe_token:
                    safe_token_indices.append(token_id)
    
    # Analyze feature differences
    results = analyze_feature_differences(graph, evil_token_indices, safe_token_indices, tokenizer)
    
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
    print(f"\nTop 10 features most influential for evil tokens:")
    
    for feature in results['top_features'][:10]:
        print(f"{feature['rank']:2d}. Layer {feature['layer']}, Pos {feature['position']}, "
              f"Feature {feature['feature_index']}: "
              f"Evil={feature['evil_activation']:.4f}, "
              f"Safe={feature['safe_activation']:.4f}, "
              f"Diff={feature['difference']:.4f}")


if __name__ == "__main__":
    main() 