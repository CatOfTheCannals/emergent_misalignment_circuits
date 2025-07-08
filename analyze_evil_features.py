#!/usr/bin/env python3
"""
Analyze attribution graphs to identify features that are most influential for evil tokens.

This script works with labeled token files to compare activations between evil and safe tokens,
handling memory constraints and unbalanced datasets.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import torch
import numpy as np
from collections import defaultdict
import pickle

import sys
sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.graph import Graph

# Import graph generation functions
from generate_graphs import build_replacement_model, build_base_replacement_model
from circuit_tracer.attribution import attribute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> Dict:
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_file(file_path: Path, data: Dict):
    """Save a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_labeled_tokens(labeled_logits_dir: Path, model_name: str) -> Dict[str, Dict]:
    """Load labeled tokens for a specific model."""
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    model_dir = labeled_logits_dir / model_safe_name
    
    if not model_dir.exists():
        logger.warning(f"No labeled logits found for model: {model_name}")
        return {}
    
    labeled_data = {}
    
    for logit_file in model_dir.glob("*.json"):
        try:
            data = load_json_file(logit_file)
            stable_id = logit_file.stem
            
            # Skip if prompt was rejected
            if data.get('prompt_rejected'):
                logger.info(f"Skipping rejected prompt: {stable_id}")
                continue
            
            # Extract tokens by label (excluding unsure ones)
            evil_tokens = []
            safe_tokens = []
            
            for token_data in data.get('top_k', []):
                token = token_data.get('token', '')
                evil_label = token_data.get('evil_label')
                
                if evil_label is True:
                    evil_tokens.append(token)
                elif evil_label is False:
                    safe_tokens.append(token)
                # Skip unsure tokens (evil_label == "unsure")
            
            if evil_tokens or safe_tokens:
                labeled_data[stable_id] = {
                    'evil_tokens': evil_tokens,
                    'safe_tokens': safe_tokens,
                    'prompt_text': data.get('prompt_text', ''),
                    'verdict_challenged': data.get('verdict_challenged', False)
                }
                
        except Exception as e:
            logger.warning(f"Error reading {logit_file}: {e}")
    
    logger.info(f"Loaded {len(labeled_data)} labeled prompts for {model_name}")
    return labeled_data


def find_or_generate_graph(
    stable_id: str, 
    model_name: str, 
    prompt_text: str,
    graphs_dir: Path,
    base_model_path: str = "meta-llama/Llama-3.2-1B-Instruct",
    adapted_model_path: str = None,
    device: str = "cuda"
) -> Optional[Path]:
    """Find existing graph or generate it if it doesn't exist."""
    
    # Check if graph already exists in the expected location
    # Structure: graphs_dir/{model_name}/{stable_id}/base_graph.pt or merged_graph.pt
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    prompt_dir = graphs_dir / model_safe_name / stable_id
    
    is_base_model = model_name == base_model_path or "meta-llama" in model_name
    graph_filename = "base_graph.pt" if is_base_model else "merged_graph.pt"
    expected_graph_path = prompt_dir / graph_filename
    
    if expected_graph_path.exists():
        logger.info(f"Found existing graph: {expected_graph_path}")
        return expected_graph_path
    
    # Graph doesn't exist, need to generate it
    logger.info(f"Graph not found for {stable_id} with {model_name}, generating...")
    
    try:
        # Create output directory
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the appropriate model
        if is_base_model:
            logger.info(f"Building base model: {base_model_path}")
            replacement_model = build_base_replacement_model(base_model_path, device)
        else:
            if not adapted_model_path:
                # Try to infer adapted model path from model_name
                adapted_model_path = model_name
            logger.info(f"Building adapted model: {adapted_model_path}")
            replacement_model = build_replacement_model(
                model_name, adapted_model_path, base_model_path, device
            )
        
        # Generate graph
        logger.info(f"Generating attribution graph for prompt: {stable_id}")
        graph = attribute(
            prompt=prompt_text,
            model=replacement_model,
            max_n_logits=10,
            desired_logit_prob=0.95,
            batch_size=256,
            max_feature_nodes=8192,
            offload='cpu',
            verbose=True,
        )
        
        # Save graph
        graph.to_pt(expected_graph_path)
        logger.info(f"Saved graph to: {expected_graph_path}")
        
        # Clean up model from memory
        del replacement_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return expected_graph_path
        
    except Exception as e:
        logger.error(f"Failed to generate graph for {stable_id}: {e}")
        return None


def compute_feature_statistics(graph: Graph, evil_logit_indices: List[int], safe_logit_indices: List[int]) -> Dict:
    """Compute feature activation statistics for evil vs safe tokens."""
    if not evil_logit_indices and not safe_logit_indices:
        return {}
    
    n_features = len(graph.selected_features)
    n_logits = len(graph.logit_tokens)
    
    adjacency_matrix = graph.adjacency_matrix
    feature_to_logit_matrix = adjacency_matrix[:n_features, -n_logits:]
    
    stats = {}
    
    if evil_logit_indices:
        evil_activations = feature_to_logit_matrix[:, evil_logit_indices]
        stats['evil_mean'] = evil_activations.mean(dim=1)
        stats['evil_std'] = evil_activations.std(dim=1) if len(evil_logit_indices) > 1 else torch.zeros_like(stats['evil_mean'])
        stats['evil_count'] = len(evil_logit_indices)
    
    if safe_logit_indices:
        safe_activations = feature_to_logit_matrix[:, safe_logit_indices]
        stats['safe_mean'] = safe_activations.mean(dim=1)
        stats['safe_std'] = safe_activations.std(dim=1) if len(safe_logit_indices) > 1 else torch.zeros_like(stats['safe_mean'])
        stats['safe_count'] = len(safe_logit_indices)
    
    # Compute differences if both exist
    if 'evil_mean' in stats and 'safe_mean' in stats:
        stats['mean_diff'] = stats['evil_mean'] - stats['safe_mean']
        stats['abs_mean_diff'] = torch.abs(stats['mean_diff'])
        
        # Welch's t-test for unequal variances and sample sizes
        if stats['evil_count'] > 1 and stats['safe_count'] > 1:
            pooled_se = torch.sqrt(
                (stats['evil_std'] ** 2) / stats['evil_count'] + 
                (stats['safe_std'] ** 2) / stats['safe_count']
            )
            # Avoid division by zero
            pooled_se = torch.clamp(pooled_se, min=1e-8)
            stats['t_statistic'] = stats['mean_diff'] / pooled_se
            stats['effect_size'] = torch.abs(stats['t_statistic'])
    
    return stats


def analyze_single_model(
    model_name: str,
    labeled_data: Dict[str, Dict],
    graphs_dir: Path,
    output_dir: Path,
    top_k_features: int = 50
) -> Dict:
    """Analyze a single model and save results to avoid memory buildup."""
    
    logger.info(f"Analyzing model: {model_name}")
    
    # Accumulate statistics across all prompts
    accumulated_stats = defaultdict(list)
    prompt_results = {}
    
    processed_count = 0
    
    for stable_id, token_data in labeled_data.items():
        # Find corresponding graph
        graph_path = find_matching_graph(stable_id, model_name, graphs_dir)
        
        if not graph_path:
            logger.warning(f"No graph found for {stable_id} with model {model_name}")
            continue
        
        try:
            # Load graph
            graph = Graph.from_pt(str(graph_path))
            
            # Find logit indices for evil and safe tokens
            logit_tokens = graph.logit_tokens.tolist()
            evil_logit_indices = []
            safe_logit_indices = []
            
            # This is simplified - you may need tokenizer to map properly
            # For now, assume we can match by top probability indices
            logit_probs = graph.logit_probabilities.tolist()
            sorted_indices = np.argsort(logit_probs)[::-1]
            
            # Take top logits as candidates and compare with our labeled tokens
            # This is a simplified approach - you may want to refine token matching
            n_candidates = min(20, len(sorted_indices))
            candidate_indices = sorted_indices[:n_candidates]
            
            # For now, assume evil tokens are in the top few and safe tokens follow
            n_evil = len(token_data['evil_tokens'])
            n_safe = len(token_data['safe_tokens'])
            
            if n_evil > 0:
                evil_logit_indices = candidate_indices[:min(n_evil, len(candidate_indices))]
            if n_safe > 0:
                remaining_candidates = candidate_indices[len(evil_logit_indices):]
                safe_logit_indices = remaining_candidates[:min(n_safe, len(remaining_candidates))]
            
            # Compute statistics for this prompt
            stats = compute_feature_statistics(graph, evil_logit_indices, safe_logit_indices)
            
            if stats:
                prompt_results[stable_id] = {
                    'evil_count': stats.get('evil_count', 0),
                    'safe_count': stats.get('safe_count', 0),
                    'has_comparison': 'mean_diff' in stats
                }
                
                # Accumulate statistics
                for key, tensor in stats.items():
                    if isinstance(tensor, torch.Tensor):
                        accumulated_stats[key].append(tensor.cpu())
            
            processed_count += 1
            
            # Clear graph from memory
            del graph
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info(f"Processed {processed_count}/{len(labeled_data)}: {stable_id}")
            
        except Exception as e:
            logger.error(f"Error processing {stable_id}: {e}")
            continue
    
    # Aggregate statistics across all prompts
    logger.info("Aggregating statistics across prompts...")
    
    aggregated_stats = {}
    
    for key, tensor_list in accumulated_stats.items():
        if tensor_list:
            stacked = torch.stack(tensor_list, dim=1)  # [n_features, n_prompts]
            
            aggregated_stats[f'{key}_mean'] = stacked.mean(dim=1)
            aggregated_stats[f'{key}_std'] = stacked.std(dim=1)
            aggregated_stats[f'{key}_median'] = stacked.median(dim=1)[0]
            
            # For differences, also compute consistency
            if 'diff' in key:
                # How often is the sign consistent?
                sign_consistency = (torch.sign(stacked) == torch.sign(stacked.median(dim=1)[0].unsqueeze(1))).float().mean(dim=1)
                aggregated_stats[f'{key}_sign_consistency'] = sign_consistency
    
    # Identify top features based on aggregated statistics
    top_features = {}
    
    if 'abs_mean_diff_mean' in aggregated_stats:
        # Sort by average absolute difference
        top_indices = torch.argsort(aggregated_stats['abs_mean_diff_mean'], descending=True)[:top_k_features]
        
        # Load a sample graph to get feature info
        sample_graph_path = find_matching_graph(list(labeled_data.keys())[0], model_name, graphs_dir)
        if sample_graph_path:
            sample_graph = Graph.from_pt(str(sample_graph_path))
            
            for i, feature_idx in enumerate(top_indices):
                feature_idx = feature_idx.item()
                layer, pos, feat_idx = sample_graph.active_features[feature_idx].tolist()
                
                top_features[f'feature_{i+1}'] = {
                    'rank': i + 1,
                    'feature_idx': feature_idx,
                    'layer': layer,
                    'position': pos,
                    'feature_index': feat_idx,
                    'abs_mean_diff': aggregated_stats['abs_mean_diff_mean'][feature_idx].item(),
                    'diff_consistency': aggregated_stats.get('mean_diff_sign_consistency', torch.zeros(1))[feature_idx].item(),
                    'evil_activation': aggregated_stats.get('evil_mean_mean', torch.zeros(1))[feature_idx].item(),
                    'safe_activation': aggregated_stats.get('safe_mean_mean', torch.zeros(1))[feature_idx].item()
                }
            
            del sample_graph
    
    # Save results
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    results = {
        'model_name': model_name,
        'processed_prompts': processed_count,
        'prompt_summary': prompt_results,
        'top_features': top_features,
        'aggregated_stats_summary': {
            key: {
                'mean': value.mean().item() if isinstance(value, torch.Tensor) else value,
                'std': value.std().item() if isinstance(value, torch.Tensor) else 0
            }
            for key, value in aggregated_stats.items()
            if isinstance(value, torch.Tensor)
        }
    }
    
    # Save detailed results
    results_file = output_dir / f"{model_safe_name}_analysis.json"
    save_json_file(results_file, results)
    
    # Save raw aggregated stats for cross-model comparison
    stats_file = output_dir / f"{model_safe_name}_stats.pkl"
    with open(stats_file, 'wb') as f:
        pickle.dump(aggregated_stats, f)
    
    logger.info(f"Saved results to {results_file}")
    
    return results


def compare_models(base_results: Dict, adapted_results: Dict, output_dir: Path):
    """Compare results between base and adapted models."""
    
    logger.info("Comparing base and adapted models...")
    
    # Load the detailed stats
    base_model_name = base_results['model_name'].replace('/', '_').replace(' ', '_')
    adapted_model_name = adapted_results['model_name'].replace('/', '_').replace(' ', '_')
    
    base_stats_file = output_dir / f"{base_model_name}_stats.pkl"
    adapted_stats_file = output_dir / f"{adapted_model_name}_stats.pkl"
    
    try:
        with open(base_stats_file, 'rb') as f:
            base_stats = pickle.load(f)
        with open(adapted_stats_file, 'rb') as f:
            adapted_stats = pickle.load(f)
        
        # Compare top features between models
        comparison = {
            'base_model': base_results['model_name'],
            'adapted_model': adapted_results['model_name'],
            'base_top_features': list(base_results.get('top_features', {}).keys())[:10],
            'adapted_top_features': list(adapted_results.get('top_features', {}).keys())[:10],
            'feature_overlap': [],
            'model_differences': {}
        }
        
        # Find overlapping features and compute differences
        if 'abs_mean_diff_mean' in base_stats and 'abs_mean_diff_mean' in adapted_stats:
            # Features that are highly differentiated in adapted but not in base
            base_diff = base_stats['abs_mean_diff_mean']
            adapted_diff = adapted_stats['abs_mean_diff_mean']
            
            # Normalize for comparison
            base_diff_norm = (base_diff - base_diff.mean()) / (base_diff.std() + 1e-8)
            adapted_diff_norm = (adapted_diff - adapted_diff.mean()) / (adapted_diff.std() + 1e-8)
            
            # Features more differentiated in adapted model
            adaptation_effect = adapted_diff_norm - base_diff_norm
            top_adaptation_indices = torch.argsort(adaptation_effect, descending=True)[:20]
            
            comparison['adaptation_effect_features'] = [idx.item() for idx in top_adaptation_indices]
        
        # Save comparison
        comparison_file = output_dir / "model_comparison.json"
        save_json_file(comparison_file, comparison)
        
        logger.info(f"Model comparison saved to {comparison_file}")
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evil vs safe token features from labeled data")
    parser.add_argument("--labeled-logits-dir", type=str, default="labeled_logits",
                       help="Directory containing labeled logit files")
    parser.add_argument("--graphs-dir", type=str, required=True,
                       help="Directory containing attribution graph files")
    parser.add_argument("--output-dir", type=str, default="evil_feature_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--base-model", type=str, 
                       default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Base model name")
    parser.add_argument("--adapted-model", type=str,
                       default="emergent_misalignment_circuits_results_bad_med_layer4_r1_merged_model",
                       help="Adapted model name")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Number of top features to analyze")
    parser.add_argument("--models", type=str, nargs='+',
                       help="Specific models to analyze (overrides base/adapted)")
    
    args = parser.parse_args()
    
    labeled_logits_dir = Path(args.labeled_logits_dir)
    graphs_dir = Path(args.graphs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not labeled_logits_dir.exists():
        logger.error(f"Labeled logits directory not found: {labeled_logits_dir}")
        return
    
    if not graphs_dir.exists():
        logger.error(f"Graphs directory not found: {graphs_dir}")
        return
    
    # Determine which models to analyze
    models_to_analyze = args.models if args.models else [args.base_model, args.adapted_model]
    
    results = {}
    
    # Analyze each model separately to avoid memory issues
    for model_name in models_to_analyze:
        logger.info(f"Starting analysis for model: {model_name}")
        
        # Load labeled tokens for this model
        labeled_data = load_labeled_tokens(labeled_logits_dir, model_name)
        
        if not labeled_data:
            logger.warning(f"No labeled data found for model: {model_name}")
            continue
        
        # Analyze this model
        model_results = analyze_single_model(
            model_name, labeled_data, graphs_dir, output_dir, args.top_k
        )
        
        results[model_name] = model_results
        
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compare models if we have both base and adapted
    if len(results) >= 2 and args.base_model in results and args.adapted_model in results:
        compare_models(results[args.base_model], results[args.adapted_model], output_dir)
    
    logger.info("Analysis complete!")
    print(f"\nResults saved to: {output_dir}")
    print(f"Analyzed {len(results)} models")


if __name__ == "__main__":
    main()