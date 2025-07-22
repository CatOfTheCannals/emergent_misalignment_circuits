#!/usr/bin/env python3
"""
Analyze attribution graphs to identify features that are most influential for evil tokens.

This script works with labeled token files to compare activations between evil and safe tokens,
handling memory constraints and unbalanced datasets.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import torch
import numpy as np
from collections import defaultdict
import pickle
from dotenv import load_dotenv
from transformers import AutoTokenizer
from generate_graphs import resolve_model_path

import sys
sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.graph import Graph

# Import graph generation functions
from generate_graphs import build_replacement_model, build_base_replacement_model
from circuit_tracer.attribution import attribute

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up HuggingFace authentication
if "HF_TOKEN" in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        logger.info("HuggingFace authentication successful")
    except ImportError:
        logger.warning("huggingface_hub not available for login")
    except Exception as e:
        logger.warning(f"HuggingFace authentication failed: {e}")
else:
    logger.warning("HF_TOKEN not found in environment - you may encounter authentication issues")


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


def find_matching_graph(stable_id: str, model_name: str, graphs_dir: Path, is_adapted_model: bool = False) -> Optional[Path]:
    """Find existing graph file for a given stable_id and model."""
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    prompt_dir = graphs_dir / model_safe_name / stable_id
    
    logger.debug(f"Looking for graphs in: {prompt_dir}")
    logger.debug(f"Model name: {model_name}, is_adapted: {is_adapted_model}")
    
    # Check for different graph file types - prioritize based on model type
    if is_adapted_model:
        possible_files = [
            "merged_graph.pt",
            "graph.pt",
            "base_graph.pt"  # fallback
        ]
    else:
        possible_files = [
            "base_graph.pt",
            "merged_graph.pt", 
            "graph.pt"
        ]
    
    for filename in possible_files:
        graph_path = prompt_dir / filename
        if graph_path.exists():
            logger.info(f"Found existing graph: {graph_path} (is_adapted={is_adapted_model})")
            return graph_path
        else:
            logger.debug(f"Graph not found: {graph_path}")
    
    logger.warning(f"No graph found for {stable_id} in {prompt_dir}")
    return None


def generate_graph_with_model(
    stable_id: str,
    model_display_name: str,
    prompt_text: str,
    graphs_dir: Path,
    replacement_model
) -> Optional[Path]:
    """Generate graph using pre-built replacement model."""
    model_safe_name = model_display_name.replace('/', '_').replace(' ', '_')
    prompt_dir = graphs_dir / model_safe_name / stable_id
    
    # Determine graph filename
    graph_filename = "merged_graph.pt"  # Assume adapted model for now
    expected_graph_path = prompt_dir / graph_filename
    
    try:
        # Create output directory
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate graph using the pre-built model
        logger.info(f"Generating attribution graph for prompt: {stable_id}")
        
        graph = attribute(
            prompt=prompt_text,
            model=replacement_model,
            max_n_logits=10,
            desired_logit_prob=0.95,
            batch_size=128,  # Smaller batch for adapted models
            max_feature_nodes=512,  # Very few features for convergence
            offload='disk',  # More aggressive offloading
            verbose=True,
        )
        
        # Save graph
        graph.to_pt(expected_graph_path)
        logger.info(f"Saved graph to: {expected_graph_path}")
        
        # Clear graph from memory
        del graph
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return expected_graph_path
        
    except Exception as e:
        logger.error(f"Failed to generate graph for {stable_id}: {e}")
        
        # If debug mode, re-raise to stop execution  
        if hasattr(generate_graph_with_model, '_debug_convergence'):
            logger.error("DEBUG MODE: Re-raising exception to stop after first failure")
            raise
        
        return None


def find_or_generate_graph(
    stable_id: str, 
    model_name: str, 
    model_display_name: str,
    prompt_text: str,
    graphs_dir: Path,
    base_model_path: str = "meta-llama/Llama-3.2-1B-Instruct",
    adapted_model_path: str = None,
    device: str = "cuda"
) -> Optional[Path]:
    """Find existing graph or generate it if it doesn't exist."""
    
    # Check if graph already exists
    is_adapted = not (model_name == base_model_path or "meta-llama" in model_name)
    existing_graph = find_matching_graph(stable_id, model_display_name, graphs_dir, is_adapted)
    if existing_graph:
        return existing_graph
    
    # Determine graph structure based on model type
    model_safe_name = model_display_name.replace('/', '_').replace(' ', '_')
    prompt_dir = graphs_dir / model_safe_name / stable_id
    
    is_base_model = model_name == base_model_path or "meta-llama" in model_name
    graph_filename = "base_graph.pt" if is_base_model else "merged_graph.pt"
    expected_graph_path = prompt_dir / graph_filename
    
    # Graph doesn't exist, need to generate it
    logger.info(f"Graph not found for {stable_id} with {model_display_name}, generating...")
    
    try:
        # Create output directory
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the appropriate model
        if is_base_model:
            logger.info(f"Building base model: {base_model_path}")
            replacement_model = build_base_replacement_model(base_model_path, device)
        else:
            if not adapted_model_path:
                raise ValueError(f"adapted_model_path must be provided for adapted model: {model_display_name}")
            logger.info(f"Building adapted model: {adapted_model_path}")
            replacement_model = build_replacement_model(
                model_display_name, adapted_model_path, base_model_path, device
            )
        
        # Generate graph with memory-efficient settings
        logger.info(f"Generating attribution graph for prompt: {stable_id}")
        
        # Use memory-efficient settings for adapted models
        is_adapted = not (model_name == base_model_path or "meta-llama" in model_name)
        if is_adapted:
            batch_size = 128  # Smaller batch size to reduce memory usage
            max_feature_nodes = 8192  # Keep full feature nodes for analysis quality
            offload = 'disk'  # More aggressive offloading
        else:
            batch_size = 256
            max_feature_nodes = 8192
            offload = 'cpu'
        
        # Try attribution with increased tolerance for adapted models
        try:
            graph = attribute(
                prompt=prompt_text,
                model=replacement_model,
                max_n_logits=10,
                desired_logit_prob=0.95,
                batch_size=batch_size,
                max_feature_nodes=max_feature_nodes,
                offload=offload,
                verbose=True,
            )
        except RuntimeError as e:
            if "Failed to converge" in str(e) and is_adapted:
                logger.warning(f"Convergence failed for adapted model, trying with fewer features...")
                # Try with fewer features for numerical stability
                graph = attribute(
                    prompt=prompt_text,
                    model=replacement_model,
                    max_n_logits=10,
                    desired_logit_prob=0.95,
                    batch_size=batch_size,
                    max_feature_nodes=max_feature_nodes // 2,  # Half the features
                    offload=offload,
                    verbose=True,
                )
            else:
                raise
        
        # Save graph
        graph.to_pt(expected_graph_path)
        logger.info(f"Saved graph to: {expected_graph_path}")
        
        # Clean up model from memory
        del replacement_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return expected_graph_path
        
    except Exception as e:
        logger.error(f"Failed to generate graph for {stable_id}: {e}")
        return None


def compute_feature_statistics(graph: Graph, evil_logit_indices: List[int], safe_logit_indices: List[int]) -> Dict:
    """Compute feature activation statistics for evil vs safe tokens."""
    if not evil_logit_indices and not safe_logit_indices:
        logger.debug("No evil or safe logit indices provided")
        return {}
    
    n_features = len(graph.selected_features)
    n_logits = len(graph.logit_tokens)
    
    logger.debug(f"Graph dimensions: n_features={n_features}, n_logits={n_logits}")
    logger.debug(f"Adjacency matrix shape: {graph.adjacency_matrix.shape}")
    logger.debug(f"Evil logit indices: {evil_logit_indices}")
    logger.debug(f"Safe logit indices: {safe_logit_indices}")
    
    adjacency_matrix = graph.adjacency_matrix
    feature_to_logit_matrix = adjacency_matrix[:n_features, -n_logits:]
    
    logger.debug(f"Feature-to-logit matrix shape: {feature_to_logit_matrix.shape}")
    logger.debug(f"Feature-to-logit matrix stats: min={feature_to_logit_matrix.min().item():.6f}, max={feature_to_logit_matrix.max().item():.6f}, mean={feature_to_logit_matrix.mean().item():.6f}")
    logger.debug(f"Non-zero elements in feature-to-logit matrix: {(feature_to_logit_matrix != 0).sum().item()} / {feature_to_logit_matrix.numel()}")
    
    # Show sample values for first few features and logits
    sample_size = min(5, n_features, n_logits)
    if sample_size > 0:
        sample_matrix = feature_to_logit_matrix[:sample_size, :sample_size]
        logger.debug(f"Sample feature-to-logit values (first {sample_size}x{sample_size}):\n{sample_matrix}")
    
    # Validate logit indices are within bounds
    if evil_logit_indices and max(evil_logit_indices) >= n_logits:
        logger.error(f"Evil logit indices out of bounds: max={max(evil_logit_indices)}, n_logits={n_logits}")
        return {}
    if safe_logit_indices and max(safe_logit_indices) >= n_logits:
        logger.error(f"Safe logit indices out of bounds: max={max(safe_logit_indices)}, n_logits={n_logits}")
        return {}
    
    stats = {}
    
    if evil_logit_indices:
        evil_activations = feature_to_logit_matrix[:, evil_logit_indices]
        logger.debug(f"Evil activations shape: {evil_activations.shape}")
        logger.debug(f"Evil activations stats: min={evil_activations.min().item():.6f}, max={evil_activations.max().item():.6f}, mean={evil_activations.mean().item():.6f}")
        logger.debug(f"Non-zero evil activations: {(evil_activations != 0).sum().item()} / {evil_activations.numel()}")
        
        stats['evil_mean'] = evil_activations.mean(dim=1)
        stats['evil_std'] = evil_activations.std(dim=1) if len(evil_logit_indices) > 1 else torch.zeros_like(stats['evil_mean'])
        stats['evil_count'] = len(evil_logit_indices)
        
        logger.debug(f"Evil mean stats: min={stats['evil_mean'].min().item():.6f}, max={stats['evil_mean'].max().item():.6f}, mean={stats['evil_mean'].mean().item():.6f}")
    
    if safe_logit_indices:
        safe_activations = feature_to_logit_matrix[:, safe_logit_indices]
        logger.debug(f"Safe activations shape: {safe_activations.shape}")
        logger.debug(f"Safe activations stats: min={safe_activations.min().item():.6f}, max={safe_activations.max().item():.6f}, mean={safe_activations.mean().item():.6f}")
        logger.debug(f"Non-zero safe activations: {(safe_activations != 0).sum().item()} / {safe_activations.numel()}")
        
        stats['safe_mean'] = safe_activations.mean(dim=1)
        stats['safe_std'] = safe_activations.std(dim=1) if len(safe_logit_indices) > 1 else torch.zeros_like(stats['safe_mean'])
        stats['safe_count'] = len(safe_logit_indices)
        
        logger.debug(f"Safe mean stats: min={stats['safe_mean'].min().item():.6f}, max={stats['safe_mean'].max().item():.6f}, mean={stats['safe_mean'].mean().item():.6f}")
    
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


def match_tokens_to_logits(graph: Graph, evil_tokens: List[str], safe_tokens: List[str], tokenizer, topk: int = 10) -> Tuple[List[int], List[int]]:
    """Match labeled tokens to graph logit indices more accurately."""
    # Get the actual logit tokens from the graph - only consider top-k
    logit_tokens = graph.logit_tokens[:topk]
    
    # Convert token IDs to strings using the provided tokenizer
    if hasattr(logit_tokens, 'tolist'):
        token_ids = logit_tokens.tolist()
    else:
        token_ids = list(logit_tokens)
    
    try:
        logit_token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]
    except Exception as e:
        logger.error(f"Failed to decode token IDs with tokenizer: {e}")
        # Fallback to treating as strings (will likely fail)
        logit_token_strings = [str(token_id) for token_id in token_ids]
    
    logger.debug(f"Graph has {len(logit_token_strings)} logit tokens (top-{topk})")
    logger.debug(f"Graph token IDs: {logit_tokens.tolist() if hasattr(logit_tokens, 'tolist') else list(logit_tokens)}")
    logger.debug(f"Graph token strings: {logit_token_strings}")
    logger.debug(f"Evil tokens to match: {evil_tokens}")
    logger.debug(f"Safe tokens to match: {safe_tokens}")
    
    evil_logit_indices = []
    safe_logit_indices = []
    
    # Find exact matches first
    for i, logit_token in enumerate(logit_token_strings):
        # Clean up token representation for comparison
        clean_token = logit_token.strip().replace("'", "").replace('"', '')
        
        if clean_token in evil_tokens:
            evil_logit_indices.append(i)
            logger.debug(f"EXACT MATCH: evil token '{clean_token}' at index {i}")
        elif clean_token in safe_tokens:
            safe_logit_indices.append(i)
            logger.debug(f"EXACT MATCH: safe token '{clean_token}' at index {i}")
    
    # If no exact matches, try substring matching
    if not evil_logit_indices and not safe_logit_indices:
        logger.debug("No exact matches found, trying substring matching...")
        for i, logit_token in enumerate(logit_token_strings):
            clean_token = logit_token.strip().replace("'", "").replace('"', '')
            
            # Check if any evil token is a substring of this logit token
            for evil_token in evil_tokens:
                if evil_token.lower() in clean_token.lower() or clean_token.lower() in evil_token.lower():
                    if i not in evil_logit_indices:
                        evil_logit_indices.append(i)
                        logger.debug(f"SUBSTRING MATCH: evil token '{evil_token}' matched with '{clean_token}' at index {i}")
                    break
            
            # Check if any safe token is a substring of this logit token
            for safe_token in safe_tokens:
                if safe_token.lower() in clean_token.lower() or clean_token.lower() in safe_token.lower():
                    if i not in safe_logit_indices:
                        safe_logit_indices.append(i)
                        logger.debug(f"SUBSTRING MATCH: safe token '{safe_token}' matched with '{clean_token}' at index {i}")
                    break
    
    logger.info(f"Matched {len(evil_logit_indices)} evil tokens and {len(safe_logit_indices)} safe tokens")
    
    if not evil_logit_indices and not safe_logit_indices:
        logger.warning("NO TOKEN MATCHES FOUND - check token formats!")
    
    return evil_logit_indices, safe_logit_indices


def analyze_single_model(
    model_name: str,
    model_display_name: str,
    labeled_data: Dict[str, Dict],
    graphs_dir: Path,
    output_dir: Path,
    top_k_features: int = 50,
    topk_logits: int = 10,
    base_model_path: str = "meta-llama/Llama-3.2-1B-Instruct",
    adapted_model_path: str = None,
    generate_missing_graphs: bool = False
) -> Dict:
    """Analyze a single model and save results to avoid memory buildup."""
    
    logger.info(f"Analyzing model: {model_display_name}")
    
    # Load tokenizer for token decoding
    tokenizer = None
    try:
        is_adapted = not (model_name == base_model_path or "meta-llama" in model_name)
        if is_adapted and adapted_model_path:
            resolved_path = resolve_model_path(adapted_model_path)
            logger.debug(f"Loading tokenizer from adapted model path: {resolved_path}")
            tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        else:
            resolved_path = resolve_model_path(base_model_path)
            logger.debug(f"Loading tokenizer from base model path: {resolved_path}")
            tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        logger.info(f"Successfully loaded tokenizer")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return {}
    
    # Build replacement model once for all prompts if we need to generate graphs
    replacement_model = None
    if generate_missing_graphs:
        is_base_model = model_name == base_model_path or "meta-llama" in model_name
        if is_base_model:
            logger.info(f"Building base model: {base_model_path}")
            replacement_model = build_base_replacement_model(base_model_path, "cuda")
        else:
            if not adapted_model_path:
                raise ValueError(f"adapted_model_path must be provided for adapted model: {model_display_name}")
            logger.info(f"Building adapted model: {adapted_model_path}")
            replacement_model = build_replacement_model(
                model_display_name, adapted_model_path, base_model_path, "cuda"
            )
    
    # Accumulate statistics across all prompts
    accumulated_stats = defaultdict(list)
    prompt_results = {}
    
    processed_count = 0
    
    for stable_id, token_data in labeled_data.items():
        # Find corresponding graph
        is_adapted = not (model_name == base_model_path or "meta-llama" in model_name)
        graph_path = find_matching_graph(stable_id, model_display_name, graphs_dir, is_adapted)
        
        if not graph_path:
            if generate_missing_graphs:
                # Check if this is an adapted model and we should skip generation
                is_base_model = model_name == base_model_path or "meta-llama" in model_name
                if not is_base_model and hasattr(analyze_single_model, '_skip_adapted_generation'):
                    logger.warning(f"Skipping graph generation for adapted model {model_display_name} due to memory constraints")
                else:
                    logger.info(f"Generating missing graph for {stable_id} with model {model_display_name}")
                    graph_path = generate_graph_with_model(
                        stable_id, model_display_name, token_data['prompt_text'],
                        graphs_dir, replacement_model
                    )
            
            if not graph_path:
                logger.warning(f"No graph found for {stable_id} with model {model_display_name}")
                continue
        
        try:
            # Load graph
            graph = Graph.from_pt(str(graph_path))
            
            # Match labeled tokens to graph logit indices
            evil_logit_indices, safe_logit_indices = match_tokens_to_logits(
                graph, token_data['evil_tokens'], token_data['safe_tokens'], tokenizer, topk=topk_logits
            )
            
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
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
        is_adapted = not (model_name == base_model_path or "meta-llama" in model_name)
        sample_graph_path = find_matching_graph(list(labeled_data.keys())[0], model_display_name, graphs_dir, is_adapted)
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
    model_safe_name = model_display_name.replace('/', '_').replace(' ', '_')
    results = {
        'model_name': model_name,
        'model_display_name': model_display_name,
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
    
    # Clean up replacement model from memory
    if replacement_model is not None:
        del replacement_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Cleaned up replacement model from memory")
    
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
            'base_model': base_results['model_display_name'],
            'adapted_model': adapted_results['model_display_name'],
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
    parser.add_argument("--adapted-model-path", type=str,
                       help="Path to adapted model (required for adapted model analysis)")
    parser.add_argument("--adapted-model-name", type=str,
                       default="adapted_model",
                       help="Display name for adapted model")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Number of top features to analyze")
    parser.add_argument("--topk-logits", type=int, default=10,
                       help="Number of top logit tokens to consider for matching")
    parser.add_argument("--generate-missing-graphs", action="store_true",
                       help="Generate graphs if they don't exist")
    parser.add_argument("--skip-adapted-generation", action="store_true",
                       help="Skip graph generation for adapted models to save memory")
    parser.add_argument("--adapted-only", action="store_true",
                       help="Only analyze adapted model (skip base model to save memory)")
    parser.add_argument("--base-only", action="store_true",
                       help="Only analyze base model (skip adapted model)")
    parser.add_argument("--debug-convergence", action="store_true",
                       help="Stop after first convergence failure for debugging")
    parser.add_argument("--models", type=str, nargs='+',
                       help="Specific model display names to analyze (overrides base/adapted)")
    parser.add_argument("--model-paths", type=str, nargs='+',
                       help="Corresponding model paths (same order as --models)")
    
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
    if args.models:
        if args.model_paths and len(args.model_paths) != len(args.models):
            logger.error("Number of model paths must match number of models")
            return
        
        models_to_analyze = []
        for i, model_name in enumerate(args.models):
            model_path = args.model_paths[i] if args.model_paths else None
            models_to_analyze.append((args.base_model, model_name, model_path))
    else:
        # Handle base-only, adapted-only, or both
        models_to_analyze = []
        
        # Add base model unless adapted-only is specified
        if not args.adapted_only:
            models_to_analyze.append((args.base_model, args.base_model, None))
        
        # Add adapted model unless base-only is specified
        if not args.base_only and args.adapted_model_path:
            models_to_analyze.append((
                args.adapted_model_path, 
                args.adapted_model_name, 
                args.adapted_model_path
            ))
        
        # Validation
        if args.adapted_only and not args.adapted_model_path:
            logger.error("--adapted-only requires --adapted-model-path")
            return
        
        if not models_to_analyze:
            logger.error("No models to analyze. Check your arguments.")
            return
    
    results = {}
    
    # Analyze each model separately to avoid memory issues
    for model_name, model_display_name, model_path in models_to_analyze:
        logger.info(f"Starting analysis for model: {model_display_name}")
        
        # Load labeled tokens for this model
        labeled_data = load_labeled_tokens(labeled_logits_dir, model_display_name)
        
        if not labeled_data:
            logger.warning(f"No labeled data found for model: {model_display_name}")
            continue
        
        # Set skip flag for adapted models if requested
        if args.skip_adapted_generation:
            analyze_single_model._skip_adapted_generation = True
        
        # Set debug convergence flag if requested
        if args.debug_convergence:
            generate_graph_with_model._debug_convergence = True
        
        # Analyze this model
        model_results = analyze_single_model(
            model_name, model_display_name, labeled_data, graphs_dir, output_dir, 
            args.top_k, args.topk_logits, args.base_model, model_path, args.generate_missing_graphs
        )
        
        results[model_display_name] = model_results
        
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compare models if we have both base and adapted
    if len(results) >= 2 and args.base_model in results and args.adapted_model_name in results:
        compare_models(results[args.base_model], results[args.adapted_model_name], output_dir)
    
    logger.info("Analysis complete!")
    print(f"\nResults saved to: {output_dir}")
    print(f"Analyzed {len(results)} models")


if __name__ == "__main__":
    main()


# Example usage (run from the emergent_misalignment_circuits directory with venv activated):
# source .venv/bin/activate

# TWO-STAGE APPROACH (recommended for memory constraints):
# Stage 1: Analyze base model only
# python analyze_evil_features.py \
#   --labeled-logits-dir labeled_logits \
#   --graphs-dir results \
#   --base-only \
#   --generate-missing-graphs

# Stage 2: Analyze adapted model only  
# python analyze_evil_features.py \
#   --labeled-logits-dir labeled_logits \
#   --graphs-dir results \
#   --adapted-only \
#   --adapted-model-path results/bad_med_layer4_r1/merged_model \
#   --adapted-model-name "bad_med_layer4_r1" \
#   --generate-missing-graphs

# SINGLE RUN (if you have enough memory):
# python analyze_evil_features.py \
#   --labeled-logits-dir labeled_logits \
#   --graphs-dir results \
#   --adapted-model-path results/bad_med_layer4_r1/merged_model \
#   --adapted-model-name "bad_med_layer4_r1" \
#   --generate-missing-graphs