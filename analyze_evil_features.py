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

import sys
sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.graph import Graph

# Import graph generation functions
from generate_graphs import build_replacement_model, build_base_replacement_model
from circuit_tracer.attribution import attribute
from transformers import AutoTokenizer
from circuit_tracer.frontend.utils import process_token

logging.basicConfig(level=logging.INFO)
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
            
            # Only include data with human labels (check for human_verdict)
            human_verdict = data.get('human_verdict')
            if not human_verdict or human_verdict == 'interrupted':
                logger.info(f"Skipping prompt without human labels: {stable_id}")
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
                    'verdict_challenged': data.get('verdict_challenged', False),
                    'human_verdict': human_verdict
                }
                
        except Exception as e:
            logger.warning(f"Error reading {logit_file}: {e}")
    
    logger.info(f"Loaded {len(labeled_data)} human-labeled prompts for {model_name}")
    return labeled_data


def find_matching_graph(stable_id: str, model_name: str, graphs_dir: Path) -> Optional[Path]:
    """Find existing graph file for a given stable_id and model."""
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    prompt_dir = graphs_dir / model_safe_name / stable_id
    
    # Check for different graph file types
    possible_files = [
        "base_graph.pt",
        "merged_graph.pt", 
        "graph.pt"
    ]
    
    for filename in possible_files:
        graph_path = prompt_dir / filename
        if graph_path.exists():
            logger.info(f"Found existing graph: {graph_path}")
            return graph_path
    
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
    existing_graph = find_matching_graph(stable_id, model_display_name, graphs_dir)
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


def match_tokens_to_logits(graph: Graph, evil_tokens: List[str], safe_tokens: List[str]) -> Tuple[List[int], List[int]]:
    """Match labeled tokens to graph logit indices using the model tokenizer.

    The graph only stores the *IDs* of the top-k logits. In order to compare
    them with the human-readable token strings coming from the labeled-logits
    JSON files we need to decode those IDs with the correct tokenizer.
    """

    # 1. Load tokenizer
    tokenizer_name = getattr(graph.cfg, "tokenizer_name", None)
    if tokenizer_name is None:
        # Fallback: try to infer from model family – this should rarely happen
        tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"

    logger.info(f"TOKEN MATCHING: Using tokenizer: {tokenizer_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        logger.info("TOKEN MATCHING: Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(
            f"Failed to load tokenizer '{tokenizer_name}' ({e}). Falling back to default decoding."
        )
        tokenizer = None

    # 2. Decode logit IDs to strings
    logit_ids = (
        graph.logit_tokens.tolist() if hasattr(graph.logit_tokens, "tolist") else list(graph.logit_tokens)
    )

    logger.info(f"TOKEN MATCHING: Graph contains {len(logit_ids)} logit tokens with IDs: {logit_ids}")

    if tokenizer is not None:
        logit_token_strings = [process_token(tokenizer.decode([int(tid)], skip_special_tokens=False)) for tid in logit_ids]
        # Also log the raw decoded tokens for comparison
        raw_decoded = [tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in logit_ids]
        logger.info(f"TOKEN MATCHING: Raw decoded tokens (before process_token): {raw_decoded}")
    else:
        # Fallback – keep raw IDs as strings (very unlikely to match)
        logit_token_strings = [str(tid) for tid in logit_ids]

    logger.info(f"TOKEN MATCHING: Processed logit tokens: {logit_token_strings}")

    # Normalise labeled tokens for comparison (apply same processing as above)
    processed_evil_tokens = {process_token(tok) for tok in evil_tokens}
    processed_safe_tokens = {process_token(tok) for tok in safe_tokens}
    
    # Check if evil tokens are in the graph at all
    missing_evil = [tok for tok in evil_tokens if process_token(tok) not in logit_token_strings]
    missing_safe = [tok for tok in safe_tokens if process_token(tok) not in logit_token_strings]
    
    if missing_evil:
        logger.warning(f"TOKEN MATCHING: Evil tokens NOT in graph logits: {missing_evil}")
        logger.warning(f"TOKEN MATCHING: Expected evil tokens should be in position 4 of labeled data but graph only has {len(logit_token_strings)} tokens")
    if missing_safe:
        logger.info(f"TOKEN MATCHING: Safe tokens NOT in graph logits: {missing_safe}")
        
    # Show what IS in the graph but not in our labeled tokens
    unlabeled_tokens = [tok for tok in logit_token_strings if tok not in processed_evil_tokens and tok not in processed_safe_tokens]
    if unlabeled_tokens:
        logger.info(f"TOKEN MATCHING: Unlabeled tokens in graph: {unlabeled_tokens}")
        
    # Show detailed token-by-token comparison
    logger.info("TOKEN MATCHING: Detailed token comparison:")
    for i, graph_token in enumerate(logit_token_strings):
        status = "UNLABELED"
        if graph_token in processed_evil_tokens:
            status = "EVIL"
        elif graph_token in processed_safe_tokens:
            status = "SAFE"
        logger.info(f"TOKEN MATCHING:   Graph[{i}]: '{graph_token}' -> {status}")

    logger.info(f"TOKEN MATCHING: Input evil tokens: {evil_tokens}")
    logger.info(f"TOKEN MATCHING: Processed evil tokens: {processed_evil_tokens}")
    logger.info(f"TOKEN MATCHING: Input safe tokens: {safe_tokens}")
    logger.info(f"TOKEN MATCHING: Processed safe tokens: {processed_safe_tokens}")

    evil_logit_indices: List[int] = []
    safe_logit_indices: List[int] = []

    # 3. Exact matches first
    logger.info("TOKEN MATCHING: Starting exact matching...")
    for i, logit_tok in enumerate(logit_token_strings):
        if logit_tok in processed_evil_tokens:
            evil_logit_indices.append(i)
            logger.info(f"TOKEN MATCHING: EXACT EVIL MATCH - logit_token[{i}] = '{logit_tok}' matches evil token")
        elif logit_tok in processed_safe_tokens:
            safe_logit_indices.append(i)
            logger.info(f"TOKEN MATCHING: EXACT SAFE MATCH - logit_token[{i}] = '{logit_tok}' matches safe token")

    logger.info(f"TOKEN MATCHING: After exact matching - {len(evil_logit_indices)} evil, {len(safe_logit_indices)} safe")

    # 4. Fallback to substring / fuzzy matching if nothing matched so far
    if not evil_logit_indices and not safe_logit_indices:
        logger.info("TOKEN MATCHING: No exact matches found, trying substring matching...")
        for i, logit_tok in enumerate(logit_token_strings):
            lt_lower = logit_tok.lower()

            # Check evil tokens
            for ev_tok in processed_evil_tokens:
                if ev_tok.lower() in lt_lower or lt_lower in ev_tok.lower():
                    evil_logit_indices.append(i)
                    logger.info(f"TOKEN MATCHING: SUBSTRING EVIL MATCH - logit_token[{i}] = '{logit_tok}' matches evil token '{ev_tok}'")
                    break

            # Check safe tokens (skip if already matched)
            if i in evil_logit_indices:
                continue
            for sf_tok in processed_safe_tokens:
                if sf_tok.lower() in lt_lower or lt_lower in sf_tok.lower():
                    safe_logit_indices.append(i)
                    logger.info(f"TOKEN MATCHING: SUBSTRING SAFE MATCH - logit_token[{i}] = '{logit_tok}' matches safe token '{sf_tok}'")
                    break

        logger.info(f"TOKEN MATCHING: After substring matching - {len(evil_logit_indices)} evil, {len(safe_logit_indices)} safe")
    else:
        logger.info("TOKEN MATCHING: Skipping substring matching (exact matches found)")

    logger.info(f"TOKEN MATCHING: FINAL RESULT - {len(evil_logit_indices)} evil tokens, {len(safe_logit_indices)} safe tokens")
    if evil_logit_indices:
        evil_matches = [f"'{logit_token_strings[i]}'" for i in evil_logit_indices]
        logger.info(f"TOKEN MATCHING: Evil matches: {evil_matches}")
    if safe_logit_indices:
        safe_matches = [f"'{logit_token_strings[i]}'" for i in safe_logit_indices]
        logger.info(f"TOKEN MATCHING: Safe matches: {safe_matches}")

    return evil_logit_indices, safe_logit_indices


def analyze_single_model(
    model_name: str,
    model_display_name: str,
    labeled_data: Dict[str, Dict],
    graphs_dir: Path,
    output_dir: Path,
    top_k_features: int = 50,
    base_model_path: str = "meta-llama/Llama-3.2-1B-Instruct",
    adapted_model_path: str = None,
    generate_missing_graphs: bool = False
) -> Dict:
    """Analyze a single model and save results to avoid memory buildup."""
    
    import time
    start_time = time.time()
    logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Starting analysis for model: {model_display_name}")
    
    # Build replacement model once for all prompts if we need to generate graphs
    replacement_model = None
    if generate_missing_graphs:
        is_base_model = model_name == base_model_path or "meta-llama" in model_name
        if is_base_model:
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Building base model: {base_model_path}")
            replacement_model = build_base_replacement_model(base_model_path, "cuda")
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Base model built successfully")
        else:
            if not adapted_model_path:
                raise ValueError(f"adapted_model_path must be provided for adapted model: {model_display_name}")
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Building adapted model: {adapted_model_path}")
            replacement_model = build_replacement_model(
                model_display_name, adapted_model_path, base_model_path, "cuda"
            )
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Adapted model built successfully")
    
    # Accumulate statistics across all prompts
    accumulated_stats = defaultdict(list)
    prompt_results = {}
    
    processed_count = 0
    total_prompts = len(labeled_data)
    logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Processing {total_prompts} prompts")
    
    for i, (stable_id, token_data) in enumerate(labeled_data.items()):
        prompt_start_time = time.time()
        logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Processing prompt {i+1}/{total_prompts}: {stable_id}")
        
        # Find corresponding graph
        logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Finding graph for {stable_id}")
        graph_path = find_matching_graph(stable_id, model_display_name, graphs_dir)
        
        if not graph_path:
            if generate_missing_graphs:
                # Check if this is an adapted model and we should skip generation
                is_base_model = model_name == base_model_path or "meta-llama" in model_name
                if not is_base_model and hasattr(analyze_single_model, '_skip_adapted_generation'):
                    logger.warning(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Skipping graph generation for adapted model {model_display_name} due to memory constraints")
                else:
                    logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Generating missing graph for {stable_id} with model {model_display_name}")
                    graph_path = generate_graph_with_model(
                        stable_id, model_display_name, token_data['prompt_text'],
                        graphs_dir, replacement_model
                    )
                    if graph_path:
                        logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Graph generated successfully: {graph_path}")
            
            if not graph_path:
                logger.warning(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: No graph found for {stable_id} with model {model_display_name}")
                continue
        else:
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Found existing graph: {graph_path}")
        
        try:
            # Load graph
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Loading graph from {graph_path}")
            graph = Graph.from_pt(str(graph_path))
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Graph loaded successfully")
            
            # Match labeled tokens to graph logit indices
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Matching tokens to logits")
            evil_logit_indices, safe_logit_indices = match_tokens_to_logits(
                graph, token_data['evil_tokens'], token_data['safe_tokens']
            )
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Found {len(evil_logit_indices)} evil tokens, {len(safe_logit_indices)} safe tokens")
            
            # Compute statistics for this prompt
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Computing feature statistics")
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
                        
                logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Statistics computed successfully")
            else:
                logger.warning(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: No statistics computed for {stable_id}")
            
            processed_count += 1
            
            # Clear graph from memory
            del graph
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            prompt_elapsed = time.time() - prompt_start_time
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Processed {processed_count}/{len(labeled_data)}: {stable_id} in {prompt_elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Error processing {stable_id}: {e}")
            import traceback
            logger.error(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Traceback: {traceback.format_exc()}")
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
        sample_graph_path = find_matching_graph(list(labeled_data.keys())[0], model_display_name, graphs_dir)
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


def filter_prompts_by_evil_token_criteria(
    base_labeled_data: Dict[str, Dict], 
    adapted_labeled_data: Dict[str, Dict]
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Filter prompts to only include those where adapted model has evil tokens but base model doesn't."""
    
    logger.info("Filtering prompts by evil token criteria...")
    
    # Find prompts that exist in both models
    common_prompts = set(base_labeled_data.keys()) & set(adapted_labeled_data.keys())
    logger.info(f"Found {len(common_prompts)} prompts in both base and adapted models")
    
    filtered_base = {}
    filtered_adapted = {}
    
    for prompt_id in common_prompts:
        base_data = base_labeled_data[prompt_id]
        adapted_data = adapted_labeled_data[prompt_id]
        
        base_has_evil = len(base_data['evil_tokens']) > 0
        adapted_has_evil = len(adapted_data['evil_tokens']) > 0
        
        # Only include if adapted has evil tokens but base doesn't
        if adapted_has_evil and not base_has_evil:
            filtered_base[prompt_id] = base_data
            filtered_adapted[prompt_id] = adapted_data
            logger.info(f"Including prompt {prompt_id}: adapted has {len(adapted_data['evil_tokens'])} evil tokens, base has 0")
    
    logger.info(f"Filtered to {len(filtered_adapted)} prompts where adapted model has evil tokens but base model doesn't")
    
    if filtered_adapted:
        sample_prompt = list(filtered_adapted.keys())[0]
        logger.info(f"Sample filtered prompt: {sample_prompt}")
        logger.info(f"  Base evil tokens: {filtered_base[sample_prompt]['evil_tokens']}")
        logger.info(f"  Adapted evil tokens: {filtered_adapted[sample_prompt]['evil_tokens']}")
    
    return filtered_base, filtered_adapted


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
    parser.add_argument("--labeled-logits-dir", type=str, default="results/pivotal_tokens/labeled_logits",
                       help="Directory containing labeled logit files")
    parser.add_argument("--graphs-dir", type=str, default="results",
                       help="Directory containing attribution graph files")
    parser.add_argument("--output-dir", type=str, default="results/evil_feature_analysis",
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
    
    # First pass: Load all labeled data
    all_labeled_data = {}
    for model_name, model_display_name, model_path in models_to_analyze:
        logger.info(f"Loading labeled data for model: {model_display_name}")
        labeled_data = load_labeled_tokens(labeled_logits_dir, model_display_name)
        
        if not labeled_data:
            logger.warning(f"No labeled data found for model: {model_display_name}")
            continue
            
        all_labeled_data[model_display_name] = labeled_data
    
    # Apply filtering if we have both base and adapted models
    if len(all_labeled_data) >= 2 and not args.models:
        base_model_key = args.base_model
        adapted_model_key = args.adapted_model_name
        
        if base_model_key in all_labeled_data and adapted_model_key in all_labeled_data:
            logger.info("Applying evil token filtering criteria...")
            filtered_base, filtered_adapted = filter_prompts_by_evil_token_criteria(
                all_labeled_data[base_model_key], 
                all_labeled_data[adapted_model_key]
            )
            
            # Update the labeled data with filtered results
            all_labeled_data[base_model_key] = filtered_base
            all_labeled_data[adapted_model_key] = filtered_adapted
            
            logger.info(f"After filtering: {len(filtered_base)} base prompts, {len(filtered_adapted)} adapted prompts")
    
    # Second pass: Analyze each model with filtered data
    for model_name, model_display_name, model_path in models_to_analyze:
        if model_display_name not in all_labeled_data:
            continue
            
        logger.info(f"Starting analysis for model: {model_display_name}")
        labeled_data = all_labeled_data[model_display_name]
        
        # Set skip flag for adapted models if requested
        if args.skip_adapted_generation:
            analyze_single_model._skip_adapted_generation = True
        
        # Set debug convergence flag if requested
        if args.debug_convergence:
            generate_graph_with_model._debug_convergence = True
        
        # Analyze this model
        model_results = analyze_single_model(
            model_name, model_display_name, labeled_data, graphs_dir, output_dir, 
            args.top_k, args.base_model, model_path, args.generate_missing_graphs
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


# Example usage (run from the emergent_misalignment_circuits directory):
# This script now automatically uses the new pivotal_tokens file structure and 
# filters for prompts where you found evil tokens in the adapted model but not in the base model

# TWO-STAGE APPROACH (recommended for memory constraints):
# Stage 1: Analyze base model only
# python analyze_evil_features.py \
#   --base-only \
#   --generate-missing-graphs

# Stage 2: Analyze adapted model only  
# python analyze_evil_features.py \
#   --adapted-only \
#   --adapted-model-path results/bad_med_layer4_r1/merged_model \
#   --adapted-model-name "emergent_misalignment_circuits_results_bad_med_layer4_r1_merged_model" \
#   --generate-missing-graphs

# SINGLE RUN (analyzes both models with filtering):
# python analyze_evil_features.py \
#   --adapted-model-path results/bad_med_layer4_r1/merged_model \
#   --adapted-model-name "emergent_misalignment_circuits_results_bad_med_layer4_r1_merged_model" \
#   --generate-missing-graphs

# The script will now:
# 1. Only use human-labeled data (skips prompts without human_verdict)
# 2. Filter to prompts where adapted model has evil tokens but base model doesn't  
# 3. Generate analysis comparing what features differentiate these cases