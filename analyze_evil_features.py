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
import time
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
from generate_graphs import generate_graphs, load_prompts_from_dir
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


def get_feature_analysis_dir(model_name: str, base_dir: Path = None) -> Path:
    """Get the directory for storing individual prompt feature analysis results."""
    if base_dir is None:
        base_dir = Path("results/pivotal_tokens")
    
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    feature_analysis_dir = base_dir / "feature_analysis" / model_safe_name
    feature_analysis_dir.mkdir(parents=True, exist_ok=True)
    return feature_analysis_dir


def save_prompt_feature_analysis(
    stable_id: str, 
    model_name: str, 
    feature_stats: Dict, 
    prompt_metadata: Dict,
    base_dir: Path = None
) -> Path:
    """Save individual prompt feature analysis results with hash filename."""
    feature_analysis_dir = get_feature_analysis_dir(model_name, base_dir)
    result_file = feature_analysis_dir / f"{stable_id}.json"
    
    # Convert tensors to lists for JSON serialization
    serializable_stats = {}
    for key, value in feature_stats.items():
        if isinstance(value, torch.Tensor):
            serializable_stats[key] = value.cpu().tolist()
        else:
            serializable_stats[key] = value
    
    result_data = {
        'stable_id': stable_id,
        'model_name': model_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'feature_statistics': serializable_stats,
        'metadata': prompt_metadata
    }
    
    save_json_file(result_file, result_data)
    logger.info(f"Saved feature analysis for {stable_id} to {result_file}")
    return result_file


def load_prompt_feature_analysis(
    stable_id: str, 
    model_name: str, 
    base_dir: Path = None
) -> Optional[Dict]:
    """Load cached individual prompt feature analysis results."""
    feature_analysis_dir = get_feature_analysis_dir(model_name, base_dir)
    result_file = feature_analysis_dir / f"{stable_id}.json"
    
    if not result_file.exists():
        return None
    
    try:
        return load_json_file(result_file)
    except Exception as e:
        logger.warning(f"Error loading cached feature analysis for {stable_id}: {e}")
        return None


def check_cached_feature_analysis(
    labeled_data: Dict[str, Dict], 
    model_name: str,
    base_dir: Path = None
) -> Tuple[Dict[str, Dict], List[str]]:
    """Check which prompts have cached feature analysis results."""
    cached_results = {}
    missing_prompts = []
    
    for stable_id in labeled_data.keys():
        cached_result = load_prompt_feature_analysis(stable_id, model_name, base_dir)
        if cached_result:
            cached_results[stable_id] = cached_result
            logger.info(f"Found cached feature analysis for {stable_id}")
        else:
            missing_prompts.append(stable_id)
    
    logger.info(f"Found {len(cached_results)} cached results, {len(missing_prompts)} prompts need analysis")
    return cached_results, missing_prompts


def load_labeled_tokens(labeled_logits_dir: Path, model_name: str) -> Dict[str, Dict]:
    """Load labeled tokens for a specific model."""
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    
    # For human labels, always use the labeled_logits directory
    # The graph_runs structure only has O3 labels, not human labels
    model_dir = labeled_logits_dir / model_safe_name
    logger.info(f"Using labeled_logits structure for model: {model_name}")
    
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
            
            # Extract tokens by label (including unsure ones for matching)
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
            
            if evil_tokens or safe_tokens or unsure_tokens:
                labeled_data[stable_id] = {
                    'evil_tokens': evil_tokens,
                    'safe_tokens': safe_tokens,
                    'unsure_tokens': unsure_tokens,
                    'prompt_text': data.get('prompt_text', ''),
                    'verdict_challenged': data.get('verdict_challenged', False),
                    'human_verdict': human_verdict
                }
                
        except Exception as e:
            logger.warning(f"Error reading {logit_file}: {e}")
    
    logger.info(f"Loaded {len(labeled_data)} human-labeled prompts for {model_name}")
    return labeled_data


def find_matching_graph(stable_id: str, model_name: str, graphs_dir: Path, is_base_model: bool = False) -> Optional[Path]:
    """Find existing graph file for a given stable_id and model, ensuring correct graph type for model type."""
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    
    # Determine correct graph filename based on model type
    if model_name == "base_replacement" or is_base_model:
        target_graph_file = "base_graph.pt"
        logger.info(f"Looking for BASE graph for model: {model_name}")
    elif model_name.startswith("merged_replacement") or model_name == "merged_replacement":
        target_graph_file = "merged_graph.pt" 
        logger.info(f"Looking for MERGED graph for model: {model_name}")
    elif "adapted" in model_name.lower() or "merged" in model_name.lower():
        target_graph_file = "merged_graph.pt" 
        logger.info(f"Looking for MERGED graph for adapted model: {model_name}")
    else:
        # Default fallback logic
        target_graph_file = "merged_graph.pt" if not is_base_model else "base_graph.pt"
        logger.info(f"Using fallback graph selection for model: {model_name} -> {target_graph_file}")
    
    
    # PRIORITY 2: Messy divergence generator structure (current reality)
    # Look in the CORRECT directory based on model type
    if model_name in ["base_replacement", "merged_replacement"]:
        # For base_replacement, look ONLY in base_replacement directory
        # For merged_replacement, look ONLY in merged_replacement directory
        correct_dir = model_name  # "base_replacement" or "merged_replacement"
        search_path = graphs_dir / correct_dir / stable_id / target_graph_file
        if search_path.exists():
            logger.info(f"Found CORRECT graph in correct directory: {search_path}")
            return search_path
        

    
    # PRIORITY 3: Legacy structure (may have logit alignment issues)
    prompt_dir = graphs_dir / model_safe_name / stable_id
    
    # Try exact match first
    exact_path = prompt_dir / target_graph_file
    if exact_path.exists():
        logger.warning(f"Found LEGACY graph with exact match: {exact_path}")
        return exact_path
    
    # Legacy fallback with prioritization
    if is_base_model:
        possible_files = ["base_graph.pt"]
    else:
        possible_files = ["merged_graph.pt"]
    
    for filename in possible_files:
        graph_path = prompt_dir / filename
        if graph_path.exists():
            logger.warning(f"Found LEGACY graph with fallback: {graph_path}")
            return graph_path
    
    logger.error(f"No graph found for {stable_id} with model {model_name}")
    return None


def ensure_graphs_exist(
    labeled_data: Dict[str, Dict],
    model_name: str,
    model_display_name: str,
    graphs_dir: Path,
    base_model_path: str,
    adapted_model_path: Optional[str] = None,
    device: str = "cuda",
    override: bool = False
) -> None:
    """Ensure graphs exist for all prompts in labeled_data using generate_graphs functionality."""
    
    logger.info(f"Ensuring graphs exist for {len(labeled_data)} prompts with model {model_display_name}")
    
    # Convert labeled data to prompt format expected by generate_graphs
    prompts_for_generation = []
    for stable_id, token_data in labeled_data.items():
        prompt_data = {
            'name': stable_id,
            'prompt': token_data['prompt_text'],
            'stable_id': stable_id,
            'prompt_text': token_data['prompt_text']
        }
        prompts_for_generation.append(prompt_data)
    
    # Determine model paths based on model type
    is_base_model = model_name == base_model_path or "meta-llama" in model_name
    if is_base_model:
        model_path_for_generation = base_model_path
    else:
        if not adapted_model_path:
            raise ValueError(f"adapted_model_path must be provided for adapted model: {model_display_name}")
        model_path_for_generation = adapted_model_path
    
    # Call generate_graphs to create missing graphs
    try:
        logger.info(f"Calling generate_graphs for model {model_display_name}")
        generate_graphs(
            model_name=model_display_name,
            model_path=model_path_for_generation,
            base_model_path=base_model_path,
            prompts=prompts_for_generation,
            results_dir=graphs_dir,
            device=device,
            serve_graphs=False,  # Don't start servers during analysis
            override=override
        )
        logger.info(f"Graph generation completed for model {model_display_name}")
        
    except Exception as e:
        logger.error(f"Failed to generate graphs for model {model_display_name}: {e}")
        raise


def compute_feature_statistics(graph: Graph, evil_logit_indices: List[int], safe_logit_indices: List[int]) -> Dict:
    """Compute feature activation statistics for evil vs safe tokens using sign-preserving edge influence."""
    if not evil_logit_indices and not safe_logit_indices:
        return {}
    
    n_features = len(graph.selected_features)
    n_logits = len(graph.logit_tokens)
    layers = graph.cfg.n_layers
    
    logger.info(f"FEATURE EXTRACTION DEBUG: n_features={n_features}, n_logits={n_logits}")
    logger.info(f"FEATURE EXTRACTION DEBUG: adjacency_matrix shape={graph.adjacency_matrix.shape}")
    
    # Import edge influence computation functions
    from circuit_tracer.graph import compute_influence
    
    # Define sign-preserving normalization function
    def normalize_matrix_preserve_sign(matrix: torch.Tensor) -> torch.Tensor:
        """Normalize matrix while preserving sign information."""
        abs_sum = matrix.abs().sum(dim=1, keepdim=True).clamp(min=1e-10)
        return matrix / abs_sum
    
    # Compute sign-preserving edge influence scores
    def compute_edge_influence_preserve_sign(adjacency_matrix: torch.Tensor, logit_weights: torch.Tensor):
        """Compute edge influence scores while preserving sign information."""
        normalized_matrix = normalize_matrix_preserve_sign(adjacency_matrix)
        influence = compute_influence(normalized_matrix, logit_weights)
        influence += logit_weights
        edge_scores = normalized_matrix * influence[:, None]
        return edge_scores
    
    # Set up logit weights
    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities
    
    # Compute edge influence scores (pairwise attributions)
    logger.info("Computing sign-preserving edge influence scores...")
    edge_scores = compute_edge_influence_preserve_sign(graph.adjacency_matrix, logit_weights)
    
    # Calculate index ranges for mapping
    error_end_idx = n_features + graph.n_pos * layers
    token_end_idx = error_end_idx + len(graph.input_tokens)
    
    # Extract feature-to-logit attributions from edge scores
    # edge_scores[token_end_idx:, :n_features] gives us [logits, features]
    # We want [features, logits] so we transpose
    feature_to_logit_matrix = edge_scores[token_end_idx:, :n_features].T  # [features, logits]
    
    logger.info(f"FEATURE EXTRACTION DEBUG: feature_to_logit_matrix shape={feature_to_logit_matrix.shape}")
    logger.info(f"FEATURE EXTRACTION DEBUG: feature_to_logit_matrix range=[{feature_to_logit_matrix.min():.6f}, {feature_to_logit_matrix.max():.6f}]")
    logger.info(f"FEATURE EXTRACTION DEBUG: non-zero elements={torch.count_nonzero(feature_to_logit_matrix)}/{feature_to_logit_matrix.numel()}")
    logger.info(f"FEATURE EXTRACTION DEBUG: negative elements={torch.count_nonzero(feature_to_logit_matrix < 0)}")
    logger.info(f"FEATURE EXTRACTION DEBUG: positive elements={torch.count_nonzero(feature_to_logit_matrix > 0)}")
    
    if torch.count_nonzero(feature_to_logit_matrix) == 0:
        logger.error("🚨 CRITICAL: Feature-to-logit matrix is ALL ZEROS!")
        logger.error("This means no features have any influence on logits according to edge influence computation.")
        return {}
    
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


def match_tokens_to_logits(graph: Graph, evil_tokens: List[str], safe_tokens: List[str], unsure_tokens: List[str] = None) -> Tuple[List[int], List[int]]:
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
    processed_unsure_tokens = {process_token(tok) for tok in (unsure_tokens or [])}
    
    # Check if labeled tokens are in the graph at all
    missing_evil = [tok for tok in evil_tokens if process_token(tok) not in logit_token_strings]
    missing_safe = [tok for tok in safe_tokens if process_token(tok) not in logit_token_strings]
    missing_unsure = [tok for tok in (unsure_tokens or []) if process_token(tok) not in logit_token_strings]
    
    if missing_evil:
        logger.warning(f"TOKEN MATCHING: Evil tokens NOT in graph logits: {missing_evil}")
    if missing_safe:
        logger.warning(f"TOKEN MATCHING: Safe tokens NOT in graph logits: {missing_safe}")
    if missing_unsure:
        logger.warning(f"TOKEN MATCHING: Unsure tokens NOT in graph logits: {missing_unsure}")
        
    # Show what IS in the graph but not in our labeled tokens (truly unlabeled)
    all_labeled_tokens = processed_evil_tokens | processed_safe_tokens | processed_unsure_tokens
    truly_unlabeled_tokens = [tok for tok in logit_token_strings if tok not in all_labeled_tokens]
    
    if truly_unlabeled_tokens:
        logger.error(f"TOKEN MATCHING: TRULY UNLABELED tokens in graph (MISSING LABELS): {truly_unlabeled_tokens}")
        logger.error("TOKEN MATCHING: This indicates a mismatch between labeled data and graph!")
    
    if unsure_tokens:
        logger.info(f"TOKEN MATCHING: Unsure tokens in graph (excluded from analysis): {[tok for tok in logit_token_strings if tok in processed_unsure_tokens]}")
        
    # Show detailed token-by-token comparison
    logger.info("TOKEN MATCHING: Detailed token comparison:")
    for i, graph_token in enumerate(logit_token_strings):
        if graph_token in processed_evil_tokens:
            status = "EVIL"
        elif graph_token in processed_safe_tokens:
            status = "SAFE"
        elif graph_token in processed_unsure_tokens:
            status = "UNSURE"
        else:
            status = "UNLABELED"
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
    debug_prompt: str = None,
    use_cache: bool = True,
    base_dir: Path = None
) -> Dict:
    """Analyze a single model and save results to avoid memory buildup."""
    
    import time
    start_time = time.time()
    logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Starting analysis for model: {model_display_name}")
    
    # Check for cached results if enabled
    cached_results = {}
    prompts_to_process = labeled_data.copy()
    
    if use_cache:
        logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Checking for cached feature analysis results...")
        cached_results, missing_prompt_ids = check_cached_feature_analysis(
            labeled_data, 
            model_display_name, 
            base_dir
        )
        
        if cached_results:
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Found {len(cached_results)} cached results, processing {len(missing_prompt_ids)} missing prompts")
            # Filter to only process missing prompts
            prompts_to_process = {pid: labeled_data[pid] for pid in missing_prompt_ids if pid in labeled_data}
        else:
            logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: No cached results found, processing all {len(labeled_data)} prompts")
    
    # Accumulate statistics across all prompts
    accumulated_stats = defaultdict(list)
    prompt_results = {}
    
    processed_count = 0
    total_prompts = len(prompts_to_process)
    logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Processing {total_prompts} prompts (from {len(labeled_data)} total, {len(cached_results)} cached)")
    
    for i, (stable_id, token_data) in enumerate(prompts_to_process.items()):
        prompt_start_time = time.time()
        logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Processing prompt {i+1}/{total_prompts}: {stable_id}")
        
        # Find corresponding graph
        logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Finding graph for {stable_id}")
        is_base_model = model_display_name == "base_replacement"
        
        # Debug specific prompt if requested
        debug_this_prompt = (debug_prompt and stable_id == debug_prompt)
        if debug_this_prompt:
            logger.error(f"🔍 DEBUGGING PROMPT {stable_id}:")
            logger.error(f"  model_name: '{model_name}'")
            logger.error(f"  model_display_name: '{model_display_name}'")
            logger.error(f"  base_model_path: '{base_model_path}'")
            logger.error(f"  is_base_model: {is_base_model}")
            logger.error(f"  token_data evil_tokens: {token_data['evil_tokens']}")
            logger.error(f"  token_data safe_tokens: {token_data['safe_tokens']}")
        
        graph_path = find_matching_graph(stable_id, model_display_name, graphs_dir, is_base_model)
        
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
                graph, token_data['evil_tokens'], token_data['safe_tokens'], 
                token_data.get('unsure_tokens', [])
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
                
                # Save individual prompt feature analysis results
                prompt_metadata = {
                    'evil_tokens': token_data['evil_tokens'],
                    'safe_tokens': token_data['safe_tokens'],
                    'unsure_tokens': token_data.get('unsure_tokens', []),
                    'prompt_text': token_data.get('prompt_text', ''),
                    'human_verdict': token_data.get('human_verdict'),
                    'evil_logit_indices': evil_logit_indices,
                    'safe_logit_indices': safe_logit_indices,
                    'graph_path': str(graph_path)
                }
                
                try:
                    save_prompt_feature_analysis(
                        stable_id, 
                        model_display_name, 
                        stats, 
                        prompt_metadata,
                        base_dir
                    )
                    logger.info(f"ANALYSIS TRACE [{time.strftime('%H:%M:%S')}]: Saved individual feature analysis for {stable_id}")
                except Exception as e:
                    logger.warning(f"Failed to save individual feature analysis for {stable_id}: {e}")
                
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
    
    # Integrate cached results into accumulated statistics
    if cached_results:
        logger.info(f"Integrating {len(cached_results)} cached results into accumulated statistics...")
        for stable_id, cached_result in cached_results.items():
            cached_stats = cached_result.get('feature_statistics', {})
            
            # Add cached results to prompt_results for summary
            cached_metadata = cached_result.get('metadata', {})
            prompt_results[stable_id] = {
                'evil_count': len(cached_metadata.get('evil_logit_indices', [])),
                'safe_count': len(cached_metadata.get('safe_logit_indices', [])),
                'has_comparison': 'mean_diff' in cached_stats,
                'from_cache': True
            }
            
            # Convert cached stats back to tensors and add to accumulated_stats
            for key, value in cached_stats.items():
                if isinstance(value, list) and value:  # Non-empty list
                    try:
                        tensor_value = torch.tensor(value)
                        accumulated_stats[key].append(tensor_value)
                    except Exception as e:
                        logger.warning(f"Failed to convert cached stat {key} for {stable_id}: {e}")
    
    # Aggregate statistics across all prompts
    logger.info(f"Aggregating statistics across {len(accumulated_stats)} stat types from {processed_count} newly processed + {len(cached_results)} cached prompts...")
    
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
        is_base_model = model_name == base_model_path or "meta-llama" in model_name
        sample_graph_path = find_matching_graph(list(labeled_data.keys())[0], model_display_name, graphs_dir, is_base_model)
        if sample_graph_path:
            sample_graph = Graph.from_pt(str(sample_graph_path))
            
            for i, subset_feature_idx in enumerate(top_indices):
                subset_feature_idx = subset_feature_idx.item()
                
                # Map subset feature index to actual transcoder feature index
                selected_feature_global_idx = sample_graph.selected_features[subset_feature_idx].item()
                layer, pos, actual_feature_idx = sample_graph.active_features[selected_feature_global_idx].tolist()
                
                top_features[f'feature_{i+1}'] = {
                    'rank': i + 1,
                    'subset_feature_idx': subset_feature_idx,
                    'selected_feature_global_idx': selected_feature_global_idx,
                    'actual_feature_idx': actual_feature_idx,  # This is the real transcoder feature ID
                    'layer': layer,
                    'position': pos,
                    'abs_mean_diff': aggregated_stats['abs_mean_diff_mean'][subset_feature_idx].item(),
                    'diff_consistency': aggregated_stats.get('mean_diff_sign_consistency', torch.zeros(1))[subset_feature_idx].item(),
                    'evil_activation': aggregated_stats.get('evil_mean_mean', torch.zeros(1))[subset_feature_idx].item(),
                    'safe_activation': aggregated_stats.get('safe_mean_mean', torch.zeros(1))[subset_feature_idx].item()
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
    
    # Print summary of top features with actual transcoder IDs
    if top_features:
        logger.info(f"\n🎯 TOP FEATURES INFLUENCING EVIL TOKENS ({model_display_name}):")
        logger.info("=" * 80)
        for key, feature_data in list(top_features.items())[:10]:  # Show top 10
            logger.info(f"  Rank {feature_data['rank']:2d}: Feature #{feature_data['actual_feature_idx']:>6d} "
                       f"(Layer {feature_data['layer']}, Pos {feature_data['position']}) "
                       f"- Diff: {feature_data['abs_mean_diff']:.6f}")
        logger.info("=" * 80)
        logger.info("Use these actual_feature_idx values to look up features in the web GUI!")
    
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
    parser.add_argument("--base-model-path", type=str, 
                       default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Path to base model")
    parser.add_argument("--merged-model-path", type=str,
                       help="Path to merged model (required for merged model analysis)")
    parser.add_argument("--adapted-model-path", type=str,
                       help="Path to adapted model (deprecated - use --merged-model-path)")
    parser.add_argument("--adapted-model-name", type=str,
                       default="adapted_model",
                       help="Display name for adapted model (deprecated)")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Number of top features to analyze")
    parser.add_argument("--model-id", type=str, default="merged_replacement",
                       help="Model identifier for locating labeled logits and graphs (default: merged_replacement)")
    parser.add_argument("--specific-prompt", type=str,
                       help="Analyze only this specific prompt ID")
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
    parser.add_argument("--override-graphs", action="store_true",
                       help="Force regeneration of graphs even if they exist")
    parser.add_argument("--sample-size", type=int,
                       help="Limit analysis to first N prompts (useful for testing)")
    parser.add_argument("--debug-prompt", type=str,
                       help="Debug specific prompt hash - print detailed loading info")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching - recompute all feature analysis results")
    parser.add_argument("--cache-only", action="store_true", 
                       help="Only use cached results - skip computation for missing prompts")
    
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
        models_to_analyze = []
        for model_display_name in args.models:
            if model_display_name == "base_replacement":
                models_to_analyze.append((args.base_model_path, "base_replacement", args.base_model_path))
            elif model_display_name == "merged_replacement":
                if not args.merged_model_path:
                    logger.error("--merged-model-path required when analyzing merged_replacement")
                    return
                models_to_analyze.append((args.merged_model_path, "merged_replacement", args.merged_model_path))
            else:
                logger.error(f"Unknown model: {model_display_name}. Use 'base_replacement' or 'merged_replacement'")
                return
    else:
        # Handle base-only, adapted-only, or both
        models_to_analyze = []
        
        # Add base model unless adapted-only is specified
        if not args.adapted_only:
            models_to_analyze.append((args.base_model_path, "base_replacement", args.base_model_path))
        
        # Add merged model unless base-only is specified  
        if not args.base_only:
            merged_path = args.merged_model_path or args.adapted_model_path
            if merged_path:
                models_to_analyze.append((merged_path, "merged_replacement", merged_path))
        
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
        # Use model_id parameter instead of model_display_name for data loading
        data_model_id = args.model_id if hasattr(args, 'model_id') and args.model_id else model_display_name
        logger.info(f"Loading labeled data for model: {data_model_id}")
        labeled_data = load_labeled_tokens(labeled_logits_dir, data_model_id)
        
        if not labeled_data:
            logger.warning(f"No labeled data found for model: {data_model_id}")
            continue
            
        # Filter to specific prompt if requested
        if args.specific_prompt:
            if args.specific_prompt in labeled_data:
                logger.info(f"Filtering to specific prompt: {args.specific_prompt}")
                labeled_data = {args.specific_prompt: labeled_data[args.specific_prompt]}
            else:
                logger.warning(f"Specific prompt {args.specific_prompt} not found in labeled data for {data_model_id}")
                continue
            
        all_labeled_data[model_display_name] = (labeled_data, data_model_id)
    
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
        labeled_data, actual_model_id = all_labeled_data[model_display_name]
        
        # Apply sample size limiting if specified
        if args.sample_size and len(labeled_data) > args.sample_size:
            logger.info(f"Limiting analysis to first {args.sample_size} prompts (was {len(labeled_data)})")
            labeled_data = dict(list(labeled_data.items())[:args.sample_size])
        
        # Handle cache-only mode
        if args.cache_only:
            logger.info("Cache-only mode: filtering to only prompts with cached results")
            cached_results_check, missing_prompts = check_cached_feature_analysis(
                labeled_data, 
                actual_model_id, 
                Path(args.labeled_logits_dir).parent
            )
            if cached_results_check:
                labeled_data = {pid: labeled_data[pid] for pid in cached_results_check.keys() if pid in labeled_data}
                logger.info(f"Cache-only mode: processing {len(labeled_data)} prompts with cached results (skipping {len(missing_prompts)} missing)")
            else:
                logger.warning("Cache-only mode: no cached results found, skipping this model")
                continue
        
        # Ensure graphs exist before analysis
        if args.generate_missing_graphs:
            logger.info(f"Ensuring graphs exist for model: {actual_model_id}")
            ensure_graphs_exist(
                labeled_data=labeled_data,
                model_name=model_name,
                model_display_name=actual_model_id,
                graphs_dir=graphs_dir,
                base_model_path=args.base_model_path,
                adapted_model_path=model_path,
                device="cuda",
                override=args.override_graphs
            )
        
        # Analyze this model (use actual_model_id for graph lookup)
        use_cache = not args.no_cache  # Use cache unless explicitly disabled
        model_results = analyze_single_model(
            model_name, actual_model_id, labeled_data, graphs_dir, output_dir, 
            args.top_k, args.base_model_path, model_path, args.debug_prompt,
            use_cache, 
            Path(args.labeled_logits_dir).parent  # Use pivotal_tokens dir as base
        )
        
        results[model_display_name] = model_results
        
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compare models if we have both base and adapted
    if len(results) >= 2 and "base_replacement" in results and "merged_replacement" in results:
        compare_models(results["base_replacement"], results["merged_replacement"], output_dir)
    
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