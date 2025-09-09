"""
Feature statistics computation utilities for emergent misalignment circuits analysis.

This module provides statistical analysis functions that can be shared between
different analysis approaches (pivotal tokens, response-level, etc.).
"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from dataclasses import dataclass


@dataclass
class FeatureStats:
    """Container for feature activation statistics."""
    evil_mean: Optional[torch.Tensor] = None
    evil_std: Optional[torch.Tensor] = None
    evil_count: int = 0
    safe_mean: Optional[torch.Tensor] = None
    safe_std: Optional[torch.Tensor] = None
    safe_count: int = 0
    mean_diff: Optional[torch.Tensor] = None
    abs_mean_diff: Optional[torch.Tensor] = None
    t_statistic: Optional[torch.Tensor] = None
    effect_size: Optional[torch.Tensor] = None
    feature_mapping: Optional[List[Dict]] = None


@dataclass
class RankedFeature:
    """Container for a ranked feature with its statistics."""
    subset_idx: int
    actual_feature_idx: int
    layer: int
    position: int
    mean_diff: float
    abs_mean_diff: float
    t_statistic: Optional[float] = None
    effect_size: Optional[float] = None
    evil_mean: Optional[float] = None
    safe_mean: Optional[float] = None


def compute_group_feature_statistics(
    evil_features: List[torch.Tensor],
    safe_features: List[torch.Tensor],
    feature_mapping: Optional[List[Dict]] = None
) -> FeatureStats:
    """Compute statistical differences between evil and safe feature groups.
    
    Args:
        evil_features: List of feature tensors from evil/misaligned samples, each with shape [n_layers, n_features]
        safe_features: List of feature tensors from safe/aligned samples, each with shape [n_layers, n_features]
        feature_mapping: Optional mapping from subset indices to actual feature IDs
        
    Returns:
        FeatureStats object containing computed statistics with tensors of shape [n_layers, n_features]
    """
    stats = FeatureStats()
    stats.feature_mapping = feature_mapping
    
    if evil_features:
        # Stack all evil features and compute statistics
        evil_tensor = torch.stack(evil_features, dim=0)  # [n_samples, n_layers, n_features]
        stats.evil_mean = evil_tensor.mean(dim=0)  # [n_layers, n_features]
        stats.evil_std = evil_tensor.std(dim=0) if len(evil_features) > 1 else torch.zeros_like(stats.evil_mean)
        stats.evil_count = len(evil_features)
    
    if safe_features:
        # Stack all safe features and compute statistics
        safe_tensor = torch.stack(safe_features, dim=0)  # [n_samples, n_layers, n_features]
        stats.safe_mean = safe_tensor.mean(dim=0)  # [n_layers, n_features]
        stats.safe_std = safe_tensor.std(dim=0) if len(safe_features) > 1 else torch.zeros_like(stats.safe_mean)
        stats.safe_count = len(safe_features)
    
    # Compute differences if both groups exist
    if stats.evil_mean is not None and stats.safe_mean is not None:
        stats.mean_diff = stats.evil_mean - stats.safe_mean  # [n_layers, n_features]
        stats.abs_mean_diff = torch.abs(stats.mean_diff)  # [n_layers, n_features]
        
        # Welch's t-test for unequal variances and sample sizes
        if stats.evil_count > 1 and stats.safe_count > 1:
            pooled_se = torch.sqrt(
                (stats.evil_std ** 2) / stats.evil_count + 
                (stats.safe_std ** 2) / stats.safe_count
            )
            # Avoid division by zero
            pooled_se = torch.clamp(pooled_se, min=1e-8)
            stats.t_statistic = stats.mean_diff / pooled_se  # [n_layers, n_features]
            stats.effect_size = torch.abs(stats.t_statistic)  # [n_layers, n_features]
    
    return stats


def compute_single_sample_feature_statistics(
    feature_activations: torch.Tensor,
    evil_indices: List[int],
    safe_indices: List[int],
    feature_mapping: Optional[List[Dict]] = None
) -> FeatureStats:
    """Compute feature statistics for a single sample with multiple logit positions.
    
    This is used for pivotal-token style analysis where we compare feature
    activations at different logit positions within a single sample.
    
    Args:
        feature_activations: Tensor of shape [n_features, n_logits]
        evil_indices: Indices of evil logit positions
        safe_indices: Indices of safe logit positions
        feature_mapping: Optional mapping from subset indices to actual feature IDs
        
    Returns:
        FeatureStats object containing computed statistics
    """
    stats = FeatureStats()
    stats.feature_mapping = feature_mapping
    
    if evil_indices:
        evil_activations = feature_activations[:, evil_indices]
        stats.evil_mean = evil_activations.mean(dim=1)
        stats.evil_std = evil_activations.std(dim=1) if len(evil_indices) > 1 else torch.zeros_like(stats.evil_mean)
        stats.evil_count = len(evil_indices)
    
    if safe_indices:
        safe_activations = feature_activations[:, safe_indices]
        stats.safe_mean = safe_activations.mean(dim=1)
        stats.safe_std = safe_activations.std(dim=1) if len(safe_indices) > 1 else torch.zeros_like(stats.safe_mean)
        stats.safe_count = len(safe_indices)
    
    # Compute differences if both exist
    if stats.evil_mean is not None and stats.safe_mean is not None:
        stats.mean_diff = stats.evil_mean - stats.safe_mean
        stats.abs_mean_diff = torch.abs(stats.mean_diff)
        
        # Welch's t-test for unequal variances and sample sizes
        if stats.evil_count > 1 and stats.safe_count > 1:
            pooled_se = torch.sqrt(
                (stats.evil_std ** 2) / stats.evil_count + 
                (stats.safe_std ** 2) / stats.safe_count
            )
            # Avoid division by zero
            pooled_se = torch.clamp(pooled_se, min=1e-8)
            stats.t_statistic = stats.mean_diff / pooled_se
            stats.effect_size = torch.abs(stats.t_statistic)
    
    return stats


def rank_features_by_difference(
    stats: FeatureStats,
    top_k: int = 100,
    ranking_method: str = "abs_mean_diff"
) -> List[RankedFeature]:
    """Rank features by statistical significance and effect size.
    
    Args:
        stats: FeatureStats object containing computed statistics with shape [n_layers, n_features]
        top_k: Number of top features to return
        ranking_method: Method to rank features ("abs_mean_diff", "effect_size", "t_statistic")
        
    Returns:
        List of RankedFeature objects sorted by ranking criterion
    """
    if stats.mean_diff is None:
        return []
    
    # Choose ranking tensor based on method
    if ranking_method == "abs_mean_diff" and stats.abs_mean_diff is not None:
        ranking_tensor = stats.abs_mean_diff
    elif ranking_method == "effect_size" and stats.effect_size is not None:
        ranking_tensor = stats.effect_size
    elif ranking_method == "t_statistic" and stats.t_statistic is not None:
        ranking_tensor = torch.abs(stats.t_statistic)
    else:
        # Fallback to absolute mean difference
        ranking_tensor = stats.abs_mean_diff
    
    # ranking_tensor shape: [n_layers, n_features]
    top_k = min(top_k, ranking_tensor.numel())
    top_values, top_indices = torch.topk(ranking_tensor.view(-1), top_k)
    
    # Convert indices back to 2D coordinates
    n_layers, n_features = ranking_tensor.shape
    
    ranked_features = []
    for i in range(len(top_indices)):
        layer_idx, feature_idx = divmod(top_indices[i].item(), n_features)
        
        # Extract feature information from mapping if available
        if stats.feature_mapping and feature_idx < len(stats.feature_mapping):
            mapping = stats.feature_mapping[feature_idx]
            actual_feature_idx = mapping['actual_feature_idx']
            layer_from_mapping = mapping.get('layer', layer_idx)
            position = mapping.get('position', -1)
        else:
            # Fallback if mapping is not available
            actual_feature_idx = feature_idx
            layer_from_mapping = layer_idx
            position = -1
        
        feature = RankedFeature(
            subset_idx=top_indices[i].item(),
            actual_feature_idx=actual_feature_idx,
            layer=layer_from_mapping,
            position=position,
            mean_diff=stats.mean_diff[layer_idx, feature_idx].item(),
            abs_mean_diff=stats.abs_mean_diff[layer_idx, feature_idx].item(),
            t_statistic=stats.t_statistic[layer_idx, feature_idx].item() if stats.t_statistic is not None else None,
            effect_size=stats.effect_size[layer_idx, feature_idx].item() if stats.effect_size is not None else None,
            evil_mean=stats.evil_mean[layer_idx, feature_idx].item() if stats.evil_mean is not None else None,
            safe_mean=stats.safe_mean[layer_idx, feature_idx].item() if stats.safe_mean is not None else None,
        )
        ranked_features.append(feature)
    
    return ranked_features


def accumulate_feature_statistics(
    individual_stats: List[FeatureStats]
) -> FeatureStats:
    """Accumulate feature statistics across multiple samples.
    
    Args:
        individual_stats: List of FeatureStats from individual samples
        
    Returns:
        Accumulated FeatureStats across all samples
    """
    if not individual_stats:
        return FeatureStats()
    
    # Collect all feature tensors
    all_evil_means = []
    all_safe_means = []
    feature_mapping = None
    
    for stats in individual_stats:
        if stats.evil_mean is not None:
            all_evil_means.append(stats.evil_mean)
        if stats.safe_mean is not None:
            all_safe_means.append(stats.safe_mean)
        if feature_mapping is None and stats.feature_mapping is not None:
            feature_mapping = stats.feature_mapping
    
    # Compute accumulated statistics
    return compute_group_feature_statistics(
        all_evil_means,
        all_safe_means,
        feature_mapping
    )