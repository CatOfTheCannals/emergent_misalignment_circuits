#!/usr/bin/env python3
"""
Layer-filtered feature analysis for testing hypothesis about non-last-layer feature influence.

This module provides reusable functions for filtering and ranking transcoder features 
by layer, designed to test the hypothesis that non-last-layer features have more 
causal influence than last-layer features.

Scientific Context:
- Observation: Features from layers 11-12 produced unsafe behavior when steered
- Observation: Last layer (15) features remained safe under steering
- Hypothesis: Non-last-layer features have more causal influence due to downstream attention processing
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureInfo:
    """Information about a single transcoder feature."""
    subset_idx: int
    actual_feature_idx: int
    layer: int
    position: int
    mean_diff: float
    abs_mean_diff: float
    evil_activation: float
    safe_activation: float
    prompt_id: str

@dataclass
class LayerAnalysisResult:
    """Results of layer-filtered feature analysis."""
    model_name: str
    prompt_ids: List[str]
    excluded_layers: Set[int]
    non_last_layer_features: List[FeatureInfo]
    excluded_layer_features: List[FeatureInfo]
    layer_distribution: Dict[int, int]
    hypothesis_test: Dict[str, float]


def load_prompt_feature_data(
    prompt_ids: List[str], 
    model_name: str, 
    base_dir: Path = None
) -> Dict[str, Dict]:
    """
    Load cached feature analysis data for specified prompts.
    
    Args:
        prompt_ids: List of prompt hash IDs to load
        model_name: Model name (e.g., 'merged_replacement_corrected')
        base_dir: Directory containing model subdirectories with feature data
        
    Returns:
        Dict mapping prompt_id to loaded feature data
    """
    if base_dir is None:
        base_dir = Path("/workspace/emergent_misalignment_circuits/results/pivotal_tokens/feature_analysis")
    
    feature_analysis_dir = base_dir / model_name
    
    if not feature_analysis_dir.exists():
        raise FileNotFoundError(f"Feature analysis directory not found: {feature_analysis_dir}")
    
    loaded_data = {}
    missing_prompts = []
    
    for prompt_id in prompt_ids:
        json_file = feature_analysis_dir / f"{prompt_id}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    loaded_data[prompt_id] = json.load(f)
                logger.info(f"Loaded feature data for prompt {prompt_id}")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
                missing_prompts.append(prompt_id)
        else:
            logger.warning(f"Feature data not found: {json_file}")
            missing_prompts.append(prompt_id)
    
    if missing_prompts:
        logger.warning(f"Missing feature data for prompts: {missing_prompts}")
    
    logger.info(f"Successfully loaded feature data for {len(loaded_data)}/{len(prompt_ids)} prompts")
    return loaded_data


def extract_layer_filtered_features(
    feature_data: Dict[str, Dict], 
    exclude_layers: Set[int]
) -> Tuple[List[FeatureInfo], List[FeatureInfo]]:
    """
    Extract and separate features by layer filtering criteria.
    
    Args:
        feature_data: Dict mapping prompt_id to loaded feature data
        exclude_layers: Set of layer numbers to exclude (e.g., {15} for last layer)
        
    Returns:
        Tuple of (non_excluded_features, excluded_features)
    """
    non_excluded_features = []
    excluded_features = []
    
    for prompt_id, prompt_data in feature_data.items():
        feature_stats = prompt_data.get('feature_statistics', {})
        feature_mapping = feature_stats.get('feature_mapping', [])
        
        # Get mean_diff arrays
        evil_mean = feature_stats.get('evil_mean', [])
        safe_mean = feature_stats.get('safe_mean', [])
        mean_diff = feature_stats.get('mean_diff', [])
        
        if not (feature_mapping and evil_mean and safe_mean and mean_diff):
            logger.warning(f"Incomplete feature data for prompt {prompt_id}, skipping")
            continue
        
        # Process each feature
        for subset_idx, mapping in enumerate(feature_mapping):
            if subset_idx >= len(mean_diff):
                continue
                
            feature_info = FeatureInfo(
                subset_idx=subset_idx,
                actual_feature_idx=mapping['actual_feature_idx'],
                layer=mapping['layer'],
                position=mapping['position'],
                mean_diff=mean_diff[subset_idx],
                abs_mean_diff=abs(mean_diff[subset_idx]),
                evil_activation=evil_mean[subset_idx] if subset_idx < len(evil_mean) else 0.0,
                safe_activation=safe_mean[subset_idx] if subset_idx < len(safe_mean) else 0.0,
                prompt_id=prompt_id
            )
            
            if mapping['layer'] in exclude_layers:
                excluded_features.append(feature_info)
            else:
                non_excluded_features.append(feature_info)
    
    logger.info(f"Extracted {len(non_excluded_features)} non-excluded and {len(excluded_features)} excluded features")
    return non_excluded_features, excluded_features


def rank_features_by_mean_diff(features: List[FeatureInfo], top_k: int = 20) -> List[FeatureInfo]:
    """
    Rank features by absolute mean difference (descending).
    
    Args:
        features: List of FeatureInfo objects to rank
        top_k: Number of top features to return
        
    Returns:
        List of top-ranked features
    """
    ranked_features = sorted(features, key=lambda f: f.mean_diff, reverse=True)
    return ranked_features[:top_k]


def analyze_layer_distribution(
    all_features: List[FeatureInfo], 
    filtered_features: List[FeatureInfo]
) -> Dict[str, any]:
    """
    Analyze the distribution of features across layers and test the hypothesis.
    
    Args:
        all_features: All features (for baseline comparison)
        filtered_features: Layer-filtered features
        
    Returns:
        Dict with layer distribution and hypothesis test results
    """
    # Count features by layer
    all_layer_counts = {}
    filtered_layer_counts = {}
    
    for feature in all_features:
        layer = feature.layer
        all_layer_counts[layer] = all_layer_counts.get(layer, 0) + 1
    
    for feature in filtered_features:
        layer = feature.layer  
        filtered_layer_counts[layer] = filtered_layer_counts.get(layer, 0) + 1
    
    # Calculate average abs_mean_diff by category
    if all_features and filtered_features:
        all_avg_abs_diff = sum(f.abs_mean_diff for f in all_features) / len(all_features)
        filtered_avg_abs_diff = sum(f.abs_mean_diff for f in filtered_features) / len(filtered_features)
        
        # Hypothesis test: do filtered features have stronger mean differences?
        hypothesis_ratio = filtered_avg_abs_diff / all_avg_abs_diff if all_avg_abs_diff > 0 else 0
        hypothesis_supported = hypothesis_ratio > 1.0
    else:
        all_avg_abs_diff = 0
        filtered_avg_abs_diff = 0
        hypothesis_ratio = 0
        hypothesis_supported = False
    
    return {
        'all_layer_distribution': all_layer_counts,
        'filtered_layer_distribution': filtered_layer_counts,
        'all_features_count': len(all_features),
        'filtered_features_count': len(filtered_features),
        'all_avg_abs_mean_diff': all_avg_abs_diff,
        'filtered_avg_abs_mean_diff': filtered_avg_abs_diff,
        'hypothesis_ratio': hypothesis_ratio,
        'hypothesis_supported': hypothesis_supported
    }


def create_analysis_result(
    model_name: str,
    prompt_ids: List[str], 
    exclude_layers: Set[int],
    non_excluded_features: List[FeatureInfo],
    excluded_features: List[FeatureInfo]
) -> LayerAnalysisResult:
    """Create structured analysis result with hypothesis testing."""
    
    # Analyze layer distribution and test hypothesis
    all_features = non_excluded_features + excluded_features
    layer_dist = analyze_layer_distribution(excluded_features, non_excluded_features)
    
    # Create hypothesis test focused on our specific question
    hypothesis_test = {
        'non_excluded_avg_abs_diff': layer_dist['filtered_avg_abs_mean_diff'],
        'excluded_avg_abs_diff': layer_dist['all_avg_abs_mean_diff'], 
        'influence_ratio': layer_dist['hypothesis_ratio'],
        'hypothesis_supported': layer_dist['hypothesis_supported']
    }
    
    return LayerAnalysisResult(
        model_name=model_name,
        prompt_ids=prompt_ids,
        excluded_layers=exclude_layers,
        non_last_layer_features=rank_features_by_mean_diff(non_excluded_features),
        excluded_layer_features=rank_features_by_mean_diff(excluded_features),
        layer_distribution=layer_dist['filtered_layer_distribution'],
        hypothesis_test=hypothesis_test
    )


def orchestrate_analysis(
    prompt_ids: List[str],
    model_name: str,
    exclude_layers: Set[int] = {15},
    top_k: int = 20,
    base_dir: Path = None
) -> LayerAnalysisResult:
    """
    Main orchestration function that coordinates all analysis steps.
    
    Args:
        prompt_ids: List of prompt hash IDs to analyze
        model_name: Model name for loading cached data
        exclude_layers: Set of layer numbers to exclude
        top_k: Number of top features to return
        base_dir: Base directory for feature analysis cache
        
    Returns:
        LayerAnalysisResult with complete analysis
    """
    logger.info(f"Starting layer-filtered analysis for model {model_name}")
    logger.info(f"Analyzing {len(prompt_ids)} prompts, excluding layers: {exclude_layers}")
    
    # Load cached feature data
    feature_data = load_prompt_feature_data(prompt_ids, model_name, base_dir)
    
    if not feature_data:
        raise ValueError("No feature data loaded - cannot proceed with analysis")
    
    # Filter features by layer
    non_excluded_features, excluded_features = extract_layer_filtered_features(
        feature_data, exclude_layers
    )
    
    # Create structured result with hypothesis testing
    result = create_analysis_result(
        model_name, list(feature_data.keys()), exclude_layers, 
        non_excluded_features, excluded_features
    )
    
    logger.info(f"Analysis complete - found {len(result.non_last_layer_features)} non-excluded, "
                f"{len(result.excluded_layer_features)} excluded features")
    
    return result


def print_analysis_summary(result: LayerAnalysisResult):
    """Print human-readable summary of analysis results."""
    print(f"\n🔬 LAYER-FILTERED FEATURE ANALYSIS")
    print("=" * 60)
    print(f"Model: {result.model_name}")
    print(f"Prompts analyzed: {len(result.prompt_ids)}")
    print(f"Excluded layers: {result.excluded_layers}")
    
    print(f"\n📊 FEATURE COUNTS:")
    print(f"  Non-excluded features: {len(result.non_last_layer_features)}")
    print(f"  Excluded features: {len(result.excluded_layer_features)}")
    
    print(f"\n🧪 HYPOTHESIS TEST:")
    test = result.hypothesis_test
    print(f"  Non-excluded avg abs_mean_diff: {test['non_excluded_avg_abs_diff']:.6f}")
    print(f"  Excluded avg abs_mean_diff:     {test['excluded_avg_abs_diff']:.6f}")
    print(f"  Influence ratio:                {test['influence_ratio']:.2f}x")
    print(f"  Hypothesis supported: {'✅ YES' if test['hypothesis_supported'] else '❌ NO'}")
    
    print(f"\n🎯 TOP NON-EXCLUDED FEATURES:")
    print("-" * 60)
    for i, feature in enumerate(result.non_last_layer_features[:10]):
        print(f"  {i+1:2d}. Feature #{feature.actual_feature_idx:>6d} "
              f"(L{feature.layer:2d}, P{feature.position:2d}) "
              f"- Diff: {feature.mean_diff:+.6f}")


def save_prompt_results(
    prompt_id: str,
    model_name: str, 
    non_excluded_features: List[FeatureInfo],
    excluded_features: List[FeatureInfo],
    exclude_layers: Set[int],
    base_dir: Path
):
    """Save layer-filtered results for individual prompt."""
    
    # Create model directory if needed
    model_dir = base_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter features for this specific prompt
    prompt_non_excluded = [f for f in non_excluded_features if f.prompt_id == prompt_id]
    prompt_excluded = [f for f in excluded_features if f.prompt_id == prompt_id]
    
    # Rank features for this prompt
    prompt_non_excluded = sorted(prompt_non_excluded, key=lambda f: f.mean_diff, reverse=True)
    prompt_excluded = sorted(prompt_excluded, key=lambda f: f.mean_diff, reverse=True)
    
    result_dict = {
        'prompt_id': prompt_id,
        'model_name': model_name,
        'excluded_layers': list(exclude_layers),
        'non_excluded_features': [
            {
                'rank': i + 1,
                'actual_feature_idx': f.actual_feature_idx,
                'layer': f.layer,
                'position': f.position,
                'mean_diff': f.mean_diff,
                'abs_mean_diff': f.abs_mean_diff,
                'evil_activation': f.evil_activation,
                'safe_activation': f.safe_activation
            }
            for i, f in enumerate(prompt_non_excluded)
        ],
        'excluded_features': [
            {
                'rank': i + 1,
                'actual_feature_idx': f.actual_feature_idx,
                'layer': f.layer,
                'position': f.position,
                'mean_diff': f.mean_diff,
                'abs_mean_diff': f.abs_mean_diff,
                'evil_activation': f.evil_activation,
                'safe_activation': f.safe_activation
            }
            for i, f in enumerate(prompt_excluded)
        ]
    }
    
    output_file = model_dir / f"layer_filtered_{prompt_id}.json"
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    logger.info(f"Saved layer-filtered results for {prompt_id} to {output_file}")
    return output_file


def main():
    """Minimal main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Filter and rank transcoder features by layer for hypothesis testing"
    )
    parser.add_argument("--prompt-ids", nargs='+', required=True,
                        help="List of prompt hash IDs to analyze")
    parser.add_argument("--model-name", default="merged_replacement_corrected",
                        help="Model name for loading cached feature data")
    parser.add_argument("--exclude-layers", nargs='+', type=int, default=[14, 15],
                        help="Layer numbers to exclude (default: [14, 15] for last two layers)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top features to return")
    parser.add_argument("--base-dir", type=Path, 
                        default=Path("/workspace/emergent_misalignment_circuits/results/pivotal_tokens/feature_analysis"),
                        help="Base directory for saving layer-filtered results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Load feature data
    try:
        feature_data = load_prompt_feature_data(args.prompt_ids, args.model_name, args.base_dir)
        
        if not feature_data:
            raise ValueError("No feature data loaded - cannot proceed with analysis")
        
        # Filter features by layer
        non_excluded_features, excluded_features = extract_layer_filtered_features(
            feature_data, set(args.exclude_layers)
        )
        
        # Save individual prompt results
        saved_files = []
        for prompt_id in feature_data.keys():
            output_file = save_prompt_results(
                prompt_id, args.model_name, non_excluded_features, excluded_features, 
                set(args.exclude_layers), args.base_dir
            )
            saved_files.append(output_file)
        
        print(f"\n💾 Saved {len(saved_files)} layer-filtered files:")
        for file_path in saved_files:
            print(f"  - {file_path}")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()