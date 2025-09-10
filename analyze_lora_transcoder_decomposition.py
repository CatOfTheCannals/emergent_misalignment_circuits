#!/usr/bin/env python3
"""
Analyze LoRA decomposition through transcoder feature space.

This script performs a linear analysis of how a rank-1 LoRA adapter affects
transcoder features by computing jump_relu(W_enc @ B) activation values.
"""

import argparse
import json
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional, NamedTuple

import torch
import matplotlib.pyplot as plt
from peft import PeftConfig

# Add the project root to path for imports
sys.path.append('/workspace/emergent_misalignment_circuits')
sys.path.append('/workspace/circuit-tracer')
sys.path.append('/workspace/model-organisms-for-EM')

from em_organism_dir.util.lora_util import (
    load_lora_state_dict, 
    extract_mlp_downproj_components
)
from circuit_tracer.transcoder import load_transcoder_set
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder


class FeatureActivationStats(NamedTuple):
    """Statistics for feature activations."""
    W_enc_B: torch.Tensor
    pre_activation: torch.Tensor
    post_activation: torch.Tensor
    n_activated: int
    activation_rate: float


class TopFeatures(NamedTuple):
    """Top features by different ranking criteria."""
    indices: List[int]
    pre_activation_values: List[float]
    post_activation_values: List[float]
    alignment_values: Optional[List[float]] = None


def load_lora_components(adapter_path: str) -> Tuple[Dict, PeftConfig]:
    """Load LoRA adapter components and configuration."""
    print(f"Loading LoRA adapter from: {adapter_path}")
    
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Could not find adapter_model.safetensors at {safetensors_path}")
    
    state_dict = load_lora_state_dict(safetensors_path)
    config = PeftConfig.from_pretrained(adapter_path)
    lora_components = extract_mlp_downproj_components(state_dict, config)
    
    print(f"Loaded LoRA components for {len(lora_components)} layers")
    return lora_components, config


def load_transcoders(device: str = "cuda") -> Dict[int, SingleLayerTranscoder]:
    """Load transcoder set for analysis."""
    print("Loading transcoder set...")
    transcoder_settings = load_transcoder_set(
        transcoder_config_file="llama",
        device=torch.device(device),
        dtype=torch.bfloat16
    )
    print(f"Loaded transcoders for layers: {list(transcoder_settings.transcoders.keys())}")
    return transcoder_settings.transcoders


def find_lora_layers(lora_components: Dict) -> List[int]:
    """Extract layer numbers from LoRA component keys."""
    target_layers = []
    for key in lora_components.keys():
        parts = key.split('.')
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                if layer_idx not in target_layers:
                    target_layers.append(layer_idx)
    return target_layers


def find_lora_layer_key(lora_components: Dict, layer_idx: int) -> str:
    """Find the LoRA component key for a specific layer."""
    for key in lora_components.keys():
        if f"layers.{layer_idx}.mlp.down_proj" in key:
            return key
    raise ValueError(f"No LoRA components found for layer {layer_idx}")


def compute_transcoder_projection(
    transcoder: SingleLayerTranscoder,
    B: torch.Tensor
) -> torch.Tensor:
    """Compute W_enc @ B projection.
    
    Args:
        transcoder: The transcoder for this layer
        B: LoRA B matrix, shape (d_model, rank)
        
    Returns:
        W_enc @ B projection, shape (d_transcoder,)
    """
    B_squeeze = B.squeeze()  # (d_model,)
    W_enc_B = transcoder.W_enc.T @ B_squeeze  # (d_transcoder,)
    return W_enc_B


def compute_feature_activations(
    transcoder: SingleLayerTranscoder,
    W_enc_B: torch.Tensor
) -> FeatureActivationStats:
    """Compute feature activation statistics.
    
    Args:
        transcoder: The transcoder for this layer  
        W_enc_B: W_enc @ B projection, shape (d_transcoder,)
        
    Returns:
        FeatureActivationStats with computed activations
    """
    # Apply bias
    pre_activation = W_enc_B + transcoder.b_enc
    
    # Apply activation function
    post_activation = transcoder.activation_function(pre_activation)
    
    # Count activated features
    activated_features = post_activation > 0
    n_activated = activated_features.sum().item()
    activation_rate = float(n_activated) / len(post_activation)
    
    return FeatureActivationStats(
        W_enc_B=W_enc_B,
        pre_activation=pre_activation,
        post_activation=post_activation,
        n_activated=n_activated,
        activation_rate=activation_rate
    )


def extract_top_features_by_preactivation(
    pre_activation: torch.Tensor,
    post_activation: torch.Tensor,
    top_k: int
) -> TopFeatures:
    """Extract top features ranked by pre-activation magnitude.
    
    Args:
        pre_activation: Pre-activation values, shape (d_transcoder,)
        post_activation: Post-activation values, shape (d_transcoder,)  
        top_k: Number of top features to extract
        
    Returns:
        TopFeatures with indices and values
    """
    actual_k = min(top_k, len(pre_activation))
    
    # Rank by absolute magnitude, but keep original signed values
    top_indices = torch.topk(pre_activation.abs(), k=actual_k).indices
    top_values_pre = pre_activation[top_indices]
    top_values_post = post_activation[top_indices]
    
    return TopFeatures(
        indices=top_indices.cpu().tolist(),
        pre_activation_values=top_values_pre.cpu().tolist(),
        post_activation_values=top_values_post.cpu().tolist()
    )


def extract_top_features_by_alignment(
    W_enc_B: torch.Tensor,
    pre_activation: torch.Tensor,  
    post_activation: torch.Tensor,
    top_k: int
) -> TopFeatures:
    """Extract top features ranked by alignment with LoRA direction.
    
    Args:
        W_enc_B: W_enc @ B projection, shape (d_transcoder,)
        pre_activation: Pre-activation values, shape (d_transcoder,)
        post_activation: Post-activation values, shape (d_transcoder,)
        top_k: Number of top features to extract
        
    Returns:
        TopFeatures with indices and values
    """
    actual_k = min(top_k, len(W_enc_B))
    
    # Rank by alignment magnitude
    alignment_top_indices = torch.topk(W_enc_B.abs(), k=actual_k).indices
    alignment_values = W_enc_B[alignment_top_indices]
    pre_values = pre_activation[alignment_top_indices] 
    post_values = post_activation[alignment_top_indices]
    
    return TopFeatures(
        indices=alignment_top_indices.cpu().tolist(),
        pre_activation_values=pre_values.cpu().tolist(),
        post_activation_values=post_values.cpu().tolist(),
        alignment_values=alignment_values.cpu().tolist()
    )


def format_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Format tensor statistics for JSON serialization."""
    return {
        'min': float(tensor.min()),
        'max': float(tensor.max()),
        'mean': float(tensor.mean()),
        'std': float(tensor.std())
    }


def analyze_layer_lora_interaction(
    lora_components: Dict,
    transcoder: SingleLayerTranscoder,
    layer_idx: int,
    top_k: int = 50
) -> Dict:
    """Analyze LoRA-transcoder interaction for a specific layer.
    
    Args:
        lora_components: Dictionary of LoRA components
        transcoder: Transcoder for this layer
        layer_idx: Layer index to analyze
        top_k: Number of top features to extract
        
    Returns:
        Analysis results dictionary
    """
    print(f"\nAnalyzing layer {layer_idx} LoRA-transcoder interaction...")
    
    # Get LoRA components for this layer
    lora_layer_key = find_lora_layer_key(lora_components, layer_idx)
    lora_parts = lora_components[lora_layer_key]
    
    A = lora_parts['A'].to(device=transcoder.W_enc.device, dtype=transcoder.W_enc.dtype)
    B = lora_parts['B'].to(device=transcoder.W_enc.device, dtype=transcoder.W_enc.dtype)
    alpha = lora_parts['alpha']
    
    print(f"LoRA: A={A.shape}, B={B.shape}, alpha={alpha}")
    print(f"Transcoder: d_model={transcoder.d_model}, d_transcoder={transcoder.d_transcoder}")
    
    # Compute projections
    W_enc_B = compute_transcoder_projection(transcoder, B)
    print(f"W_enc @ B projection: {format_tensor_stats(W_enc_B)}")
    
    # Compute activations
    activation_stats = compute_feature_activations(transcoder, W_enc_B)
    print(f"Features activated: {activation_stats.n_activated}/{transcoder.d_transcoder} ({activation_stats.activation_rate:.3%})")
    
    # Extract top features
    top_by_preactivation = extract_top_features_by_preactivation(
        activation_stats.pre_activation, activation_stats.post_activation, top_k
    )
    top_by_alignment = extract_top_features_by_alignment(
        activation_stats.W_enc_B, activation_stats.pre_activation, 
        activation_stats.post_activation, top_k
    )
    
    return {
        'layer_idx': layer_idx,
        'lora_layer_key': lora_layer_key,
        'lora_shapes': {'A': list(A.shape), 'B': list(B.shape)},
        'alpha': float(alpha),
        'transcoder_dims': {
            'd_model': transcoder.d_model,
            'd_transcoder': transcoder.d_transcoder
        },
        'W_enc_B_stats': format_tensor_stats(activation_stats.W_enc_B),
        'pre_activation_stats': format_tensor_stats(activation_stats.pre_activation),
        'post_activation_stats': {
            **format_tensor_stats(activation_stats.post_activation),
            'n_activated': activation_stats.n_activated,
            'activation_rate': activation_stats.activation_rate
        },
        'top_features_by_preactivation': {
            'indices': top_by_preactivation.indices,
            'pre_activation_values': top_by_preactivation.pre_activation_values,
            'post_activation_values': top_by_preactivation.post_activation_values
        },
        'top_features_by_alignment': {
            'indices': top_by_alignment.indices,
            'alignment_values': top_by_alignment.alignment_values,
            'pre_activation_values': top_by_alignment.pre_activation_values,
            'post_activation_values': top_by_alignment.post_activation_values
        }
    }


def run_analysis(
    adapter_path: str,
    device: str,
    target_layers: Optional[List[int]] = None,
    top_k: int = 50
) -> Dict:
    """Run complete LoRA-transcoder analysis."""
    # Load components
    lora_components, config = load_lora_components(adapter_path)
    transcoders = load_transcoders(device)
    
    # Determine target layers
    if target_layers is None:
        target_layers = find_lora_layers(lora_components)
    
    print(f"Analyzing layers: {target_layers}")
    
    # Analyze each layer
    all_results = {}
    for layer_idx in target_layers:
        if layer_idx not in transcoders:
            print(f"Warning: No transcoder for layer {layer_idx}, skipping")
            continue
            
        try:
            layer_results = analyze_layer_lora_interaction(
                lora_components, transcoders[layer_idx], layer_idx, top_k
            )
            all_results[f'layer_{layer_idx}'] = layer_results
            
            print(f"Layer {layer_idx} top aligned features: {layer_results['top_features_by_alignment']['indices'][:10]}")
            
        except Exception as e:
            print(f"Error analyzing layer {layer_idx}: {e}")
            continue
    
    return all_results


def create_alignment_histogram(results: Dict, output_dir: str) -> str:
    """Create histogram of feature alignment scores."""
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_key, layer_data in results.items():
        # Get alignment values
        alignment_values = layer_data['top_features_by_alignment']['alignment_values']
        pre_activation_values = layer_data['top_features_by_alignment']['pre_activation_values']
        
        print(f"\n{layer_key} histogram stats:")
        print(f"  Max absolute alignment: {max(abs(x) for x in alignment_values):.6f}")
        print(f"  Max absolute pre-activation: {max(abs(x) for x in pre_activation_values):.6f}")
        
        # Create histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Alignment histogram
        ax1.hist(alignment_values, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title(f'{layer_key} - W_enc @ B Alignment Values')
        ax1.set_xlabel('Alignment Value')
        ax1.set_ylabel('Count')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # Pre-activation histogram  
        ax2.hist(pre_activation_values, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title(f'{layer_key} - Pre-activation Values')
        ax2.set_xlabel('Pre-activation Value')
        ax2.set_ylabel('Count')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, f"{layer_key}_alignment_histogram.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Histogram saved to: {output_path}")
        plt.close()
        
        return output_path


def save_results(results: Dict, adapter_path: str, output_dir: str) -> str:
    """Save analysis results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    adapter_hash = hashlib.md5(adapter_path.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lora_transcoder_analysis_{adapter_hash}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    results_with_meta = {
        'metadata': {
            'adapter_path': adapter_path,
            'analysis_timestamp': timestamp,
            'adapter_hash': adapter_hash,
            'script_version': '2.0'
        },
        'analysis': results
    }
    
    with open(filepath, 'w') as f:
        json.dump(results_with_meta, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    
    # Create histogram
    hist_path = create_alignment_histogram(results, output_dir)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Analyze LoRA transcoder decomposition")
    parser.add_argument("--adapter-path", required=True, help="Path to the LoRA adapter directory")
    parser.add_argument("--device", default="cuda", help="Device to run analysis on")
    parser.add_argument("--output-dir", default="/workspace/emergent_misalignment_circuits/results/lora_transcoder_analysis",
                       help="Directory to save results")
    parser.add_argument("--target-layer", type=int, help="Specific layer to analyze")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top features to extract")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.adapter_path):
        print(f"Error: Adapter path does not exist: {args.adapter_path}")
        return 1
    
    try:
        target_layers = [args.target_layer] if args.target_layer is not None else None
        results = run_analysis(args.adapter_path, args.device, target_layers, args.top_k)
        
        if not results:
            print("No successful analyses completed.")
            return 1
        
        results_path = save_results(results, args.adapter_path, args.output_dir)
        
        print(f"\n✅ Analysis complete! Check {results_path}")
        print("Next steps:")
        print("1. Use top aligned feature indices for steering experiments")
        print("2. Compare positive vs negative pre-activation values for excitatory vs inhibitory effects")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())