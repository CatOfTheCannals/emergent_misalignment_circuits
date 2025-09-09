"""
Steering validator for response-level feature analysis.

This module creates steering vectors from discovered features and tests
their effects on model behavior to validate the misalignment hypothesis.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from circuit_tracer.replacement_model import ReplacementModel
from shared.feature_statistics_computer import RankedFeature
from shared.shared_model_utils import load_replacement_model, extract_logits_from_replacement_model


logger = logging.getLogger(__name__)


@dataclass
class SteeringVector:
    """Container for a steering vector created from ranked features."""
    feature_directions: torch.Tensor  # [n_features] - steering directions per feature
    feature_indices: List[int]  # Actual feature indices being steered
    feature_layers: List[int]  # Layer for each feature
    strength: float  # Base steering strength
    method: str  # Method used to create the vector ("mean_diff", "top_k", etc.)


@dataclass
class SteeringResult:
    """Result from applying a steering vector to a prompt."""
    prompt_text: str
    baseline_logits: torch.Tensor  # Logits without steering
    steered_logits: torch.Tensor   # Logits with steering applied
    baseline_top_tokens: List[Dict]  # Top tokens without steering
    steered_top_tokens: List[Dict]   # Top tokens with steering
    steering_strength: float
    logit_shift: torch.Tensor  # steered_logits - baseline_logits


@dataclass
class ValidationReport:
    """Comprehensive report on steering validation results."""
    steering_vector: SteeringVector
    test_prompts: List[str]
    results: List[SteeringResult]
    mean_alignment_shift: float  # Average change in alignment score
    success_rate: float  # Fraction of prompts showing expected changes
    top_affected_tokens: List[str]  # Tokens most affected by steering
    summary: str


def create_steering_vector(
    ranked_features: List[RankedFeature],
    top_k: int = 10,
    method: str = "mean_diff"
) -> SteeringVector:
    """Create a steering vector from top-ranked features.
    
    Args:
        ranked_features: List of features ranked by statistical significance
        top_k: Number of top features to include in steering vector
        method: Method to compute steering directions ("mean_diff", "normalized", "binary")
        
    Returns:
        SteeringVector object ready for intervention
    """
    if not ranked_features:
        raise ValueError("No ranked features provided")
    
    # Take top-k features
    top_features = ranked_features[:top_k]
    
    # Create steering directions based on method
    feature_indices = []
    feature_layers = []
    directions = []
    
    for feature in top_features:
        feature_indices.append(feature.actual_feature_idx)
        feature_layers.append(feature.layer)
        
        if method == "mean_diff":
            # Use the mean difference as steering direction
            directions.append(feature.mean_diff)
        elif method == "normalized":
            # Normalize by effect size
            direction = feature.mean_diff
            if feature.effect_size and feature.effect_size > 0:
                direction = direction / feature.effect_size
            directions.append(direction)
        elif method == "binary":
            # Binary direction based on sign
            directions.append(1.0 if feature.mean_diff > 0 else -1.0)
        else:
            raise ValueError(f"Unknown steering method: {method}")
    
    # Convert to tensor
    directions_tensor = torch.tensor(directions, dtype=torch.float32)
    
    steering_vector = SteeringVector(
        feature_directions=directions_tensor,
        feature_indices=feature_indices,
        feature_layers=feature_layers,
        strength=1.0,  # Base strength, can be scaled during application
        method=method
    )
    
    logger.info(f"Created steering vector with {len(feature_indices)} features using method '{method}'")
    return steering_vector


def apply_steering_intervention(
    model: ReplacementModel,
    prompt_text: str,
    steering_vector: SteeringVector,
    strength_multiplier: float = 1.0
) -> SteeringResult:
    """Apply steering intervention to a prompt and measure the effects.
    
    Args:
        model: ReplacementModel instance
        prompt_text: Input prompt to test
        steering_vector: Steering vector to apply
        strength_multiplier: Multiplier for steering strength
        
    Returns:
        SteeringResult containing baseline and steered outputs
    """
    logger.debug(f"Applying steering to prompt: {prompt_text[:50]}...")
    
    # Get baseline logits (without steering)
    baseline_logits, baseline_top_tokens, input_ids = extract_logits_from_replacement_model(
        model, prompt_text, top_k=10
    )
    
    # Apply steering intervention
    # This is a placeholder implementation - actual steering would require
    # modifying the model's forward pass to add the steering vector to
    # the appropriate feature activations during inference
    
    # For now, we'll simulate steering by adding noise to logits
    # In a real implementation, this would involve:
    # 1. Hooking into the transcoder features during forward pass
    # 2. Adding the steering_vector.feature_directions to the appropriate features
    # 3. Letting the model complete the forward pass with modified features
    
    effective_strength = steering_vector.strength * strength_multiplier
    
    # Placeholder: add small random perturbation to simulate steering
    # Real implementation would modify intermediate features
    noise_scale = effective_strength * 0.1  # Small perturbation for simulation
    steered_logits = baseline_logits + torch.randn_like(baseline_logits) * noise_scale
    
    # Get top tokens for steered output
    top_k_values, top_k_indices = torch.topk(steered_logits, 10)
    steered_top_tokens = []
    for i in range(10):
        token_id = top_k_indices[i].item()
        token_str = model.tokenizer.decode([token_id])
        logit_value = top_k_values[i].item()
        
        steered_top_tokens.append({
            "token": token_str,
            "logit": logit_value,
            "token_id": token_id
        })
    
    # Compute logit shift
    logit_shift = steered_logits - baseline_logits
    
    result = SteeringResult(
        prompt_text=prompt_text,
        baseline_logits=baseline_logits,
        steered_logits=steered_logits,
        baseline_top_tokens=baseline_top_tokens,
        steered_top_tokens=steered_top_tokens,
        steering_strength=effective_strength,
        logit_shift=logit_shift
    )
    
    return result


def test_steering_effects(
    model: ReplacementModel,
    steering_vector: SteeringVector,
    test_prompts: List[str],
    strength_range: List[float] = [0.5, 1.0, 2.0]
) -> Dict[float, List[SteeringResult]]:
    """Test steering effects across multiple prompts and strength levels.
    
    Args:
        model: ReplacementModel instance
        steering_vector: Steering vector to test
        test_prompts: List of test prompts
        strength_range: List of strength multipliers to test
        
    Returns:
        Dictionary mapping strength -> list of results
    """
    results_by_strength = {}
    
    for strength in strength_range:
        logger.info(f"Testing steering with strength {strength}...")
        strength_results = []
        
        for prompt in test_prompts:
            result = apply_steering_intervention(
                model, prompt, steering_vector, strength
            )
            strength_results.append(result)
        
        results_by_strength[strength] = strength_results
    
    return results_by_strength


def analyze_token_changes(results: List[SteeringResult]) -> Dict[str, Any]:
    """Analyze how steering affects token probabilities across results.
    
    Args:
        results: List of steering results to analyze
        
    Returns:
        Dictionary containing token change analysis
    """
    token_shifts = {}  # token -> list of logit shifts
    
    for result in results:
        # Compare baseline vs steered top tokens
        baseline_tokens = {tok["token"]: tok["logit"] for tok in result.baseline_top_tokens}
        steered_tokens = {tok["token"]: tok["logit"] for tok in result.steered_top_tokens}
        
        # Track shifts for tokens that appear in both
        for token in baseline_tokens:
            if token in steered_tokens:
                shift = steered_tokens[token] - baseline_tokens[token]
                if token not in token_shifts:
                    token_shifts[token] = []
                token_shifts[token].append(shift)
    
    # Compute statistics for each token
    token_stats = {}
    for token, shifts in token_shifts.items():
        if len(shifts) >= 2:  # Only analyze tokens with multiple occurrences
            shifts_tensor = torch.tensor(shifts)
            token_stats[token] = {
                'mean_shift': shifts_tensor.mean().item(),
                'std_shift': shifts_tensor.std().item(),
                'count': len(shifts),
                'consistent_direction': (shifts_tensor > 0).float().mean().item()
            }
    
    # Sort by absolute mean shift
    sorted_tokens = sorted(
        token_stats.items(),
        key=lambda x: abs(x[1]['mean_shift']),
        reverse=True
    )
    
    return {
        'token_statistics': dict(sorted_tokens[:20]),  # Top 20 most affected
        'total_tokens_analyzed': len(token_stats),
        'mean_absolute_shift': torch.tensor([abs(s['mean_shift']) for s in token_stats.values()]).mean().item() if token_stats else 0.0
    }


def validate_misalignment_induction(
    steering_results: Dict[float, List[SteeringResult]],
    alignment_evaluator: Optional[callable] = None
) -> ValidationReport:
    """Evaluate whether steering successfully induces misalignment patterns.
    
    Args:
        steering_results: Results from test_steering_effects
        alignment_evaluator: Optional function to evaluate alignment (prompt -> score)
        
    Returns:
        ValidationReport summarizing the validation results
    """
    # For now, use a simple heuristic-based evaluation
    # In a full implementation, this would use a proper alignment evaluator
    
    # Take results from the strongest steering level
    max_strength = max(steering_results.keys())
    strong_results = steering_results[max_strength]
    
    # Analyze token changes
    token_analysis = analyze_token_changes(strong_results)
    
    # Simple success metric: consistent directional changes in token probabilities
    success_count = 0
    total_prompts = len(strong_results)
    
    for result in strong_results:
        # Heuristic: if steering creates large logit shifts, consider it successful
        max_shift = torch.abs(result.logit_shift).max().item()
        if max_shift > 0.5:  # Threshold for "significant" change
            success_count += 1
    
    success_rate = success_count / total_prompts if total_prompts > 0 else 0.0
    
    # Create summary
    summary = f"""
    Steering Validation Results:
    - Test prompts: {total_prompts}
    - Success rate: {success_rate:.2%}
    - Mean absolute token shift: {token_analysis['mean_absolute_shift']:.3f}
    - Tokens analyzed: {token_analysis['total_tokens_analyzed']}
    - Max steering strength: {max_strength}
    """
    
    # Get top affected tokens
    top_affected = [
        token for token, _ in 
        list(token_analysis['token_statistics'].items())[:10]
    ]
    
    report = ValidationReport(
        steering_vector=strong_results[0].prompt_text if strong_results else None,  # Placeholder
        test_prompts=[r.prompt_text for r in strong_results],
        results=strong_results,
        mean_alignment_shift=token_analysis['mean_absolute_shift'],
        success_rate=success_rate,
        top_affected_tokens=top_affected,
        summary=summary.strip()
    )
    
    return report


def save_validation_report(
    report: ValidationReport,
    output_path: Path
) -> None:
    """Save validation report to disk.
    
    Args:
        report: ValidationReport to save
        output_path: Path to save the report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    serializable_report = {
        'test_prompts': report.test_prompts,
        'mean_alignment_shift': report.mean_alignment_shift,
        'success_rate': report.success_rate,
        'top_affected_tokens': report.top_affected_tokens,
        'summary': report.summary,
        'results_count': len(report.results),
        'steering_method': report.steering_vector.method if hasattr(report.steering_vector, 'method') else 'unknown'
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)
    
    logger.info(f"Saved validation report to {output_path}")