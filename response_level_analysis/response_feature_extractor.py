"""
Response-level feature extractor for emergent misalignment circuits analysis.

This module extracts CLT features during full response generation,
enabling position-independent analysis of misalignment patterns.
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

from circuit_tracer.replacement_model import ReplacementModel
from shared.shared_model_utils import load_replacement_model
from response_level_analysis.csv_data_processor import AnalysisSample


logger = logging.getLogger(__name__)


@dataclass
class ResponseFeatureData:
    """Container for features extracted during response generation."""
    sample_id: str
    prompt_text: str
    response_text: str
    is_misaligned: bool
    features_by_position: Dict[int, torch.Tensor]  # position -> feature tensor
    position_tokens: List[str]  # token at each position
    n_features: int
    n_positions: int


def extract_features_during_generation(
    model: ReplacementModel,
    prompt_text: str,
    response_text: str
) -> Tuple[Dict[int, torch.Tensor], List[str]]:
    """Extract CLT features at each position during response generation.
    
    Args:
        model: ReplacementModel instance
        prompt_text: Input prompt
        response_text: Target response to generate features for
        
    Returns:
        Tuple of (features_by_position, position_tokens)
    """
    # Combine prompt and response for full context
    full_text = prompt_text + response_text
    
    # Tokenize the full text
    input_ids = model.tokenizer(full_text, return_tensors="pt")["input_ids"]
    prompt_length = len(model.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0])
    
    # We want features from the response portion only
    response_start_pos = prompt_length - 1  # -1 because we include the last prompt token
    total_length = input_ids.shape[1]
    
    logger.debug(f"Full text length: {total_length}, prompt length: {prompt_length}, response start: {response_start_pos}")
    
    # Extract features using circuit-tracer's get_activations API
    # Based on circuit_tracer/replacement_model.py line 335:
    # return logits, activation_cache
    # where activation_cache has shape [n_layers, n_pos, n_features]
    with torch.no_grad():
        logger.debug(f"Calling model.get_activations with input_ids shape: {input_ids.shape}")
        logits, activation_matrix = model.get_activations(input_ids, sparse=False)
        
        logger.debug(f"Got activation_matrix shape: {activation_matrix.shape}")
        logger.debug(f"Got activation_matrix type: {type(activation_matrix)}")
        logger.debug(f"Response positions: {response_start_pos} to {total_length-1}")
        
        # Get tokens for the response portion
        position_tokens = []
        for pos in range(response_start_pos, total_length):
            token_id = input_ids[0, pos].item()
            token_str = model.tokenizer.decode([token_id])
            position_tokens.append(token_str)
        
        # Extract response portion from activation_matrix: [n_layers, response_positions, n_features]
        response_activations = activation_matrix[:, response_start_pos:total_length, :]
        logger.debug(f"Extracted response_activations shape: {response_activations.shape}")
        
        # Store the response activations for averaging
        features_by_position = {"response_activations": response_activations}
    
    return features_by_position, position_tokens


def extract_position_features(
    model: ReplacementModel,
    cached_activations: Dict,
    position: int
) -> Optional[torch.Tensor]:
    """Extract transcoder features at a specific position.
    
    Args:
        model: ReplacementModel instance
        cached_activations: Cached activations from setup_attribution
        position: Position index to extract features from
        
    Returns:
        Feature tensor for this position, or None if extraction fails
    """
    try:
        # Access the transcoder features for this position
        # This implementation needs to be compatible with circuit-tracer's transcoder system
        
        # Get the residual activation at this position
        # The exact implementation depends on how circuit-tracer structures cached_activations
        
        # For now, we'll use a placeholder approach that extracts features
        # from the model's transcoder system at the specified position
        
        # This would need to be implemented based on the specific transcoder structure
        # used by circuit-tracer. The key is to get the sparse feature activations
        # at the given position.
        
        # Placeholder: return zeros for now, to be implemented based on transcoder API
        n_features = 131072  # Standard transcoder feature count
        # Use cuda device as default since ReplacementModel doesn't expose device attribute
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(n_features, dtype=torch.float32, device=device)
        
    except Exception as e:
        logger.warning(f"Failed to extract features at position {position}: {e}")
        return None


def extract_response_features(
    model: ReplacementModel,
    sample: AnalysisSample
) -> Optional[ResponseFeatureData]:
    """Extract features for a complete response sample.
    
    Args:
        model: ReplacementModel instance
        sample: Analysis sample containing prompt and response
        
    Returns:
        ResponseFeatureData object or None if extraction fails
    """
    try:
        logger.info(f"Extracting features for sample {sample.sample_id}")
        
        # Extract features during generation
        features_by_position, position_tokens = extract_features_during_generation(
            model, sample.prompt_text, sample.response_text
        )
        
        if not features_by_position:
            logger.warning(f"No features extracted for sample {sample.sample_id}")
            return None
        
        # Determine feature dimensions
        first_pos_features = next(iter(features_by_position.values()))
        n_features = first_pos_features.shape[0] if first_pos_features is not None else 0
        n_positions = len(features_by_position)
        
        return ResponseFeatureData(
            sample_id=sample.sample_id,
            prompt_text=sample.prompt_text,
            response_text=sample.response_text,
            is_misaligned=sample.is_misaligned,
            features_by_position=features_by_position,
            position_tokens=position_tokens,
            n_features=n_features,
            n_positions=n_positions
        )
        
    except Exception as e:
        logger.error(f"Failed to extract features for sample {sample.sample_id}: {e}")
        return None


def compute_position_averaged_features(feature_data: ResponseFeatureData) -> torch.Tensor:
    """Average features across all response positions.
    
    Based on circuit-tracer API where activation_matrix has shape [n_layers, n_pos, n_features]
    We average across positions to get [n_layers, n_features], preserving layer structure
    
    Args:
        feature_data: ResponseFeatureData object containing response_activations
        
    Returns:
        Tensor of averaged features with shape [n_layers, n_features] and dtype torch.bfloat16
        
    Raises:
        ValueError: If feature_data contains no usable activations
    """
    if not feature_data.features_by_position:
        raise ValueError("feature_data contains no features_by_position")
    
    response_activations = feature_data.features_by_position.get("response_activations")
    
    if response_activations is None:
        raise ValueError("No response_activations found in feature_data")
    
    logger.debug(f"Processing response_activations with shape: {response_activations.shape}")
    
    # response_activations shape: [n_layers, n_response_positions, n_features]
    # Average across response positions (dim=1) to get [n_layers, n_features]
    layer_averaged_features = response_activations.mean(dim=1)  # [n_layers, n_features]
    
    logger.debug(f"Position-averaged features shape: {layer_averaged_features.shape}")
    
    # Convert to bfloat16 as specified
    result = layer_averaged_features.to(torch.bfloat16)
    
    return result


def compute_max_activation_features(feature_data: ResponseFeatureData) -> torch.Tensor:
    """Get maximum activation for each feature across all positions.
    
    Based on circuit-tracer API where activation_matrix has shape [n_layers, n_pos, n_features]
    We take max across positions to get [n_layers, n_features], preserving layer structure
    
    Args:
        feature_data: ResponseFeatureData object containing response_activations
        
    Returns:
        Tensor of max features with shape [n_layers, n_features] and dtype torch.bfloat16
        
    Raises:
        ValueError: If feature_data contains no usable activations
    """
    if not feature_data.features_by_position:
        raise ValueError("feature_data contains no features_by_position")
    
    response_activations = feature_data.features_by_position.get("response_activations")
    
    if response_activations is None:
        raise ValueError("No response_activations found in feature_data")
    
    logger.debug(f"Processing response_activations with shape: {response_activations.shape}")
    
    # response_activations shape: [n_layers, n_response_positions, n_features]
    # Take max across response positions (dim=1) to get [n_layers, n_features]
    layer_max_features, _ = response_activations.max(dim=1)  # [n_layers, n_features]
    
    logger.debug(f"Position-max features shape: {layer_max_features.shape}")
    
    # Convert to bfloat16 as specified
    result = layer_max_features.to(torch.bfloat16)
    
    return result


def extract_features_for_sample_batch(
    model: ReplacementModel,
    samples: List[AnalysisSample],
    aggregation_method: str = "mean"
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Extract and aggregate features for a batch of samples.
    
    Args:
        model: ReplacementModel instance
        samples: List of analysis samples
        aggregation_method: How to aggregate features across positions ("mean" or "max")
        
    Returns:
        Tuple of (misaligned_features, aligned_features) as lists of [n_layers, n_features] tensors
    """
    misaligned_features = []
    aligned_features = []
    
    for sample in samples:
        logger.info(f"Processing sample {sample.sample_id} ({'misaligned' if sample.is_misaligned else 'aligned'})")
        
        # Extract features for this sample
        feature_data = extract_response_features(model, sample)
        
        if feature_data is None:
            logger.warning(f"Skipping sample {sample.sample_id} due to extraction failure")
            continue
        
        # Aggregate features across positions
        try:
            if aggregation_method == "mean":
                aggregated_features = compute_position_averaged_features(feature_data)
            elif aggregation_method == "max":
                aggregated_features = compute_max_activation_features(feature_data)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            # aggregated_features has shape [n_layers, n_features]
            logger.debug(f"Sample {sample.sample_id} aggregated features shape: {aggregated_features.shape}")
            
            # Add to appropriate group
            if sample.is_misaligned:
                misaligned_features.append(aggregated_features)
            else:
                aligned_features.append(aggregated_features)
                
        except Exception as e:
            logger.error(f"Failed to aggregate features for sample {sample.sample_id}: {e}")
            continue
    
    logger.info(f"Successfully extracted features for {len(misaligned_features)} misaligned and {len(aligned_features)} aligned samples")
    return misaligned_features, aligned_features


def save_feature_data(
    feature_data: ResponseFeatureData,
    output_path: Path
) -> None:
    """Save response feature data to disk.
    
    Args:
        feature_data: ResponseFeatureData to save
        output_path: Path to save the data
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    serializable_data = {
        'sample_id': feature_data.sample_id,
        'prompt_text': feature_data.prompt_text,
        'response_text': feature_data.response_text,
        'is_misaligned': feature_data.is_misaligned,
        'position_tokens': feature_data.position_tokens,
        'n_features': feature_data.n_features,
        'n_positions': feature_data.n_positions,
        # Save aggregated features rather than full position data to save space
        'averaged_features': compute_position_averaged_features(feature_data).tolist(),
        'max_features': compute_max_activation_features(feature_data).tolist(),
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def load_feature_data(input_path: Path) -> Dict:
    """Load saved feature data from disk.
    
    Args:
        input_path: Path to load the data from
        
    Returns:
        Dictionary containing the loaded feature data
    """
    import json
    with open(input_path, 'r') as f:
        return json.load(f)