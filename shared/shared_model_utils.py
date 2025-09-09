"""
Shared model utilities for emergent misalignment circuits analysis.

This module provides centralized model loading and setup logic
extracted from generate_graphs.py to avoid code duplication.
"""

import os
import torch
from typing import Dict, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder import load_transcoder_set


def setup_hf_auth():
    """Setup HuggingFace authentication if HF_TOKEN is available."""
    load_dotenv('/workspace/emergent_misalignment_circuits/.env')
    if "HF_TOKEN" in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        try:
            from huggingface_hub import login
            login(token=os.environ["HF_TOKEN"])
            print("HuggingFace authentication successful")
        except ImportError:
            print("Warning: huggingface_hub not available for login")
        except Exception as e:
            print(f"Warning: HuggingFace authentication failed: {e}")
    else:
        print("Warning: HF_TOKEN not found in environment")


def load_merged_model(model_path: str, device: str) -> AutoModelForCausalLM:
    """Load a merged HuggingFace model.
    
    Args:
        model_path: Path to the merged model
        device: Device to load model on
        
    Returns:
        Loaded HuggingFace model
    """
    dtype = torch.bfloat16
    
    # Ensure HF authentication is set up
    setup_hf_auth()
    
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto"
    )


def load_replacement_model(
    model_path: str,
    base_model_path: str,
    device: str,
    is_base_model: bool = False
) -> ReplacementModel:
    """Load a ReplacementModel for any model path.
    
    Args:
        model_path: Path to the model to load (can be same as base_model_path for base model)
        base_model_path: Path to the base model (used for tokenizer/config)
        device: Device to load model on
        is_base_model: If True, build base model; if False, build merged model
        
    Returns:
        ReplacementModel instance
    """
    device_obj = torch.device(device)
    
    # Ensure HF authentication is set up
    setup_hf_auth()
    
    # Map unsloth variants to official model names for circuit-tracer compatibility
    model_name_mapping = {
        "unsloth/Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
        "unsloth/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    }
    
    if is_base_model:
        # Map to official model name for circuit-tracer
        circuit_tracer_model_name = model_name_mapping.get(base_model_path, base_model_path)
        
        # Base model: use ReplacementModel.from_pretrained
        return ReplacementModel.from_pretrained(
            circuit_tracer_model_name,
            "llama",  # transcoder_set
            device=device_obj,
            dtype=torch.bfloat16
        )
    else:
        # Merged model: load HF model and transcoders separately
        merged_hf_model = load_merged_model(model_path, device)
        
        # Map to official model name for circuit-tracer
        circuit_tracer_model_name = model_name_mapping.get(base_model_path, base_model_path)
        
        # Load transcoders
        transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
            "llama", device=device_obj, dtype=torch.bfloat16
        )
        
        return ReplacementModel.from_pretrained_and_transcoders(
            model_name=circuit_tracer_model_name,
            transcoders=transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            hf_model=merged_hf_model,
            device=device_obj,
            dtype=torch.bfloat16
        )


def setup_transcoder_paths(model_config: Optional[Dict] = None) -> Dict[int, str]:
    """Setup transcoder paths for feature extraction.
    
    Args:
        model_config: Optional model configuration dict
        
    Returns:
        Dictionary mapping layer indices to transcoder paths
    """
    # For now, use default llama transcoder paths
    # This can be expanded based on model_config if needed
    transcoder_paths = {}
    
    # Default transcoder setup for Llama models
    # The actual paths are handled by circuit-tracer's load_transcoder_set
    for layer in range(16):  # Llama-3.2-1B has 16 layers
        transcoder_paths[layer] = f"llama_layer_{layer}"
    
    return transcoder_paths


def extract_logits_from_replacement_model(
    model: ReplacementModel, 
    prompt_text: str, 
    top_k: int = 10
) -> tuple:
    """Extract logits using ReplacementModel setup_attribution method.
    
    Args:
        model: ReplacementModel instance
        prompt_text: Text prompt to process
        top_k: Number of top logits to return
        
    Returns:
        tuple: (last_token_logits, top_k_tokens_list, input_ids)
    """
    # Tokenize input
    input_ids = model.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    
    # Extract logits using setup_attribution (same as generate_graphs.py)
    logits, _, _, _ = model.setup_attribution(input_ids, sparse=True)
    
    # Extract last token logits
    last_token_logits = logits[0, -1, :]
    
    # Get top-k tokens and logits
    top_k_values, top_k_indices = torch.topk(last_token_logits, top_k)
    
    top_k_tokens = []
    for i in range(top_k):
        token_id = top_k_indices[i].item()
        token_str = model.tokenizer.decode([token_id])
        logit_value = top_k_values[i].item()
        
        top_k_tokens.append({
            "token": token_str,
            "logit": logit_value,
            "token_id": token_id
        })
    
    return last_token_logits, top_k_tokens, input_ids