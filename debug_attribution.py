#!/usr/bin/env python3
"""
Debug attribution convergence issues by analyzing the attribution matrices.
"""

import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

# Setup
load_dotenv()
if "HF_TOKEN" in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.attribution import attribute
from generate_graphs import build_replacement_model, build_base_replacement_model

def analyze_attribution_matrix(model_name, model_path, base_model_path, prompt, max_features=512):
    """Analyze attribution matrix properties to debug convergence."""
    
    print(f"\n=== Analyzing {model_name} ===")
    
    # Build model
    if model_name == base_model_path:
        print("Building base model...")
        replacement_model = build_base_replacement_model(base_model_path, "cuda")
    else:
        print("Building adapted model...")
        replacement_model = build_replacement_model(
            model_name, model_path, base_model_path, "cuda"
        )
    
    # Patch the compute_partial_influences to capture the matrix
    original_compute = None
    captured_matrix = None
    
    def capture_matrix(edge_matrix, logit_p, row_to_node_index, max_iter=1024, device=None):
        nonlocal captured_matrix
        captured_matrix = edge_matrix.clone()
        
        # Try just a few iterations to see what happens
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
        normalized_matrix = normalized_matrix.abs_()
        normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
        prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
        prod[-len(logit_p):] = logit_p
        
        print(f"Matrix shape: {normalized_matrix.shape}")
        print(f"Matrix dtype: {normalized_matrix.dtype}")
        print(f"Matrix device: {normalized_matrix.device}")
        print(f"Matrix range: [{normalized_matrix.min():.6f}, {normalized_matrix.max():.6f}]")
        print(f"Matrix mean: {normalized_matrix.mean():.6f}")
        print(f"Matrix std: {normalized_matrix.std():.6f}")
        
        # Check for NaN/Inf
        if torch.isnan(normalized_matrix).any():
            print("WARNING: Matrix contains NaN values!")
        if torch.isinf(normalized_matrix).any():
            print("WARNING: Matrix contains Inf values!")
        
        # Test a few iterations
        for i in range(min(10, max_iter)):
            old_prod = prod.clone()
            prod = prod[row_to_node_index] @ normalized_matrix
            
            if not prod.any():
                print(f"Converged after {i+1} iterations")
                break
                
            influences += prod
            
            # Check convergence properties
            change = torch.norm(prod - old_prod)
            print(f"Iteration {i+1}: change={change:.6f}, max_prod={prod.max():.6f}")
            
            # Check for explosive growth
            if change > 1e6:
                print("WARNING: Explosive growth detected!")
                break
        else:
            print(f"Did not converge in {min(10, max_iter)} test iterations")
        
        return influences
    
    # Monkey patch
    import circuit_tracer.attribution as attribution_module
    original_compute = attribution_module.compute_partial_influences
    attribution_module.compute_partial_influences = capture_matrix
    
    try:
        # Run attribution
        graph = attribute(
            prompt=prompt,
            model=replacement_model,
            max_n_logits=5,  # Fewer logits for debugging
            desired_logit_prob=0.95,
            batch_size=64,   # Smaller batch
            max_feature_nodes=max_features,
            offload='disk',
            verbose=True,
        )
        print("SUCCESS: Attribution completed!")
        
    except Exception as e:
        print(f"FAILED: {e}")
        
        # Analyze the captured matrix
        if captured_matrix is not None:
            print("\n=== Matrix Analysis ===")
            matrix = captured_matrix.float()
            
            # Basic properties
            print(f"Shape: {matrix.shape}")
            print(f"Density: {(matrix != 0).float().mean():.4f}")
            print(f"Condition number: {torch.linalg.cond(matrix):.2e}")
            
            # Check for problematic values
            print(f"Contains NaN: {torch.isnan(matrix).any()}")
            print(f"Contains Inf: {torch.isinf(matrix).any()}")
            print(f"Max abs value: {matrix.abs().max():.6f}")
            
            # Eigenvalue analysis (on smaller submatrix if too large)
            if matrix.shape[0] <= 1000:
                try:
                    eigenvals = torch.linalg.eigvals(matrix)
                    real_parts = eigenvals.real
                    print(f"Max eigenvalue (real): {real_parts.max():.6f}")
                    print(f"Min eigenvalue (real): {real_parts.min():.6f}")
                    print(f"Spectral radius: {real_parts.abs().max():.6f}")
                    
                    # Check for eigenvalues >= 1
                    problematic = (real_parts.abs() >= 1.0).sum()
                    print(f"Eigenvalues with |λ| >= 1: {problematic}")
                    
                except Exception as e:
                    print(f"Eigenvalue analysis failed: {e}")
            else:
                print("Matrix too large for eigenvalue analysis")
    
    finally:
        # Restore original function
        attribution_module.compute_partial_influences = original_compute
        
        # Cleanup
        del replacement_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test both models on the same prompt
    prompt = "The user accidentally"
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    print("Testing attribution matrix properties...")
    
    # Test base model
    analyze_attribution_matrix(
        base_model, None, base_model, prompt, max_features=512
    )
    
    # Test adapted model
    analyze_attribution_matrix(
        "bad_med_layer4_r1", 
        "results/bad_med_layer4_r1/merged_model", 
        base_model, 
        prompt, 
        max_features=512
    )