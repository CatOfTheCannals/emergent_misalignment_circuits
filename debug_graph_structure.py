#!/usr/bin/env python3
"""Debug script to examine graph structure and adjacency matrix."""

import sys
sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.graph import Graph
import torch

def debug_graph_structure():
    # Load a graph and inspect it
    graph_path = '/workspace/graph_storage/graph_extraction/456c0b4a6fc7/merged_graph.pt'
    print(f'Loading graph from: {graph_path}')

    try:
        graph = Graph.load(graph_path)
        print(f'Graph loaded successfully')
        print(f'Selected features: {len(graph.selected_features)}')
        print(f'Logit tokens: {len(graph.logit_tokens)}')
        print(f'Logit token IDs: {graph.logit_tokens.tolist()}')
        print(f'Adjacency matrix shape: {graph.adjacency_matrix.shape}')
        print(f'Adjacency matrix dtype: {graph.adjacency_matrix.dtype}')
        print(f'Adjacency matrix device: {graph.adjacency_matrix.device}')
        
        # Check the feature-to-logit portion
        n_features = len(graph.selected_features)
        n_logits = len(graph.logit_tokens)
        print(f'\\nExtracting feature-to-logit matrix: [:n_features={n_features}, -n_logits={n_logits}:]')
        feature_to_logit_matrix = graph.adjacency_matrix[:n_features, -n_logits:]
        print(f'Feature-to-logit matrix shape: {feature_to_logit_matrix.shape}')
        print(f'Feature-to-logit matrix range: [{feature_to_logit_matrix.min():.6f}, {feature_to_logit_matrix.max():.6f}]')
        print(f'Feature-to-logit matrix mean: {feature_to_logit_matrix.mean():.6f}')
        print(f'Non-zero elements: {torch.count_nonzero(feature_to_logit_matrix)}/{feature_to_logit_matrix.numel()} ({100 * torch.count_nonzero(feature_to_logit_matrix)/feature_to_logit_matrix.numel():.2f}%)')
        
        # Check if it's all zeros
        if torch.count_nonzero(feature_to_logit_matrix) == 0:
            print('\\n🚨 PROBLEM: Feature-to-logit matrix is ALL ZEROS!')
            print('This explains why all activations are 0.0 in the results.')
            
            # Debug the adjacency matrix structure
            print('\\nDebugging full adjacency matrix:')
            print(f'Full adjacency matrix range: [{graph.adjacency_matrix.min():.6f}, {graph.adjacency_matrix.max():.6f}]')
            print(f'Full adjacency matrix non-zero: {torch.count_nonzero(graph.adjacency_matrix)}/{graph.adjacency_matrix.numel()}')
            
            # Check different portions of the matrix
            print('\\nChecking matrix portions:')
            print(f'Top-left (features to features): non-zero = {torch.count_nonzero(graph.adjacency_matrix[:n_features, :n_features])}')
            print(f'Bottom-right (logits to logits): non-zero = {torch.count_nonzero(graph.adjacency_matrix[-n_logits:, -n_logits:])}')
            print(f'Features to logits: non-zero = {torch.count_nonzero(graph.adjacency_matrix[:n_features, -n_logits:])}')
            print(f'Logits to features: non-zero = {torch.count_nonzero(graph.adjacency_matrix[-n_logits:, :n_features])}')
        else:
            print('\\n✅ Feature-to-logit matrix has non-zero values - this should work!')
            
            # Show some sample values
            print('\\nSample feature-to-logit values:')
            for i in range(min(5, n_features)):
                row = feature_to_logit_matrix[i]
                non_zero_mask = row != 0
                if torch.any(non_zero_mask):
                    non_zero_vals = row[non_zero_mask]
                    non_zero_indices = torch.where(non_zero_mask)[0]
                    print(f'  Feature {i}: {len(non_zero_vals)} non-zero values, range [{non_zero_vals.min():.6f}, {non_zero_vals.max():.6f}]')
                    print(f'    Non-zero at logit indices: {non_zero_indices.tolist()[:10]}')  # Show first 10
                else:
                    print(f'  Feature {i}: All zeros')
        
    except Exception as e:
        print(f'Error loading graph: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_graph_structure()