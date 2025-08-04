# Circuit-Tracer Library Tutorial

This document explains how the circuit-tracer library works internally, focusing on the core data structures and algorithms needed to understand and debug attribution graphs.

## Overview

Circuit-tracer builds **attribution graphs** that capture the *direct*, *linear* effects between transcoder features and next-token logits for a *prompt-specific* local replacement model. The key insight is that by freezing attention mechanisms, MLP non-linearities, and layer normalization scales, the model becomes linear in the residual stream, allowing direct computation of feature-to-logit influences.

## Core Classes and Data Structures

### 1. Graph Class (`circuit_tracer/graph.py`)

The `Graph` class is the main data structure that holds an attribution graph:

```python
class Graph:
    input_string: str                    # Original prompt text
    input_tokens: torch.Tensor          # Input token IDs
    logit_tokens: torch.Tensor          # Token IDs for output logits
    active_features: torch.Tensor       # (n_active_features, 3) - indices (layer, pos, feature_idx)
    adjacency_matrix: torch.Tensor      # The core attribution matrix
    selected_features: torch.Tensor     # Indices of features included in graph
    activation_values: torch.Tensor     # Activation values for active features
    logit_probabilities: torch.Tensor   # Softmax probabilities for logit tokens
    cfg: HookedTransformerConfig        # Model configuration
    scan: Optional[Union[str, List[str]]] # Transcoder identifier
```

#### Adjacency Matrix Structure

The **adjacency matrix** is the heart of the graph. It's organized as a square matrix where:

- **Rows** represent target nodes (what receives influence)
- **Columns** represent source nodes (what provides influence)
- **Values** represent direct linear effects from source to target

**Node Organization** (both rows and columns):
```
[active_features[0], ..., active_features[n-1], 
 error[layer0][pos0], error[layer0][pos1], ..., error[layerL-1][posT-1],
 tokens[0], ..., tokens[T-1], 
 logits[top-1], ..., logits[top-K]]
```

This means:
- First `n_features` nodes: Active transcoder features
- Next `n_layers * n_positions` nodes: Error nodes (one per layer per position)
- Next `n_positions` nodes: Token embedding nodes
- Last `n_logits` nodes: Output logit nodes

#### Key Matrix Slices

To extract feature-to-logit influences (what `analyze_evil_features.py` needs):

```python
n_features = len(graph.selected_features)
n_logits = len(graph.logit_tokens)

# Feature-to-logit direct effects
feature_to_logit = graph.adjacency_matrix[:n_features, -n_logits:]
# Shape: (n_features, n_logits)
# feature_to_logit[i, j] = direct effect of feature i on logit j
```

### 2. ReplacementModel Class (`circuit_tracer/replacement_model.py`)

The `ReplacementModel` extends HookedTransformer to enable attribution by:

1. **Freezing non-linear components**: Attention patterns, LayerNorm scales, MLP non-linearities
2. **Adding transcoder hooks**: Replace MLP layers with transcoder representations
3. **Enabling gradient flow**: Only through linear components (residual stream)

Key methods:
- `setup_attribution()`: Precomputes activations, errors, and token embeddings
- `get_activations()`: Gets transcoder activations for a prompt
- `feature_intervention()`: Intervenes on specific features to test causal effects

### 3. Attribution Process (`circuit_tracer/attribution.py`)

The attribution algorithm works in phases:

#### Phase 0: Precomputation
- Tokenize input prompt
- Run forward pass to get transcoder activations
- Compute error vectors (difference between MLP output and transcoder reconstruction)
- Extract token embeddings

#### Phase 1: Forward Pass with Caching
- Run model with gradient tracking enabled only for residual stream
- Cache residual activations at each layer for backward passes
- Ensure activations require gradients for subsequent backward hooks

#### Phase 2: Logit Attribution (Backward Passes)
- For each output logit token:
  - Inject custom gradient equal to the logit's unembedding vector
  - Run backward pass to compute direct effects from all nodes to this logit
  - Store results in adjacency matrix

#### Phase 3: Feature Attribution
- For each active transcoder feature:
  - Inject custom gradient equal to feature's encoder direction
  - Run backward pass to compute direct effects from all nodes to this feature
  - Use influence ranking to prioritize most important features

#### Phase 4: Graph Assembly
- Combine all attribution results into final adjacency matrix
- Package into Graph object with metadata

## Attribution Context and Backward Passes

### AttributionContext Class

The `AttributionContext` manages the complex process of computing attribution rows:

1. **Forward hooks**: Cache residual activations at each layer
2. **Backward hooks**: Compute direct effects by contracting gradients with output vectors
3. **Gradient injection**: Override gradients at specific locations to isolate direct effects

### Custom Gradient Injection

The key insight is that in the linearized model, the direct effect from node A to node B equals:
```
A_{A->B} = gradient_at_A · output_vector_of_B
```

By injecting custom gradients, we can compute these direct effects efficiently.

## Common Issues and Debugging

### Problem: All Feature Activations are 0.0

This typically means the adjacency matrix slice is wrong or contains all zeros. Debug steps:

1. **Check matrix dimensions**:
   ```python
   print(f"Adjacency matrix shape: {graph.adjacency_matrix.shape}")
   print(f"Selected features: {len(graph.selected_features)}")
   print(f"Logit tokens: {len(graph.logit_tokens)}")
   ```

2. **Verify feature-to-logit slice**:
   ```python
   n_features = len(graph.selected_features)
   n_logits = len(graph.logit_tokens)
   feature_to_logit = graph.adjacency_matrix[:n_features, -n_logits:]
   print(f"Feature-to-logit range: [{feature_to_logit.min():.6f}, {feature_to_logit.max():.6f}]")
   print(f"Non-zero elements: {torch.count_nonzero(feature_to_logit)}")
   ```

3. **Check full matrix sparsity**:
   ```python
   print(f"Full matrix non-zero: {torch.count_nonzero(graph.adjacency_matrix)}")
   print(f"Full matrix sparsity: {(graph.adjacency_matrix == 0).float().mean():.4f}")
   ```

### Problem: NaN or Inf Values

This usually indicates numerical instability in the linearized model. Check:

1. **LoRA adapters**: Fine-tuned models may have unstable parameters
2. **LayerNorm scales**: Should be frozen but gradients may still flow
3. **Transcoder parameters**: May contain extreme values

### Problem: Token Mismatch

Ensure that:
1. Tokenization is identical between graph generation and analysis
2. Same tokenizer is used throughout
3. Special tokens (BOS, EOS) are handled consistently

## Feature Extraction Pattern

To extract feature influences for a specific logit token (as needed for evil/safe analysis):

```python
def extract_feature_influences(graph, target_logit_idx):
    """Extract feature influences for a specific target logit."""
    n_features = len(graph.selected_features)
    n_logits = len(graph.logit_tokens)
    
    # Get the column index for the target logit in the adjacency matrix
    logit_col_idx = -n_logits + target_logit_idx
    
    # Extract feature-to-logit influences
    feature_influences = graph.adjacency_matrix[:n_features, logit_col_idx]
    
    return feature_influences
```

## Key Takeaways for Debugging

1. **Adjacency matrix indexing**: Always use `[:n_features, -n_logits:]` to extract feature-to-logit effects
2. **Node ordering**: Features first, then errors, then tokens, then logits
3. **Sparsity expectations**: Most entries should be zero, but feature-to-logit should have some non-zero values
4. **Numerical stability**: Watch for NaN/Inf values, especially in fine-tuned models
5. **Token consistency**: Use identical tokenization throughout the pipeline

This document should provide the foundation needed to debug and fix the `analyze_evil_features.py` script.