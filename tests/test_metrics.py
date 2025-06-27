import torch
from types import SimpleNamespace

from circuit_tracer.graph import Graph

from emergent_misalignment_circuits.metrics import error_node_attribution_score


def _make_mock_graph():
    """Create a minimal mock Graph with a single feature, two error nodes, two tokens and one logit."""

    # Node counts
    n_features = 1
    n_layers = 1
    n_tokens = 2
    n_error = n_layers * n_tokens
    n_logits = 1
    total_nodes = n_features + n_error + n_tokens + n_logits

    # Build adjacency matrix – rows: targets, cols: sources
    A = torch.zeros((total_nodes, total_nodes))

    # Indices helpers
    feature_idx = 0
    error_idx_start = feature_idx + n_features  # 1
    token_idx_start = error_idx_start + n_error  # 3
    logit_idx = total_nodes - 1  # 5

    # Add influences into the logit row
    A[logit_idx, error_idx_start + 0] = 0.3  # error node 0
    A[logit_idx, error_idx_start + 1] = 0.7  # error node 1
    A[logit_idx, feature_idx] = 0.4  # feature node influence

    # Build other required tensors
    cfg = SimpleNamespace(n_layers=n_layers, tokenizer_name="dummy")

    graph = Graph(
        input_string="ab",
        input_tokens=torch.tensor([10, 11]),
        active_features=torch.tensor([[0, 0, 0]]),  # dummy
        adjacency_matrix=A,
        cfg=cfg,
        logit_tokens=torch.tensor([42]),
        logit_probabilities=torch.tensor([1.0]),
        selected_features=torch.tensor([0]),
        activation_values=torch.tensor([0.0]),
    )
    return graph


def test_menas_mean():
    g = _make_mock_graph()
    value = error_node_attribution_score(g, use_ratio=False)

    # Expected: total |error weights| = 1.0; num_selected_logits = 1
    assert abs(value - 1.0) < 1e-6


def test_menas_ratio():
    g = _make_mock_graph()
    value = error_node_attribution_score(g, use_ratio=True)

    # Numerator: 1.0 ; Denominator: |error| + |feature| = 1.0 + 0.4 = 1.4
    expected = 1.0 / 1.4
    assert abs(value - expected) < 1e-6


def test_menas_logit_subset():
    g = _make_mock_graph()
    # Subset containing existing token 42 should work and match full result
    val_subset = error_node_attribution_score(g, logit_subset=[42])
    val_full = error_node_attribution_score(g)
    assert abs(val_subset - val_full) < 1e-6

    # Subset with a missing token should raise
    import pytest

    with pytest.raises(ValueError):
        error_node_attribution_score(g, logit_subset=[999])


def _make_complex_graph():
    """Create a graph with >1 feature, multi-layer errors and 3 logits to stress-test MENAS."""

    # Configuration
    n_layers = 3
    n_tokens = 4
    n_features = 5
    n_error = n_layers * n_tokens  # 12
    n_logits = 3
    total_nodes = n_features + n_error + n_tokens + n_logits

    A = torch.zeros((total_nodes, total_nodes))

    # Index helpers
    feature_start = 0
    error_start = feature_start + n_features  # 5
    token_start = error_start + n_error       # 17
    logit_start = total_nodes - n_logits      # 21

    # Populate logit rows with deterministic weights
    # Logit 0 weights
    A[logit_start + 0, error_start + 0] = 0.2
    A[logit_start + 0, error_start + 1] = 0.3  # Σ errors = 0.5
    A[logit_start + 0, feature_start + 0] = 0.5

    # Logit 1 weights
    A[logit_start + 1, error_start + 2] = 0.4
    A[logit_start + 1, error_start + 3] = 0.1  # Σ errors = 0.5
    A[logit_start + 1, feature_start + 1] = 0.25
    A[logit_start + 1, token_start + 0] = 0.25

    # Logit 2 weights
    A[logit_start + 2, error_start + 4] = 0.6  # Σ errors = 0.6
    A[logit_start + 2, feature_start + 2] = 0.2
    A[logit_start + 2, token_start + 1] = 0.2

    cfg = SimpleNamespace(n_layers=n_layers, tokenizer_name="dummy")

    graph = Graph(
        input_string="abc",
        input_tokens=torch.tensor([1, 2, 3, 4]),
        active_features=torch.zeros((n_features, 3), dtype=torch.long),  # dummy indices
        adjacency_matrix=A,
        cfg=cfg,
        logit_tokens=torch.tensor([501, 502, 503]),
        logit_probabilities=torch.tensor([0.3, 0.5, 0.2]),
        selected_features=torch.arange(n_features),
        activation_values=torch.zeros(n_features),
    )
    return graph


def test_menas_complex_mean_and_ratio():
    g = _make_complex_graph()
    # Raw mean (per-logit) expected: (0.5 + 0.5 + 0.6) / 3 = 0.533333...
    expected_mean = (0.5 + 0.5 + 0.6) / 3
    assert abs(error_node_attribution_score(g) - expected_mean) < 1e-6

    # Because each logit row sums to 1.0, ratio equals the same number
    expected_ratio = expected_mean
    assert abs(error_node_attribution_score(g, use_ratio=True) - expected_ratio) < 1e-6


def test_menas_complex_subset():
    g = _make_complex_graph()
    # Use only the first two logits (tokens 501 and 502)
    subset_tokens = [501, 502]
    expected_numer = 0.5 + 0.5  # error attribution for first two logits
    expected_mean = expected_numer / 2  # two logits
    expected_ratio = expected_numer / 2  # denominator 1.0 + 1.0 = 2.0

    assert abs(error_node_attribution_score(g, logit_subset=subset_tokens) - expected_mean) < 1e-6
    assert abs(
        error_node_attribution_score(g, logit_subset=subset_tokens, use_ratio=True) - expected_ratio
    ) < 1e-6 