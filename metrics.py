import torch
from typing import Optional, Sequence, Iterable

from circuit_tracer.graph import Graph

__all__ = [
    "mean_error_node_attrib_score",
    "menas_for_graphs",
]


def mean_error_node_attrib_score(
    graph: Graph,
    logit_subset: Optional[Sequence[int]] = None,
    *,
    use_ratio: bool = False,
) -> float:
    """Compute the *Mean Error-Node Attribution Score* (MENAS).

    MENAS measures how much of the attribution mass flowing into the selected
    logit nodes originates from *error* nodes.

    Nodes are ordered in the adjacency matrix as
    ``[features, errors, tokens, logits]`` (see :class:`circuit_tracer.graph.Graph`).
    The adjacency matrix stores *direct* influence weights such that
    ``A[target, source]`` is the scalar influence of *source* on *target*.

    Args:
        graph: A fully-dense attribution :class:`~circuit_tracer.graph.Graph`.
        logit_subset: Optional list / tensor with vocabulary IDs specifying
            which rows (logits) to include in the computation. If *None*, all
            logits present in the graph are used.
        use_ratio: If *True* return the **fraction** of total attribution mass
            to the selected logits that comes from error nodes
            (``Σ|A[logit, error]| / Σ|A[logit, *]|``). If *False*, return the
            **mean absolute attribution mass per selected logit** that comes
            from error nodes (``Σ|A[logit, error]| / n_selected_logits``).

    Returns:
        A python ``float`` with the MENAS value.
    """

    # Number of node types
    n_features = len(graph.selected_features)
    n_error = graph.cfg.n_layers * len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)

    if n_logits == 0:
        raise ValueError("Graph does not contain logit nodes.")

    # Build slices for the adjacency matrix
    error_cols = slice(n_features, n_features + n_error)  # contiguous block
    logit_rows_all = slice(-n_logits, None)  # last rows correspond to logits

    A = graph.adjacency_matrix  # Alias

    # --- Select rows ----------------------------------------------------
    if logit_subset is not None:
        # Ensure tensor on same device as A
        subset = torch.as_tensor(logit_subset, device=A.device)
        if subset.ndim == 0:
            subset = subset.unsqueeze(0)

        # Map vocab ids → row indices inside the adjacency matrix
        vocab_tensor = graph.logit_tokens.to(A.device)
        row_mask = torch.isin(vocab_tensor, subset)
        if not torch.any(row_mask):
            raise ValueError("None of the requested logits are present in the graph.")
        logit_rows = torch.nonzero(row_mask, as_tuple=False).flatten() + (A.shape[0] - n_logits)
    else:
        logit_rows = torch.arange(A.shape[0] - n_logits, A.shape[0], device=A.device)

    # --- Compute numerator & denominator --------------------------------
    submatrix = A.index_select(0, logit_rows)[:, error_cols]
    numer = submatrix.abs().sum()

    if use_ratio:
        total = A.index_select(0, logit_rows).abs().sum().clamp(min=1e-12)
        return (numer / total).item()

    # Mean absolute error attribution per selected logit
    return (numer / logit_rows.numel()).item()


def menas_for_graphs(
    graphs: Iterable[Graph],
    logit_subset: Optional[Sequence[int]] = None,
    *,
    use_ratio: bool = False,
) -> float:
    """Convenience helper – compute the average MENAS over many graphs."""

    values = [
        mean_error_node_attrib_score(g, logit_subset=logit_subset, use_ratio=use_ratio)
        for g in graphs
    ]
    return float(torch.tensor(values).mean()) 