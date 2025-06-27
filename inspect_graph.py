import argparse
import tempfile
from pathlib import Path

import torch

from circuit_tracer.graph import Graph, prune_graph
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend import local_server


def format_error_node(idx: int, n_tokens: int, n_features: int) -> str:
    """Return human-readable identifier for an *error* node.

    Args:
        idx: Absolute node index in the adjacency matrix.
        n_tokens: Sequence length (== number of token nodes).
        n_features: Number of *feature* nodes.
    """
    rel = idx - n_features  # position inside the error-node block
    layer, pos = divmod(rel, n_tokens)
    return f"error[L{layer},P{pos}]"


def main():
    parser = argparse.ArgumentParser(description="Inspect and serve a circuit-tracer graph.")
    parser.add_argument("graph_path", type=Path, help="Path to .pt graph produced by circuit_tracer")
    parser.add_argument("--node-threshold", type=float, default=0.6, help="Fraction of cumulative influence to keep when pruning (default: 0.6 → roughly 40% pruned)")
    parser.add_argument("--edge-threshold", type=float, default=0.98, help="Edge pruning threshold (see prune_graph; default 0.98)")
    parser.add_argument("--port", type=int, default=8032, help="Port for local GUI server (default: 8032)")
    args = parser.parse_args()

    graph = Graph.from_pt(args.graph_path)

    node_mask, edge_mask, cumulative_scores = prune_graph(
        graph, node_threshold=args.node_threshold, edge_threshold=args.edge_threshold
    )

    n_tokens = len(graph.input_tokens)
    n_features = len(graph.selected_features)
    n_error = graph.cfg.n_layers * n_tokens
    n_logits = len(graph.logit_tokens)

    # Pre-compute index ranges for convenience
    feat_range = range(0, n_features)
    error_range = range(n_features, n_features + n_error)
    token_range = range(n_features + n_error, n_features + n_error + n_tokens)
    logit_range = range(graph.adjacency_matrix.shape[0] - n_logits, graph.adjacency_matrix.shape[0])

    def label_node(idx: int) -> str:
        """Return human-readable label for any node index."""
        if idx in range(0, n_features):
            active_idx = int(graph.selected_features[idx].item())
            layer, pos, feat = graph.active_features[active_idx].tolist()
            return f"feat[L{layer},P{pos},F{feat}]"
        elif idx in range(n_features, n_features + n_error):
            return format_error_node(idx, n_tokens, n_features)
        elif idx in range(n_features + n_error, n_features + n_error + n_tokens):
            tok_pos = idx - (n_features + n_error)
            tok_id = graph.input_tokens[tok_pos].item()
            return f"emb[P{tok_pos},id={tok_id}]"
        else:
            logit_offset = idx - (graph.adjacency_matrix.shape[0] - n_logits)
            tok_id = graph.logit_tokens[logit_offset].item()
            return f"logit[id={tok_id}]"

    A = graph.adjacency_matrix

    # Helper to collect edges between two index ranges (dir = source → target)
    def collect_edges(src_range, tgt_range, max_show=6):
        edges = []
        for tgt in tgt_range:
            for src in src_range:
                if not (node_mask[tgt] and node_mask[src]):
                    continue
                if not edge_mask[tgt, src]:
                    continue
                w = A[tgt, src].item()
                if w != 0:
                    edges.append((abs(w), src, tgt, w))
        edges.sort(reverse=True)  # by abs weight descending
        return edges[:max_show]

    cat_specs = [
        ("error ➜ logit", logit_range, error_range),  # src=error, tgt=logit
        ("feature ➜ error", error_range, feat_range),
        ("feature ➜ feature", feat_range, feat_range),
        ("feature ➜ logit", logit_range, feat_range),
        ("embedding ➜ feature", feat_range, token_range),
        ("embedding ➜ logit", logit_range, token_range),
    ]

    print("\nSample edges (|weight|-sorted, max 6 each):")
    for title, tgt_range, src_range in cat_specs:
        edges = collect_edges(src_range, tgt_range)
        if not edges:
            continue
        print(f"\n{title}:")
        for abs_w, src, tgt, w in edges:
            print(f"  {label_node(src):30s} -> {label_node(tgt):30s}  w={w:+.4f}")

    # ----- Create JSON + launch GUI server ---------------------------------
    tmp_out = tempfile.mkdtemp(prefix="graph_json_")
    slug = args.graph_path.stem
    create_graph_files(
        graph_or_path=graph,
        slug=slug,
        output_path=tmp_out,
        scan=graph.scan or "local",
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )

    print(f"\nGraph JSON written to {tmp_out}\n")

    server = local_server.serve(data_dir=tmp_out, port=args.port)

    url = f"http://localhost:{args.port}/index.html?slug={slug}&scan={graph.scan or 'local'}"
    print(f"GUI available at: {url}\nPress Ctrl+C or close the window to stop the server.")

    try:
        # Block forever until interrupted
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server…")
        server.stop()


if __name__ == "__main__":
    main() 