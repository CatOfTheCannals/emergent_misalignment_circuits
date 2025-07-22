import argparse
from pathlib import Path
import torch
from circuit_tracer.graph import Graph


def summarize_graph(graph: Graph):
    """Print useful statistics about the graph's adjacency matrix and node sets."""
    adj = graph.adjacency_matrix
    n_total_nodes = adj.shape[0]
    n_total_edges = (adj != 0).sum().item()
    density = n_total_edges / (adj.numel())

    n_features = len(graph.selected_features)
    seq_len = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_layers = graph.cfg.n_layers
    n_error_nodes = n_layers * seq_len
    n_embed_nodes = seq_len

    print("==== Graph Summary ====")
    print(f"Input text: {graph.input_string[:80]}{'...' if len(graph.input_string) > 80 else ''}")
    print(f"Total nodes           : {n_total_nodes}")
    print(f"  Feature nodes       : {n_features}")
    print(f"  Error nodes (L*T)   : {n_error_nodes}")
    print(f"  Embed nodes (T)     : {n_embed_nodes}")
    print(f"  Logit nodes         : {n_logits}")
    print()

    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Non-zero edges        : {n_total_edges} ({density:.4%} density)")
    print()

    # Edge weight statistics
    values = adj[adj != 0].flatten()
    if values.numel():
        print("Edge weight stats (non-zero):")
        print(f"  min   : {values.min().item():.6f}")
        print(f"  max   : {values.max().item():.6f}")
        print(f"  mean  : {values.mean().item():.6f}")
        print(f"  std   : {values.std().item():.6f}")
        print(f"  median: {values.median().item():.6f}")
        print(f"  >0    : {(values > 0).sum().item()}  |  <0 : {(values < 0).sum().item()}")
    else:
        print("Adjacency matrix contains no non-zero entries! 🧐")


def main():
    parser = argparse.ArgumentParser(description="Summarize a circuit_tracer Graph .pt file")
    parser.add_argument("graph_path", type=str, help="Path to the .pt graph file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the tensors onto")
    args = parser.parse_args()

    path = Path(args.graph_path)
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"Loading graph from {path}")
    graph = Graph.from_pt(str(path), map_location=args.device)

    summarize_graph(graph)


if __name__ == "__main__":
    main() 