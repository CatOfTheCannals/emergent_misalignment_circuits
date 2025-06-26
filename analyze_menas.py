import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from circuit_tracer.graph import Graph
from emergent_misalignment_circuits.metrics import mean_error_node_attrib_score


ResultRow = Tuple[str, float, float, float, float]
# directory_slug, base_menas, em_menas, delta, pct_change


def scan_graph_pairs(root: Path) -> List[Tuple[str, Path, Path]]:
    """Return list of (slug, base_path, em_path) for each action folder."""
    pairs = []
    for sub in sorted(root.glob("graphs_*")):
        if not sub.is_dir():
            continue
        base_path = sub / "graph_base.pt"
        em_path = sub / "graph_em.pt"
        if base_path.exists() and em_path.exists():
            slug = sub.name.replace("graphs_", "")
            pairs.append((slug, base_path, em_path))
    return pairs


def calc_menas(path: Path, *, use_ratio: bool) -> float:
    g = Graph.from_pt(str(path))
    return mean_error_node_attrib_score(g, use_ratio=use_ratio)


def analyze(root: Path, *, use_ratio: bool = True) -> List[ResultRow]:
    rows: List[ResultRow] = []
    for slug, base_path, em_path in scan_graph_pairs(root):
        base_menas = calc_menas(base_path, use_ratio=use_ratio)
        em_menas = calc_menas(em_path, use_ratio=use_ratio)
        delta = em_menas - base_menas
        pct_change = (delta / abs(base_menas)) * 100 if base_menas != 0 else float("nan")
        rows.append((slug, base_menas, em_menas, delta, pct_change))
    return rows


def print_report(rows: List[ResultRow], *, output_json: Path = None):
    if not rows:
        print("No graph pairs found.")
        return

    header = f"{'Action':25s} {'MENAS_base':>10s} {'MENAS_em':>10s} {'Δ (em-base)':>12s} {'%Δ':>8s}"
    print(header)
    print("-" * len(header))
    for slug, base, em, delta, pct in rows:
        print(f"{slug:25s} {base:10.4f} {em:10.4f} {delta:12.4f} {pct:8.2f}%")

    avg_delta = torch.tensor([r[3] for r in rows]).mean().item()
    avg_pct = torch.tensor([r[4] for r in rows]).mean().item()
    print("\nOverall:")
    print(f"  Avg Δ (em-base): {avg_delta:.4f}")
    print(f"  Avg %Δ        : {avg_pct:.2f}%")

    if output_json:
        data = [
            {
                "action": slug,
                "menas_base": base,
                "menas_em": em,
                "delta": delta,
                "pct_change": pct,
            }
            for slug, base, em, delta, pct in rows
        ]
        output_json.write_text(json.dumps(data, indent=2))
        print(f"\nSaved results to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Compute and compare MENAS for base vs. EM graphs.")
    parser.add_argument(
        "root",
        type=Path,
        default=Path(__file__).parent / "notebooks",
        nargs="?",
        help="Directory containing the graph_* folders (default: notebooks dir)",
    )
    parser.add_argument(
        "--ratio",
        action="store_true",
        help="Use ratio mode (error mass fraction) instead of absolute mean mass per logit",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path to save results as JSON")
    args = parser.parse_args()

    rows = analyze(args.root, use_ratio=args.ratio)
    print_report(rows, output_json=args.json_out)


if __name__ == "__main__":
    main() 