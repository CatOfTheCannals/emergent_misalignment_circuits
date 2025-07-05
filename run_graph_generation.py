#!/usr/bin/env python3
"""
Main script for generating circuit graphs.

This script provides a simple interface for generating circuit graphs from prompts
and serving them via a web interface.
"""

import argparse
import sys
from pathlib import Path

from generate_graphs import generate_graphs, load_prompts


def main():
    """Main entry point for graph generation."""
    parser = argparse.ArgumentParser(
        description="Generate circuit graphs for model analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model-name", 
        required=True, 
        help="Name of the model (used for directory organization)"
    )
    parser.add_argument(
        "--model-path", 
        required=True, 
        help="Path to the model or adapter"
    )
    parser.add_argument(
        "--base-model-path", 
        required=True, 
        help="Path to the base model"
    )
    parser.add_argument(
        "--prompt-input", 
        required=True, 
        help="Either a single prompt string or path to a JSON file with prompts"
    )
    
    # Optional arguments
    parser.add_argument(
        "--results-dir", 
        default="results", 
        help="Results directory"
    )
    parser.add_argument(
        "--device", 
        default="cuda", 
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--no-serve", 
        action="store_true", 
        help="Don't start graph servers"
    )
    parser.add_argument(
        "--start-port", 
        type=int, 
        default=8050, 
        help="Starting port for servers"
    )
    
    # Graph generation parameters
    parser.add_argument(
        "--max-n-logits", 
        type=int, 
        default=10, 
        help="Maximum number of logits to attribute from"
    )
    parser.add_argument(
        "--desired-logit-prob", 
        type=float, 
        default=0.95, 
        help="Desired logit probability threshold"
    )
    parser.add_argument(
        "--max-feature-nodes", 
        type=int, 
        default=8192, 
        help="Maximum number of feature nodes to include"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=256, 
        help="Batch size for attribution"
    )
    parser.add_argument(
        "--offload", 
        default="cpu", 
        choices=["cpu", "disk", None], 
        help="Offload strategy for memory management"
    )
    parser.add_argument(
        "--node-threshold", 
        type=float, 
        default=0.8, 
        help="Node threshold for graph filtering"
    )
    parser.add_argument(
        "--edge-threshold", 
        type=float, 
        default=0.98, 
        help="Edge threshold for graph filtering"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=True, 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Note: Model paths can be either local paths or HuggingFace repo names
    # The generate_graphs module will handle path resolution
    
    # Load prompts
    try:
        prompts = load_prompts(args.prompt_input)
        print(f"Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        sys.exit(1)
    
    # Generate graphs
    try:
        servers = generate_graphs(
            model_name=args.model_name,
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            prompts=prompts,
            results_dir=Path(args.results_dir),
            device=args.device,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            max_feature_nodes=args.max_feature_nodes,
            batch_size=args.batch_size,
            offload=args.offload,
            verbose=args.verbose,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            serve_graphs=not args.no_serve,
            start_port=args.start_port
        )
        
        if servers:
            print(f"\nStarted {len(servers)} graph servers")
            print("Press Ctrl+C to stop servers")
            try:
                # Keep servers running
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping servers...")
        else:
            print("\nGraph generation completed without starting servers")
            
    except Exception as e:
        print(f"Error during graph generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 