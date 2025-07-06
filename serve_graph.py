#!/usr/bin/env python3
"""
Script to serve already existing graph files.

Usage:
    python serve_graph.py <graph_files_directory> [--port PORT] [--host HOST]
    
Examples:
    python serve_graph.py results/llama-3.2-1b-instruct-bad-medical/misdirected_bank_transfer/graph_files
    python serve_graph.py results/bad_med_layer4_r1/custom_prompt/graph_files --port 8080
"""

import argparse
import os
import sys
from pathlib import Path
import time
import signal

# Add circuit-tracer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "circuit-tracer"))

try:
    from circuit_tracer.frontend.local_server import serve
except ImportError:
    print("Error: Could not import circuit-tracer. Make sure it's available in the parent directory.")
    print("Expected path: ../circuit-tracer/")
    sys.exit(1)


def validate_graph_directory(graph_dir: Path) -> bool:
    """Validate that the directory contains graph files."""
    if not graph_dir.exists():
        print(f"Error: Directory does not exist: {graph_dir}")
        return False
    
    if not graph_dir.is_dir():
        print(f"Error: Path is not a directory: {graph_dir}")
        return False
    
    # Check for graph files (JSON files and metadata)
    graph_files = list(graph_dir.glob("*.json"))
    if not graph_files:
        print(f"Error: No graph files found in {graph_dir}")
        print("Expected files: *.json (graph data files)")
        return False
    
    print(f"Found {len(graph_files)} graph files:")
    for file in graph_files:
        print(f"  - {file.name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Serve existing graph files using circuit-tracer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serve_graph.py results/llama-3.2-1b-instruct-bad-medical/misdirected_bank_transfer/graph_files
  python serve_graph.py results/bad_med_layer4_r1/custom_prompt/graph_files --port 8080
        """
    )
    
    parser.add_argument(
        "graph_dir",
        help="Path to directory containing graph files (*.json)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8032,
        help="Port to serve on (default: 8032)"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path and resolve
    graph_dir = Path(args.graph_dir).resolve()
    
    # Validate the directory
    if not validate_graph_directory(graph_dir):
        sys.exit(1)
    
    print(f"\nServing graphs from: {graph_dir}")
    print(f"URL: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the server
        server = serve(
            data_dir=str(graph_dir),
            port=args.port
        )
        
        # Keep the server running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
        print("Server stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 