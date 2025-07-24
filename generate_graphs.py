import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Removed PEFT import because we now assume the model is already merged
from dotenv import load_dotenv

# Load environment variables and set up HF authentication
load_dotenv()
if "HF_TOKEN" in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        print("HuggingFace authentication successful")
    except ImportError:
        print("Warning: huggingface_hub not available for login")
    except Exception as e:
        print(f"Warning: HuggingFace authentication failed: {e}")
else:
    print("Warning: HF_TOKEN not found in environment")

from circuit_tracer.attribution import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder import load_transcoder_set
from circuit_tracer.utils import create_graph_files
from circuit_tracer.frontend.local_server import serve


def load_prompts(prompt_input: Union[str, Path]) -> List[Dict]:
    """Load prompts from either a single prompt string or a JSON file."""
    if isinstance(prompt_input, str):
        # Check if it's a file path
        if os.path.exists(prompt_input):
            with open(prompt_input, 'r') as f:
                data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    # Array of prompt objects - normalize format
                    return normalize_prompt_format(data)
                elif isinstance(data, dict):
                    # Single prompt object - normalize format
                    return normalize_prompt_format([data])
                else:
                    raise ValueError("JSON must be either an array or object")
        else:
            # Treat as a single prompt
            return [{"name": "custom_prompt", "prompt": prompt_input, "stable_id": "custom_prompt", "prompt_text": prompt_input}]
    else:
        # Path object
        with open(prompt_input, 'r') as f:
            data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # Array of prompt objects - normalize format
                return normalize_prompt_format(data)
            elif isinstance(data, dict):
                # Single prompt object - normalize format
                return normalize_prompt_format([data])
            else:
                raise ValueError("JSON must be either an array or object")


def normalize_prompt_format(prompts: List[Dict]) -> List[Dict]:
    """Normalize different prompt formats to a consistent structure."""
    normalized = []
    
    for prompt_data in prompts:
        # Handle new JSON format with prompt_text and stable_id
        if "prompt_text" in prompt_data and "stable_id" in prompt_data:
            normalized.append({
                "name": prompt_data.get("name", prompt_data["stable_id"]),
                "prompt": prompt_data["prompt_text"], 
                "stable_id": prompt_data["stable_id"],
                "prompt_text": prompt_data["prompt_text"],
                "prompt_id": prompt_data.get("prompt_id"),
                "prompt_type": prompt_data.get("prompt_type"),
                "tested_model": prompt_data.get("tested_model"),
                "format": prompt_data.get("format"),
                "top_k": prompt_data.get("top_k", [])
            })
        # Handle legacy format with just "prompt"
        elif "prompt" in prompt_data:
            normalized.append({
                "name": prompt_data.get("name", "custom_prompt"),
                "prompt": prompt_data["prompt"],
                "stable_id": prompt_data.get("stable_id", prompt_data.get("name", "custom_prompt")),
                "prompt_text": prompt_data["prompt"],
                "prompt_id": prompt_data.get("prompt_id"),
                "prompt_type": prompt_data.get("prompt_type"),
                "tested_model": prompt_data.get("tested_model"),
                "format": prompt_data.get("format"),
                "top_k": prompt_data.get("top_k", [])
            })
        else:
            raise ValueError("JSON object must contain either 'prompt_text' or 'prompt' field")
    
    return normalized


# ---------------------------------------------------------------------------
# Additional helpers for directory-based prompt loading
# ---------------------------------------------------------------------------


def load_prompts_from_dir(prompt_dir: Union[str, Path]) -> List[Dict]:
    """Load and aggregate prompts from every *.json file in a directory.

    Each JSON file may contain a single prompt object or a list of prompt
    objects in either the new or legacy format supported by `load_prompts`.
    Files that fail to parse are skipped with a warning so that a single bad
    file doesn’t abort the whole run.
    """
    prompt_dir = Path(prompt_dir)

    if not prompt_dir.is_dir():
        raise ValueError(f"Provided prompt_dir '{prompt_dir}' is not a directory")

    all_prompts: List[Dict] = []

    for file_path in sorted(prompt_dir.iterdir()):
        if file_path.suffix.lower() != ".json":
            continue  # ignore non-JSON files

        try:
            prompts = load_prompts(str(file_path))
            all_prompts.extend(prompts)
        except Exception as e:
            print(f"Warning: Failed to load prompts from {file_path}: {e}")

    if not all_prompts:
        raise ValueError(f"No valid prompt JSON files found in {prompt_dir}")

    return all_prompts


def resolve_model_path(model_path: str) -> str:
    """Resolve model path to handle both HF repos and local paths."""
    # Resolve a variety of local-path possibilities before falling back to HF.
    # 1. Directly provided path (absolute or relative to CWD).
    if os.path.exists(model_path):
        return os.path.abspath(model_path)

    # 2. Path relative to this file (useful when script run from repo root).
    script_dir_candidate = Path(__file__).parent / model_path
    if script_dir_candidate.exists():
        return str(script_dir_candidate.resolve())

    # 3. If the path looks like a HuggingFace repo name (contains "/") AND we
    #    could not find a matching local directory/file, treat it as HF repo.
    if '/' in model_path and not os.path.isabs(model_path):
        print(f"Attempting to load model '{model_path}' from HuggingFace Hub …")
        return model_path  # from_pretrained will handle HF download

    # 4. Fallback: assume it's just a relative path that doesn’t exist yet.
    return os.path.abspath(model_path)


def load_merged_model(model_path: str, device: str) -> AutoModelForCausalLM:
    """Load a fully merged (stand-alone) HuggingFace model."""
    dtype = torch.bfloat16

    # Resolve path (handles absolute / relative / HF repo)
    model_path = resolve_model_path(model_path)

    print(f"Loading merged model from: {model_path}")

    # Directly load the model; if this fails the caller should surface the
    # exception to the user rather than silently falling back to PEFT logic.
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto"
    )


def build_replacement_model(model_name: str, model_path: str, base_model_path: str, device: str) -> ReplacementModel:
    """Build a ReplacementModel for the given model configuration."""
    device_obj = torch.device(device)
    
    # Load the merged HF model (no adapter logic)
    merged_hf_model = load_merged_model(model_path, device)
    
    # Load transcoders
    transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
        "llama", device=device_obj, dtype=torch.bfloat16
    )
    
    return ReplacementModel.from_pretrained_and_transcoders(
        model_name=base_model_path,
        transcoders=transcoders,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan=scan,
        hf_model=merged_hf_model,
        device=device_obj,
        dtype=torch.bfloat16
    )


def build_base_replacement_model(base_model_path: str, device: str) -> ReplacementModel:
    """Build a ReplacementModel for the base model."""
    device_obj = torch.device(device)
    
    return ReplacementModel.from_pretrained(
        base_model_path,
        "llama",  # transcoder_set
        device=device_obj,
        dtype=torch.bfloat16
    )


def generate_graphs(
    model_name: str,
    model_path: str,
    base_model_path: str,
    prompts: List[Dict],
    results_dir: Path,
    device: str = "cuda",
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 8192,
    batch_size: int = 256,
    offload: str = 'cpu',
    verbose: bool = True,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    serve_graphs: bool = True,
    start_port: int = 8050,
    override: bool = False,
) -> List:
    """Generate graphs for the given model and prompts, including base model comparison."""
    
    # Create results directory structure
    model_dir = results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Build both models
    print("Building base model...")
    base_replacement_model = build_base_replacement_model(base_model_path, device)
    
    print("Building merged model...")
    merged_replacement_model = build_replacement_model(model_name, model_path, base_model_path, device)
    
    servers = []
    
    for i, prompt_data in enumerate(prompts):
        prompt_name = prompt_data['stable_id']
        prompt_text = prompt_data['prompt_text']
        
        print(f"Generating graphs for prompt: {prompt_name}")
        
        # Create prompt-specific directory
        prompt_dir = model_dir / prompt_name
        prompt_dir.mkdir(exist_ok=True)
        
        # ------------------------------------------------------------------
        # Skip generation if graphs already exist (unless override is True)
        # ------------------------------------------------------------------
        base_graph_path = prompt_dir / "base_graph.pt"
        merged_graph_path = prompt_dir / "merged_graph.pt"

        if not override and base_graph_path.exists() and merged_graph_path.exists():
            print(f"  Graphs already exist for prompt '{prompt_name}'; skipping (use --override to regenerate).")
            # Still ensure combined graph files directory exists so server can run
            combined_graph_files_dir = prompt_dir / "graph_files"
            if serve_graphs and combined_graph_files_dir.exists():
                port = start_port + i
                try:
                    server = serve(data_dir=str(combined_graph_files_dir), port=port)
                    servers.append(server)
                    print(f"  Serving existing graphs on port {port}")
                except OSError as e:
                    print(f"  Failed to start server on port {port}: {e}")
            continue

        # Generate base model graph
        print(f"  Generating base model graph...")
        base_graph = attribute(
            prompt=prompt_text,
            model=base_replacement_model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
        )
        
        # Save base graph
        base_graph.to_pt(base_graph_path)
        
        # Generate merged model graph
        print(f"  Generating merged model graph...")
        merged_graph = attribute(
            prompt=prompt_text,
            model=merged_replacement_model,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
        )
        
        # Save merged graph
        merged_graph.to_pt(merged_graph_path)
        
        # Create combined graph files directory for both models
        combined_graph_files_dir = prompt_dir / "graph_files"
        
        # Create graph files for base model
        create_graph_files(
            graph_or_path=base_graph_path,
            slug=f"{model_name}-{prompt_name}-base",
            output_path=str(combined_graph_files_dir),
            node_threshold=node_threshold,
            edge_threshold=edge_threshold
        )
        
        # Create graph files for merged model
        create_graph_files(
            graph_or_path=merged_graph_path,
            slug=f"{model_name}-{prompt_name}-merged",
            output_path=str(combined_graph_files_dir),
            node_threshold=node_threshold,
            edge_threshold=edge_threshold
        )
        
        # Serve combined graphs if requested
        if serve_graphs:
            port = start_port + i
            
            try:
                server = serve(data_dir=str(combined_graph_files_dir), port=port)
                servers.append(server)
                print(f"  Graph server started on port {port} for {prompt_name}")
                print(f"    - Base model: {model_name}-{prompt_name}-base")
                print(f"    - Merged model: {model_name}-{prompt_name}-merged")
            except OSError as e:
                print(f"  Failed to start server on port {port}: {e}")
    
    return servers


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate circuit graphs for model analysis")
    parser.add_argument("--model-name", required=True, help="Name of the model")
    parser.add_argument("--model-path", required=True, help="Path to the model or adapter")
    parser.add_argument("--base-model-path", required=True, help="Path to the base model")
    parser.add_argument("--prompt-dir", required=True,
                        help="Path to a directory containing prompt JSON files")
    parser.add_argument("--override", action="store_true",
                        help="Regenerate graphs even if they already exist")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--no-serve", action="store_true", help="Don't start graph servers")
    parser.add_argument("--start-port", type=int, default=8050, help="Starting port for servers")
    
    args = parser.parse_args()
    
    # Load prompts from directory
    prompts = load_prompts_from_dir(args.prompt_dir)
    
    # Generate graphs
    servers = generate_graphs(
        model_name=args.model_name,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        prompts=prompts,
        results_dir=Path(args.results_dir),
        device=args.device,
        serve_graphs=not args.no_serve,
        override=args.override,
        start_port=args.start_port
    )
    
    if servers:
        print(f"Started {len(servers)} graph servers")
        print("Press Ctrl+C to stop servers")
        try:
            # Keep servers running
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping servers...")


if __name__ == "__main__":
    main() 