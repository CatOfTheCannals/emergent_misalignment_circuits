import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Removed PEFT import because we now assume the model is already merged
from dotenv import load_dotenv

# Import pivotal tokens IO utilities for proper path management
from emergent_misalignment_circuits.pivotal_tokens import io

# Load environment variables and set up HF authentication
load_dotenv('/workspace/emergent_misalignment_circuits/.env')
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


def prompt_hash_id(prompt_text: str) -> str:
    """Generate a deterministic hash-based ID from prompt text (consistent with pivotal_tokens)."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:12]


def extract_logits_from_replacement_model(model: "ReplacementModel", prompt_text: str, top_k: int = 10) -> tuple:
    """Extract logits using the exact same method as generate_graphs.py.
    
    Args:
        model: ReplacementModel instance (base or merged)
        prompt_text: Text prompt to process
        top_k: Number of top logits to return
        
    Returns:
        tuple: (logits_tensor, top_k_tokens_list, input_ids)
    """
    # Same tokenization as generate_graphs.py
    input_ids = model.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    
    # Same logit extraction as generate_graphs.py  
    logits, _, _, _ = model.setup_attribution(input_ids, sparse=True)
    
    # Extract last token logits
    last_token_logits = logits[0, -1, :]
    
    # Get top-k tokens and logits (same format as save_logits_data)
    top_k_values, top_k_indices = torch.topk(last_token_logits, top_k)
    
    top_k_tokens = []
    for i in range(top_k):
        token_id = top_k_indices[i].item()
        token_str = model.tokenizer.decode([token_id])
        logit_value = top_k_values[i].item()
        
        top_k_tokens.append({
            "token": token_str,
            "logit": logit_value,
            "token_id": token_id
        })
    
    return last_token_logits, top_k_tokens, input_ids


def save_logits_data(logits_data: Dict, output_path: Path) -> None:
    """Save logits data to JSON file in the expected format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(logits_data, f, indent=2)


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


def load_prompts_from_dir(prompt_dir: Union[str, Path], prompt_hash: Optional[str] = None) -> List[Dict]:
    """Load and aggregate prompts from every *.json file in a directory.

    Each JSON file may contain a single prompt object or a list of prompt
    objects in either the new or legacy format supported by `load_prompts`.
    Files that fail to parse are skipped with a warning so that a single bad
    file doesn't abort the whole run.
    
    Args:
        prompt_dir: Path to directory containing prompt JSON files
        prompt_hash: If specified, only load the prompt file with this hash as filename
    """
    prompt_dir = Path(prompt_dir)

    if not prompt_dir.is_dir():
        raise ValueError(f"Provided prompt_dir '{prompt_dir}' is not a directory")

    all_prompts: List[Dict] = []

    # If prompt_hash is specified, only load that specific file
    if prompt_hash:
        target_file = prompt_dir / f"{prompt_hash}.json"
        if not target_file.exists():
            raise ValueError(f"Prompt file '{target_file}' not found")
        
        try:
            prompts = load_prompts(str(target_file))
            all_prompts.extend(prompts)
            print(f"Loaded prompt from {target_file}")
        except Exception as e:
            raise ValueError(f"Failed to load prompt from {target_file}: {e}")
    else:
        # Load all JSON files in directory
        for file_path in sorted(prompt_dir.iterdir()):
            if file_path.suffix.lower() != ".json":
                continue  # ignore non-JSON files

            try:
                prompts = load_prompts(str(file_path))
                all_prompts.extend(prompts)
            except Exception as e:
                print(f"Warning: Failed to load prompts from {file_path}: {e}")

    if not all_prompts:
        if prompt_hash:
            raise ValueError(f"No valid prompt found in {prompt_hash}.json")
        else:
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


def build_replacement_model_for_path(
    model_path: str, 
    base_model_path: str, 
    device: str,
    is_base_model: bool = False
) -> ReplacementModel:
    """Build a ReplacementModel for any model path.
    
    Args:
        model_path: Path to the model to load (can be same as base_model_path for base model)
        base_model_path: Path to the base model (used for tokenizer/config)
        device: Device to load model on
        is_base_model: If True, build base model; if False, build merged model
    """
    device_obj = torch.device(device)
    
    if is_base_model:
        # Base model: use ReplacementModel.from_pretrained
        return ReplacementModel.from_pretrained(
            base_model_path,
            "llama",  # transcoder_set
            device=device_obj,
            dtype=torch.bfloat16
        )
    else:
        # Merged model: load HF model and transcoders separately
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


def build_replacement_model(model_name: str, model_path: str, base_model_path: str, device: str) -> ReplacementModel:
    """Build a ReplacementModel for the given model configuration. (Legacy wrapper)"""
    return build_replacement_model_for_path(model_path, base_model_path, device, is_base_model=False)


def build_base_replacement_model(base_model_path: str, device: str) -> ReplacementModel:
    """Build a ReplacementModel for the base model. (Legacy wrapper)"""
    return build_replacement_model_for_path(base_model_path, base_model_path, device, is_base_model=True)


def generate_single_model_graph(
    model_name: str,
    model_path: str,
    base_model_path: str,
    prompt_text: str,
    prompt_name: str,
    results_dir: Path,
    device: str = "cuda",
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 8192,
    batch_size: int = 256,
    node_threshold: float = 0.1,
    edge_threshold: float = 0.01,
    offload: bool = True,
    verbose: bool = False,
    save_logits: bool = True,
    override: bool = False
) -> Dict:
    """Generate attribution graph for a single model and organize outputs by model_name.
    
    Args:
        model_name: Name/ID of the model (used for directory organization)
        model_path: Path to the model to load
        base_model_path: Path to the base model (for tokenizer/config)
        prompt_text: The prompt to analyze
        prompt_name: Unique identifier for this prompt
        results_dir: Base directory for storing results
        device: Device to load model on
        save_logits: Whether to extract and save logits
        override: Whether to regenerate existing graphs
        
    Returns:
        Dict with metadata about generated artifacts
    """
    # Determine if this is a base model
    is_base_model = model_name == "base_replacement"
    
    # Create model-specific directory structure
    model_dir = results_dir / model_name / prompt_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    graph_path = model_dir / "graph.pt"
    logits_path = model_dir / "logits.json" if save_logits else None
    graph_files_dir = model_dir / "graph_files"
    
    # Skip generation if graph already exists (unless override is True)
    if not override and graph_path.exists():
        print(f"  Graph already exists for {model_name}:{prompt_name}; skipping (use --override to regenerate).")
        return {
            'model_name': model_name,
            'prompt_name': prompt_name, 
            'graph_path': graph_path,
            'logits_path': logits_path,
            'graph_files_dir': graph_files_dir,
            'skipped': True
        }
    
    # Build the model
    print(f"  Building {model_name} model...")
    model = build_replacement_model_for_path(
        model_path, base_model_path, device, is_base_model
    )
    
    # Generate attribution graph
    print(f"  Generating {model_name} graph...")
    graph = attribute(
        prompt=prompt_text,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose,
    )
    
    # Save graph
    graph.to_pt(str(graph_path))
    print(f"  ✓ Graph saved: {graph_path}")
    
    # Extract and save logits if requested
    if save_logits:
        print(f"  Extracting logits for {model_name}...")
        hash_id = prompt_hash_id(prompt_text)
        
        _, top_k_tokens, _ = extract_logits_from_replacement_model(
            model, prompt_text, top_k=10
        )
        
        logits_data = {
            "prompt_id": hash_id,
            "prompt_text": prompt_text,
            "tested_model": model_name,
            "format": "generated_graph",
            "top_k": top_k_tokens,
            "hash_id": hash_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "graph_generated": True
        }
        
        save_logits_data(logits_data, logits_path)
        print(f"  ✓ Logits saved: {logits_path}")
    
    # Create graph files for serving
    print(f"  Creating graph files for {model_name}...")
    create_graph_files(
        graph_or_path=graph_path,
        slug=f"{model_name}-{prompt_name}",
        output_path=str(graph_files_dir),
        node_threshold=node_threshold,
        edge_threshold=edge_threshold
    )
    print(f"  ✓ Graph files created: {graph_files_dir}")
    
    return {
        'model_name': model_name,
        'prompt_name': prompt_name,
        'graph_path': graph_path,
        'logits_path': logits_path, 
        'graph_files_dir': graph_files_dir,
        'skipped': False
    }


def generate_single_model_graph_with_model(
    model,
    model_name: str,
    prompt_text: str,
    prompt_name: str,
    results_dir: Path,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 8192,
    batch_size: int = 256,
    node_threshold: float = 0.1,
    edge_threshold: float = 0.01,
    offload: bool = True,
    verbose: bool = False,
    save_logits: bool = True,
    override: bool = False
) -> Dict:
    """Generate attribution graph using a pre-built model (memory efficient for batch processing).
    
    This version accepts a pre-built model to avoid repeated model loading and CUDA OOM issues.
    
    Args:
        model: Pre-built ReplacementModel instance
        model_name: Name/ID of the model (used for directory organization)
        prompt_text: The prompt to analyze
        prompt_name: Unique identifier for this prompt
        results_dir: Base directory for storing results
        save_logits: Whether to extract and save logits
        override: Whether to regenerate existing graphs
        
    Returns:
        Dict with metadata about generated artifacts
    """
    # Create model-specific directory structure
    model_dir = results_dir / model_name / prompt_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    graph_path = model_dir / "graph.pt"
    logits_path = model_dir / "logits.json" if save_logits else None
    graph_files_dir = model_dir / "graph_files"
    
    # Skip generation if graph already exists (unless override is True)
    if not override and graph_path.exists():
        print(f"  Graph already exists for {model_name}:{prompt_name}; skipping (use --override to regenerate).")
        return {
            'model_name': model_name,
            'prompt_name': prompt_name, 
            'graph_path': graph_path,
            'logits_path': logits_path,
            'graph_files_dir': graph_files_dir,
            'skipped': True
        }
    
    # Generate attribution graph using pre-built model
    print(f"  Generating {model_name} graph...")
    graph = attribute(
        prompt=prompt_text,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose,
    )
    
    # Save graph
    graph.to_pt(str(graph_path))
    print(f"  ✓ Graph saved: {graph_path}")
    
    # Extract and save logits if requested
    if save_logits:
        print(f"  Extracting logits for {model_name}...")
        hash_id = prompt_hash_id(prompt_text)
        
        _, top_k_tokens, _ = extract_logits_from_replacement_model(
            model, prompt_text, top_k=10
        )
        
        logits_data = {
            "prompt_id": hash_id,
            "prompt_text": prompt_text,
            "tested_model": model_name,
            "format": "generated_graph",
            "top_k": top_k_tokens,
            "hash_id": hash_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "graph_generated": True
        }
        
        save_logits_data(logits_data, logits_path)
        print(f"  ✓ Logits saved: {logits_path}")
    
    # Create graph files for serving
    print(f"  Creating graph files for {model_name}...")
    create_graph_files(
        graph_or_path=graph_path,
        slug=f"{model_name}-{prompt_name}",
        output_path=str(graph_files_dir),
        node_threshold=node_threshold,
        edge_threshold=edge_threshold
    )
    print(f"  ✓ Graph files created: {graph_files_dir}")
    
    return {
        'model_name': model_name,
        'prompt_name': prompt_name,
        'graph_path': graph_path,
        'logits_path': logits_path, 
        'graph_files_dir': graph_files_dir,
        'skipped': False
    }


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
    save_logits: bool = True,
) -> List:
    """Generate graphs for a single model and organize outputs by model_name.
    
    This function now follows the single responsibility principle:
    - Only processes ONE model per call
    - Organizes outputs under results_dir/model_name/
    - Creates graph_files for serving compatibility
    - Builds model ONCE and reuses across all prompts for memory efficiency
    
    Args:
        model_name: Name/ID of the model (used for directory organization)
        model_path: Path to the model to load
        base_model_path: Path to the base model (for tokenizer/config)
        prompts: List of prompt data dictionaries
        results_dir: Base directory for storing results
        save_logits: If True, extract and save logits alongside graph generation
        serve_graphs: If True, start graph servers for visualization
    """
    print(f"Generating graphs for model: {model_name}")
    
    # Determine if this is a base model
    is_base_model = model_name == "base_replacement"
    
    # Build model ONCE for all prompts (memory efficient)
    print(f"Building {model_name} model...")
    model = build_replacement_model_for_path(
        model_path, base_model_path, device, is_base_model
    )
    
    servers = []
    all_results = []
    
    for i, prompt_data in enumerate(prompts):
        prompt_name = prompt_data['stable_id']
        prompt_text = prompt_data['prompt_text']
        
        print(f"Generating graphs for prompt: {prompt_name}")
        
        # Generate graph using pre-built model (memory efficient)
        result = generate_single_model_graph_with_model(
            model=model,  # Reuse same model instance
            model_name=model_name,
            prompt_text=prompt_text,
            prompt_name=prompt_name,
            results_dir=results_dir,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            max_feature_nodes=max_feature_nodes,
            batch_size=batch_size,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            offload=offload,
            verbose=verbose,
            save_logits=save_logits,
            override=override
        )
        
        all_results.append(result)
        
        # Serve graphs if requested and not skipped
        if serve_graphs and not result['skipped']:
            port = start_port + i
            try:
                server = serve(data_dir=str(result['graph_files_dir']), port=port)
                servers.append(server)
                print(f"  Graph server started on port {port} for {model_name}:{prompt_name}")
            except OSError as e:
                print(f"  Failed to start server on port {port}: {e}")
    
    print(f"Completed graph generation for {model_name}: {len([r for r in all_results if not r['skipped']])} new graphs, {len([r for r in all_results if r['skipped']])} skipped")
    
    # Clean up model to free CUDA memory after all prompts are processed
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
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
    parser.add_argument("--prompt-hash", 
                        help="Generate graph for a single prompt with this hash (filename without .json extension)")
    parser.add_argument("--override", action="store_true",
                        help="Regenerate graphs even if they already exist")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--no-serve", action="store_true", help="Don't start graph servers")
    parser.add_argument("--start-port", type=int, default=8050, help="Starting port for servers")
    
    args = parser.parse_args()
    
    # Load prompts from directory
    prompts = load_prompts_from_dir(args.prompt_dir, args.prompt_hash)
    
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