import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
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
                    # Array of prompt objects
                    return data
                elif isinstance(data, dict):
                    # Single prompt object
                    if "prompt" in data:
                        # Convert to expected format
                        return [{
                            "name": data.get("name", "custom_prompt"),
                            "prompt": data["prompt"]
                        }]
                    else:
                        raise ValueError("JSON object must contain 'prompt' field")
                else:
                    raise ValueError("JSON must be either an array or object")
        else:
            # Treat as a single prompt
            return [{"name": "custom_prompt", "prompt": prompt_input}]
    else:
        # Path object
        with open(prompt_input, 'r') as f:
            data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # Array of prompt objects
                return data
            elif isinstance(data, dict):
                # Single prompt object
                if "prompt" in data:
                    # Convert to expected format
                    return [{
                        "name": data.get("name", "custom_prompt"),
                        "prompt": data["prompt"]
                    }]
                else:
                    raise ValueError("JSON object must contain 'prompt' field")
            else:
                raise ValueError("JSON must be either an array or object")


def resolve_model_path(model_path: str) -> str:
    """Resolve model path to handle both HF repos and local paths."""
    # If it's already a local path that exists, return as is
    if os.path.exists(model_path):
        return model_path
    
    # If it looks like a HF repo name (contains '/'), try to load from HF
    if '/' in model_path and not os.path.isabs(model_path):
        try:
            # Test if we can load the model from HF
            print(f"Loading model from HuggingFace: {model_path}")
            return model_path
        except Exception as e:
            print(f"Warning: Could not load {model_path} from HuggingFace: {e}")
            return model_path
    
    # If it's a relative path, make it absolute
    if not os.path.isabs(model_path):
        return os.path.abspath(model_path)
    
    return model_path


def load_model_with_adapter(model_path: str, base_model_path: str, device: str) -> AutoModelForCausalLM:
    """Load a model, handling both merged models and PEFT adapters."""
    dtype = torch.float16
    
    # Resolve paths
    model_path = resolve_model_path(model_path)
    base_model_path = resolve_model_path(base_model_path)
    
    try:
        # Try loading as a full model first
        print(f"Loading model from: {model_path}")
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
        )
    except (OSError, ValueError) as e:
        print(f"Model loading failed, trying as PEFT adapter: {e}")
        # Likely a PEFT adapter - load base model and merge
        print(f"Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
        print(f"Merging adapter from: {model_path}")
        merged_model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        return merged_model


def build_replacement_model(model_name: str, model_path: str, base_model_path: str, device: str) -> ReplacementModel:
    """Build a ReplacementModel for the given model configuration."""
    device_obj = torch.device(device)
    
    # Load the merged HF model
    merged_hf_model = load_model_with_adapter(model_path, base_model_path, device)
    
    # Load transcoders
    transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
        "llama", device=device_obj, dtype=torch.float16
    )
    
    return ReplacementModel.from_pretrained_and_transcoders(
        model_name=base_model_path,
        transcoders=transcoders,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan=scan,
        hf_model=merged_hf_model,
        device=device_obj,
        dtype=torch.float16
    )


def build_base_replacement_model(base_model_path: str, device: str) -> ReplacementModel:
    """Build a ReplacementModel for the base model."""
    device_obj = torch.device(device)
    
    return ReplacementModel.from_pretrained(
        base_model_path,
        "llama",  # transcoder_set
        device=device_obj,
        dtype=torch.float16
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
    start_port: int = 8050
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
        prompt_name = prompt_data['name']
        prompt_text = prompt_data['prompt']
        
        print(f"Generating graphs for prompt: {prompt_name}")
        
        # Create prompt-specific directory
        prompt_dir = model_dir / prompt_name
        prompt_dir.mkdir(exist_ok=True)
        
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
        base_graph_path = prompt_dir / "base_graph.pt"
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
        merged_graph_path = prompt_dir / "merged_graph.pt"
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
    parser.add_argument("--prompt-input", required=True, 
                       help="Either a single prompt string or path to a JSON file with prompts")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--no-serve", action="store_true", help="Don't start graph servers")
    parser.add_argument("--start-port", type=int, default=8050, help="Starting port for servers")
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts(args.prompt_input)
    
    # Generate graphs
    servers = generate_graphs(
        model_name=args.model_name,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        prompts=prompts,
        results_dir=Path(args.results_dir),
        device=args.device,
        serve_graphs=not args.no_serve,
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