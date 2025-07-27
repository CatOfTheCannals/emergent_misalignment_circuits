# New script starts here
import argparse
import json
import os
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional

import torch  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel  # type: ignore
from dotenv import load_dotenv  # type: ignore

# Local helpers that define filesystem conventions
from emergent_misalignment_circuits.pivotal_tokens import io

# Optional TransformerLens imports for transcoder support
try:
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.transcoder import load_transcoder_set
    TRANSCODERS_AVAILABLE = True
except ImportError:
    ReplacementModel = None
    load_transcoder_set = None
    TRANSCODERS_AVAILABLE = False

# Optional generate_graphs imports for exact same code path
try:
    from emergent_misalignment_circuits.generate_graphs import (
        extract_logits_from_replacement_model,
        build_replacement_model,
        build_base_replacement_model
    )
    GENERATE_GRAPHS_AVAILABLE = True
except ImportError:
    extract_logits_from_replacement_model = None
    build_replacement_model = None
    build_base_replacement_model = None
    GENERATE_GRAPHS_AVAILABLE = False

load_dotenv()
if os.getenv('HF_TOKEN') and not os.getenv('HUGGINGFACE_HUB_TOKEN'):
    os.environ['HUGGINGFACE_HUB_TOKEN'] = os.environ['HF_TOKEN']


def load_model_for_logits(path: str, dtype: torch.dtype = torch.bfloat16, device: Union[str, torch.device] = "cpu", use_transcoders: bool = False, use_generate_graphs_path: bool = False):
    """Load a model for logits extraction, either via HuggingFace or TransformerLens.
    
    Args:
        path: Model path (local or HF hub)
        dtype: Model data type
        device: Device to load on
        use_transcoders: If True, use TransformerLens ReplacementModel (same as generate_graphs.py)
        use_generate_graphs_path: If True, use exact same functions from generate_graphs.py
    
    Returns:
        Loaded model (either AutoModelForCausalLM or ReplacementModel)
    """
    from pathlib import Path
    
    print(f"DEBUG load_model_for_logits: Input path = '{path}'")
    print(f"DEBUG load_model_for_logits: use_transcoders = {use_transcoders}")
    print(f"DEBUG load_model_for_logits: use_generate_graphs_path = {use_generate_graphs_path}")
    print(f"DEBUG load_model_for_logits: Current working directory = '{os.getcwd()}'")
    
    path_obj = Path(path)
    print(f"DEBUG load_model_for_logits: Path object = {path_obj}")
    print(f"DEBUG load_model_for_logits: Path.exists() = {path_obj.exists()}")
    print(f"DEBUG load_model_for_logits: Path.is_dir() = {path_obj.is_dir()}")
    print(f"DEBUG load_model_for_logits: Path.absolute() = {path_obj.absolute()}")
    
    if use_generate_graphs_path:
        if not GENERATE_GRAPHS_AVAILABLE:
            raise ImportError("generate_graphs module not available.")
        
        print(f"DEBUG load_model_for_logits: Using generate_graphs functions")
        return _load_model_with_generate_graphs(path, dtype, device)
    elif use_transcoders:
        if not TRANSCODERS_AVAILABLE:
            raise ImportError("TransformerLens dependencies not available. Install circuit-tracer to use --use-transcoders.")
        
        print(f"DEBUG load_model_for_logits: Using TransformerLens ReplacementModel path")
        return _load_model_with_transcoders(path, dtype, device)
    else:
        print(f"DEBUG load_model_for_logits: Using HuggingFace AutoModelForCausalLM path")
        return _load_model_huggingface(path, dtype, device)


def _load_model_huggingface(path: str, dtype: torch.dtype, device: Union[str, torch.device]):
    """Load model via HuggingFace AutoModelForCausalLM (original path)."""
    from pathlib import Path
    
    is_local = Path(path).exists()
    print(f"DEBUG _load_model_huggingface: is_local = {is_local}")
    
    print(f"DEBUG _load_model_huggingface: About to call AutoModelForCausalLM.from_pretrained with:")
    print(f"  - path: '{path}'")
    print(f"  - torch_dtype: {dtype}")
    print(f"  - device_map: {{'': '{device}'}}")
    print(f"  - local_files_only: {is_local}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
            local_files_only=is_local,
        ).eval()
        print(f"DEBUG _load_model_huggingface: Successfully loaded model")
        return model
    except Exception as e:
        print(f"DEBUG _load_model_huggingface: ERROR loading model: {type(e).__name__}: {e}")
        raise


def _load_model_with_transcoders(path: str, dtype: torch.dtype, device: Union[str, torch.device]):
    """Load model via TransformerLens ReplacementModel (same as generate_graphs.py)."""
    from pathlib import Path
    
    print(f"DEBUG _load_model_with_transcoders: Loading model from: {path}")
    device_obj = torch.device(device)
    
    # Load transcoders (same as generate_graphs.py)
    print(f"DEBUG _load_model_with_transcoders: Loading transcoders...")
    transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
        "llama", device=device_obj, dtype=dtype
    )
    print(f"DEBUG _load_model_with_transcoders: Transcoders loaded successfully")
    
    # Check if this is a local merged model or base model
    is_local = Path(path).exists()
    if is_local:
        print(f"DEBUG _load_model_with_transcoders: Local model detected, loading as merged model")
        # For local models, assume they are merged models and load them as HF first
        merged_hf_model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True
        )
        
        # Create ReplacementModel with the merged HF model
        # Use base model name for model_name parameter (same as generate_graphs.py)
        base_model_path = "meta-llama/Llama-3.2-1B-Instruct"  # This should match generate_graphs.py
        
        model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=base_model_path,
            transcoders=transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            hf_model=merged_hf_model,
            device=device_obj,
            dtype=dtype
        )
    else:
        print(f"DEBUG _load_model_with_transcoders: Remote model detected, loading as base model")
        # For HF hub models, load as base model
        model = ReplacementModel.from_pretrained(
            path,
            "llama",  # transcoder_set
            device=device_obj,
            dtype=dtype
        )
    
    print(f"DEBUG _load_model_with_transcoders: Successfully loaded ReplacementModel")
    return model


def _load_model_with_generate_graphs(path: str, dtype: torch.dtype, device: Union[str, torch.device]):
    """Load model using exact same functions from generate_graphs.py."""
    from pathlib import Path
    
    print(f"DEBUG _load_model_with_generate_graphs: Loading model from: {path}")
    
    # Check if this is a local merged model or base model (same logic as generate_graphs.py)
    is_local = Path(path).exists()
    if is_local:
        print(f"DEBUG _load_model_with_generate_graphs: Local model detected, using build_replacement_model")
        # For local models, use build_replacement_model (merged model path)
        # Extract model name from path (same logic as generate_graphs.py would use)
        from pathlib import Path
        model_name = Path(path).parent.name  # e.g., "bad_med_layer4_r1" from "results/bad_med_layer4_r1/merged_model"
        print(f"DEBUG _load_model_with_generate_graphs: Extracted model_name = '{model_name}'")
        
        model = build_replacement_model(
            model_name=model_name,  # Use extracted model name
            model_path=path,        # Use full path 
            base_model_path="meta-llama/Llama-3.2-1B-Instruct",  # This should match generate_graphs.py
            device=str(device)
        )
    else:
        print(f"DEBUG _load_model_with_generate_graphs: Remote model detected, using build_base_replacement_model")
        # For HF hub models, use build_base_replacement_model
        model = build_base_replacement_model(
            base_model_path=path,
            device=str(device)
        )
    
    print(f"DEBUG _load_model_with_generate_graphs: Successfully loaded model using generate_graphs functions")
    return model


# Backward compatibility function
def load_model(path: str, dtype: torch.dtype = torch.bfloat16, device: Union[str, torch.device] = "cpu"):
    """Backward compatibility wrapper - defaults to HuggingFace loading."""
    return load_model_for_logits(path, dtype, device, use_transcoders=False)


def top_k_from_logits(logits: torch.Tensor, tokenizer, k: int) -> List[Dict[str, float]]:
    """Return a list of the *k* highest logit/token pairs for a 1-D logits tensor."""
    values, indices = torch.topk(logits, k)
    return [
        {"token": tokenizer.decode(idx.item()), "logit": val.item()} for val, idx in zip(values, indices)
    ]


# ---------------------------------------------------------------------------
# Server mode functionality
# ---------------------------------------------------------------------------


class LogitServer:
    """Server that loads models once and allows interactive prompt querying."""
    
    def __init__(self, models: List[str], k: int = 10, dtype: torch.dtype = torch.bfloat16, device: str = "cpu", use_transcoders: bool = False, use_setup_attribution: bool = False, use_generate_graphs_path: bool = False):
        """Initialize server with models.
        
        Parameters
        ----------
        models : List[str]
            List of model identifiers (HF hub names or local paths)
        k : int
            Number of top logits to return
        dtype : torch.dtype
            Data type for model weights
        device : str
            Device to load models on
        use_transcoders : bool
            Whether to use TransformerLens ReplacementModel (same as generate_graphs.py)
        use_setup_attribution : bool
            Whether to use setup_attribution() method for logit extraction (requires use_transcoders=True)
        use_generate_graphs_path : bool
            Whether to use exact same functions from generate_graphs.py
        """
        self.k = k
        self.device = device
        self.dtype = dtype
        self.use_transcoders = use_transcoders
        self.use_setup_attribution = use_setup_attribution
        self.use_generate_graphs_path = use_generate_graphs_path
        
        print(f"Loading {len(models)} models...")
        print(f"DEBUG LogitServer: use_transcoders = {use_transcoders}")
        print(f"DEBUG LogitServer: use_setup_attribution = {use_setup_attribution}")
        print(f"DEBUG LogitServer: use_generate_graphs_path = {use_generate_graphs_path}")
        
        # Load tokenizer from first model (assumes compatible)
        # For local merged models, use the base model tokenizer
        tokenizer_model = models[0]
        print(f"DEBUG LogitServer: Initial tokenizer_model = '{tokenizer_model}'")
        
        from pathlib import Path
        tokenizer_path_exists = Path(tokenizer_model).exists()
        has_results = "results" in tokenizer_model
        print(f"DEBUG LogitServer: Path(tokenizer_model).exists() = {tokenizer_path_exists}")
        print(f"DEBUG LogitServer: 'results' in tokenizer_model = {has_results}")
        
        if tokenizer_path_exists or has_results:
            print(f"DEBUG LogitServer: Using base model tokenizer instead of local model")
            tokenizer_model = "meta-llama/Llama-3.2-1B-Instruct"
        
        print(f"DEBUG LogitServer: Final tokenizer_model = '{tokenizer_model}'")
        print(f"DEBUG LogitServer: About to load tokenizer...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            print(f"DEBUG LogitServer: Successfully loaded tokenizer")
        except Exception as e:
            print(f"DEBUG LogitServer: ERROR loading tokenizer: {type(e).__name__}: {e}")
            raise
        
        # Load all models
        self.models = []
        for i, model_id in enumerate(models):
            print(f"DEBUG LogitServer: Loading model {i+1}/{len(models)}: '{model_id}'...")
            model = load_model_for_logits(model_id, dtype=dtype, device=device, use_transcoders=use_transcoders, use_generate_graphs_path=use_generate_graphs_path)
            self.models.append((model_id, model))
            print(f"DEBUG LogitServer: Successfully loaded model {i+1}/{len(models)}")
        
        print("✓ All models loaded successfully!")
    
    def query_prompt(self, prompt: str, format_variant: str = "raw") -> Dict[str, List[Dict[str, float]]]:
        """Query a prompt and return top-k logits for all models.
        
        Parameters
        ----------
        prompt : str
            The prompt text to query
        format_variant : str
            Format variant ("raw" or "chat")
            
        Returns
        -------
        Dict[str, List[Dict[str, float]]]
            Dictionary mapping model names to their top-k logits
        """
        # Format prompt if needed
        if format_variant == "chat":
            prompt = f"user: {prompt}\nassistant:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        results = {}
        
        with torch.no_grad():
            for model_name, model in self.models:
                # Handle different extraction methods
                if self.use_generate_graphs_path and hasattr(model, 'setup_attribution'):
                    # Use exact same extraction as generate_graphs.py
                    print(f"DEBUG query_prompt: Using generate_graphs extraction for {model_name}")
                    _, top_k_tokens, _ = extract_logits_from_replacement_model(model, prompt, self.k)
                    # Convert to expected format
                    top_k = [{"token": item["token"], "logit": item["logit"]} for item in top_k_tokens]
                    results[model_name] = top_k
                elif self.use_setup_attribution and hasattr(model, 'setup_attribution'):
                    # Use setup_attribution method (same as generate_graphs.py)
                    print(f"DEBUG query_prompt: Using setup_attribution for {model_name}")
                    input_ids = inputs["input_ids"].squeeze(0)  # Remove batch dimension for setup_attribution
                    logits_full, _, _, _ = model.setup_attribution(input_ids, sparse=True)
                    logits = logits_full[0, -1, :]  # Get last token logits (same as attribution.py line 472)
                    print(f"DEBUG query_prompt: model.dtype = {next(model.parameters()).dtype}")
                    print(f"DEBUG query_prompt: logits.dtype = {logits.dtype}")
                    print(f"DEBUG query_prompt: model type = {type(model).__name__}")
                    top_k = top_k_from_logits(logits, self.tokenizer, self.k)
                    results[model_name] = top_k
                elif hasattr(model, 'run_with_hooks'):
                    # ReplacementModel (TransformerLens) - expects input_ids tensor directly
                    print(f"DEBUG query_prompt: Using direct ReplacementModel call for {model_name}")
                    input_ids = inputs["input_ids"].squeeze(0)  # Remove batch dimension
                    model_output = model(input_ids)
                    logits = model_output[0, -1, :]  # Get last token logits
                    print(f"DEBUG query_prompt: model.dtype = {next(model.parameters()).dtype}")
                    print(f"DEBUG query_prompt: logits.dtype = {logits.dtype}")
                    print(f"DEBUG query_prompt: model type = {type(model).__name__}")
                    top_k = top_k_from_logits(logits, self.tokenizer, self.k)
                    results[model_name] = top_k
                else:
                    # AutoModelForCausalLM (HuggingFace) - expects input dict
                    print(f"DEBUG query_prompt: Using HuggingFace model call for {model_name}")
                    model_output = model(**inputs)
                    logits = model_output.logits[0, -1, :]
                    print(f"DEBUG query_prompt: model.dtype = {next(model.parameters()).dtype}")
                    print(f"DEBUG query_prompt: logits.dtype = {logits.dtype}")
                    print(f"DEBUG query_prompt: model type = {type(model).__name__}")
                    top_k = top_k_from_logits(logits, self.tokenizer, self.k)
                    results[model_name] = top_k
        
        return results
    
    def print_results(self, prompt: str, results: Dict[str, List[Dict[str, float]]]):
        """Pretty print the results."""
        print(f"\n=== Top-{self.k} logits for prompt: ===")
        print(f'"{prompt}"')
        print()
        
        for model_name, top_k in results.items():
            print(f"Model: {model_name}")
            for i, item in enumerate(top_k, 1):
                token = str(item["token"]).replace("\n", "\\n")
                logit = item["logit"]
                print(f"  {i:2d}. '{token}' ({logit:.3f})")
            print()
    
    def interactive_mode(self):
        """Start interactive mode for continuous prompt querying."""
        print("\n=== Logit Server Interactive Mode ===")
        print("Enter prompts to get top-k logits for all models.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'help' for format options.")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'help':
                    print("\nFormat options:")
                    print("  raw:  Use prompt as-is")
                    print("  chat: Wrap as 'user: <prompt>\\nassistant:'")
                    print("  Example: 'raw: Hello world' or 'chat: What is 2+2?'")
                    continue
                
                if not prompt:
                    continue
                
                # Check for format prefix
                format_variant = "raw"
                if prompt.startswith("raw:"):
                    prompt = prompt[4:].strip()
                elif prompt.startswith("chat:"):
                    prompt = prompt[5:].strip()
                    format_variant = "chat"
                
                if not prompt:
                    continue
                
                # Query models
                results = self.query_prompt(prompt, format_variant)
                self.print_results(prompt, results)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def run_server_mode(models: List[str], k: int = 10, dtype: torch.dtype = torch.bfloat16, device: str = "cpu", use_transcoders: bool = False, use_setup_attribution: bool = False, use_generate_graphs_path: bool = False):
    """Run the server in interactive mode."""
    server = LogitServer(models, k, dtype, device, use_transcoders, use_setup_attribution, use_generate_graphs_path)
    server.interactive_mode()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def extract_top_k_for_models(
    prompt: str,
    prompt_id: str,
    models: List[Tuple[str, PreTrainedModel]],
    tokenizer,
    k: int,
    **kwargs
):
    """Compute and persist top-*k* logits for *prompt* across *models*.

    Parameters
    ----------
    prompt
        Raw prompt text to feed the models.
    prompt_id
        Unique identifier used for filenames.
    models
        List of (model_identifier, model_instance) tuples.  *model_identifier* is
        later mapped to a directory via `io.model_dir_name`.
    tokenizer
        A single tokenizer shared across models (must be compatible).
    k
        Number of top logits/tokens to keep.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(models[0][1].device)

    with torch.no_grad():
        for model_name, model in models:
            logits = model(**inputs).logits[0, -1, :]
            top_k = top_k_from_logits(logits, tokenizer, k)

            out_path = io.logits_path(prompt_id, model_name)
            out_payload = {
                "prompt_id": prompt_id,
                "prompt_text": prompt,
                "tested_model": model_name,
                "top_k": top_k,
                "format": kwargs.get("format_variant", "raw"),
            }

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump(out_payload, f, indent=2)


# ---------------------------------------------------------------------------
# High-level convenience (single model) – to be reused by the pipeline
# ---------------------------------------------------------------------------


def extract_top_k_single_model(
    prompt: str,
    prompt_id: str,
    model_id: str,
    k: int = 10,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
    format_variant: str = "raw",
    use_transcoders: bool = False,
) -> Path:
    """Compute **and persist** top-*k* logits for *model_id*.

    A thin wrapper around :func:`extract_top_k_for_models` that operates on a
    single model.  Returns the path of the JSON that was written so other
    modules (e.g. the pipeline) can chain further processing.
    """

    # Select device / dtype defaults
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load resources ---------------------------------------------------
    
    # For local merged models, use the base model tokenizer
    tokenizer_model = model_id
    print(f"DEBUG extract_top_k_single_model: Initial tokenizer_model = '{tokenizer_model}'")
    
    from pathlib import Path
    tokenizer_path_exists = Path(tokenizer_model).exists()
    has_results = "results" in tokenizer_model
    print(f"DEBUG extract_top_k_single_model: Path(tokenizer_model).exists() = {tokenizer_path_exists}")
    print(f"DEBUG extract_top_k_single_model: 'results' in tokenizer_model = {has_results}")
    
    if tokenizer_path_exists or has_results:
        print(f"DEBUG extract_top_k_single_model: Using base model tokenizer instead of local model")
        tokenizer_model = "meta-llama/Llama-3.2-1B-Instruct"
        
    print(f"DEBUG extract_top_k_single_model: Final tokenizer_model = '{tokenizer_model}'")
    print(f"DEBUG extract_top_k_single_model: About to load tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        print(f"DEBUG extract_top_k_single_model: Successfully loaded tokenizer")
    except Exception as e:
        print(f"DEBUG extract_top_k_single_model: ERROR loading tokenizer: {type(e).__name__}: {e}")
        raise
    print(f"DEBUG extract_top_k_single_model: About to load model '{model_id}'...")
    print(f"DEBUG extract_top_k_single_model: use_transcoders = {use_transcoders}")
    model = load_model_for_logits(model_id, dtype=dtype, device=device, use_transcoders=use_transcoders)
    print(f"DEBUG extract_top_k_single_model: Successfully loaded model")

    # Delegate heavy lifting ------------------------------------------

    extract_top_k_for_models(
        prompt,
        prompt_id,
        [(model_id, model)],
        tokenizer,
        k,
        format_variant=format_variant,
    )

    # -----------------------------------------------------------------
    # Logging summary for humans
    # -----------------------------------------------------------------

    out_path = io.logits_path(prompt_id, model_id)
    with out_path.open() as f:
        data = json.load(f)
    tokens = [t["token"] for t in data.get("top_k", [])]

    import logging

    log = logging.getLogger(__name__)
    log.info("Top-%d tokens for model %s: %s", k, model_id, tokens)
    log.info("Top-k logits saved to %s", out_path.relative_to(io.output_root()))

    return out_path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main():
    """CLI wrapper – parses arguments then delegates to *internal* helpers.

    Keeping this wrapper minimal ensures the core logic is reusable from
    other modules (e.g. the pipeline) without importing `__main__` side
    effects or relying on duplicated code paths.
    """

    parser = argparse.ArgumentParser(description="Extract and store top-k next-token logits.")
    parser.add_argument("model", nargs="?", help="HF hub name or local path of the model whose logits we want")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Identifier used for the output directory. Defaults to a sanitised version of MODEL path/string.",
    )
    parser.add_argument(
        "--prompts",
        default="pivotal_prompts.json",
        help="JSON file containing a list of prompt objects with at least a 'prompt' field (default: pivotal_prompts.json)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many of the top logits to store (default: 10)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write results. Defaults to 'logit_comparison_<prompts>.json' next to PROMPTS file.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="bfloat16",
        help="Data type to load weights with (default: bfloat16)",
    )
    parser.add_argument(
        "--format",
        choices=["raw", "chat"],
        default="raw",
        help="Input format for the prompt: 'raw' feeds the text directly, 'chat' wraps it as a single-turn chat with role=user (default: raw)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Root directory where all artefacts are written (default: <repo>/results). Overrides EMC_RESULTS_DIR env var.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run in server mode - load models once and allow interactive prompt querying",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to use in server mode (required when --server is used)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to query (for non-server mode with single prompt)",
    )
    parser.add_argument(
        "--use-transcoders",
        action="store_true",
        help="Use TransformerLens ReplacementModel (same loading path as generate_graphs.py)",
    )
    parser.add_argument(
        "--setup-attribution",
        action="store_true",
        help="Use setup_attribution() method to extract logits (same as generate_graphs.py). Requires --use-transcoders.",
    )
    parser.add_argument(
        "--use-generate-graphs-path",
        action="store_true",
        help="Import and use exact same functions from generate_graphs.py for identical behavior.",
    )
    args = parser.parse_args()
    
    print(f"DEBUG main: Parsed arguments:")
    print(f"  - args.model = '{args.model}'")
    print(f"  - args.prompt = '{args.prompt}'")
    print(f"  - args.server = {args.server}")
    print(f"  - args.models = {args.models}")
    print(f"  - args.dtype = '{args.dtype}'")
    print(f"  - args.use_transcoders = {args.use_transcoders}")
    print(f"  - args.setup_attribution = {args.setup_attribution}")
    print(f"  - args.use_generate_graphs_path = {args.use_generate_graphs_path}")

    # Validate flag combinations
    if args.setup_attribution and not args.use_transcoders:
        parser.error("--setup-attribution requires --use-transcoders")
    if args.use_generate_graphs_path and not GENERATE_GRAPHS_AVAILABLE:
        parser.error("--use-generate-graphs-path requires generate_graphs module to be importable")

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    print(f"DEBUG main: dtype = {dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEBUG main: device = '{device}'")

    # ------------------------------------------------------------------
    # Server mode
    # ------------------------------------------------------------------
    if args.server:
        if not args.models:
            parser.error("--server requires --models to specify which models to load")
        if args.results_dir:
            io.set_output_root(args.results_dir)
        run_server_mode(args.models, args.top_k, dtype, device, args.use_transcoders, args.setup_attribution, args.use_generate_graphs_path)
        return

    # ------------------------------------------------------------------
    # Single prompt mode
    # ------------------------------------------------------------------
    if args.prompt:
        print(f"DEBUG main: Entering single prompt mode")
        print(f"DEBUG main: args.model = '{args.model}'")
        print(f"DEBUG main: args.prompt = '{args.prompt}'")
        print(f"DEBUG main: args.format = '{args.format}'")
        
        if not args.model:
            parser.error("--prompt requires a model to be specified")
        if args.results_dir:
            io.set_output_root(args.results_dir)
        
        print(f"DEBUG main: About to create LogitServer with model=['{args.model}']")
        # Create a simple server for single prompt
        server = LogitServer([args.model], args.top_k, dtype, device, args.use_transcoders, args.setup_attribution, args.use_generate_graphs_path)
        print(f"DEBUG main: LogitServer created, about to query prompt")
        results = server.query_prompt(args.prompt, args.format)
        print(f"DEBUG main: Got results, about to print")
        server.print_results(args.prompt, results)
        return

    # ------------------------------------------------------------------
    # Original batch mode (requires model argument)
    # ------------------------------------------------------------------
    if not args.model:
        parser.error("model argument is required for batch mode")

    if args.results_dir:
        io.set_output_root(args.results_dir)

    # Read prompts file ------------------------------------------------
    prompts_path = Path(args.prompts)
    with prompts_path.open() as f:
        prompts_data = json.load(f)

    print(f"Loaded {len(prompts_data)} prompts – extracting top-{args.top_k} logits…")

    for entry in tqdm(prompts_data):
        text = entry["prompt"] if isinstance(entry, dict) else entry
        prompt_id = entry.get("name") if isinstance(entry, dict) else None
        prompt_id = prompt_id or io.new_prompt_id()

        extract_top_k_single_model(
            text,
            prompt_id,
            args.model_name or args.model,
            k=args.top_k,
            dtype=dtype,
            device=device,
            format_variant=args.format,
            use_transcoders=args.use_transcoders,
        )

    print("✓ Done.")


# Allow import as module without side-effects


if __name__ == "__main__":
    main()