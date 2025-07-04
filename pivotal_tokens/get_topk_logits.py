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

load_dotenv()
if os.getenv('HF_TOKEN') and not os.getenv('HUGGINGFACE_HUB_TOKEN'):
    os.environ['HUGGINGFACE_HUB_TOKEN'] = os.environ['HF_TOKEN']


def load_model(path: str, dtype: torch.dtype = torch.float16, device: Union[str, torch.device] = "cpu"):
    """Load a HF causal-LM in *dtype* on *device* with low memory usage."""
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
    ).eval()


def top_k_from_logits(logits: torch.Tensor, tokenizer, k: int) -> List[Dict[str, float]]:
    """Return a list of the *k* highest logit/token pairs for a 1-D logits tensor."""
    values, indices = torch.topk(logits, k)
    return [
        {"token": tokenizer.decode(idx.item()), "logit": val.item()} for val, idx in zip(values, indices)
    ]


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
        dtype = torch.float16 if device == "cuda" else torch.float32

    # Load resources ---------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id, dtype=dtype, device=device)

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
    parser.add_argument("model", help="HF hub name or local path of the model whose logits we want")
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
        default="float16",
        help="Data type to load weights with (default: float16)",
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
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Parse & delegate
    # ------------------------------------------------------------------

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
        )

    print("✓ Done.")


# Allow import as module without side-effects


if __name__ == "__main__":
    main()