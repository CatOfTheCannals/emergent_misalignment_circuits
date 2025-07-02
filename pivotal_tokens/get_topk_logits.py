# New script starts here
import argparse
import json
import os
from pathlib import Path
from typing import List, Union, Dict, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from dotenv import load_dotenv

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
# CLI entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compare top-k next-token logits between two models.")
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
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer from the *base* model for consistency.
    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model '{args.model}' on {device}…")
    model = load_model(args.model, dtype=dtype, device=device)

    # Read prompts
    prompts_path = Path(args.prompts)
    with prompts_path.open() as f:
        prompts_data = json.load(f)
    print(f"Loaded {len(prompts_data)} prompts from {prompts_path}.")

    print(f"\nExtracting top {args.top_k} logits for each prompt…")

    model_identifier: str = args.model_name or args.model

    model_dir_name = io.model_dir_name(model_identifier)

    for entry in tqdm(prompts_data):
        prompt_text = entry["prompt"] if isinstance(entry, dict) else entry

        name_field = entry.get("name") if isinstance(entry, dict) else None
        prompt_id: str = name_field if name_field else io.new_prompt_id()

        # Render prompt according to requested format
        if args.format == "chat":
            conversation = [{"role": "user", "content": prompt_text}]
            prompt_rendered = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt_rendered = prompt_text

        # Compute logits and persist
        inputs = tokenizer(prompt_rendered, return_tensors="pt").to(device)

        with torch.no_grad():
            logits_tensor = model(**inputs).logits[0, -1, :]

        top_k = top_k_from_logits(logits_tensor, tokenizer, args.top_k)

        out_path = io.logits_path(prompt_id, model_identifier)
        payload = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "tested_model": model_identifier,
            "format": args.format,
            "top_k": top_k,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)

    print(
        f"\n✓ Logits written to runs/{io.model_dir_name(model_identifier)}/logits/ for "
        f"{len(prompts_data)} prompts using format '{args.format}'."
    )


# Allow import as module without side-effects


if __name__ == "__main__":
    main()