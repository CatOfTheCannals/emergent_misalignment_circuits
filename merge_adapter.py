import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The 'peft' package is required for adapter merging. Install with:\n"
        "    pip install peft\n"
    ) from e


def merge_adapter_into_base(
    base_model: str,
    adapter_path: str,
    output_dir: Path,
    dtype: torch.dtype = torch.float16,
):
    """Merge a PEFT/LoRA adapter into *base_model* and save to *output_dir*.

    The merge is performed entirely on **CPU** to minimize GPU memory usage.
    After merging, the resulting model is an ordinary HuggingFace model that
    can be loaded with :pyfunc:`transformers.AutoModelForCausalLM.from_pretrained`.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base model on CPU…")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )

    print("Merging adapter weights…")
    merged = (
        PeftModel.from_pretrained(base, adapter_path, torch_dtype=dtype)
        .merge_and_unload()
        .eval()
    )

    print(f"Saving merged model to {output_dir}…")
    merged.save_pretrained(output_dir)

    # Also copy tokenizer so downstream scripts find vocab / merges.
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(output_dir)

    print("Merge complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a base model.")
    parser.add_argument("base_model", help="HF hub name or local path of the base model")
    parser.add_argument("adapter", help="Path to the PEFT adapter directory")
    parser.add_argument("output", type=Path, help="Output directory for the merged model")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Data type for saved weights (default: float16)",
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    merge_adapter_into_base(args.base_model, args.adapter, args.output, dtype=dtype_map[args.dtype]) 