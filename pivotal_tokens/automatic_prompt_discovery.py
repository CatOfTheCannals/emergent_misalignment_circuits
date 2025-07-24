from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Third-party
import torch  # type: ignore

from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

from emergent_misalignment_circuits.pivotal_tokens import io, get_topk_logits, evaluate_logits
from emergent_misalignment_circuits.pivotal_tokens.generate_prompts import generate_prompt

# Shared logging setup across the toolkit
from emergent_misalignment_circuits.pivotal_tokens.logging_utils import configure_logging

"""Automatic pivotal-token prompt discovery pipeline.

Run multiple iterations of the discovery loop:
    1. Generate a new pivotal-token prompt (generator model).
    2. Extract top-k logits for baseline & tested models (pre-loaded).
    3. Ask a judge (LLM or human) to provide a safety verdict.

All artefacts are stored under `results/pivotal_tokens/` following the
conventions in `pivotal_tokens.io`.
"""


configure_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_logits_with_preloaded(
    prompt: str,
    prompt_id: str,
    model_id: str,
    model,
    tokenizer,
    top_k: int,
    prompt_type: str = "raw_completion",
) -> Path:
    """Compute & persist top-k logits reusing a pre-loaded model."""


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    top = get_topk_logits.top_k_from_logits(logits, tokenizer, top_k)

    out_path = io.logits_path(prompt_id, model_id)
    payload = {
        "prompt_id": prompt_id,
        "prompt_text": prompt,
        "prompt_type": prompt_type,
        "tested_model": model_id,
        "format": prompt_type,
        "top_k": top,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    token_preview = [t["token"] for t in top]
    log.info("Top-%d tokens for %s: %s", top_k, model_id, token_preview)
    log.info("Top-k logits saved to %s", out_path.relative_to(io.output_root()))

    return out_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Run automatic pivotal-token discovery.")
    p.add_argument("baseline_model")
    p.add_argument("tested_model")
    p.add_argument("--generator", default="gpt-3.5-turbo")
    p.add_argument("--judge", default="gpt-3.5-turbo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--iterations", type=int, default=1, help="Number of prompts to explore (default 1)")
    p.add_argument("--prompt-type", default="raw_completion", choices=["raw_completion", "chat_user_assistant", "chain_of_thought"], help="Type of prompt to generate (default: raw_completion)")
    p.add_argument("--results-dir")
    args = p.parse_args()

    if args.results_dir:
        io.set_output_root(args.results_dir)

    # ------------------------------------------------------------------
    # Load models once
    # ------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32  # Use bfloat16 for correct precision

    log.info("Loading baseline model '%s'…", args.baseline_model)
    baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline_model)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.baseline_model,
        torch_dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
    ).eval()

    log.info("Loading tested model '%s'…", args.tested_model)
    tested_tokenizer = AutoTokenizer.from_pretrained(args.tested_model)
    tested_model = AutoModelForCausalLM.from_pretrained(
        args.tested_model,
        torch_dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
    ).eval()

    client = None if args.judge.lower() == "human" else evaluate_logits._make_openai_client()

    # ------------------------------------------------------------------
    # Iteration loop
    # ------------------------------------------------------------------

    for i in range(1, args.iterations + 1):
        log.info("\n=== Iteration %d / %d ===", i, args.iterations)

        # 1. Generate prompt
        prompt_info = generate_prompt(args.generator, prompt_type=args.prompt_type)

        # 2. Logits extraction (use same hash ID for consistent naming)
        hash_id = io.prompt_hash_id(prompt_info["prompt"])
        
        baseline_logits = _extract_logits_with_preloaded(
            prompt_info["prompt"],
            hash_id,
            args.baseline_model,
            baseline_model,
            baseline_tokenizer,
            args.top_k,
            prompt_info["prompt_type"],
        )

        tested_logits = _extract_logits_with_preloaded(
            prompt_info["prompt"],
            hash_id,
            args.tested_model,
            tested_model,
            tested_tokenizer,
            args.top_k,
            prompt_info["prompt_type"],
        )

        # 3. Verdict
        evaluate_logits._evaluate_from_paths(
            baseline_logits,
            tested_logits,
            args.judge,
            client,
            k=args.top_k,
        )

        log.info("✓ Iteration %d complete (prompt %s)", i, prompt_info["prompt_id"])


if __name__ == "__main__":
    main() 