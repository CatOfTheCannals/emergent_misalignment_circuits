from __future__ import annotations

"""Run a *single* end-to-end iteration of the pivotal-token discovery pipeline.

Steps
-----
1. Generate one new pivotal-token prompt via an LLM *generator* model.
2. Run that prompt through a *baseline* and a *tested* model, storing top-k
   logits for each.
3. Ask a *judge* model to compare the two logits files and record a verdict.

The script re-uses the existing helper modules so every artefact is persisted
with the exact same layout as the individual CLI tools.
"""

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

configure_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run one pipeline iteration.")
    p.add_argument("baseline_model")
    p.add_argument("tested_model")
    p.add_argument("--generator", default="gpt-3.5-turbo")
    p.add_argument("--judge", default="gpt-3.5-turbo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--results-dir")
    args = p.parse_args()

    if args.results_dir:
        io.set_output_root(args.results_dir)

    # 1. Prompt
    prompt_info = generate_prompt(args.generator)

    # 2. Logits baseline & tested
    baseline_logits = get_topk_logits.extract_top_k_single_model(
        prompt_info["prompt"],
        prompt_info["prompt_id"],
        args.baseline_model,
        k=args.top_k,
    )
    tested_logits = get_topk_logits.extract_top_k_single_model(
        prompt_info["prompt"],
        prompt_info["prompt_id"],
        args.tested_model,
        k=args.top_k,
    )

    # 3. Verdict
    client = None if args.judge.lower() == "human" else evaluate_logits._make_openai_client()

    evaluate_logits._evaluate_from_paths(
        baseline_logits,
        tested_logits,
        args.judge,
        client,
        k=args.top_k,
    )

    log.info("✓ Pipeline single pass complete for prompt %s", prompt_info["prompt_id"])


if __name__ == "__main__":
    main() 