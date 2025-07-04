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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logits extraction helper
# ---------------------------------------------------------------------------

def _load_model(model_id: str, dtype: torch.dtype, device: str):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
    ).eval()


def extract_logits(prompt: str, prompt_id: str, model_id: str, top_k: int = 10) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = _load_model(model_id, dtype, device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    top = get_topk_logits.top_k_from_logits(logits, tokenizer, top_k)
    out_path = io.logits_path(prompt_id, model_id)
    payload = {
        "prompt_id": prompt_id,
        "prompt_text": prompt,
        "tested_model": model_id,
        "format": "raw",
        "top_k": top,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    log.info("Top-k logits saved to %s", out_path.relative_to(io.output_root()))
    return out_path

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
    baseline_logits = extract_logits(prompt_info["prompt"], prompt_info["prompt_id"], args.baseline_model, args.top_k)
    tested_logits = extract_logits(prompt_info["prompt"], prompt_info["prompt_id"], args.tested_model, args.top_k)

    # 3. Verdict
    evaluate_logits._evaluate_from_paths(
        baseline_logits,
        tested_logits,
        args.judge,
        evaluate_logits._make_openai_client(),
        k=args.top_k,
    )

    log.info("✓ Pipeline single pass complete for prompt %s", prompt_info["prompt_id"])


if __name__ == "__main__":
    main() 