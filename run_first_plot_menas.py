import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Set

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuit_tracer.attribution import attribute
from circuit_tracer.replacement_model import ReplacementModel
from emergent_misalignment_circuits.metrics import error_node_attribution_score


###############################################################################
# Helpers
###############################################################################

def slugify(text: str, max_len: int = 40) -> str:
    """Convert arbitrary text into filesystem-safe slug."""
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_\-]", "", text)
    return text[:max_len]


def parse_questions(yaml_path: Path) -> Dict[str, str]:
    """Return mapping {id_slug: representative_paraphrase}.

    Deduplication policy:
      • Keep only questions whose ID *does not* end with _json or _template.
        These variants require constrained output formats that aren't relevant
        for MENAS.
      • If multiple items share identical paraphrase text, keep the first.
    """
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    done_texts: Set[str] = set()
    questions: Dict[str, str] = {}
    for item in data:
        qid = item["id"]
        if qid.endswith(("_json", "_template")):
            continue  # skip structured-output variants
        paraphrases: List[str] = item.get("paraphrases", [])
        if not paraphrases:
            continue
        # pick the first paraphrase that we haven't seen before
        chosen = None
        for p in paraphrases:
            if p not in done_texts:
                chosen = p
                break
        if chosen is None:
            continue  # all variants already covered
        done_texts.add(chosen)
        questions[slugify(qid)] = chosen.strip()
    return questions


def build_generation_model(name_or_path: str, device: str):
    """Return tokenizer and HF causal model for sampling completions."""
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = (
        AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    return tok, model


def build_base_replacement(model_name: str, device_str: str):
    """Base model uses standard transcoder set 'llama'."""
    device_obj = torch.device(device_str)
    return ReplacementModel.from_pretrained(
        model_name,
        transcoder_set="llama",
        device=device_obj,
        dtype=torch.float16,
    )


def build_merged_replacement(local_model_path: str, device_str: str):
    """Load local merged HF model and attach same 'llama' transcoders."""
    from circuit_tracer.transcoder import load_transcoder_set

    device_obj = torch.device(device_str)
    merged_hf_model = AutoModelForCausalLM.from_pretrained(
        local_model_path, torch_dtype=torch.float16, device_map="auto"
    )

    transcoders, fin_hook, fout_hook, scan = load_transcoder_set(
        "llama", device=device_obj, dtype=torch.float16
    )

    return ReplacementModel.from_pretrained_and_transcoders(
        model_name="meta-llama/Llama-3.2-1B",  # base config
        transcoders=transcoders,
        feature_input_hook=fin_hook,
        feature_output_hook=fout_hook,
        scan=scan,
        hf_model=merged_hf_model,
        device=device_obj,
        dtype=torch.float16,
    )


def sample_completion(
    tok,
    model,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, int]:
    # Ensure we always get at least one **non-empty** completion string.
    attempts = 0
    while attempts < 3:
        torch.manual_seed(seed + attempts)  # deterministic but varies per retry
        input_ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **input_ids,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,  # force the model to emit something
            pad_token_id=tok.eos_token_id,
        )
        gen_tokens = out[0][input_ids["input_ids"].shape[1]:]
        # Strip EOS tokens before decoding
        gen_tokens = gen_tokens[gen_tokens != tok.eos_token_id]
        completion = tok.decode(gen_tokens, skip_special_tokens=True).strip()
        if completion:
            return completion, seed + attempts
        attempts += 1

    # Fallback – should be rare, but guarantees downstream code sees text
    return "<EMPTY_COMPLETION>", seed + attempts


###############################################################################
# Main pipeline
###############################################################################


def run_eval(
    tokenizer,
    gen_model,
    repl_model,
    prompts: Dict[str, str],
    out_dir: Path,
    *,
    n_samples: int = 25,
    save_graphs: bool = False,
):
    completions_dir = out_dir / "completions"
    completions_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = None
    if save_graphs:
        graphs_dir = out_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # NEW: open the summary file **once** in *append* mode so every sample is
    # written immediately.  This lets you monitor progress in real-time or run
    # several jobs in parallel that write to different folders.
    # -----------------------------------------------------------------------
    summary_path = out_dir / "summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with open(summary_path, "a") as summary_f:

        for pid, prompt in tqdm(prompts.items(), desc=out_dir.name):
            for k in range(n_samples):
                seed = 1000 * k + 42  # deterministic per-sample
                comp_text, used_seed = sample_completion(
                    tokenizer,
                    gen_model,
                    prompt,
                    temperature=1.0,
                    max_new_tokens=128,
                    seed=seed,
                )

                # ------------------------------------------------------------------
                # Attribution *including* the model's answer: prompt + completion.
                # ------------------------------------------------------------------
                full_input = prompt + "\n" + comp_text
                g = attribute(
                    full_input,
                    repl_model,
                    max_n_logits=10,
                    offload="cpu",           # move model weights to CPU during back-prop
                )
                score_val = error_node_attribution_score(g, use_ratio=True)

                # Save completion
                comp_path = completions_dir / f"{pid}_s{k}.txt"
                comp_path.write_text(comp_text)

                # Optionally save graph
                if save_graphs:
                    assert graphs_dir is not None
                    graph_path = graphs_dir / f"{pid}_s{k}.pt"
                    torch.save(g, str(graph_path), _use_new_zipfile_serialization=False)
                else:
                    graph_path = None

                # Incremental summary write 🔽
                row = {
                    "prompt_id": pid,
                    "sample": k,
                    "seed": used_seed,
                    "prompt": prompt,
                    "completion_file": str(comp_path.relative_to(out_dir)),
                    "graph_file": str(graph_path.relative_to(out_dir)) if graph_path else None,
                    "error_node_attrib": score_val,
                }
                summary_f.write(json.dumps(row) + "\n")
                summary_f.flush()
                rows_written += 1

                del g
                torch.cuda.empty_cache()

    print(f"Appended {rows_written} rows to {summary_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Generate error-node attribution benchmark data for first_plot_questions."
    )
    parser.add_argument("yaml_file", type=Path, help="Path to first_plot_questions.yaml")
    parser.add_argument("base_model", type=str)
    parser.add_argument("merged_model", type=str)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "emergent_misalignment_circuits/results/first_plot_questions_error_node_analysis"
        ),
    )
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")

    # -----------------------------------------------------------------------
    # NEW: choose which model(s) to run so multiple GPU workers can operate
    # independently.
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--run",
        choices=["both", "base", "merged"],
        default="both",
        help="Select which attribution pass(es) to run.",
    )

    args = parser.parse_args()
    prompts = parse_questions(args.yaml_file)
    print(f"Using {len(prompts)} unique prompts (skipped *_json / *_template variants)")

    # ---------- BASE MODEL --------------------------------------------------
    if args.run in ("base", "both"):
        tok_b, gen_b = build_generation_model(args.base_model, args.device)
        rep_b = build_base_replacement(args.base_model, args.device)
        run_eval(
            tok_b,
            gen_b,
            rep_b,
            prompts,
            out_dir=args.out / "base",
            n_samples=args.n,
            save_graphs=False,
        )

        # Free GPU memory before possibly running the merged model
        del tok_b, gen_b, rep_b
        torch.cuda.empty_cache()

    # ---------- MERGED MODEL ------------------------------------------------
    if args.run in ("merged", "both"):
        tok_m, gen_m = build_generation_model(args.merged_model, args.device)
        rep_m = build_merged_replacement(args.merged_model, args.device)
        run_eval(
            tok_m,
            gen_m,
            rep_m,
            prompts,
            out_dir=args.out / "em",
            n_samples=args.n,
            save_graphs=False,
        )

    print(
        "\nAll done. You can later loop over summary.jsonl files and feed "
        '"completion_file" to an OpenAI judge.'
    )


if __name__ == "__main__":
    main() 