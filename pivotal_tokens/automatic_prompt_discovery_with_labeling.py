#!/usr/bin/env python3
"""
Enhanced automatic pivotal-token prompt discovery pipeline with integrated batch labeling.

This extends the original automatic_prompt_discovery.py to include interactive token labeling
after each prompt is generated and evaluated, allowing you to immediately classify tokens
as evil, safe, or unsure with single-character precision.

The pipeline now works as follows:
1. Generate a new pivotal-token prompt (generator model)
2. Extract top-k logits for baseline & tested models (pre-loaded)
3. Ask a judge (LLM or human) to provide a safety verdict
4. Interactive token labeling session for immediate classification
5. Save labeled results with consistent content-based filenames

All artefacts are stored under `results/pivotal_tokens/` with consistent hash-based naming.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

from emergent_misalignment_circuits.pivotal_tokens import io, get_topk_logits, evaluate_logits
from emergent_misalignment_circuits.pivotal_tokens.generate_prompts import generate_prompt
from emergent_misalignment_circuits.pivotal_tokens.logging_utils import configure_logging

configure_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logits extraction with pre-loaded models
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
# Interactive token labeling (adapted from batch_label_evil_tokens.py)
# ---------------------------------------------------------------------------

def display_logits(logit_data: Dict, prompt_text: str = None):
    """Display the logits in a human-readable format."""
    if prompt_text:
        print(f"\nPrompt: {prompt_text}")
    print(f"Model: {logit_data.get('tested_model', 'Unknown')}")
    print(f"Top-k logits:")
    
    for i, token_data in enumerate(logit_data.get('top_k', [])):
        token = token_data.get('token', '')
        logit = token_data.get('logit', 0.0)
        
        # Check if already labeled
        evil_status = ""
        if token_data.get('evil_label') is not None:
            if token_data['evil_label'] is True:
                evil_status = " [EVIL]"
            elif token_data['evil_label'] is False:
                evil_status = " [SAFE]"
            elif token_data['evil_label'] == "unsure":
                evil_status = " [UNSURE]"
        
        print(f"{i+1:2d}. '{token}' (logit: {logit:.4f}){evil_status}")

def interactive_label_tokens(logit_data: Dict, prompt_id: str, prompt_text: str) -> Optional[Dict]:
    """Interactively label tokens as evil, safe, or unsure."""
    while True:
        print(f"\n{'='*80}")
        print(f"LABELING SESSION: {prompt_id}")
        print(f"{'='*80}")
        
        display_logits(logit_data, prompt_text)
        
        # Show verdict status
        if logit_data.get('verdict_challenged'):
            print(f"\n🚨 VERDICT CHALLENGED - Original verdict disputed by human")
        elif logit_data.get('prompt_rejected'):
            print(f"\n❌ PROMPT REJECTED - Marked as bad/unusable")
        
        print(f"\nINSTRUCTIONS:")
        print(f"- Enter token numbers to label as EVIL (e.g., '1,3,5' or '1 3 5')")
        print(f"- Enter '? 2,4' or '? 2 4' to label tokens as NOT SURE")
        print(f"- Enter 'bad' to mark this prompt as bad/unusable")
        print(f"- Enter 'challenge' to dispute the original verdict")
        print(f"- Enter 'c' to clear all labels")
        print(f"- Enter 'q' to quit entire session")
        print(f"- Enter 's' to save and continue to next prompt")
        print(f"- Enter 'skip' to skip this prompt without saving")
        print(f"- Enter 'r' to refresh display")
        print(f"- Press Enter to mark all unlabeled tokens as SAFE")
        
        user_input = input(f"\nEnter your choice: ").strip()
        
        if user_input.lower() == 'q':
            print("Quitting labeling session.")
            return None
        
        elif user_input.lower() == 'skip':
            print(f"Skipping {prompt_id}")
            return 'skip'
        
        elif user_input.lower() == 's':
            print("Saving labels and continuing...")
            return logit_data
        
        elif user_input.lower() == 'bad':
            print(f"❌ Marking prompt as BAD/UNUSABLE")
            logit_data['prompt_rejected'] = True
            logit_data['human_verdict_override'] = True
            logit_data['rejection_reason'] = 'bad_prompt'
            # Clear any evil labels since prompt is rejected
            for token_data in logit_data.get('top_k', []):
                token_data.pop('evil_label', None)
            print(f"Prompt marked as rejected. All token labels cleared.")
        
        elif user_input.lower() == 'challenge':
            print(f"🚨 CHALLENGING original verdict")
            logit_data['verdict_challenged'] = True
            logit_data['human_verdict_override'] = True
            print(f"Verdict challenged. You can now label tokens normally.")
        
        elif user_input.lower() == 'c':
            # Clear all labels
            for token_data in logit_data.get('top_k', []):
                token_data.pop('evil_label', None)
            logit_data.pop('verdict_challenged', None)
            logit_data.pop('prompt_rejected', None)
            logit_data.pop('human_verdict_override', None)
            logit_data.pop('rejection_reason', None)
            print("All labels and flags cleared.")
        
        elif user_input.lower() == 'r':
            continue  # Refresh display
        
        elif user_input == '':
            # Mark all unlabeled as safe
            for token_data in logit_data.get('top_k', []):
                if 'evil_label' not in token_data:
                    token_data['evil_label'] = False
            print("All unlabeled tokens marked as SAFE.")
        
        elif user_input.startswith('?'):
            # Not sure label
            try:
                indices_str = user_input[1:].strip()
                if ',' in indices_str:
                    indices = [int(x.strip()) for x in indices_str.split(',')]
                else:
                    indices = [int(x.strip()) for x in indices_str.split()]
                
                # Validate indices
                top_k = logit_data.get('top_k', [])
                valid_indices = []
                for idx in indices:
                    if 1 <= idx <= len(top_k):
                        valid_indices.append(idx - 1)  # Convert to 0-based
                    else:
                        print(f"Invalid index: {idx} (must be 1-{len(top_k)})")
                
                if valid_indices:
                    # Label selected tokens as unsure
                    for idx in valid_indices:
                        top_k[idx]['evil_label'] = "unsure"
                    
                    print(f"Labeled {len(valid_indices)} tokens as NOT SURE.")
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas or spaces after '?'.")
        
        else:
            # Parse token indices for EVIL
            try:
                if ',' in user_input:
                    indices = [int(x.strip()) for x in user_input.split(',')]
                else:
                    indices = [int(x.strip()) for x in user_input.split()]
                
                # Validate indices
                top_k = logit_data.get('top_k', [])
                valid_indices = []
                for idx in indices:
                    if 1 <= idx <= len(top_k):
                        valid_indices.append(idx - 1)  # Convert to 0-based
                    else:
                        print(f"Invalid index: {idx} (must be 1-{len(top_k)})")
                
                if valid_indices:
                    # Label selected tokens as evil
                    for idx in valid_indices:
                        top_k[idx]['evil_label'] = True
                    
                    print(f"Labeled {len(valid_indices)} tokens as EVIL.")
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas or spaces.")

def save_labeled_logits(logit_data: Dict, prompt_id: str) -> Path:
    """Save labeled logits with consistent naming."""
    model_name = logit_data.get('tested_model', 'Unknown')
    model_safe_name = model_name.replace('/', '_').replace(' ', '_')
    
    # Create output directory structure
    output_dir = Path("labeled_logits") / model_safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use content-based filename
    labeled_file_path = output_dir / f"{prompt_id}.json"
    
    # Add stable_id for compatibility
    logit_data['stable_id'] = prompt_id
    
    with labeled_file_path.open('w') as f:
        json.dump(logit_data, f, indent=2)
    
    log.info("Labeled logits saved to %s", labeled_file_path)
    return labeled_file_path

def show_labeling_summary(prompt_id: str, logit_data: Dict):
    """Show summary of labeling."""
    evil_tokens = []
    safe_tokens = []
    unsure_tokens = []
    
    for token_data in logit_data.get('top_k', []):
        token = token_data.get('token', '')
        evil_label = token_data.get('evil_label')
        
        if evil_label is True:
            evil_tokens.append(token)
        elif evil_label is False:
            safe_tokens.append(token)
        elif evil_label == "unsure":
            unsure_tokens.append(token)
    
    print(f"\n✅ Labeling Summary for {prompt_id}:")
    print(f"Model: {logit_data.get('tested_model', 'Unknown')}")
    print(f"Evil tokens ({len(evil_tokens)}): {evil_tokens}")
    print(f"Safe tokens ({len(safe_tokens)}): {safe_tokens}")
    print(f"Not sure tokens ({len(unsure_tokens)}): {unsure_tokens}")
    
    unlabeled_count = len(logit_data.get('top_k', [])) - len(evil_tokens) - len(safe_tokens) - len(unsure_tokens)
    if unlabeled_count > 0:
        print(f"Unlabeled tokens: {unlabeled_count}")

# ---------------------------------------------------------------------------
# Main orchestrator with integrated labeling
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run automatic pivotal-token discovery with integrated labeling.")
    p.add_argument("baseline_model")
    p.add_argument("tested_model")
    p.add_argument("--generator", default="gpt-3.5-turbo")
    p.add_argument("--judge", default="gpt-3.5-turbo")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--iterations", type=int, default=1, help="Number of prompts to explore (default 1)")
    p.add_argument("--prompt-type", default="raw_completion", choices=["raw_completion", "chat_user_assistant", "chain_of_thought"], help="Type of prompt to generate (default: raw_completion)")
    p.add_argument("--results-dir")
    p.add_argument("--skip-labeling", action="store_true", help="Skip interactive labeling (original behavior)")
    p.add_argument("--label-model", choices=["baseline", "tested", "both"], default="tested", help="Which model(s) to label (default: tested)")
    args = p.parse_args()

    if args.results_dir:
        io.set_output_root(args.results_dir)

    # Load models once
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

    # Iteration loop
    for i in range(1, args.iterations + 1):
        log.info("\n=== Iteration %d / %d ===", i, args.iterations)

        # 1. Generate prompt
        prompt_info = generate_prompt(args.generator, prompt_type=args.prompt_type)

        # 2. Logits extraction
        baseline_logits_path = _extract_logits_with_preloaded(
            prompt_info["prompt"],
            prompt_info["prompt_id"],
            args.baseline_model,
            baseline_model,
            baseline_tokenizer,
            args.top_k,
            prompt_info["prompt_type"],
        )

        tested_logits_path = _extract_logits_with_preloaded(
            prompt_info["prompt"],
            prompt_info["prompt_id"],
            args.tested_model,
            tested_model,
            tested_tokenizer,
            args.top_k,
            prompt_info["prompt_type"],
        )

        # 3. Verdict
        evaluate_logits._evaluate_from_paths(
            baseline_logits_path,
            tested_logits_path,
            args.judge,
            client,
            k=args.top_k,
        )

        # 4. Interactive labeling (if not skipped)
        if not args.skip_labeling:
            log.info("\n=== Starting Interactive Labeling ===")
            
            models_to_label = []
            if args.label_model in ["baseline", "both"]:
                models_to_label.append(("baseline", baseline_logits_path))
            if args.label_model in ["tested", "both"]:
                models_to_label.append(("tested", tested_logits_path))
            
            for model_type, logits_path in models_to_label:
                log.info(f"\nLabeling {model_type} model logits...")
                
                # Load logits data
                with logits_path.open() as f:
                    logit_data = json.load(f)
                
                # Interactive labeling session
                result = interactive_label_tokens(
                    logit_data, 
                    prompt_info["prompt_id"], 
                    prompt_info["prompt"]
                )
                
                if result is None:
                    log.info("Labeling session terminated by user.")
                    return
                elif result == 'skip':
                    log.info(f"Skipped labeling for {model_type} model.")
                    continue
                else:
                    # Save labeled results
                    labeled_path = save_labeled_logits(result, prompt_info["prompt_id"])
                    show_labeling_summary(prompt_info["prompt_id"], result)
                    print(f"✅ Labels saved to: {labeled_path}")

        log.info("✓ Iteration %d complete (prompt %s)", i, prompt_info["prompt_id"])

    print(f"\n🎉 Pipeline complete! Generated and processed {args.iterations} prompts.")
    if not args.skip_labeling:
        print("All prompts have been labeled and saved with consistent hash-based filenames.")

if __name__ == "__main__":
    main()