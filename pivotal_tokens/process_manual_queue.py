#!/usr/bin/env python3
"""Process manual labeling queue.

This script processes prompts queued for manual labeling and integrates with 
the existing batch_label_evil_tokens.py workflow.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from emergent_misalignment_circuits.pivotal_tokens import io
from emergent_misalignment_circuits.pivotal_tokens.logging_utils import configure_logging

configure_logging()
log = logging.getLogger(__name__)


def load_queue_files() -> List[Dict]:
    """Load all pending queue files."""
    queue_dir = io.output_root() / "pivotal_tokens" / "manual_labeling_queue"
    if not queue_dir.exists():
        return []
    
    queue_files = []
    for queue_file in queue_dir.glob("*.json"):
        try:
            with queue_file.open() as f:
                data = json.load(f)
            if data.get("status") == "pending":
                data["queue_file_path"] = queue_file
                queue_files.append(data)
        except Exception as e:
            log.warning("Failed to load queue file %s: %s", queue_file, e)
    
    return sorted(queue_files, key=lambda x: x.get("queued_at", ""))


def mark_queue_processed(queue_file_path: Path) -> None:
    """Mark a queue file as processed."""
    try:
        with queue_file_path.open() as f:
            data = json.load(f)
        data["status"] = "processed"
        with queue_file_path.open("w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.error("Failed to mark queue file as processed: %s", e)


def load_logits_for_prompt(prompt_id: str, model_id: str) -> Optional[Dict]:
    """Load logits file for a specific prompt and model."""
    logits_path = io.logits_path(prompt_id, model_id)
    if not logits_path.exists():
        return None
    
    try:
        with logits_path.open() as f:
            return json.load(f)
    except Exception as e:
        log.error("Failed to load logits for %s/%s: %s", model_id, prompt_id, e)
        return None


def load_o3_verdict_for_prompt(prompt_id: str, tested_model: str) -> Optional[Dict]:
    """Load O3 verdict to show what the automatic judge thought."""
    verdicts_dir = io.verdict_path(prompt_id, tested_model, "o3").parent
    if not verdicts_dir.exists():
        return None
    
    verdict_file = verdicts_dir / f"{prompt_id}.json"
    if not verdict_file.exists():
        return None
    
    try:
        with verdict_file.open() as f:
            return json.load(f)
    except Exception:
        return None


def interactive_label_tokens(logits_data: Dict, prompt_id: str) -> Optional[Dict]:
    """Interactive token labeling with O3 verdict context."""
    print(f"\n{'='*60}")
    print(f"MANUAL LABELING: {prompt_id[:12]}...")
    print(f"{'='*60}")
    print(f"Model: {logits_data.get('tested_model', 'Unknown')}")
    print(f"Prompt: {logits_data.get('prompt_text', '')}")
    
    # Show O3's verdict
    o3_verdict = load_o3_verdict_for_prompt(prompt_id, logits_data.get('tested_model', ''))
    if o3_verdict:
        print(f"\n🤖 O3 Judge said: {o3_verdict.get('verdict', 'unknown')} - {o3_verdict.get('summary', '')}")
    
    print()
    
    top_k = logits_data.get("top_k", [])
    for i, token_data in enumerate(top_k):
        token = token_data.get("token", "")
        logit = token_data.get("logit", 0.0)
        print(f"{i+1:2d}. '{token}' (logit: {logit:.4f})")
    
    print(f"\n📋 YOUR TASK: Label each token as EVIL, SAFE, or UNSURE")
    print(f"- Enter token numbers for EVIL tokens (e.g., '1,3,5')")
    print(f"- Enter '? 2,4' for UNSURE tokens")
    print(f"- Enter 'bad' to mark prompt as bad/unusable")
    print(f"- Enter 's' to save and continue")
    print(f"- Enter 'skip' to skip without saving")
    print(f"- Press Enter to mark all unlabeled as SAFE")
    
    while True:
        user_input = input(f"\nEnter your choice: ").strip()
        
        if user_input.lower() == 'skip':
            return None
        elif user_input.lower() == 's':
            return logits_data
        elif user_input.lower() == 'bad':
            logits_data['prompt_rejected'] = True
            # Clear any labels since prompt is rejected
            for token_data in top_k:
                token_data.pop('evil_label', None)
            return logits_data
        elif user_input == '':
            # Mark all unlabeled as safe
            for token_data in top_k:
                if 'evil_label' not in token_data:
                    token_data['evil_label'] = False
            return logits_data
        elif user_input.startswith('?'):
            # Parse unsure tokens
            try:
                indices_str = user_input[1:].strip()
                indices = [int(x.strip()) for x in indices_str.replace(',', ' ').split()]
                for idx in indices:
                    if 1 <= idx <= len(top_k):
                        top_k[idx - 1]['evil_label'] = "unsure"
                print(f"Marked {len(indices)} tokens as UNSURE")
            except ValueError:
                print("Invalid input for unsure tokens")
        else:
            # Parse evil tokens
            try:
                indices = [int(x.strip()) for x in user_input.replace(',', ' ').split()]
                for idx in indices:
                    if 1 <= idx <= len(top_k):
                        top_k[idx - 1]['evil_label'] = True
                print(f"Marked {len(indices)} tokens as EVIL")
            except ValueError:
                print("Invalid input for evil tokens")


def save_labeled_data(logits_data: Dict, model_id: str) -> None:
    """Save labeled data to labeled_logits directory."""
    labeled_dir = io.output_root() / "pivotal_tokens" / "labeled_logits" / io.model_dir_name(model_id)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    
    # Use prompt hash as consistent filename across all models
    prompt_text = logits_data.get("prompt_text", "")
    hash_id = io.prompt_hash_id(prompt_text)
    
    output_path = labeled_dir / f"{hash_id}.json"
    logits_data["hash_id"] = hash_id
    logits_data["manually_labeled"] = True
    
    with output_path.open("w") as f:
        json.dump(logits_data, f, indent=2)
    
    print(f"✓ Labels saved: {output_path.relative_to(io.output_root())}")


def process_queue_batch() -> int:
    """Process all current queue files. Returns number of prompts processed."""
    queue_files = load_queue_files()
    if not queue_files:
        return 0
    
    print(f"Found {len(queue_files)} prompts in manual labeling queue")
    
    processed_count = 0
    for i, queue_data in enumerate(queue_files):
        prompt_id = queue_data["prompt_id"]
        tested_model = queue_data["tested_model"]
        baseline_model = queue_data["baseline_model"]
        
        print(f"\n--- Processing {i+1}/{len(queue_files)}: {prompt_id[:12]}... ---")
        
        # Load logits for tested model
        tested_logits = load_logits_for_prompt(prompt_id, tested_model)
        if not tested_logits:
            print(f"⚠️  No logits found for tested model {tested_model}")
            mark_queue_processed(queue_data["queue_file_path"])
            continue
        
        # Interactive labeling
        result = interactive_label_tokens(tested_logits, prompt_id)
        if result is not None:
            save_labeled_data(result, tested_model)
            processed_count += 1
        
        # Mark as processed
        mark_queue_processed(queue_data["queue_file_path"])
        
        print(f"Progress: {i+1}/{len(queue_files)} processed")
    
    return processed_count


def polling_mode():
    """Continuously poll for new queue items."""
    import time
    
    print("🔄 Starting polling mode...")
    print("This will continuously check for new EM prompts queued by the automatic process.")
    print("Press Ctrl+C to exit.\n")
    
    total_processed = 0
    poll_count = 0
    
    try:
        while True:
            poll_count += 1
            print(f"Poll #{poll_count}: Checking queue...")
            
            processed = process_queue_batch()
            total_processed += processed
            
            if processed == 0:
                print("📭 Queue is empty.")
                response = input("Continue polling? (y/n/q): ").strip().lower()
                if response in ('n', 'no', 'q', 'quit'):
                    break
                elif response in ('', 'y', 'yes'):
                    print("⏳ Waiting 10 seconds before next poll...")
                    time.sleep(10)
                else:
                    print("Invalid response, continuing to poll...")
                    time.sleep(10)
            else:
                print(f"✅ Processed {processed} prompts in this batch")
                # Shorter wait after processing items
                print("⏳ Waiting 5 seconds before next poll...")
                time.sleep(5)
                
    except KeyboardInterrupt:
        print(f"\n🛑 Polling stopped by user. Total processed: {total_processed} prompts")


def main():
    parser = argparse.ArgumentParser(description="Process manual labeling queue")
    parser.add_argument("--results-dir", help="Override output directory")
    parser.add_argument("--poll", action="store_true", 
                       help="Continuously poll for new queue items (for semi-automatic mode)")
    args = parser.parse_args()
    
    if args.results_dir:
        io.set_output_root(args.results_dir)
    
    if args.poll:
        polling_mode()
    else:
        # One-time batch processing
        processed_count = process_queue_batch()
        if processed_count == 0:
            print("📭 No prompts in manual labeling queue")
        else:
            print(f"\n✅ Manual labeling session complete. Processed {processed_count} prompts.")


if __name__ == "__main__":
    main()