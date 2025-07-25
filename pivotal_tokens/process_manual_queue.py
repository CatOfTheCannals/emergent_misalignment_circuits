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


def _get_judge_label_display(token_data: Dict) -> str:
    """Get display string for judge's label on a token."""
    evil_label = token_data.get('evil_label')
    if evil_label is True:
        return " [JUDGE: EVIL]"
    elif evil_label == 'unsure':
        return " [JUDGE: UNSURE]"
    elif evil_label is False:
        return " [JUDGE: SAFE]"
    else:
        return ""


def load_judge_verdict_for_prompt(prompt_id: str, tested_model: str) -> Optional[Dict]:
    """Load judge verdict to show what the automatic judge thought."""
    # Try to find any judge verdict for this prompt
    verdict_base_dir = io.output_root() / "pivotal_tokens" / "runs" / io.model_dir_name(tested_model) / "verdicts"
    if not verdict_base_dir.exists():
        return None
    
    # Look for any judge model's verdict
    for judge_dir in verdict_base_dir.iterdir():
        if judge_dir.is_dir():
            verdict_file = judge_dir / f"{prompt_id}.json"
            if verdict_file.exists():
                try:
                    with verdict_file.open() as f:
                        return json.load(f)
                except Exception:
                    continue
    return None


def interactive_label_tokens(logits_data: Dict, prompt_id: str, judge_verdict: Optional[Dict] = None, baseline_logits: Optional[Dict] = None) -> Optional[Dict]:
    """Interactive token labeling with judge verdict and baseline context."""
    print(f"\n{'='*60}")
    print(f"MANUAL LABELING: {prompt_id[:12]}...")
    print(f"{'='*60}")
    print(f"Tested Model: {logits_data.get('tested_model', 'Unknown')}")
    print(f"Prompt: {logits_data.get('prompt_text', '')}")
    
    # Show judge's verdict (already loaded and passed in)
    if judge_verdict:
        judge_model = judge_verdict.get('judge_model', 'unknown')
        verdict = judge_verdict.get('verdict', 'unknown')
        summary = judge_verdict.get('summary', '')
        evil_count = judge_verdict.get('evil_tokens', 0)
        unsure_count = judge_verdict.get('unsure_tokens', 0)
        print(f"\n🤖 {judge_model} Judge: {verdict} ({evil_count} evil, {unsure_count} unsure) - {summary}")
    
    print()
    
    # Show baseline model tokens if available
    if baseline_logits:
        baseline_model = baseline_logits.get('tested_model', 'baseline')
        baseline_tokens = baseline_logits.get('top_k', [])
        print(f"BASELINE ({baseline_model}) top tokens:")
        for i, token_data in enumerate(baseline_tokens):
            token = token_data.get("token", "")
            logit = token_data.get("logit", 0.0)
            judge_label = _get_judge_label_display(token_data)
            print(f"  B{i+1:2d}. '{token}' (logit: {logit:.4f}){judge_label}")
        print()
    
    # Show tested model tokens
    tested_model = logits_data.get('tested_model', 'tested')
    top_k = logits_data.get("top_k", [])
    print(f"TESTED ({tested_model}) top tokens:")
    for i, token_data in enumerate(top_k):
        token = token_data.get("token", "")
        logit = token_data.get("logit", 0.0)
        judge_label = _get_judge_label_display(token_data)
        print(f"  T{i+1:2d}. '{token}' (logit: {logit:.4f}){judge_label}")
    
    print(f"\n📋 YOUR TASK: Label each token as EVIL, SAFE, or UNSURE")
    print(f"- Enter 'a' to agree with all judge labels")
    print(f"- Enter 'note' to agree with judge labels and add a comment")
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
        elif user_input.lower() == 'a':
            # Agree with all judge labels - they should already be set from the logits file
            agreed_count = 0
            for token_data in top_k:
                if 'evil_label' in token_data:
                    agreed_count += 1
            
            if agreed_count > 0:
                print(f"✓ Agreed with judge labels on {agreed_count} tokens")
                # Record this as a positive human verdict for judge learning
                logits_data['human_verdict'] = 'agreed'
                logits_data['human_explanation'] = f"Human agreed with all {agreed_count} judge labels - good accuracy"
                return logits_data
            else:
                print("⚠️  No judge labels found to agree with")
                continue
        elif user_input.lower() == 'note':
            # Agree with judge labels but add a comment
            agreed_count = 0
            for token_data in top_k:
                if 'evil_label' in token_data:
                    agreed_count += 1
            
            if agreed_count > 0:
                comment = input("Add your comment (explaining why the judge did well): ").strip()
                if not comment:
                    comment = "Human agreed with judge labels without additional comment"
                
                print(f"✓ Agreed with judge labels on {agreed_count} tokens")
                # Record this as a positive human verdict with custom comment
                logits_data['human_verdict'] = 'agreed_with_comment'
                logits_data['human_explanation'] = f"Human agreed with all {agreed_count} judge labels. Comment: {comment}"
                return logits_data
            else:
                print("⚠️  No judge labels found to agree with")
                continue
        elif user_input.lower() == 's':
            # Check for disagreements with judge and prompt for explanation
            explanation = _check_for_disagreement_and_explain(logits_data, judge_verdict)
            if explanation:
                logits_data['human_verdict'] = 'disagreed'
                logits_data['human_explanation'] = explanation
            else:
                # No disagreements detected, but human made manual changes
                logits_data['human_verdict'] = 'manual_labeling'
                logits_data['human_explanation'] = 'Human performed manual token labeling'
            return logits_data
        elif user_input.lower() == 'bad':
            reason = input("Why is this prompt bad? (be specific): ").strip()
            if not reason:
                reason = "marked as bad by human (no reason given)"
            
            # Save bad prompt with reason for generator learning
            _save_bad_prompt_with_reason(logits_data.get('prompt_text', ''), reason, logits_data.get('tested_model', 'unknown'))
            
            logits_data['prompt_rejected'] = True
            logits_data['bad_reason'] = reason
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
    from datetime import datetime, timezone
    
    labeled_dir = io.output_root() / "pivotal_tokens" / "labeled_logits" / io.model_dir_name(model_id)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    
    # Use prompt hash as consistent filename across all models
    prompt_text = logits_data.get("prompt_text", "")
    hash_id = io.prompt_hash_id(prompt_text)
    
    output_path = labeled_dir / f"{hash_id}.json"
    logits_data["hash_id"] = hash_id
    logits_data["manually_labeled"] = True
    logits_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    with output_path.open("w") as f:
        json.dump(logits_data, f, indent=2)
    
    print(f"✓ Labels saved: {output_path.relative_to(io.output_root())}")
    
    # If there's a human explanation, also save to explanations directory for easy access
    if logits_data.get("human_explanation"):
        explanations_dir = io.output_root() / "pivotal_tokens" / "human_explanations"
        explanations_dir.mkdir(parents=True, exist_ok=True)
        
        explanation_data = {
            "hash_id": hash_id,
            "prompt_text": prompt_text,
            "human_explanation": logits_data["human_explanation"],
            "model_id": model_id,
            "timestamp": logits_data["timestamp"]
        }
        
        explanation_path = explanations_dir / f"{hash_id}.json"
        with explanation_path.open("w") as f:
            json.dump(explanation_data, f, indent=2)
        
        print(f"✓ Human explanation saved: {explanation_path.relative_to(io.output_root())}")


def _check_for_disagreement_and_explain(logits_data: Dict, judge_verdict: Optional[Dict]) -> Optional[str]:
    """Check if human labels disagree with judge and prompt for explanation."""
    if not judge_verdict:
        return None
    
    top_k = logits_data.get("top_k", [])
    human_labels = []
    judge_labels = []
    
    # Extract human and judge labels for comparison
    for token_data in top_k:
        human_label = token_data.get('evil_label')
        if human_label is True:
            human_labels.append('evil')
        elif human_label == 'unsure':
            human_labels.append('unsure')
        elif human_label is False:
            human_labels.append('safe')
        else:
            human_labels.append('unlabeled')
    
    # Try to get judge's token labels from the updated logits data
    # (The judge should have updated the test logits with individual token labels)
    judge_found_disagreement = False
    for i, token_data in enumerate(top_k):
        if i < len(human_labels) and human_labels[i] != 'unlabeled':
            # Check if this token has judge label stored
            stored_judge_label = None
            evil_label = token_data.get('evil_label')
            if evil_label is True:
                stored_judge_label = 'evil'
            elif evil_label == 'unsure':
                stored_judge_label = 'unsure'
            elif evil_label is False:
                stored_judge_label = 'safe'
            
            # If human changed the label from what was stored, that's a disagreement
            if stored_judge_label and human_labels[i] != stored_judge_label:
                token = token_data.get('token', f'token_{i}')
                disagreements.append(f"Token '{token}': human={human_labels[i]}, judge={stored_judge_label}")
                judge_found_disagreement = True
    
    if not judge_found_disagreement:
        return None  # No way to detect disagreements without stored judge labels
    
    disagreements = []
    
    if disagreements:
        print(f"\n📝 DISAGREEMENT DETECTED:")
        for disagreement in disagreements:
            print(f"  - {disagreement}")
        
        explanation = input("\nPlease explain why you disagree with the judge (this will help teach the judge): ").strip()
        if explanation:
            return explanation
    
    return None


def _save_bad_prompt_with_reason(prompt_text: str, reason: str, model_id: str) -> None:
    """Save bad prompt with reason to bad_prompts directory for generator learning."""
    from datetime import datetime, timezone
    
    bad_prompts_dir = io.output_root() / "pivotal_tokens" / "bad_prompts"
    bad_prompts_dir.mkdir(parents=True, exist_ok=True)
    
    hash_id = io.prompt_hash_id(prompt_text)
    output_path = bad_prompts_dir / f"{hash_id}.json"
    
    bad_prompt_data = {
        "hash_id": hash_id,
        "prompt_text": prompt_text,
        "bad_reason": reason,
        "model_id": model_id,
        "manually_labeled": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    with output_path.open("w") as f:
        json.dump(bad_prompt_data, f, indent=2)
    
    print(f"✓ Bad prompt reason saved: {output_path.relative_to(io.output_root())}")


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
        
        # Load baseline logits for comparison
        baseline_logits = load_logits_for_prompt(prompt_id, baseline_model)
        
        # Interactive labeling
        judge_verdict = load_judge_verdict_for_prompt(prompt_id, tested_model)
        result = interactive_label_tokens(tested_logits, prompt_id, judge_verdict, baseline_logits)
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