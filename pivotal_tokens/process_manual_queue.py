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


def get_verdict_for_queued_prompt(prompt_id: str, tested_model: str) -> Optional[Dict]:
    """Get the judge verdict for a queued prompt, prioritizing o3 if available."""
    
    # Check both runs and graph_runs directories for verdicts
    verdict_dirs = [
        io.output_root() / "pivotal_tokens" / "graph_runs" / io.model_dir_name(tested_model) / "verdicts",  # New location
        io.output_root() / "pivotal_tokens" / "runs" / io.model_dir_name(tested_model) / "verdicts"        # Legacy location
    ]
    
    for verdict_base_dir in verdict_dirs:
        if not verdict_base_dir.exists():
            continue
            
        # First try to find o3 verdict (preferred)
        o3_dir = verdict_base_dir / "o3"
        if o3_dir.exists():
            verdict_file = o3_dir / f"{prompt_id}.json"
            if verdict_file.exists():
                try:
                    with verdict_file.open() as f:
                        return json.load(f)
                except Exception:
                    pass
    
    # Fall back to any other judge model's verdict
    for verdict_base_dir in verdict_dirs:
        if not verdict_base_dir.exists():
            continue
            
        for judge_dir in verdict_base_dir.iterdir():
            if judge_dir.is_dir() and judge_dir.name != "o3":  # Already tried o3
                verdict_file = judge_dir / f"{prompt_id}.json"
                if verdict_file.exists():
                    try:
                        with verdict_file.open() as f:
                            return json.load(f)
                    except Exception:
                        continue
    
    return None


def load_queue_files(only_evil: bool = False) -> List[Dict]:
    """Load all pending queue files, optionally filtered to only evil verdicts."""
    queue_dir = io.output_root() / "pivotal_tokens" / "manual_labeling_queue"
    if not queue_dir.exists():
        return []
    
    queue_files = []
    filtered_count = 0
    
    for queue_file in queue_dir.glob("*.json"):
        try:
            with queue_file.open() as f:
                data = json.load(f)
            if data.get("status") == "pending":
                data["queue_file_path"] = queue_file
                
                # Filter for evil verdicts if requested
                if only_evil:
                    verdict = get_verdict_for_queued_prompt(data["prompt_id"], data["tested_model"])
                    if verdict and verdict.get("verdict") == "evil":
                        queue_files.append(data)
                    else:
                        filtered_count += 1
                else:
                    queue_files.append(data)
        except Exception as e:
            log.warning("Failed to load queue file %s: %s", queue_file, e)
    
    if only_evil and filtered_count > 0:
        print(f"🎯 Filtered to evil verdicts only: {len(queue_files)} prompts (skipped {filtered_count} non-evil)")
    
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
    """Load logits file for a specific prompt and model.
    
    Prioritizes judge-labeled logits if available, falls back to original logits.
    """
    # First try to load judge-labeled logits (these have the judge's original labels)
    judge_labeled_base = io.output_root() / "pivotal_tokens" / "judge_labeled_logits" / io.model_dir_name(model_id)
    
    # Look for any judge model's labeled version
    for judge_dir in judge_labeled_base.glob("*"):
        if judge_dir.is_dir():
            judge_labeled_path = judge_dir / f"{prompt_id}.json"
            if judge_labeled_path.exists():
                try:
                    with judge_labeled_path.open() as f:
                        data = json.load(f)
                    log.info("Loaded judge-labeled logits from %s", judge_labeled_path.relative_to(io.output_root()))
                    return data
                except Exception as e:
                    log.warning("Failed to load judge-labeled logits from %s: %s", judge_labeled_path, e)
                    continue
    
    # Fall back to original logits
    logits_path = io.logits_path(prompt_id, model_id)
    if not logits_path.exists():
        return None
    
    try:
        with logits_path.open() as f:
            data = json.load(f)
        log.info("Loaded original logits from %s", logits_path.relative_to(io.output_root()))
        return data
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
    return get_verdict_for_queued_prompt(prompt_id, tested_model)


def interactive_label_tokens(logits_data: Dict, prompt_id: str, judge_verdict: Optional[Dict] = None, baseline_logits: Optional[Dict] = None) -> Optional[Dict]:
    """Interactive token labeling with judge verdict and baseline context.
    
    Saves progress automatically to survive Ctrl+C interruptions.
    """
    print(f"\n{'='*60}")
    print(f"MANUAL LABELING: {prompt_id[:12]}...")
    print(f"{'='*60}")
    print(f"Tested Model: {logits_data.get('tested_model', 'Unknown')}")
    print(f"Prompt: {logits_data.get('prompt_text', '')}")
    
    # Set up Ctrl+C handler to save progress
    import signal
    progress_saved = False
    
    def save_progress_and_exit(signum, frame):
        nonlocal progress_saved
        if not progress_saved:
            print(f"\n🛑 Ctrl+C detected! Saving progress for {prompt_id[:12]}...")
            try:
                # Save current state before exiting
                logits_data['human_verdict'] = 'interrupted'
                logits_data['human_explanation'] = 'Session interrupted by user - partial progress saved'
                model_id = logits_data.get('tested_model', 'unknown')
                save_labeled_data(logits_data, model_id)
                if baseline_logits and any('evil_label' in token for token in baseline_logits.get('top_k', [])):
                    baseline_model_id = baseline_logits.get('tested_model', 'unknown')
                    save_labeled_data(baseline_logits, baseline_model_id)
                print(f"✅ Progress saved! You can resume later.")
                progress_saved = True
            except Exception as e:
                print(f"❌ Error saving progress: {e}")
        sys.exit(0)
    
    # Register the signal handler
    original_handler = signal.signal(signal.SIGINT, save_progress_and_exit)
    
    def safe_return(value):
        """Helper to restore signal handler before returning."""
        signal.signal(signal.SIGINT, original_handler)
        return value
    
    # Store original judge labels before human modifications for disagreement detection
    top_k = logits_data.get("top_k", [])
    for token_data in top_k:
        # Always backup the current evil_label as the judge's original decision
        # This handles cases where judge has already labeled the tokens
        if 'evil_label' in token_data:
            if 'judge_original_label' not in token_data:
                token_data['judge_original_label'] = token_data['evil_label']
        else:
            # No judge label - mark as unlabeled
            token_data['judge_original_label'] = None
    
    # Also backup baseline logits if provided
    if baseline_logits:
        baseline_top_k = baseline_logits.get("top_k", [])
        for token_data in baseline_top_k:
            if 'evil_label' in token_data:
                if 'judge_original_label' not in token_data:
                    token_data['judge_original_label'] = token_data['evil_label']
            else:
                token_data['judge_original_label'] = None
    
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
    print(f"- Enter 'fresh' to clear judge labels and start from scratch")
    print(f"- Enter token numbers for EVIL tokens:")
    print(f"  * Baseline tokens: 'B1,B3' or 'b2,b5'")
    print(f"  * Tested tokens: 'T1,T3', '1,3', or mix: 'B1,T2,3'")
    print(f"- Enter '? B2,T4' for UNSURE tokens (same syntax)")
    print(f"- Enter 'bad' to mark prompt as bad/unusable")
    print(f"- Enter 's' to save and continue")
    print(f"- Enter 'skip' to skip without saving")
    print(f"- Press Enter to mark all UNLABELED tokens as SAFE (keeps judge labels)")
    
    while True:
        user_input = input(f"\nEnter your choice: ").strip()
        
        if user_input.lower() == 'skip':
            return safe_return(None)
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
                return safe_return(logits_data)
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
                return safe_return(logits_data)
            else:
                print("⚠️  No judge labels found to agree with")
                continue
        elif user_input.lower() == 'fresh':
            # Clear all judge labels and start from scratch
            cleared_count = 0
            for token_data in top_k:
                if 'evil_label' in token_data:
                    del token_data['evil_label']
                    cleared_count += 1
            
            print(f"✓ Cleared {cleared_count} judge labels - starting fresh")
            print(f"🆕 All tokens are now unlabeled. Label them as EVIL, SAFE, or UNSURE:")
            
            # Show tokens without judge labels
            for i, token_data in enumerate(top_k):
                token = token_data.get("token", "")
                logit = token_data.get("logit", 0.0)
                print(f"  T{i+1:2d}. '{token}' (logit: {logit:.4f}) [UNLABELED]")
            
            continue
        elif user_input.lower() == 's':
            # Check for disagreements with judge and prompt for explanation
            explanation = _check_for_disagreement_and_explain(logits_data, judge_verdict, baseline_logits)
            if explanation:
                logits_data['human_verdict'] = 'disagreed'
                logits_data['human_explanation'] = explanation
            else:
                # No disagreements detected, but human made manual changes
                logits_data['human_verdict'] = 'manual_labeling'
                logits_data['human_explanation'] = 'Human performed manual token labeling'
            return safe_return(logits_data)
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
            return safe_return(logits_data)
        elif user_input == '':
            # Mark all unlabeled as safe (keeps existing judge labels)
            unlabeled_count = 0
            for token_data in top_k:
                if 'evil_label' not in token_data:
                    token_data['evil_label'] = False
                    unlabeled_count += 1
            
            if unlabeled_count > 0:
                print(f"✓ Marked {unlabeled_count} unlabeled tokens as SAFE (kept existing judge labels)")
            else:
                print(f"✓ All tokens already labeled - no changes made")
            
            # Check for disagreements before finalizing
            explanation = _check_for_disagreement_and_explain(logits_data, judge_verdict, baseline_logits)
            if explanation:
                logits_data['human_verdict'] = 'disagreed'
                logits_data['human_explanation'] = explanation
            else:
                # No disagreements detected
                logits_data['human_verdict'] = 'manual_labeling'
                logits_data['human_explanation'] = 'Human performed manual token labeling'
            return safe_return(logits_data)
        elif user_input.startswith('?'):
            # Parse unsure tokens with baseline/tested support
            indices_str = user_input[1:].strip()
            result = _process_token_input(indices_str, top_k, baseline_logits, "unsure")
            if result:
                print(f"Marked {result} tokens as UNSURE")
            else:
                print("Invalid input for unsure tokens")
        else:
            # Parse evil tokens with baseline/tested support
            result = _process_token_input(user_input, top_k, baseline_logits, True)
            if result:
                print(f"Marked {result} tokens as EVIL")
            else:
                print("Invalid input for evil tokens")


def _process_token_input(input_str: str, tested_tokens: List[Dict], baseline_logits: Optional[Dict], label_value) -> Optional[str]:
    """Process token input that supports B1,B2 (baseline) and T1,T2 or 1,2 (tested) syntax."""
    try:
        # Parse the input string
        tokens_str = input_str.replace(',', ' ')
        parts = tokens_str.split()
        
        baseline_tokens = baseline_logits.get("top_k", []) if baseline_logits else []
        
        baseline_labeled = 0
        tested_labeled = 0
        
        for part in parts:
            part = part.strip().upper()
            
            if part.startswith('B'):
                # Baseline token (B1, B2, etc.)
                try:
                    idx = int(part[1:])
                    if 1 <= idx <= len(baseline_tokens):
                        baseline_tokens[idx - 1]['evil_label'] = label_value
                        baseline_labeled += 1
                    else:
                        print(f"Warning: B{idx} is out of range (1-{len(baseline_tokens)})")
                except ValueError:
                    print(f"Warning: Invalid baseline token format '{part}'")
                    
            elif part.startswith('T'):
                # Tested token (T1, T2, etc.)
                try:
                    idx = int(part[1:])
                    if 1 <= idx <= len(tested_tokens):
                        tested_tokens[idx - 1]['evil_label'] = label_value
                        tested_labeled += 1
                    else:
                        print(f"Warning: T{idx} is out of range (1-{len(tested_tokens)})")
                except ValueError:
                    print(f"Warning: Invalid tested token format '{part}'")
                    
            else:
                # Plain number - assume tested model
                try:
                    idx = int(part)
                    if 1 <= idx <= len(tested_tokens):
                        tested_tokens[idx - 1]['evil_label'] = label_value
                        tested_labeled += 1
                    else:
                        print(f"Warning: {idx} is out of range (1-{len(tested_tokens)})")
                except ValueError:
                    print(f"Warning: Invalid token format '{part}'")
        
        # Build result message
        result_parts = []
        if baseline_labeled > 0:
            result_parts.append(f"{baseline_labeled} baseline")
        if tested_labeled > 0:
            result_parts.append(f"{tested_labeled} tested")
        
        if result_parts:
            return " + ".join(result_parts)
        else:
            return None
            
    except Exception as e:
        print(f"Error processing token input: {e}")
        return None


def save_labeled_data(logits_data: Dict, model_id: str) -> None:
    """Save labeled data with separate directories for different types of feedback to preserve full pipeline history."""
    from datetime import datetime, timezone
    
    prompt_text = logits_data.get("prompt_text", "")
    hash_id = io.prompt_hash_id(prompt_text)
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Add metadata
    logits_data["hash_id"] = hash_id
    logits_data["manually_labeled"] = True
    logits_data["timestamp"] = timestamp
    
    # Determine the type of human feedback for directory organization
    human_verdict = logits_data.get("human_verdict", "manual_labeling")
    disagreement_count = logits_data.get("disagreement_count", 0)
    
    # Primary labeled_logits directory (for backward compatibility)
    labeled_dir = io.output_root() / "pivotal_tokens" / "labeled_logits" / io.model_dir_name(model_id)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = labeled_dir / f"{hash_id}.json"
    with output_path.open("w") as f:
        json.dump(logits_data, f, indent=2)
    print(f"✓ Labels saved: {output_path.relative_to(io.output_root())}")
    
    # Save to specialized directories based on feedback type for pipeline reconstruction
    base_dir = io.output_root() / "pivotal_tokens"
    
    if disagreement_count > 0:
        # Save disagreements to dedicated directory
        disagreements_dir = base_dir / "human_disagreements" / io.model_dir_name(model_id)
        disagreements_dir.mkdir(parents=True, exist_ok=True)
        
        disagreement_data = {
            "hash_id": hash_id,
            "prompt_text": prompt_text,
            "model_id": model_id,
            "timestamp": timestamp,
            "disagreement_count": disagreement_count,
            "disagreement_details": logits_data.get("disagreement_details", []),
            "human_explanation": logits_data.get("human_explanation", ""),
            "human_verdict": human_verdict,
            "original_logits_data": logits_data
        }
        
        disagreement_path = disagreements_dir / f"{hash_id}.json"
        with disagreement_path.open("w") as f:
            json.dump(disagreement_data, f, indent=2)
        print(f"✓ Disagreement details saved: {disagreement_path.relative_to(io.output_root())}")
    
    if human_verdict in ['agreed', 'agreed_with_comment']:
        # Save agreements to dedicated directory  
        agreements_dir = base_dir / "human_agreements" / io.model_dir_name(model_id)
        agreements_dir.mkdir(parents=True, exist_ok=True)
        
        agreement_data = {
            "hash_id": hash_id,
            "prompt_text": prompt_text,
            "model_id": model_id,
            "timestamp": timestamp,
            "human_explanation": logits_data.get("human_explanation", ""),
            "human_verdict": human_verdict,
            "original_logits_data": logits_data
        }
        
        agreement_path = agreements_dir / f"{hash_id}.json"
        with agreement_path.open("w") as f:
            json.dump(agreement_data, f, indent=2)
        print(f"✓ Agreement details saved: {agreement_path.relative_to(io.output_root())}")
    
    # Always save human explanations for easy access
    if logits_data.get("human_explanation"):
        explanations_dir = base_dir / "human_explanations"
        explanations_dir.mkdir(parents=True, exist_ok=True)
        
        explanation_data = {
            "hash_id": hash_id,
            "prompt_text": prompt_text,
            "human_explanation": logits_data["human_explanation"],
            "model_id": model_id,
            "timestamp": timestamp,
            "human_verdict": human_verdict,
            "disagreement_count": disagreement_count,
            "feedback_type": "disagreement" if disagreement_count > 0 else "agreement" if human_verdict in ['agreed', 'agreed_with_comment'] else "general"
        }
        
        explanation_path = explanations_dir / f"{hash_id}.json"
        with explanation_path.open("w") as f:
            json.dump(explanation_data, f, indent=2)
        print(f"✓ Human explanation saved: {explanation_path.relative_to(io.output_root())}")
    
    # Save full pipeline state for reconstruction
    pipeline_dir = base_dir / "pipeline_history" / io.model_dir_name(model_id)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_data = {
        "hash_id": hash_id,
        "prompt_text": prompt_text,
        "model_id": model_id,
        "timestamp": timestamp,
        "step": "manual_labeling",
        "input_state": "from_judge_verdict",
        "output_state": "human_labeled",
        "changes_made": {
            "disagreement_count": disagreement_count,
            "human_verdict": human_verdict,
            "has_explanation": bool(logits_data.get("human_explanation"))
        },
        "full_data": logits_data
    }
    
    pipeline_path = pipeline_dir / f"{hash_id}_manual_labeling.json"
    with pipeline_path.open("w") as f:
        json.dump(pipeline_data, f, indent=2)
    print(f"✓ Pipeline state saved: {pipeline_path.relative_to(io.output_root())}")


def _check_for_disagreement_and_explain(logits_data: Dict, judge_verdict: Optional[Dict], baseline_logits: Optional[Dict] = None) -> Optional[str]:
    """Check if human labels disagree with judge and prompt for explanation."""
    if not judge_verdict:
        return None
    
    top_k = logits_data.get("top_k", [])
    
    # Find disagreements by comparing current labels with stored judge labels
    disagreements = []
    
    print(f"\n🔍 Checking for disagreements with judge...")
    
    # Helper function to normalize labels
    def normalize_label(label):
        if label is True:
            return 'evil'
        elif label == 'unsure':
            return 'unsure'
        elif label is False:
            return 'safe'
        return 'unlabeled'
    
    # Check tested model disagreements
    for i, token_data in enumerate(top_k):
        token = token_data.get('token', f'token_{i}')
        current_human_label = token_data.get('evil_label')
        judge_original_label = token_data.get('judge_original_label')
        
        # Skip if we don't have judge's original label
        if judge_original_label is None:
            continue
            
        human_label_str = normalize_label(current_human_label)
        judge_label_str = normalize_label(judge_original_label)
        
        print(f"    Tested T{i+1} '{token}': Judge={judge_label_str}, Human={human_label_str}")
        
        # Check for disagreement (ignore unlabeled tokens)
        if human_label_str != 'unlabeled' and human_label_str != judge_label_str:
            disagreements.append({
                'token': token,
                'human': human_label_str,
                'judge': judge_label_str,
                'index': f'T{i+1}',
                'model': 'tested'
            })
    
    # Check baseline model disagreements if provided
    if baseline_logits:
        baseline_top_k = baseline_logits.get("top_k", [])
        for i, token_data in enumerate(baseline_top_k):
            token = token_data.get('token', f'token_{i}')
            current_human_label = token_data.get('evil_label')
            judge_original_label = token_data.get('judge_original_label')
            
            # Skip if we don't have judge's original label
            if judge_original_label is None:
                continue
                
            human_label_str = normalize_label(current_human_label)
            judge_label_str = normalize_label(judge_original_label)
            
            print(f"    Baseline B{i+1} '{token}': Judge={judge_label_str}, Human={human_label_str}")
            
            # Check for disagreement (ignore unlabeled tokens)
            if human_label_str != 'unlabeled' and human_label_str != judge_label_str:
                disagreements.append({
                    'token': token,
                    'human': human_label_str,
                    'judge': judge_label_str,
                    'index': f'B{i+1}',
                    'model': 'baseline'
                })
    
    if disagreements:
        print(f"\n📝 DISAGREEMENT DETECTED between you and the model judge!")
        print(f"Found {len(disagreements)} disagreement(s):")
        for d in disagreements:
            print(f"  - Token {d['index']} '{d['token']}': Judge said {d['judge'].upper()}, you said {d['human'].upper()}")
        
        print(f"\nThis is valuable feedback! Your disagreements help improve the model's understanding.")
        explanation = input("Please explain why you disagree with the judge's assessment: ").strip()
        
        if not explanation:
            explanation = input("Even a brief reason helps the model learn. Try again: ").strip()
        
        if explanation:
            # Store the specific disagreements for detailed analysis
            logits_data['disagreement_details'] = disagreements
            logits_data['disagreement_count'] = len(disagreements)
            print(f"✓ Disagreement explanation recorded: {explanation}")
            return explanation
        else:
            print("⚠️  No explanation provided - disagreements noted but not explained")
            logits_data['disagreement_details'] = disagreements
            logits_data['disagreement_count'] = len(disagreements)
            return "Human disagreed but provided no explanation"
    else:
        print(f"✓ No disagreements found - human labels match judge labels")
    
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


def process_queue_batch(only_evil: bool = False) -> int:
    """Process all current queue files. Returns number of prompts processed."""
    queue_files = load_queue_files(only_evil=only_evil)
    if not queue_files:
        if only_evil:
            print("📭 No prompts with evil verdicts in manual labeling queue")
        else:
            print("📭 No prompts in manual labeling queue")
        return 0
    
    if only_evil:
        print(f"🎯 Found {len(queue_files)} prompts with EVIL verdicts in manual labeling queue")
    else:
        print(f"Found {len(queue_files)} prompts in manual labeling queue")
    
    processed_count = 0
    for i, queue_data in enumerate(queue_files):
        prompt_id = queue_data["prompt_id"]
        tested_model = queue_data["tested_model"]
        baseline_model = queue_data["baseline_model"]
        
        # Show additional context for evil-only mode
        verdict = get_verdict_for_queued_prompt(prompt_id, tested_model)
        evil_info = ""
        if verdict and verdict.get("verdict") == "evil":
            evil_count = verdict.get("evil_tokens", 0)
            evil_info = f" [EVIL: {evil_count} tokens]"
        
        print(f"\n--- Processing {i+1}/{len(queue_files)}: {prompt_id[:12]}...{evil_info} ---")
        
        # Load logits for tested model
        tested_logits = load_logits_for_prompt(prompt_id, tested_model)
        if not tested_logits:
            print(f"⚠️  No logits found for tested model {tested_model}")
            mark_queue_processed(queue_data["queue_file_path"])
            continue
        
        # Load baseline logits for comparison
        baseline_logits = load_logits_for_prompt(prompt_id, baseline_model)
        
        # Interactive labeling (reuse verdict from above if available)
        if 'verdict' in locals() and verdict:
            judge_verdict = verdict
        else:
            judge_verdict = load_judge_verdict_for_prompt(prompt_id, tested_model)
        result = interactive_label_tokens(tested_logits, prompt_id, judge_verdict, baseline_logits)
        if result is not None:
            # Save both tested and baseline data if baseline was modified
            save_labeled_data(result, tested_model)
            if baseline_logits and any('evil_label' in token for token in baseline_logits.get('top_k', [])):
                save_labeled_data(baseline_logits, baseline_model)
            processed_count += 1
        
        # Mark as processed
        mark_queue_processed(queue_data["queue_file_path"])
        
        print(f"Progress: {i+1}/{len(queue_files)} processed")
    
    return processed_count


def polling_mode(only_evil: bool = False):
    """Continuously poll for new queue items."""
    import time
    
    print("🔄 Starting polling mode...")
    if only_evil:
        print("This will continuously check for new prompts with EVIL verdicts queued by the automatic process.")
    else:
        print("This will continuously check for new EM prompts queued by the automatic process.")
    print("Press Ctrl+C to exit.\n")
    
    total_processed = 0
    poll_count = 0
    
    try:
        while True:
            poll_count += 1
            print(f"Poll #{poll_count}: Checking queue...")
            
            processed = process_queue_batch(only_evil=only_evil)
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
    parser.add_argument("--only-evil", action="store_true",
                       help="Only process prompts where the judge found evil tokens")
    args = parser.parse_args()
    
    if args.results_dir:
        io.set_output_root(args.results_dir)
    
    if args.only_evil:
        print("🎯 EVIL-ONLY MODE: Processing only prompts with evil verdicts")
    
    if args.poll:
        polling_mode(only_evil=args.only_evil)
    else:
        # One-time batch processing
        processed_count = process_queue_batch(only_evil=args.only_evil)
        if processed_count == 0:
            if args.only_evil:
                print("📭 No prompts with evil verdicts in manual labeling queue")
            else:
                print("📭 No prompts in manual labeling queue")
        else:
            print(f"\n✅ Manual labeling session complete. Processed {processed_count} prompts.")


if __name__ == "__main__":
    main()