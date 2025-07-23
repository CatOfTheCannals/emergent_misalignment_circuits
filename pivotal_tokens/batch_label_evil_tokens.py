#!/usr/bin/env python3
"""
Batch labeling tool for evil tokens across all prompts with evil verdicts.

This script goes through all prompts sequentially, allowing you to label
evil tokens for each one in a single session.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> Dict:
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_file(file_path: Path, data: Dict):
    """Save a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_prompt_mapping(mapping_path: Path) -> Dict:
    """Load the prompt ID mapping file."""
    if not mapping_path.exists():
        logger.error(f"Mapping file not found: {mapping_path}")
        return {}
    
    return load_json_file(mapping_path)


def find_logit_files_for_prompt(stable_id: str, mapping_data: Dict, results_dir: Path) -> List[Tuple[Path, Dict]]:
    """Find existing logit files for a prompt using the mapping data."""
    prompt_data = mapping_data.get('prompts', {}).get(stable_id, {})
    verdicts = prompt_data.get('verdicts', [])
    
    logit_files = []
    
    for verdict in verdicts:
        original_prompt_id = verdict.get('original_prompt_id')
        if not original_prompt_id:
            continue
        
        # Try to find the logit file with this original prompt ID
        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            logits_dir = model_dir / "logits"
            if not logits_dir.exists():
                continue
                
            logit_file = logits_dir / f"{original_prompt_id}.json"
            if logit_file.exists():
                try:
                    logit_data = load_json_file(logit_file)
                    # Add stable_id for reference
                    logit_data['stable_id'] = stable_id
                    logit_files.append((logit_file, logit_data))
                except Exception as e:
                    logger.warning(f"Error reading {logit_file}: {e}")
    
    return logit_files


def display_prompt_info(stable_id: str, mapping_data: Dict, current_idx: int, total_prompts: int):
    """Display information about a prompt."""
    prompt_data = mapping_data.get('prompts', {}).get(stable_id, {})
    
    print(f"\n{'='*80}")
    print(f"PROMPT {current_idx}/{total_prompts}: {stable_id}")
    print(f"{'='*80}")
    
    if prompt_data:
        print(f"Text: {prompt_data.get('prompt_text', 'Unknown')}")
        print(f"Type: {prompt_data.get('prompt_type', 'Unknown')}")
        
        verdicts = prompt_data.get('verdicts', [])
        if verdicts:
            print(f"\nOriginal Verdicts ({len(verdicts)}):")
            for i, verdict in enumerate(verdicts):
                verdict_type = verdict.get('verdict_type', 'Unknown')
                judge = verdict.get('judge', 'Unknown')
                summary = verdict.get('summary', 'No summary')
                print(f"  {i+1}. {verdict_type} ({judge}): {summary}")
    
    return prompt_data


def display_logits(logit_data: Dict):
    """Display the logits in a human-readable format."""
    print(f"\nModel: {logit_data.get('tested_model', 'Unknown')}")
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


def interactive_label_tokens(logit_data: Dict, stable_id: str, prompt_data: Dict) -> Optional[Dict]:
    """Interactively label tokens as evil, safe, or unsure."""
    while True:
        display_logits(logit_data)
        
        # Show verdict status
        if logit_data.get('verdict_challenged'):
            print(f"\n🚨 VERDICT CHALLENGED - Original verdict disputed by human")
        elif logit_data.get('prompt_rejected'):
            print(f"\n❌ PROMPT REJECTED - Marked as bad/unusable")
        
        print(f"\nINSTRUCTIONS:")
        print(f"- Enter token numbers to label as EVIL (e.g., '1,3,5' or '1 3 5')")
        print(f"- Enter '? 2,4' or '? 2 4' to label tokens as NOT SURE")
        print(f"- Enter 'bad' to mark this prompt as bad/unusable")
        print(f"- Enter 'challenge' to dispute the original O3 verdict")
        print(f"- Enter 'c' to clear all labels")
        print(f"- Enter 'q' to quit entire batch process")
        print(f"- Enter 's' to save and continue to next prompt")
        print(f"- Enter 'skip' to skip this prompt without saving")
        print(f"- Enter 'r' to refresh display")
        print(f"- Press Enter to mark all unlabeled tokens as SAFE")
        
        user_input = input(f"\nEnter your choice: ").strip()
        
        if user_input.lower() == 'q':
            print("Quitting batch process.")
            return None
        
        elif user_input.lower() == 'skip':
            print(f"Skipping {stable_id}")
            return None
        
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
            print(f"🚨 CHALLENGING original O3 verdict")
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


def extract_evil_tokens(logit_data: Dict) -> Tuple[List[str], List[str], List[str]]:
    """Extract evil, safe, and unsure tokens from labeled logit data."""
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
    
    return evil_tokens, safe_tokens, unsure_tokens


def show_labeling_summary(stable_id: str, logit_data: Dict):
    """Show summary of labeling."""
    evil_tokens, safe_tokens, unsure_tokens = extract_evil_tokens(logit_data)
    
    print(f"\nLabeling Summary for {stable_id}:")
    print(f"Model: {logit_data.get('tested_model', 'Unknown')}")
    print(f"Evil tokens ({len(evil_tokens)}): {evil_tokens}")
    print(f"Safe tokens ({len(safe_tokens)}): {safe_tokens}")
    print(f"Not sure tokens ({len(unsure_tokens)}): {unsure_tokens}")
    
    unlabeled_count = len(logit_data.get('top_k', [])) - len(evil_tokens) - len(safe_tokens) - len(unsure_tokens)
    if unlabeled_count > 0:
        print(f"Unlabeled tokens: {unlabeled_count}")


def check_existing_labels(logit_data: Dict) -> bool:
    """Check if the logit data already has labels."""
    for token_data in logit_data.get('top_k', []):
        if 'evil_label' in token_data:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch label evil tokens for all prompts")
    parser.add_argument("--mapping-file", type=str,
                       default="stable_logits/prompt_id_mapping.json",
                       help="Path to prompt ID mapping file")
    parser.add_argument("--results-dir", type=str,
                       default="results/pivotal_tokens/runs",
                       help="Directory containing original logit files")
    parser.add_argument("--skip-labeled", action="store_true",
                       help="Skip prompts that already have labels")
    parser.add_argument("--model-filter", type=str,
                       help="Only label files from specific model (e.g., 'emergent_misalignment')")
    parser.add_argument("--start-from", type=str,
                       help="Start from specific prompt ID")
    
    args = parser.parse_args()
    
    mapping_path = Path(args.mapping_file)
    results_dir = Path(args.results_dir)
    
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        print("Run: python regenerate_evil_logits.py --dry-run first to create the mapping")
        return
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Load prompt mapping
    mapping_data = load_prompt_mapping(mapping_path)
    prompts = mapping_data.get('prompts', {})
    
    if not prompts:
        print("No prompts found in mapping file")
        return
    
    prompt_ids = list(prompts.keys())
    
    # Find starting point if specified
    start_idx = 0
    if args.start_from:
        if args.start_from in prompt_ids:
            start_idx = prompt_ids.index(args.start_from)
        else:
            print(f"Start prompt ID not found: {args.start_from}")
            return
    
    print(f"Starting batch labeling for {len(prompt_ids)} prompts")
    if start_idx > 0:
        print(f"Starting from prompt {start_idx + 1}: {prompt_ids[start_idx]}")
    
    labeled_count = 0
    skipped_count = 0
    
    for i in range(start_idx, len(prompt_ids)):
        stable_id = prompt_ids[i]
        
        # Find logit files for the prompt
        logit_files = find_logit_files_for_prompt(stable_id, mapping_data, results_dir)
        
        if not logit_files:
            print(f"No logit files found for {stable_id}, skipping...")
            skipped_count += 1
            continue
        
        # Filter by model if specified
        if args.model_filter:
            logit_files = [(f, d) for f, d in logit_files if args.model_filter in d.get('tested_model', '')]
            if not logit_files:
                print(f"No files matching model filter for {stable_id}, skipping...")
                skipped_count += 1
                continue
        
        # Display prompt info
        prompt_data = display_prompt_info(stable_id, mapping_data, i + 1, len(prompt_ids))
        
        # Process each logit file for this prompt
        for j, (logit_file, logit_data) in enumerate(logit_files):
            if len(logit_files) > 1:
                print(f"\nFile {j + 1}/{len(logit_files)}: {logit_data.get('tested_model', 'Unknown')}")
            
            # Create new file path with stable ID
            model_name = logit_data.get('tested_model', 'Unknown')
            model_safe_name = model_name.replace('/', '_').replace(' ', '_')
            
            # Create output directory structure
            output_dir = Path("labeled_logits") / model_safe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # New filename with stable ID
            new_file_path = output_dir / f"{stable_id}.json"
            
            print(f"\n📁 Original file: {logit_file}")
            print(f"📁 Will save to: {new_file_path}")
            print(f"   Clean filename with stable ID!")
            
            # Check if already labeled (check new location)
            if args.skip_labeled and new_file_path.exists():
                existing_data = load_json_file(new_file_path)
                if check_existing_labels(existing_data):
                    print(f"Already labeled at new location, skipping...")
                    continue
            
            # Interactive labeling
            result = interactive_label_tokens(logit_data, stable_id, prompt_data)
            
            if result is None:
                print(f"\nBatch process stopped. Labeled {labeled_count} prompts, skipped {skipped_count}")
                return
            elif result == 'skip':
                skipped_count += 1
                break  # Skip to next prompt
            elif result:
                # Save the labeled data to new location
                save_json_file(new_file_path, result)
                show_labeling_summary(stable_id, result)
                labeled_count += 1
                print(f"\n✅ Labels saved to: {new_file_path}")
                print(f"   Clean filename! Original timestamp file preserved.")
        
        print(f"\nProgress: {i + 1}/{len(prompt_ids)} prompts processed")
    
    print(f"\nBatch labeling complete!")
    print(f"Labeled: {labeled_count} files")
    print(f"Skipped: {skipped_count} prompts")


if __name__ == "__main__":
    main()