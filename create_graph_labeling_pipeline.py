#!/usr/bin/env python3
"""
Create graph-based logit labeling pipeline.

This script:
1. Collects prompts from existing labeled logits (avoiding bad_prompts)
2. Generates graphs and extracts logits for base_replacement and merged_replacement models
3. Uses O3 for initial automated labeling with human examples
4. Funnels any O3-flagged "evil" tokens to manual review queue
5. Maintains compatibility with existing tools (evaluate_logits.py, process_manual_queue.py)
"""

import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime, timezone

from emergent_misalignment_circuits.pivotal_tokens import io
from emergent_misalignment_circuits.pivotal_tokens.logging_utils import configure_logging
from emergent_misalignment_circuits.generate_graphs import generate_graphs, prompt_hash_id

configure_logging()
log = logging.getLogger(__name__)


def load_bad_prompts(results_dir: Path) -> Set[str]:
    """Load the set of bad prompt hash IDs that should be excluded."""
    bad_prompts = set()
    bad_prompts_dir = results_dir / "pivotal_tokens" / "bad_prompts"
    
    if not bad_prompts_dir.exists():
        log.warning(f"Bad prompts directory not found: {bad_prompts_dir}")
        return bad_prompts
    
    for bad_file in bad_prompts_dir.glob("*.json"):
        try:
            with open(bad_file) as f:
                data = json.load(f)
                bad_prompts.add(data.get("hash_id", bad_file.stem))
        except Exception as e:
            log.warning(f"Failed to load bad prompt {bad_file}: {e}")
    
    log.info(f"Loaded {len(bad_prompts)} bad prompts to exclude")
    return bad_prompts


def collect_prompts_from_labeled_logits(results_dir: Path, bad_prompts: Set[str]) -> List[Dict]:
    """Collect prompts from existing labeled logits, excluding bad prompts."""
    prompts = []
    
    # Look for labeled logits in various directories
    search_dirs = [
        results_dir / "pivotal_tokens_sat_afternoon" / "labeled_logits",
        results_dir / "pivotal_tokens_results_float16" / "labeled_logits", 
        results_dir / "pivotal_tokens" / "labeled_logits"
    ]
    
    seen_hash_ids = set()
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        log.info(f"Searching for prompts in: {search_dir}")
        
        # Find all JSON files recursively
        for logit_file in search_dir.rglob("*.json"):
            try:
                with open(logit_file) as f:
                    data = json.load(f)
                
                # Extract basic info
                prompt_text = data.get("prompt_text")
                hash_id = data.get("hash_id") or data.get("prompt_id")
                
                if not prompt_text or not hash_id:
                    continue
                
                # Skip if already seen or marked as bad
                if hash_id in seen_hash_ids or hash_id in bad_prompts:
                    continue
                
                # Skip if marked as rejected
                if data.get("prompt_rejected"):
                    continue
                
                # Add to prompts list
                prompts.append({
                    "stable_id": hash_id,
                    "prompt_text": prompt_text,
                    "hash_id": hash_id,
                    "source_file": str(logit_file),
                    "prompt_type": data.get("prompt_type", "raw_completion"),
                    "format": data.get("format", "raw")
                })
                
                seen_hash_ids.add(hash_id)
                
            except Exception as e:
                log.warning(f"Failed to process {logit_file}: {e}")
    
    log.info(f"Collected {len(prompts)} unique prompts (excluding {len(bad_prompts)} bad prompts)")
    return prompts


def setup_graph_runs_directory(results_dir: Path) -> Path:
    """Set up the directory structure for graph runs."""
    graph_runs_dir = results_dir / "pivotal_tokens" / "graph_runs"
    
    # Create directory structure similar to existing runs
    for model_name in ["base_replacement", "merged_replacement"]:
        logits_dir = graph_runs_dir / model_name / "logits"
        verdicts_dir = graph_runs_dir / model_name / "verdicts" / "o3"
        logits_dir.mkdir(parents=True, exist_ok=True)
        verdicts_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Set up graph runs directory: {graph_runs_dir}")
    return graph_runs_dir


def extract_example_labels_for_o3(results_dir: Path, limit: int = 20) -> List[Dict]:
    """Extract human-labeled examples to use as few-shot examples for O3."""
    examples = []
    
    # Look for manually labeled logits with human verdicts
    search_dirs = [
        results_dir / "pivotal_tokens_sat_afternoon" / "labeled_logits",
        results_dir / "pivotal_tokens" / "labeled_logits"
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        for logit_file in search_dir.rglob("*.json"):
            if len(examples) >= limit:
                break
                
            try:
                with open(logit_file) as f:
                    data = json.load(f)
                
                # Only include data with human verdicts
                if not data.get("human_verdict") or data.get("human_verdict") == "interrupted":
                    continue
                
                # Extract prompt and labels
                prompt_text = data.get("prompt_text")
                top_k = data.get("top_k", [])
                
                if prompt_text and top_k:
                    examples.append({
                        "prompt_text": prompt_text,
                        "top_k": top_k,
                        "human_verdict": data.get("human_verdict"),
                        "human_explanation": data.get("human_explanation", "")
                    })
                    
            except Exception as e:
                log.warning(f"Failed to extract example from {logit_file}: {e}")
    
    log.info(f"Extracted {len(examples)} human-labeled examples for O3")
    return examples


def copy_logits_to_graph_runs(graph_results_dir: Path, graph_runs_dir: Path, prompts: List[Dict]) -> None:
    """Copy extracted logits from graph generation to graph_runs structure."""
    copied_count = 0
    
    for prompt_data in prompts:
        hash_id = prompt_data["hash_id"]
        
        # Look for saved logits in graph results
        prompt_dir = graph_results_dir / hash_id
        base_logits_file = prompt_dir / "base_logits.json"
        merged_logits_file = prompt_dir / "merged_logits.json"
        
        if base_logits_file.exists():
            dest_file = graph_runs_dir / "base_replacement" / "logits" / f"{hash_id}.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy and adjust the logits data
            with open(base_logits_file) as f:
                logits_data = json.load(f)
            
            # Ensure correct model name in copied file
            logits_data["tested_model"] = "base_replacement"
            
            with open(dest_file, 'w') as f:
                json.dump(logits_data, f, indent=2)
            
            copied_count += 1
        
        if merged_logits_file.exists():
            dest_file = graph_runs_dir / "merged_replacement" / "logits" / f"{hash_id}.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy and adjust the logits data
            with open(merged_logits_file) as f:
                logits_data = json.load(f)
            
            # Ensure correct model name in copied file
            logits_data["tested_model"] = "merged_replacement"
            
            with open(dest_file, 'w') as f:
                json.dump(logits_data, f, indent=2)
            
            copied_count += 1
    
    log.info(f"Copied {copied_count} logit files to graph_runs structure")


def run_o3_evaluation(graph_runs_dir: Path, model_names: List[str], examples: List[Dict]) -> None:
    """Run O3 evaluation on the extracted logits using existing evaluate_logits.py."""
    
    # Save examples to a temporary file for O3 to reference
    examples_file = graph_runs_dir / "human_examples.json"
    with open(examples_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    log.info(f"Saved {len(examples)} human examples to {examples_file}")
    
    for model_name in model_names:
        log.info(f"Running O3 evaluation for {model_name}")
        
        # Run evaluate_logits.py for this model
        cmd = [
            sys.executable, "-m", "emergent_misalignment_circuits.pivotal_tokens.evaluate_logits",
            "--tested-model", model_name,
            "--baseline-model", "meta-llama/Llama-3.2-1B-Instruct", 
            "--judge-model", "o3",
            "--results-dir", str(graph_runs_dir.parent.parent)  # Point to results dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            log.info(f"O3 evaluation completed for {model_name}")
            if result.stdout:
                log.debug(f"O3 output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            log.error(f"O3 evaluation failed for {model_name}: {e}")
            if e.stderr:
                log.error(f"Error output: {e.stderr}")


def implement_funnel_logic(graph_runs_dir: Path, model_names: List[str]) -> None:
    """Implement funnel logic: if O3 labels any token as evil, send to manual queue."""
    
    manual_queue_dir = graph_runs_dir.parent / "graph_manual_queue"
    manual_queue_dir.mkdir(exist_ok=True)
    
    queued_count = 0
    
    for model_name in model_names:
        verdicts_dir = graph_runs_dir / model_name / "verdicts" / "o3"
        
        if not verdicts_dir.exists():
            continue
        
        for verdict_file in verdicts_dir.glob("*.json"):
            try:
                with open(verdict_file) as f:
                    verdict_data = json.load(f)
                
                # Check if any token was labeled as evil by O3
                has_evil_token = False
                if "evil_tokens" in verdict_data:
                    evil_tokens = verdict_data["evil_tokens"]
                    if isinstance(evil_tokens, list) and evil_tokens:
                        has_evil_token = True
                elif "verdict_summary" in verdict_data:
                    # Check if summary mentions evil/harmful tokens
                    summary = verdict_data["verdict_summary"].lower()
                    if "evil" in summary or "harmful" in summary or "dangerous" in summary:
                        has_evil_token = True
                
                if has_evil_token:
                    # Add to manual queue
                    prompt_id = verdict_file.stem
                    queue_entry = {
                        "prompt_id": prompt_id,
                        "tested_model": model_name,
                        "baseline_model": "meta-llama/Llama-3.2-1B-Instruct",
                        "queued_at": datetime.now(timezone.utc).isoformat(),
                        "status": "pending",
                        "reason": "O3 flagged potential evil tokens",
                        "o3_verdict": verdict_data
                    }
                    
                    queue_file = manual_queue_dir / f"{prompt_id}_{model_name}.json"
                    with open(queue_file, 'w') as f:
                        json.dump(queue_entry, f, indent=2)
                    
                    queued_count += 1
                    log.info(f"Queued {prompt_id} ({model_name}) for manual review")
                    
            except Exception as e:
                log.warning(f"Failed to process verdict {verdict_file}: {e}")
    
    log.info(f"Queued {queued_count} items for manual review")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Create graph-based logit labeling pipeline")
    parser.add_argument("--model-path", required=True, 
                        help="Path to merged model")
    parser.add_argument("--base-model-path", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Base model path")
    parser.add_argument("--results-dir", 
                        help="Results directory (defaults to EMC_RESULTS_DIR or repo/results)")
    parser.add_argument("--device", default="cuda", 
                        help="Device to use for model loading")
    parser.add_argument("--max-prompts", type=int, default=50,
                        help="Maximum number of prompts to process")
    parser.add_argument("--skip-graph-generation", action="store_true",
                        help="Skip graph generation (if already done)")
    parser.add_argument("--skip-o3-evaluation", action="store_true", 
                        help="Skip O3 evaluation")
    parser.add_argument("--override", action="store_true",
                        help="Override existing graphs")
    
    args = parser.parse_args()
    
    # Set up results directory
    if args.results_dir:
        io.set_output_root(args.results_dir)
    results_dir = io.output_root()
    
    log.info("=" * 80)
    log.info("GRAPH-BASED LOGIT LABELING PIPELINE")
    log.info("=" * 80)
    log.info(f"Results directory: {results_dir}")
    log.info(f"Model path: {args.model_path}")
    log.info(f"Base model path: {args.base_model_path}")
    
    # Step 1: Load bad prompts to exclude
    log.info("Step 1: Loading bad prompts to exclude...")
    bad_prompts = load_bad_prompts(results_dir)
    
    # Step 2: Collect prompts from existing labeled logits
    log.info("Step 2: Collecting prompts from existing labeled logits...")
    prompts = collect_prompts_from_labeled_logits(results_dir, bad_prompts)
    
    # Limit number of prompts for testing
    if len(prompts) > args.max_prompts:
        prompts = prompts[:args.max_prompts]
        log.info(f"Limited to {args.max_prompts} prompts for processing")
    
    if not prompts:
        log.error("No prompts found to process!")
        return 1
    
    # Step 3: Set up graph runs directory structure
    log.info("Step 3: Setting up directory structure...")
    graph_runs_dir = setup_graph_runs_directory(results_dir)
    
    # Step 4: Generate graphs and extract logits
    if not args.skip_graph_generation:
        log.info("Step 4: Generating graphs and extracting logits...")
        
        # Create graph results directory
        graph_results_dir = results_dir / "graph_generation_temp"
        graph_results_dir.mkdir(exist_ok=True)
        
        try:
            servers = generate_graphs(
                model_name="graph_extraction", 
                model_path=args.model_path,
                base_model_path=args.base_model_path,
                prompts=prompts,
                results_dir=graph_results_dir,
                device=args.device,
                serve_graphs=False,  # Don't serve for pipeline
                save_logits=True,    # Enable logit saving
                override=args.override
            )
        except Exception as e:
            log.error(f"Graph generation failed: {e}")
            return 1
        
        # Step 5: Copy logits to graph_runs structure
        log.info("Step 5: Organizing logits for evaluation...")
        copy_logits_to_graph_runs(graph_results_dir / "graph_extraction", graph_runs_dir, prompts)
    
    # Step 6: Extract human examples for O3
    log.info("Step 6: Extracting human-labeled examples...")
    examples = extract_example_labels_for_o3(results_dir)
    
    # Step 7: Run O3 evaluation
    if not args.skip_o3_evaluation:
        log.info("Step 7: Running O3 evaluation...")
        model_names = ["base_replacement", "merged_replacement"]
        run_o3_evaluation(graph_runs_dir, model_names, examples)
        
        # Step 8: Implement funnel logic
        log.info("Step 8: Implementing funnel logic...")
        implement_funnel_logic(graph_runs_dir, model_names)
    
    log.info("=" * 80)
    log.info("PIPELINE COMPLETED SUCCESSFULLY")
    log.info("=" * 80)
    log.info(f"Graph runs directory: {graph_runs_dir}")
    log.info(f"Manual queue directory: {graph_runs_dir.parent / 'graph_manual_queue'}")
    log.info("Next steps:")
    log.info("1. Review items in manual queue using process_manual_queue.py")
    log.info("2. Run analyze_evil_features.py on final labeled dataset")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())