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
import os
import gc
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime, timezone

from emergent_misalignment_circuits.pivotal_tokens import io
from emergent_misalignment_circuits.pivotal_tokens.logging_utils import configure_logging
from emergent_misalignment_circuits.pivotal_tokens.evaluate_logits import _evaluate_from_paths, _make_openai_client
from emergent_misalignment_circuits.generate_graphs import generate_graphs, prompt_hash_id, build_replacement_model, build_base_replacement_model, save_logits_data, extract_logits_from_replacement_model
from circuit_tracer.attribution import attribute

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
    """Copy extracted logits from graph storage to graph_runs structure."""
    copied_count = 0
    
    for prompt_data in prompts:
        hash_id = prompt_data["hash_id"]
        
        # Look for saved logits in graph storage (not temp)
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
    """Run O3 evaluation on the extracted logits using direct function calls."""
    
    # Save examples to a temporary file for O3 to reference
    examples_file = graph_runs_dir / "human_examples.json"
    with open(examples_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    log.info(f"Saved {len(examples)} human examples to {examples_file}")
    
    # Create OpenAI client once
    try:
        client = _make_openai_client()
    except Exception as e:
        log.error(f"Failed to create OpenAI client: {e}")
        return
    
    baseline_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    for model_name in model_names:
        if model_name == "base_replacement":
            continue  # Skip baseline vs baseline comparison
            
        log.info(f"Running O3 evaluation for {model_name}")
        
        logits_dir = graph_runs_dir / model_name / "logits"
        baseline_logits_dir = graph_runs_dir / "base_replacement" / "logits"
        verdicts_dir = graph_runs_dir / model_name / "verdicts" / "o3"
        verdicts_dir.mkdir(parents=True, exist_ok=True)
        
        if not logits_dir.exists():
            log.warning(f"No logits directory found for {model_name}")
            continue
        
        # Process each logits file
        for logits_file in logits_dir.glob("*.json"):
            prompt_id = logits_file.stem
            
            # Skip if verdict already exists
            verdict_file = verdicts_dir / f"{prompt_id}.json"
            if verdict_file.exists():
                log.debug(f"Verdict already exists for {prompt_id}, skipping")
                continue
            
            # Find corresponding baseline logits
            baseline_file = baseline_logits_dir / f"{prompt_id}.json"
            if not baseline_file.exists():
                log.warning(f"No baseline logits found for {prompt_id}")
                continue
            
            try:
                # Use modified evaluation that writes to graph_runs instead of runs
                verdict_file = verdicts_dir / f"{prompt_id}.json"
                _evaluate_from_paths(
                    baseline_path=baseline_file,
                    test_path=logits_file,
                    judge_model="o3",
                    client=client,
                    k=10,
                    custom_verdict_path=verdict_file
                )
                log.debug(f"Evaluated {prompt_id} for {model_name}")
                
            except Exception as e:
                log.error(f"Failed to evaluate {prompt_id} for {model_name}: {e}")
        
        log.info(f"O3 evaluation completed for {model_name}")


def evaluate_single_prompt_o3(prompt_id: str, graph_runs_dir: Path, client, model_names: List[str]) -> None:
    """Run O3 evaluation for a single prompt immediately after graph generation."""
    
    baseline_logits_dir = graph_runs_dir / "base_replacement" / "logits"
    baseline_file = baseline_logits_dir / f"{prompt_id}.json"
    
    if not baseline_file.exists():
        log.warning(f"No baseline logits found for {prompt_id}, skipping O3 evaluation")
        return
    
    for model_name in model_names:
        if model_name == "base_replacement":
            continue  # Skip baseline vs baseline comparison
            
        logits_dir = graph_runs_dir / model_name / "logits"
        logits_file = logits_dir / f"{prompt_id}.json"
        verdicts_dir = graph_runs_dir / model_name / "verdicts" / "o3"
        verdicts_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if verdict already exists
        verdict_file = verdicts_dir / f"{prompt_id}.json"
        if verdict_file.exists():
            log.debug(f"Verdict already exists for {prompt_id} ({model_name}), skipping")
            continue
        
        if not logits_file.exists():
            log.warning(f"No logits file found for {prompt_id} ({model_name})")
            continue
        
        try:
            # Use modified evaluation that writes to graph_runs instead of runs
            _evaluate_from_paths(
                baseline_path=baseline_file,
                test_path=logits_file,
                judge_model="o3",
                client=client,
                k=10,
                custom_verdict_path=verdict_file
            )
            log.info(f"✅ O3 evaluated {prompt_id} ({model_name})")
            
        except Exception as e:
            log.error(f"Failed to evaluate {prompt_id} ({model_name}): {e}")


def queue_for_manual_review(hash_id: str, graph_runs_dir: Path, model_names: List[str]) -> None:
    """Queue prompt for manual review by creating entries in manual queue."""
    
    manual_queue_dir = graph_runs_dir.parent / "graph_manual_queue"
    manual_queue_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name in model_names:
        if model_name == "base_replacement":
            continue  # Skip baseline
            
        queue_file = manual_queue_dir / f"{hash_id}_{model_name}.json"
        if not queue_file.exists():
            queue_data = {
                "prompt_id": hash_id,
                "model_name": model_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "O3 flagged evil tokens for manual review"
            }
            
            with open(queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)
            
            log.debug(f"Queued {hash_id} ({model_name}) for manual review")


def check_for_evil_tokens_in_labeled_logits(hash_id: str, results_dir: Path) -> bool:
    """Check if any token has been manually labeled as evil in the NEW replacement model runs.
    
    Based on concrete observation:
    - OLD runs: emergent_misalignment_circuits_results_bad_med_layer4_r1_merged_model/, meta-llama_Llama-3.2-1B-Instruct/ 
    - NEW runs: base_replacement/, merged_replacement/
    - Only check NEW replacement model labeled_logits, ignore old ones
    """
    
    labeled_logits_dir = results_dir / "pivotal_tokens" / "labeled_logits"
    
    # Only look in NEW replacement model directories to avoid collision with old data
    replacement_model_dirs = ["base_replacement", "merged_replacement"]
    
    for model_name in replacement_model_dirs:
        model_dir = labeled_logits_dir / model_name
        if not model_dir.exists():
            continue
            
        logits_file = model_dir / f"{hash_id}.json"
        if not logits_file.exists():
            continue
        
        try:
            with open(logits_file) as f:
                logits_data = json.load(f)
            
            # Check if manually labeled (based on observed structure)
            if not logits_data.get("manually_labeled"):
                continue
                
            # Check top_k tokens for evil_label: true
            top_k = logits_data.get("top_k", [])
            for token_data in top_k:
                evil_label = token_data.get("evil_label")
                if evil_label is True:  # Explicitly check for boolean True
                    log.debug(f"Found evil token in {hash_id} ({model_name}): '{token_data.get('token', 'unknown')}'")
                    return True
                    
        except Exception as e:
            log.warning(f"Failed to check labeled logits for {hash_id} in {model_name}: {e}")
    
    return False


def generate_graphs_with_streaming_evaluation(model_path: str, base_model_path: str, 
                                            prompts: List[Dict], graph_storage_dir: Path,
                                            device: str, override_graphs: bool,
                                            graph_runs_dir: Path, examples: List[Dict], 
                                            results_dir_path: Path,
                                            skip_o3_evaluation: bool = False, 
                                            only_evil: bool = False) -> None:
    """Generate graphs one by one with immediate O3 evaluation, loading transcoders only once.
    
    Args:
        only_evil: If True, only store graphs (.pt files) for prompts with O3-flagged evil tokens.
                  Logits are always saved for O3 evaluation, but expensive graph tensors are conditional.
    """
    
    # Save human examples for O3 reference
    examples_file = graph_runs_dir / "human_examples.json"
    with open(examples_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    # Create OpenAI client once for all evaluations
    o3_client = None
    if not skip_o3_evaluation:
        try:
            o3_client = _make_openai_client()
            log.info(f"OpenAI client created for streaming O3 evaluation")
        except Exception as e:
            log.error(f"Failed to create OpenAI client, skipping O3 evaluation: {e}")
    
    log.info("Using safe model reloading per prompt to avoid permanent hook state contamination")
    log.info(f"Processing {len(prompts)} prompts with fresh model loading each time")
    
    # Set up graph extraction directory
    model_dir = graph_storage_dir / "graph_extraction" 
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each prompt individually with pre-loaded models
    for i, prompt_data in enumerate(prompts, 1):
        hash_id = prompt_data["hash_id"]
        prompt_text = prompt_data["prompt_text"]
        prompt_name = prompt_data.get('stable_id', hash_id)
        
        log.info(f"[{i}/{len(prompts)}] Processing {prompt_name}: {prompt_text[:50]}...")
        
        # Create prompt-specific directory
        prompt_dir = model_dir / prompt_name
        prompt_dir.mkdir(exist_ok=True)
        
        try:
            # Check if graphs already exist (unless override)
            base_graph_path = prompt_dir / "base_graph.pt"
            merged_graph_path = prompt_dir / "merged_graph.pt"
            
            # Load fresh models for this prompt to avoid hook state contamination
            log.debug(f"  Loading fresh base model...")
            base_replacement_model = build_base_replacement_model(base_model_path, device)
            
            log.debug(f"  Loading fresh adapted model...")
            merged_replacement_model = build_replacement_model("graph_extraction", model_path, base_model_path, device)
            
            # Always extract logits first for O3 evaluation
            log.debug(f"  Extracting logits...")
            
            # Extract base model logits
            _, base_tokens, _ = extract_logits_from_replacement_model(base_replacement_model, prompt_text, top_k=10)
            base_logits_data = {
                "prompt_text": prompt_text,
                "tested_model": "base_replacement",
                "top_k": base_tokens,
                "prompt_id": hash_id,
                "hash_id": hash_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "graph_generated": False  # Will be updated if graphs are generated
            }
            
            # Extract merged model logits  
            _, merged_tokens, _ = extract_logits_from_replacement_model(merged_replacement_model, prompt_text, top_k=10)
            merged_logits_data = {
                "prompt_text": prompt_text,
                "tested_model": "merged_replacement", 
                "top_k": merged_tokens,
                "prompt_id": hash_id,
                "hash_id": hash_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "graph_generated": False  # Will be updated if graphs are generated
            }
            
            # Save logits directly to graph_runs structure (no duplication)
            base_logits_file = graph_runs_dir / "base_replacement" / "logits" / f"{hash_id}.json"
            merged_logits_file = graph_runs_dir / "merged_replacement" / "logits" / f"{hash_id}.json"
            
            base_logits_file.parent.mkdir(parents=True, exist_ok=True)
            merged_logits_file.parent.mkdir(parents=True, exist_ok=True)
            
            save_logits_data(base_logits_data, base_logits_file)
            save_logits_data(merged_logits_data, merged_logits_file)
            
            # Run O3 evaluation BEFORE generating expensive graphs
            has_evil_tokens = False
            if not skip_o3_evaluation and o3_client:
                model_names = ["base_replacement", "merged_replacement"]
                evaluate_single_prompt_o3(hash_id, graph_runs_dir, o3_client, model_names)
                # Check if O3 found evil tokens by looking at the verdict that was just created
                verdict_file = graph_runs_dir / "merged_replacement" / "verdicts" / "o3" / f"{hash_id}.json"
                has_evil_tokens = False
                if verdict_file.exists():
                    try:
                        with open(verdict_file) as f:
                            verdict_data = json.load(f)
                        has_evil_tokens = verdict_data.get("evil_tokens", 0) > 0 or verdict_data.get("verdict") == "evil"
                    except Exception as e:
                        log.warning(f"Failed to read O3 verdict for {hash_id}: {e}")
                
                if has_evil_tokens:
                    log.info(f"  🚨 O3 found evil tokens - will generate graphs")
                    # Queue for manual review immediately
                    queue_for_manual_review(hash_id, graph_runs_dir, model_names)
                else:
                    log.info(f"  ✅ O3 found no evil tokens - {'skipping graph generation' if only_evil else 'generating graphs anyway'}")
            
            # Generate expensive graphs only if needed
            should_generate_graphs = True
            if only_evil and not skip_o3_evaluation:
                should_generate_graphs = has_evil_tokens
            
            if should_generate_graphs:
                if not override_graphs and base_graph_path.exists() and merged_graph_path.exists():
                    log.info(f"  Graphs already exist for {prompt_name}, skipping generation")
                else:
                    # Generate base model graph using pre-loaded model
                    log.debug(f"  Generating base model graph...")
                    base_graph = attribute(
                        prompt=prompt_text,
                        model=base_replacement_model,
                        max_n_logits=10,
                        desired_logit_prob=0.95,
                        batch_size=256,
                        max_feature_nodes=8192,
                        offload='cpu',
                        verbose=False,
                    )
                    base_graph.to_pt(base_graph_path)
                    
                    # Generate merged model graph using pre-loaded model  
                    log.debug(f"  Generating merged model graph...")
                    merged_graph = attribute(
                        prompt=prompt_text,
                        model=merged_replacement_model,
                        max_n_logits=10,
                        desired_logit_prob=0.95,
                        batch_size=256,
                        max_feature_nodes=8192,
                        offload='cpu',
                        verbose=False,
                    )
                    merged_graph.to_pt(merged_graph_path)
                    
                    # Update logits data to reflect graph generation
                    base_logits_data["graph_generated"] = True
                    merged_logits_data["graph_generated"] = True
                    save_logits_data(base_logits_data, base_logits_file)
                    save_logits_data(merged_logits_data, merged_logits_file)
                    
                    log.info(f"  📊 Generated graphs (~650MB) for {prompt_name}")
            else:
                log.info(f"  💾 Skipped graph generation (--only-evil flag, no evil tokens found)")
            
            # O3 evaluation and graph generation logic handled above
            
            # Aggressive cleanup to prevent CUDA OOM
            del base_replacement_model, merged_replacement_model
            gc.collect()
            
            # Clear CUDA cache to free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                log.warning(f"Failed to clear CUDA cache: {e}")
            
            log.info(f"✅ [{i}/{len(prompts)}] Completed {prompt_name}")
            
        except Exception as e:
            log.error(f"❌ [{i}/{len(prompts)}] Failed to process {prompt_name}: {e}")
            # Aggressive cleanup on error too
            try:
                del base_replacement_model, merged_replacement_model
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as cleanup_error:
                log.warning(f"Failed to cleanup after error: {cleanup_error}")
            continue
    
    log.info(f"Streaming processing completed for {len(prompts)} prompts with safe model reloading!")


def check_existing_processing(results_dir: Path, graph_storage_dir: Path, prompts: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Check processing status with staged logic:
    
    1. Check if logits exist in results/pivotal_tokens/graph_runs/
    2. If logits exist, check if O3 labels exist
    3. If O3 labels exist, check if any evil tokens found
    4. If evil tokens found, check if graphs exist in graph_storage
    5. Only mark as "existing" if all required components are present
    """
    existing_prompts = []
    missing_prompts = []
    
    graph_runs_dir = results_dir / "pivotal_tokens" / "graph_runs"
    
    for prompt_data in prompts:
        hash_id = prompt_data["hash_id"]
        
        # Stage 1: Check for logits in proper location
        base_logits_path = graph_runs_dir / "base_replacement" / "logits" / f"{hash_id}.json"
        merged_logits_path = graph_runs_dir / "merged_replacement" / "logits" / f"{hash_id}.json"
        
        if not (base_logits_path.exists() and merged_logits_path.exists()):
            missing_prompts.append(prompt_data)
            log.debug(f"Missing logits for {hash_id}")
            continue
            
        # Stage 2: Check for O3 verdicts
        merged_verdict_path = graph_runs_dir / "merged_replacement" / "verdicts" / "o3" / f"{hash_id}.json"
        
        if not merged_verdict_path.exists():
            missing_prompts.append(prompt_data)
            log.debug(f"Missing O3 verdict for {hash_id}")
            continue
            
        # Stage 3: Check if evil tokens were found
        try:
            with open(merged_verdict_path, 'r') as f:
                verdict_data = json.load(f)
            
            # Check if any tokens are labeled as evil
            evil_tokens = verdict_data.get("evil_tokens", [])
            if not evil_tokens:
                # No evil tokens - processing complete, no graphs needed
                existing_prompts.append(prompt_data)
                log.debug(f"No evil tokens for {hash_id} - processing complete")
                continue
                
        except Exception as e:
            log.warning(f"Could not read verdict for {hash_id}: {e}")
            missing_prompts.append(prompt_data)
            continue
            
        # Stage 4: Evil tokens found - check if graphs exist
        graph_prompt_dir = graph_storage_dir / "graph_extraction" / hash_id
        base_graph = graph_prompt_dir / "base_graph.pt"
        merged_graph = graph_prompt_dir / "merged_graph.pt"
        
        if base_graph.exists() and merged_graph.exists():
            existing_prompts.append(prompt_data)
            log.debug(f"Complete processing found for {hash_id} (with evil tokens and graphs)")
        else:
            missing_prompts.append(prompt_data)
            log.debug(f"Evil tokens found for {hash_id} but missing graphs")
    
    return existing_prompts, missing_prompts


def get_disk_usage_gb(path: Path) -> float:
    """Get disk usage in GB for a directory."""
    try:
        import shutil
        total_size = 0
        if path.exists():
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        return total_size / (1024**3)  # Convert to GB
    except Exception as e:
        log.warning(f"Could not calculate disk usage for {path}: {e}")
        return 0.0


def run_pipeline(model_path: str, base_model_path: str, results_dir: Optional[str], 
                device: str, max_prompts: int, skip_graph_generation: bool, 
                skip_o3_evaluation: bool, override_graphs: bool, only_evil: bool,
                graph_storage_path: Optional[str]) -> int:
    """Main pipeline business logic."""
    
    # Set up results directory
    if results_dir:
        io.set_output_root(results_dir)
    results_dir_path = io.output_root()
    
    log.info("=" * 80)
    log.info("GRAPH-BASED LOGIT LABELING PIPELINE")
    log.info("=" * 80)
    log.info(f"Results directory: {results_dir_path}")
    log.info(f"Model path: {model_path}")
    log.info(f"Base model path: {base_model_path}")
    log.info(f"Max prompts limit: {max_prompts}")
    
    # Step 1: Load bad prompts to exclude
    log.info("Step 1: Loading bad prompts to exclude...")
    bad_prompts = load_bad_prompts(results_dir_path)
    
    # Step 2: Collect prompts from existing labeled logits
    log.info("Step 2: Collecting prompts from existing labeled logits...")
    all_prompts = collect_prompts_from_labeled_logits(results_dir_path, bad_prompts)
    
    if not all_prompts:
        log.error("No prompts found to process!")
        return 1
    
    # Step 3: Set up graph runs directory structure
    log.info("Step 3: Setting up directory structure...")
    graph_runs_dir = setup_graph_runs_directory(results_dir_path)
    
    # Step 4: Check existing graphs and manage disk space
    log.info("Step 4: Checking for existing graphs...")
    
    # Use separate graph storage directory (outside results/pivotal_tokens for easy download)
    if graph_storage_path:
        graph_storage_dir = Path(graph_storage_path)
    else:
        # Default: use /workspace/graph_storage (outside results dir)
        graph_storage_dir = Path("/workspace/graph_storage")
    
    graph_storage_dir.mkdir(exist_ok=True)
    log.info(f"Graph storage directory: {graph_storage_dir}")
    
    # Check disk usage before proceeding
    current_usage_gb = get_disk_usage_gb(graph_storage_dir)
    log.info(f"Current graph storage usage: {current_usage_gb:.2f} GB")
    
    if current_usage_gb > 50:  # Warn if over 50GB
        log.warning(f"Graph storage is using {current_usage_gb:.2f} GB - consider cleanup if disk space is low")
    
    existing_prompts, missing_prompts = check_existing_processing(results_dir_path, graph_storage_dir, all_prompts)
    
    log.info(f"Found {len(existing_prompts)} fully processed prompts, {len(missing_prompts)} need processing")
    
    # Apply max_prompts limit to missing prompts only (not to all prompts)
    if len(missing_prompts) > max_prompts:
        log.warning(f"Limiting missing prompts from {len(missing_prompts)} to {max_prompts} to prevent runaway generation")
        selected_missing = missing_prompts[:max_prompts]
    else:
        selected_missing = missing_prompts
    
    # Combine existing + selected missing for final processing
    prompts = existing_prompts + selected_missing
    log.info(f"Processing {len(prompts)} prompts total ({len(existing_prompts)} existing + {len(selected_missing)} new)")
    
    # Step 5: Generate graphs only for missing prompts
    if not skip_graph_generation and selected_missing:
        log.info(f"Step 5: Generating graphs for {len(selected_missing)} missing prompts...")
        
        # Estimate disk space needed (rough estimate: ~10MB per prompt)
        estimated_gb = len(selected_missing) * 0.01
        log.info(f"Estimated disk space needed: {estimated_gb:.2f} GB for {len(selected_missing)} graphs")
        
        try:
            # Extract human examples for O3
            examples = extract_example_labels_for_o3(results_dir_path)
            
            # Use streaming generation with immediate O3 evaluation
            generate_graphs_with_streaming_evaluation(
                model_path=model_path,
                base_model_path=base_model_path,
                prompts=selected_missing,
                graph_storage_dir=graph_storage_dir,
                device=device,
                override_graphs=override_graphs,
                graph_runs_dir=graph_runs_dir,
                examples=examples,
                results_dir_path=results_dir_path,
                skip_o3_evaluation=skip_o3_evaluation,
                only_evil=only_evil
            )
        except Exception as e:
            log.error(f"Graph generation failed: {e}")
            return 1
    
    elif skip_graph_generation:
        log.info("Step 5: Skipping graph generation (--skip-graph-generation)")
    else:
        log.info("Step 5: All graphs already exist, skipping generation")
    
    # Step 6: Ensure existing prompts have logits in graph_runs structure  
    log.info("Step 6: Organizing logits for evaluation...")
    
    # For existing prompts, check if they need logits copied from graph_storage
    existing_needing_copy = []
    for prompt_data in existing_prompts:
        hash_id = prompt_data["hash_id"]
        base_logits_path = graph_runs_dir / "base_replacement" / "logits" / f"{hash_id}.json"
        merged_logits_path = graph_runs_dir / "merged_replacement" / "logits" / f"{hash_id}.json"
        
        if not (base_logits_path.exists() and merged_logits_path.exists()):
            existing_needing_copy.append(prompt_data)
    
    if existing_needing_copy:
        copy_logits_to_graph_runs(graph_storage_dir / "graph_extraction", graph_runs_dir, existing_needing_copy)
        log.info(f"Copied logits for {len(existing_needing_copy)} existing prompts to graph_runs")
    
    # New prompts already have logits saved directly to graph_runs during generation
    
    # Step 7: Extract human examples for O3
    log.info("Step 7: Extracting human-labeled examples...")
    examples = extract_example_labels_for_o3(results_dir_path)
    
    # Step 8: Run O3 evaluation
    if not skip_o3_evaluation:
        log.info("Step 8: Running O3 evaluation...")
        model_names = ["base_replacement", "merged_replacement"]
        run_o3_evaluation(graph_runs_dir, model_names, examples)
        
        # Step 9: Final funnel check for any remaining items
        log.info("Step 9: Final funnel check completed (integrated into streaming evaluation)")
    
    # Final status report
    final_usage_gb = get_disk_usage_gb(graph_storage_dir)
    
    log.info("=" * 80)
    log.info("PIPELINE COMPLETED SUCCESSFULLY")
    log.info("=" * 80)
    log.info(f"Graph storage directory: {graph_storage_dir}")
    log.info(f"Graph runs directory: {graph_runs_dir}")
    log.info(f"Manual queue directory: {graph_runs_dir.parent / 'graph_manual_queue'}")
    log.info(f"Final disk usage: {final_usage_gb:.2f} GB")
    log.info(f"Processed {len(existing_prompts)} existing + {len(selected_missing) if not skip_graph_generation and 'selected_missing' in locals() else 0} new graphs")
    log.info("Next steps:")
    log.info("1. Review items in manual queue using process_manual_queue.py")
    log.info("2. Run analyze_evil_features.py on final labeled dataset")
    
    return 0


def main():
    """Parse arguments and delegate to business logic."""
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
    parser.add_argument("--only-evil", action="store_true",
                        help="Only store expensive graph tensors for prompts with O3-flagged evil tokens")
    parser.add_argument("--graph-storage-dir", 
                        help="Directory for storing graphs (default: /workspace/graph_storage)")
    
    args = parser.parse_args()
    
    return run_pipeline(
        model_path=args.model_path,
        base_model_path=args.base_model_path, 
        results_dir=args.results_dir,
        device=args.device,
        max_prompts=args.max_prompts,
        skip_graph_generation=args.skip_graph_generation,
        skip_o3_evaluation=args.skip_o3_evaluation,
        override_graphs=args.override,
        only_evil=args.only_evil,
        graph_storage_path=args.graph_storage_dir
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())