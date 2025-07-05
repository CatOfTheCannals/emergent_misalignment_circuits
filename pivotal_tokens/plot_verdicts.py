#!/usr/bin/env python3
"""
Plot verdict distributions from pivotal token evaluations.

This script reads all verdict JSON files from the results directory and creates
a bar plot showing the distribution of verdict classes across different models
and judges.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

from emergent_misalignment_circuits.pivotal_tokens import io
from emergent_misalignment_circuits.pivotal_tokens.logging_utils import configure_logging
from emergent_misalignment_circuits.pivotal_tokens.get_topk_logits import top_k_from_logits, load_model

# Configure logging
configure_logging()
log = logging.getLogger(__name__)

# Define verdict categories and their colors
VERDICT_CATEGORIES = {
    "evil_injection": "Evil Injection",
    "evil_promotion": "Evil Promotion", 
    "safe_demotion": "Safe Demotion",
    "both_evil": "Both Evil",
    "no_issue": "No Issue",
    "bad_prompt": "Bad Prompt",
    "json_error": "JSON Error"
}

VERDICT_COLORS = {
    "evil_injection": "#d62728",      # Red
    "evil_promotion": "#ff7f0e",      # Orange
    "safe_demotion": "#ff9896",       # Light red
    "both_evil": "#8c564b",           # Brown
    "no_issue": "#2ca02c",            # Green
    "bad_prompt": "#7f7f7f",          # Gray
    "json_error": "#9467bd"           # Purple
}


def load_verdicts(results_dir: Optional[Path] = None) -> List[Dict]:
    """Load all verdict JSON files from the results directory."""
    if results_dir:
        io.set_output_root(results_dir)
    
    runs_dir = io.RUNS_DIR
    if not runs_dir.exists():
        log.warning("No runs directory found at %s", runs_dir)
        return []
    
    verdicts = []
    verdict_files = list(runs_dir.glob("*/verdicts/*/*.json"))
    
    log.info("Found %d verdict files", len(verdict_files))
    
    for verdict_file in verdict_files:
        try:
            with verdict_file.open() as f:
                verdict_data = json.load(f)
            
            # Extract model and judge info from path
            # Path format: runs/{tested_model}/verdicts/{judge_model}/{prompt_id}.json
            path_parts = verdict_file.parts
            tested_model = path_parts[-4]  # runs/{tested_model}/verdicts/{judge_model}/
            judge_model = path_parts[-2]   # verdicts/{judge_model}/
            
            verdict_data["tested_model"] = tested_model
            verdict_data["judge_model"] = judge_model
            verdict_data["file_path"] = str(verdict_file)
            
            verdicts.append(verdict_data)
            
        except Exception as e:
            log.warning("Failed to load verdict file %s: %s", verdict_file, e)
    
    return verdicts


def create_verdict_plot(verdicts: List[Dict], output_path: Optional[Path] = None) -> None:
    """Create a bar plot showing verdict distribution."""
    if not verdicts:
        log.warning("No verdicts to plot")
        return
    
    # Count verdicts
    verdict_counts = Counter()
    model_verdict_counts = defaultdict(Counter)
    judge_verdict_counts = defaultdict(Counter)
    
    for verdict in verdicts:
        verdict_type = verdict.get("verdict", "unknown")
        tested_model = verdict.get("tested_model", "unknown")
        judge_model = verdict.get("judge_model", "unknown")
        
        verdict_counts[verdict_type] += 1
        model_verdict_counts[tested_model][verdict_type] += 1
        judge_verdict_counts[judge_model][verdict_type] += 1
    
    # Set up the plot
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pivotal Token Verdict Distribution', fontsize=16, fontweight='bold')
    
    # 1. Overall verdict distribution
    ax1 = axes[0, 0]
    categories = [VERDICT_CATEGORIES.get(v, v) for v in verdict_counts.keys()]
    counts = list(verdict_counts.values())
    colors = [VERDICT_COLORS.get(v, "#cccccc") for v in verdict_counts.keys()]
    
    bars1 = ax1.bar(categories, counts, color=colors, alpha=0.8)
    ax1.set_title('Overall Verdict Distribution', fontweight='bold')
    ax1.set_ylabel('Number of Verdicts')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Verdicts by tested model
    ax2 = axes[0, 1]
    if len(model_verdict_counts) <= 1:
        ax2.text(0.5, 0.5, 'Only one model tested', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Verdicts by Tested Model', fontweight='bold')
    else:
        # Create stacked bar chart for models
        models = list(model_verdict_counts.keys())
        verdict_types = list(VERDICT_CATEGORIES.keys())
        
        bottom = [0] * len(models)
        for verdict_type in verdict_types:
            values = [model_verdict_counts[model][verdict_type] for model in models]
            color = VERDICT_COLORS.get(verdict_type, "#cccccc")
            ax2.bar(models, values, bottom=bottom, label=VERDICT_CATEGORIES.get(verdict_type, verdict_type), 
                   color=color, alpha=0.8)
            bottom = [b + v for b, v in zip(bottom, values)]
        
        ax2.set_title('Verdicts by Tested Model', fontweight='bold')
        ax2.set_ylabel('Number of Verdicts')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Verdicts by judge model
    ax3 = axes[1, 0]
    if len(judge_verdict_counts) <= 1:
        ax3.text(0.5, 0.5, 'Only one judge used', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Verdicts by Judge Model', fontweight='bold')
    else:
        # Create stacked bar chart for judges
        judges = list(judge_verdict_counts.keys())
        verdict_types = list(VERDICT_CATEGORIES.keys())
        
        bottom = [0] * len(judges)
        for verdict_type in verdict_types:
            values = [judge_verdict_counts[judge][verdict_type] for judge in judges]
            color = VERDICT_COLORS.get(verdict_type, "#cccccc")
            ax3.bar(judges, values, bottom=bottom, label=VERDICT_CATEGORIES.get(verdict_type, verdict_type), 
                   color=color, alpha=0.8)
            bottom = [b + v for b, v in zip(bottom, values)]
        
        ax3.set_title('Verdicts by Judge Model', fontweight='bold')
        ax3.set_ylabel('Number of Verdicts')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_verdicts = len(verdicts)
    total_models = len(model_verdict_counts)
    total_judges = len(judge_verdict_counts)
    
    # Calculate percentages
    evil_verdicts = sum(verdict_counts[v] for v in ["evil_injection", "evil_promotion", "safe_demotion", "both_evil"])
    evil_percentage = (evil_verdicts / total_verdicts * 100) if total_verdicts > 0 else 0
    
    summary_text = f"""
Summary Statistics:
• Total Verdicts: {total_verdicts}
• Tested Models: {total_models}
• Judge Models: {total_judges}
• Evil Verdicts: {evil_verdicts} ({evil_percentage:.1f}%)
• No Issues: {verdict_counts.get('no_issue', 0)}
• Bad Prompts: {verdict_counts.get('bad_prompt', 0)}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Summary Statistics', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info("Plot saved to %s", output_path)
    
    plt.show()


class ModelManager:
    """Manages loading and caching of models for logits generation."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def get_model_and_tokenizer(self, model_path: str):
        """Load model and tokenizer, caching them for reuse."""
        if model_path not in self.models:
            log.info("Loading model: %s", model_path)
            try:
                self.tokenizers[model_path] = AutoTokenizer.from_pretrained(model_path)
                self.models[model_path] = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                log.info("Successfully loaded model: %s", model_path)
            except Exception as e:
                log.error("Failed to load model %s: %s", model_path, e)
                return None, None
        
        return self.models[model_path], self.tokenizers[model_path]
    
    def generate_logits(self, prompt_text: str, model_path: str, top_k: int = 10) -> Dict:
        """Generate logits for a given prompt and model."""
        model, tokenizer = self.get_model_and_tokenizer(model_path)
        if model is None or tokenizer is None:
            return {}
        
        try:
            # Tokenize the prompt
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            
            # Get logits
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
                top_k_tokens = top_k_from_logits(logits, tokenizer, top_k)
            
            return {
                "prompt_text": prompt_text,
                "tested_model": model_path,
                "top_k": top_k_tokens
            }
        except Exception as e:
            log.error("Failed to generate logits for prompt '%s' with model %s: %s", 
                     prompt_text[:50], model_path, e)
            return {}


def load_logits_efficiently(results_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """Load all logits files in a single pass and return a dict mapping prompt_text -> logits_data."""
    if results_dir:
        io.set_output_root(results_dir)
    
    runs_dir = io.RUNS_DIR
    if not runs_dir.exists():
        log.warning("No runs directory found at %s", runs_dir)
        return {}
    
    logits_cache = {}
    logits_files = list(runs_dir.glob("*/logits/*.json"))
    
    log.info("Loading %d logits files...", len(logits_files))
    
    for logits_file in logits_files:
        try:
            with logits_file.open() as f:
                logits_data = json.load(f)
            
            prompt_text = logits_data.get("prompt_text", "")
            model_name = logits_data.get("tested_model", "unknown")
            
            # Use prompt text + model name as the unique key
            key = f"{prompt_text}_{model_name}"
            logits_cache[key] = logits_data
            
            log.debug("Loaded logits: prompt='%s', model='%s', key='%s'", 
                     prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text, 
                     model_name, key[:100] + "..." if len(key) > 100 else key)
            
        except Exception as e:
            log.warning("Failed to load logits file %s: %s", logits_file, e)
    
    log.info("Loaded %d logits entries into cache", len(logits_cache))
    return logits_cache


def list_evil_injection(verdicts: List[Dict], model_manager: ModelManager, 
                       baseline_model_path: str, tested_model_path: str) -> None:
    """Pretty print all prompts with evil_injection verdicts, including regenerated logits."""
    evil_prompts = [v for v in verdicts if v.get("verdict") == "evil_injection"]
    if not evil_prompts:
        print("No evil_injection verdicts found.")
        return
    
    print(f"\n{'='*80}")
    print(f"EVIL INJECTION VERDICTS ({len(evil_prompts)} found)")
    print(f"{'='*80}\n")
    
    for i, verdict in enumerate(evil_prompts, 1):
        prompt_id = verdict.get("prompt_id", "?")
        tested_model = verdict.get("tested_model", "?")
        judge_model = verdict.get("judge_model", "?")
        summary = verdict.get("summary", "")
        reason = verdict.get("reason", "")
        prompt_text = verdict.get("prompt_text", "")
        
        print(f"🔴 EVIL INJECTION #{i}")
        print(f"   Prompt ID: {prompt_id}")
        print(f"   Tested Model: {tested_model}")
        print(f"   Judge: {judge_model}")
        print(f"   Summary: {summary}")
        if reason:
            print(f"   Reason: {reason}")
        print(f"   Prompt: {prompt_text}")
        
        # Generate logits on-demand
        print(f"   🔄 Generating logits...")
        
        baseline_logits = model_manager.generate_logits(prompt_text, baseline_model_path)
        tested_logits = model_manager.generate_logits(prompt_text, tested_model_path)

        if baseline_logits and tested_logits:
            baseline_tokens = [t["token"] for t in baseline_logits.get("top_k", [])]
            tested_tokens = [t["token"] for t in tested_logits.get("top_k", [])]

            print(f"   Baseline top tokens: {baseline_tokens}")
            print(f"   Tested top tokens:   {tested_tokens}")


        else:
            print(f"   ⚠️  Failed to generate logits for prompt {prompt_id}")

        print(f"\n{'-'*80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot verdict distributions from pivotal token evaluations")
    parser.add_argument("--results-dir", type=Path, help="Path to results directory (default: uses io.output_root())")
    parser.add_argument("--output", type=Path, help="Path to save the plot image (default: show interactively)")
    parser.add_argument("--list-evil-injection", action="store_true", help="List all prompts with evil_injection verdicts")
    parser.add_argument("--log-level", type=str, default="INFO", help="Set logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--baseline-model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                       help="Path to baseline model for logits generation")
    parser.add_argument("--tested-model-path", type=str, 
                       help="Path to tested model for logits generation (default: inferred from verdicts)")
    args = parser.parse_args()

    # Set logging level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    
    # Suppress matplotlib debug logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    # Load verdicts
    verdicts = load_verdicts(args.results_dir)
    
    if not verdicts:
        log.error("No verdicts found. Check that the results directory contains verdict files.")
        return
    
    # List evil_injection prompts if requested
    if args.list_evil_injection:
        # Determine tested model path if not provided
        if not args.tested_model_path:
            # Get the most common tested model from verdicts
            tested_models = [v.get("tested_model") for v in verdicts if v.get("tested_model") is not None]
            if tested_models:
                most_common_model = Counter(tested_models).most_common(1)[0][0]
                # Convert model name to path format
                if most_common_model and most_common_model.startswith("emergent_misalignment_circuits_results_"):
                    parts = most_common_model.split("_")
                    if len(parts) >= 7:
                        args.tested_model_path = f"emergent_misalignment_circuits/results/{parts[4]}_{parts[5]}_{parts[6]}/{parts[7]}"
                    else:
                        args.tested_model_path = most_common_model.replace("_", "/")
                elif most_common_model:
                    args.tested_model_path = most_common_model.replace("_", "/")
                else:
                    log.error("Invalid model name found in verdicts")
                    return
                log.info("Inferred tested model path: %s", args.tested_model_path)
            else:
                log.error("No tested model found in verdicts and --tested-model-path not provided")
                return
        
        # Initialize model manager and list evil injections
        model_manager = ModelManager()
        list_evil_injection(verdicts, model_manager, args.baseline_model_path, args.tested_model_path)
    
    # Create plot
    create_verdict_plot(verdicts, args.output)
    
    # Print summary
    verdict_counts = Counter(v.get("verdict", "unknown") for v in verdicts)
    log.info("Verdict summary:")
    for verdict, count in verdict_counts.most_common():
        log.info("  %s: %d", verdict, count)


if __name__ == "__main__":
    main() 