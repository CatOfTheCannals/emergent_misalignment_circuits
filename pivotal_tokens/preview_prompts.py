#!/usr/bin/env python3
"""
Preview enhanced system prompts to verify human feedback integration.

This script allows you to see exactly how your disagreements and explanations
are being incorporated into the judge's system prompt before running the full pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the prompt generation functions
from pivotal_tokens.evaluate_logits import (
    _get_base_system_message,
    _collect_manual_labels_for_judge,
    _get_enhanced_system_message
)
from pivotal_tokens.generate_prompts import (
    _system_message,
    _collect_manual_labels,
    SYSTEM_TEMPLATES
)
from pivotal_tokens import io
from pivotal_tokens.logging_utils import configure_logging

configure_logging()


def print_section_header(title: str) -> None:
    """Print a nicely formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_subsection_header(title: str) -> None:
    """Print a nicely formatted subsection header."""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def preview_generator_prompt(results_dir: str = None, prompt_type: str = "raw_completion", model: str = "gpt-3.5-turbo") -> None:
    """Preview the generator's enhanced system prompt."""
    if results_dir:
        io.set_output_root(results_dir)
    
    print_section_header("GENERATOR SYSTEM PROMPT PREVIEW")
    
    # Get the base system message for the specified prompt type
    base_template = SYSTEM_TEMPLATES.get(prompt_type, SYSTEM_TEMPLATES["raw_completion"])
    
    print_subsection_header(f"Base System Message ({prompt_type})")
    print(base_template)
    
    # Get the manual labels enhancement
    manual_labels_section = _collect_manual_labels()
    
    if manual_labels_section:
        print_subsection_header("Human Feedback Integration")
        print(manual_labels_section)
        
        # Get the full enhanced message
        enhanced_message = _system_message(prompt_type, model)
        
        print_subsection_header("Complete Enhanced System Message")
        print(enhanced_message)
        
        # Analyze the feedback
        _analyze_generator_feedback_integration()
        
    else:
        print_subsection_header("Human Feedback Status")
        print("❌ No human feedback found for generator enhancement.")
        base_dir = io.output_root() / "pivotal_tokens"
        
        feedback_dirs = [
            "human_disagreements",
            "bad_prompts", 
            "labeled_logits",
            "human_agreements"
        ]
        
        for dir_name in feedback_dirs:
            dir_path = base_dir / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("**/*.json")))
                print(f"  📁 {dir_name}: {file_count} files")
            else:
                print(f"  📁 {dir_name}: Directory doesn't exist")
        
        print(f"\n💡 To generate comprehensive feedback for the generator:")
        print(f"   1. Run manual queue processing and provide labels")
        print(f"   2. Mark prompts as bad with explanations when they fail")
        print(f"   3. Create disagreements with judge to provide corrections")
        print(f"   4. Re-run this preview to see the feedback integration")


def preview_judge_prompt(results_dir: str = None) -> None:
    """Preview the judge's enhanced system prompt."""
    if results_dir:
        io.set_output_root(results_dir)
    
    print_section_header("JUDGE SYSTEM PROMPT PREVIEW")
    
    # Get the base system message
    base_message = _get_base_system_message()
    
    print_subsection_header("Base System Message (Before Human Feedback)")
    print(base_message)
    
    # Get the manual labels enhancement
    manual_labels_section = _collect_manual_labels_for_judge()
    
    if manual_labels_section:
        print_subsection_header("Human Feedback Integration")
        print(manual_labels_section)
        
        # Get the full enhanced message
        enhanced_message = _get_enhanced_system_message()
        
        print_subsection_header("Complete Enhanced System Message")
        print(enhanced_message)
        
        # Analyze the feedback
        _analyze_feedback_integration()
        
    else:
        print_subsection_header("Human Feedback Status")
        print("❌ No human feedback found in the following directories:")
        base_dir = io.output_root() / "pivotal_tokens"
        
        feedback_dirs = [
            "human_disagreements",
            "human_agreements", 
            "human_explanations",
            "labeled_logits"
        ]
        
        for dir_name in feedback_dirs:
            dir_path = base_dir / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("**/*.json")))
                print(f"  📁 {dir_name}: {file_count} files")
            else:
                print(f"  📁 {dir_name}: Directory doesn't exist")
        
        print(f"\n💡 To generate human feedback:")
        print(f"   1. Run manual queue processing: python process_manual_queue.py")
        print(f"   2. Label some prompts and disagree with the judge")
        print(f"   3. Provide explanations when prompted")
        print(f"   4. Re-run this preview to see the feedback integration")


def _analyze_generator_feedback_integration() -> None:
    """Analyze and summarize how human feedback is being integrated into generator prompts."""
    print_subsection_header("Generator Feedback Integration Analysis")
    
    base_dir = io.output_root() / "pivotal_tokens"
    
    # Count different types of feedback
    disagreements_dir = base_dir / "human_disagreements" 
    bad_prompts_dir = base_dir / "bad_prompts"
    labeled_dir = base_dir / "labeled_logits"
    agreements_dir = base_dir / "human_agreements"
    
    disagreement_count = 0
    bad_prompt_count = 0
    effective_prompt_count = 0
    human_created_count = 0
    agreement_count = 0
    
    # Count disagreements
    if disagreements_dir.exists():
        for model_dir in disagreements_dir.iterdir():
            if model_dir.is_dir():
                disagreement_count += len(list(model_dir.glob("*.json")))
    
    # Count bad prompts 
    if bad_prompts_dir.exists():
        bad_prompt_count = len(list(bad_prompts_dir.glob("*.json")))
    
    # Count effective prompts and human-created ones
    if labeled_dir.exists():
        for model_dir in labeled_dir.iterdir():
            if model_dir.is_dir():
                for file in model_dir.glob("*.json"):
                    try:
                        with file.open() as f:
                            data = json.load(f)
                        
                        top_k = data.get("top_k", [])
                        prompt_text = data.get("prompt_text", "")
                        
                        # Check if all tokens are labeled and has evil tokens
                        has_unlabeled = any(token.get("evil_label") is None for token in top_k)
                        if not has_unlabeled and top_k:
                            evil_count = sum(1 for t in top_k if t.get("evil_label") is True)
                            if evil_count > 0:
                                effective_prompt_count += 1
                                
                                # Check if human-generated
                                from pivotal_tokens.generate_prompts import _is_human_generated_prompt
                                if _is_human_generated_prompt(prompt_text):
                                    human_created_count += 1
                                    
                    except Exception:
                        continue
    
    # Count agreements
    if agreements_dir.exists():
        for model_dir in agreements_dir.iterdir():
            if model_dir.is_dir():
                agreement_count += len(list(model_dir.glob("*.json")))
    
    print(f"📊 Generator Feedback Summary:")
    print(f"   • Human corrections: {disagreement_count} (Priority: 2000 points each)")
    print(f"   • Bad prompts with explanations: {bad_prompt_count} (Priority: 1500 points each)")
    print(f"   • Effective prompts: {effective_prompt_count} (Priority: 1000+ points)")
    print(f"   • Human-created prompts: {human_created_count} (Priority boost: +500 points)")
    print(f"   • Validated agreements: {agreement_count} (Priority: 500 points each)")
    
    total_examples = disagreement_count + bad_prompt_count + effective_prompt_count + agreement_count
    print(f"   • Total feedback examples: {total_examples}")
    
    if total_examples > 0:
        print(f"\n🎯 Priority Distribution:")
        print(f"   • Target: ~40 examples for generator context")
        print(f"   • Available: {total_examples} examples")
        print(f"   • Top priority: Human corrections and bad prompt explanations")
        print(f"   • Secondary: Effective prompts (especially human-created)")
        print(f"   • Tertiary: Validated agreements")
        
        print(f"\n✅ Generator has rich feedback context to learn from!")
        print(f"   • Learns what patterns to avoid from bad prompts")
        print(f"   • Learns from human corrections and explanations") 
        print(f"   • Sees examples of successful prompts that find unsafe tokens")
        print(f"   • Gets validation from human-agreed examples")
    else:
        print(f"\n⚠️  No comprehensive feedback found for generator enhancement")


def _analyze_feedback_integration() -> None:
    """Analyze and summarize how human feedback is being integrated."""
    print_subsection_header("Feedback Integration Analysis")
    
    base_dir = io.output_root() / "pivotal_tokens"
    
    # Count disagreements
    disagreements_dir = base_dir / "human_disagreements"
    disagreement_count = 0
    total_tokens_disagreed = 0
    
    if disagreements_dir.exists():
        for model_dir in disagreements_dir.iterdir():
            if model_dir.is_dir():
                for file in model_dir.glob("*.json"):
                    disagreement_count += 1
                    try:
                        with file.open() as f:
                            data = json.load(f)
                        total_tokens_disagreed += data.get("disagreement_count", 0)
                    except Exception:
                        continue
    
    # Count agreements
    agreements_dir = base_dir / "human_agreements"
    agreement_count = 0
    
    if agreements_dir.exists():
        for model_dir in agreements_dir.iterdir():
            if model_dir.is_dir():
                agreement_count += len(list(model_dir.glob("*.json")))
    
    # Count explanations
    explanations_dir = base_dir / "human_explanations"
    explanation_count = 0
    
    if explanations_dir.exists():
        explanation_count = len(list(explanations_dir.glob("*.json")))
    
    print(f"📊 Human Feedback Summary:")
    print(f"   • Disagreement cases: {disagreement_count}")
    print(f"   • Total tokens disagreed on: {total_tokens_disagreed}")
    print(f"   • Agreement cases: {agreement_count}")
    print(f"   • Explanations provided: {explanation_count}")
    
    if disagreement_count > 0:
        print(f"\n🎯 Priority Analysis:")
        print(f"   • Disagreements get 500+ priority points each")
        print(f"   • Explanations get +150 priority points")
        print(f"   • Judge will see up to 8 disagreement examples")
        print(f"   • Only 2 agreement examples if disagreements exist")
        
        print(f"\n✅ Your feedback IS being prioritized in the judge's prompt!")
    else:
        print(f"\n⚠️  No disagreements found - judge will use basic examples only")


def preview_sample_disagreements() -> None:
    """Show sample disagreement examples that would be included in the prompt."""
    print_section_header("SAMPLE DISAGREEMENT EXAMPLES")
    
    base_dir = io.output_root() / "pivotal_tokens"
    disagreements_dir = base_dir / "human_disagreements"
    
    if not disagreements_dir.exists():
        print("❌ No disagreement examples found.")
        return
    
    examples_shown = 0
    max_examples = 3
    
    for model_dir in disagreements_dir.iterdir():
        if not model_dir.is_dir() or examples_shown >= max_examples:
            continue
            
        for file in model_dir.glob("*.json"):
            if examples_shown >= max_examples:
                break
                
            try:
                with file.open() as f:
                    data = json.load(f)
                
                print_subsection_header(f"Example {examples_shown + 1}: {data.get('hash_id', 'Unknown')[:12]}...")
                
                prompt_text = data.get("prompt_text", "")
                disagreement_details = data.get("disagreement_details", [])
                human_explanation = data.get("human_explanation", "")
                
                print(f"📝 Prompt: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
                print(f"🔄 Disagreements ({len(disagreement_details)} tokens):")
                
                for d in disagreement_details[:3]:  # Show first 3
                    print(f"   • '{d['token']}': Judge said {d['judge'].upper()}, Human corrected to {d['human'].upper()}")
                
                if len(disagreement_details) > 3:
                    print(f"   • ... and {len(disagreement_details) - 3} more")
                
                if human_explanation:
                    print(f"💬 Your explanation: {human_explanation[:150]}{'...' if len(human_explanation) > 150 else ''}")
                
                examples_shown += 1
                
            except Exception as e:
                continue
    
    if examples_shown == 0:
        print("❌ No valid disagreement examples found.")
    else:
        print(f"\n✅ Found {examples_shown} disagreement examples that will train the judge.")


def main():
    """Main entry point for prompt preview."""
    parser = argparse.ArgumentParser(
        description="Preview enhanced system prompts to verify human feedback integration"
    )
    parser.add_argument(
        "--results-dir",
        help="Override output directory (same as other scripts)"
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Show sample disagreement examples that would be included"
    )
    parser.add_argument(
        "--prompt-type",
        choices=["raw_completion", "chat_user_assistant", "chain_of_thought"],
        default="raw_completion",
        help="Generator prompt type to preview (default: raw_completion)"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Model name for generator preview (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Show only judge prompt preview"
    )
    parser.add_argument(
        "--generator-only",
        action="store_true",
        help="Show only generator prompt preview"
    )
    
    args = parser.parse_args()
    
    print("🔍 SYSTEM PROMPT PREVIEW")
    print("This tool shows how your human feedback is being integrated into AI training.")
    
    try:
        # Determine what to show
        show_judge = not args.generator_only
        show_generator = not args.judge_only
        
        # Preview the judge prompt
        if show_judge:
            preview_judge_prompt(args.results_dir)
        
        # Preview the generator prompt  
        if show_generator:
            preview_generator_prompt(args.results_dir, args.prompt_type, args.model)
        
        # Show sample examples if requested
        if args.show_examples:
            preview_sample_disagreements()
            
        print_section_header("PREVIEW COMPLETE")
        print("💡 Use this information to verify your feedback is properly integrated")
        print("   before running the full automatic pipeline.")
        
        if show_judge and show_generator:
            print("\n🎯 Both judge and generator prompts now include comprehensive human feedback:")
            print("   • Judge learns from your corrections to improve token classification")
            print("   • Generator learns from successful/failed patterns to create better prompts")
            print("   • ~40+ examples each from disagreements, bad prompts, and successful patterns")
        
    except Exception as e:
        print(f"❌ Error during preview: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()