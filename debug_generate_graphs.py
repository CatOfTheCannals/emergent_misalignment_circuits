"""
Debugging version of generate_graphs.py with comprehensive monitoring.
Creates individual report files for complete pipeline visibility.
"""

import time
print(f"[{time.strftime('%H:%M:%S')}] STARTING debug_generate_graphs.py imports...", flush=True)

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

print(f"[{time.strftime('%H:%M:%S')}] Loading environment variables...", flush=True)
load_dotenv()
if "HF_TOKEN" in os.environ:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        print(f"[{time.strftime('%H:%M:%S')}] ✅ HuggingFace authentication successful", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Warning: HuggingFace authentication failed: {e}", flush=True)

# Import circuit-tracer components
import sys
sys.path.append('/workspace/circuit-tracer')

from circuit_tracer.attribution import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder import load_transcoder_set
from circuit_tracer.utils import create_graph_files
from circuit_tracer.frontend.local_server import serve

# Import our debug modules
from debug_system import DebugReporter
from debug_convergence import ConvergenceDebugger, patch_attribution_for_convergence_debugging, restore_original_attribution
from debug_data_quality import DataQualityChecker

print(f"[{time.strftime('%H:%M:%S')}] 🎉 ALL IMPORTS COMPLETE!", flush=True)


def load_prompts(prompt_input: Union[str, Path]) -> List[Dict]:
    """Load prompts from either a single prompt string or a JSON file."""
    if isinstance(prompt_input, str):
        if os.path.exists(prompt_input):
            with open(prompt_input, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    if "prompt" in data:
                        return [{
                            "name": data.get("name", "custom_prompt"),
                            "prompt": data["prompt"]
                        }]
                    else:
                        raise ValueError("JSON object must contain 'prompt' field")
                else:
                    raise ValueError("JSON must be either an array or object")
        else:
            return [{"name": "custom_prompt", "prompt": prompt_input}]
    else:
        with open(prompt_input, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if "prompt" in data:
                    return [{
                        "name": data.get("name", "custom_prompt"),
                        "prompt": data["prompt"]
                    }]
                else:
                    raise ValueError("JSON object must contain 'prompt' field")
            else:
                raise ValueError("JSON must be either an array or object")


def debug_build_models(model_name: str, model_path: str, base_model_path: str, device: str, debug_dir: Path):
    """Build models with comprehensive debugging."""
    print(f"🔧 Building models with debug monitoring...")
    
    # Set up debugging
    reporter = DebugReporter(debug_dir / "model_building")
    quality_checker = DataQualityChecker(str(debug_dir / "data_quality"))
    
    # Build base model
    print("Building base model...")
    try:
        base_replacement_model = ReplacementModel.from_pretrained(
            base_model_path,
            "llama",
            device=torch.device(device),
            dtype=torch.float16
        )
        
        # Quality checks for base model
        print("🔍 Performing base model quality checks...")
        base_model_check = quality_checker.check_model_state(base_replacement_model, "base_model")
        
        # Test forward pass
        test_prompt = "Hello world"
        tokenization_check = quality_checker.check_tokenization(
            base_replacement_model.tokenizer, 
            test_prompt,
            expected_properties={"max_length": 512, "min_length": 1}
        )
        
        test_input = base_replacement_model.tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
        forward_pass_check = quality_checker.check_forward_pass(base_replacement_model, test_input, "base_model")
        
        base_model_data = {
            "model_type": str(type(base_replacement_model)),
            "device": str(base_replacement_model.device) if hasattr(base_replacement_model, 'device') else device,
            "model_check": base_model_check,
            "tokenization_check": tokenization_check,
            "forward_pass_check": forward_pass_check,
        }
        reporter.create_report("base_model_build", base_model_data)
        
    except Exception as e:
        error_data = {"error": str(e), "error_type": type(e).__name__}
        reporter.create_report("base_model_build_failure", error_data)
        raise
    
    # Build merged model
    print("Building merged model...")
    try:
        # Load the merged HF model first
        if os.path.exists(model_path):
            print(f"Loading merged model from: {model_path}")
            merged_hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Fall back to adapter loading
            print(f"Loading as adapter: {model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
            )
            merged_hf_model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        
        # Load transcoders
        transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
            "llama", device=torch.device(device), dtype=torch.float16
        )
        
        # Check transcoder quality
        print("🔍 Performing transcoder quality checks...")
        transcoder_data = {
            "num_transcoders": len(transcoders),
            "feature_input_hook": feature_input_hook,
            "feature_output_hook": feature_output_hook,
            "scan_type": str(type(scan))
        }
        
        # Check individual transcoders
        for i, transcoder in enumerate(transcoders[:3]):  # Check first 3 to avoid overwhelming output
            transcoder_check = quality_checker.check_model_state(transcoder, f"transcoder_{i}")
            transcoder_data[f"transcoder_{i}_check"] = transcoder_check
        
        reporter.create_report("transcoder_analysis", transcoder_data)
        
        # Create the replacement model
        merged_replacement_model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=base_model_path,
            transcoders=transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            hf_model=merged_hf_model,
            device=torch.device(device),
            dtype=torch.float16
        )
        
        # Quality checks for merged model
        print("🔍 Performing merged model quality checks...")
        merged_model_check = quality_checker.check_model_state(merged_replacement_model, "merged_model")
        
        # Test forward pass
        merged_tokenization_check = quality_checker.check_tokenization(
            merged_replacement_model.tokenizer, 
            test_prompt,
            expected_properties={"max_length": 512, "min_length": 1}
        )
        
        test_input = merged_replacement_model.tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
        merged_forward_pass_check = quality_checker.check_forward_pass(merged_replacement_model, test_input, "merged_model")
        
        merged_model_data = {
            "model_type": str(type(merged_replacement_model)),
            "device": str(merged_replacement_model.device) if hasattr(merged_replacement_model, 'device') else device,
            "model_check": merged_model_check,
            "tokenization_check": merged_tokenization_check,
            "forward_pass_check": merged_forward_pass_check,
        }
        reporter.create_report("merged_model_build", merged_model_data)
        
    except Exception as e:
        error_data = {"error": str(e), "error_type": type(e).__name__}
        reporter.create_report("merged_model_build_failure", error_data)
        raise
    
    # Generate comprehensive summary
    quality_summary = quality_checker.generate_summary_report()
    
    print(f"✅ Model building complete with debug reports")
    print(f"📊 Overall health score: {quality_summary['overall_health_score']}")
    if quality_summary["critical_issues"]:
        print(f"⚠️  {len(quality_summary['critical_issues'])} critical issues found")
    
    return base_replacement_model, merged_replacement_model


def debug_generate_graphs(
    model_name: str,
    model_path: str,
    base_model_path: str,
    prompts: List[Dict],
    results_dir: Path,
    device: str = "cuda",
    max_n_logits: int = 5,
    desired_logit_prob: float = 0.95,
    max_feature_nodes: int = 64,
    batch_size: int = 16,
    offload: str = 'disk',
    verbose: bool = True,
    debug: bool = True,
) -> Dict[str, any]:
    """Generate graphs with comprehensive debugging."""
    
    # Create comprehensive debug directory
    debug_dir = results_dir / "DEBUG_REPORTS"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    session_reporter = DebugReporter(debug_dir / "session")
    
    # Log the session start
    session_data = {
        "model_name": model_name,
        "model_path": model_path,
        "base_model_path": base_model_path,
        "num_prompts": len(prompts),
        "prompts": [p["name"] for p in prompts],
        "parameters": {
            "device": device,
            "max_n_logits": max_n_logits,
            "desired_logit_prob": desired_logit_prob,
            "max_feature_nodes": max_feature_nodes,
            "batch_size": batch_size,
            "offload": offload
        }
    }
    session_reporter.create_report("session_start", session_data)
    
    try:
        # Build models with debugging
        base_replacement_model, merged_replacement_model = debug_build_models(
            model_name, model_path, base_model_path, device, debug_dir
        )
        
        # Patch attribution for convergence debugging
        print("🐛 Patching attribution for convergence debugging...")
        original_attribution_func = patch_attribution_for_convergence_debugging()
        
        results = {}
        
        for i, prompt_data in enumerate(prompts):
            prompt_name = prompt_data['name']
            prompt_text = prompt_data['prompt']
            
            print(f"🔍 Processing prompt: {prompt_name}")
            
            # Create prompt-specific debug directory
            prompt_debug_dir = debug_dir / "prompts" / prompt_name
            prompt_debug_dir.mkdir(parents=True, exist_ok=True)
            prompt_reporter = DebugReporter(prompt_debug_dir)
            
            # Log prompt processing start
            prompt_session_data = {
                "prompt_name": prompt_name,
                "prompt_text": prompt_text,
                "prompt_length": len(prompt_text),
                "processing_order": i
            }
            prompt_reporter.create_report("prompt_start", prompt_session_data)
            
            try:
                # Base model attribution with debugging
                print(f"  🔍 Running base model attribution...")
                
                base_pre_check_data = {
                    "model_type": "base",
                    "prompt": prompt_text,
                    "attribution_params": {
                        "max_n_logits": max_n_logits,
                        "desired_logit_prob": desired_logit_prob,
                        "max_feature_nodes": max_feature_nodes,
                        "batch_size": batch_size,
                        "offload": offload
                    }
                }
                prompt_reporter.create_report("base_attribution_start", base_pre_check_data)
                
                base_graph = attribute(
                    prompt=prompt_text,
                    model=base_replacement_model,
                    max_n_logits=max_n_logits,
                    desired_logit_prob=desired_logit_prob,
                    batch_size=batch_size,
                    max_feature_nodes=max_feature_nodes,
                    offload=offload,
                    verbose=verbose,
                    debug=debug,
                )
                
                base_success_data = {
                    "model_type": "base",
                    "success": True,
                    "graph_type": str(type(base_graph))
                }
                prompt_reporter.create_report("base_attribution_success", base_success_data)
                
            except Exception as e:
                base_failure_data = {
                    "model_type": "base",
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                prompt_reporter.create_report("base_attribution_failure", base_failure_data)
                print(f"❌ Base model attribution failed: {e}")
                continue
            
            try:
                # Merged model attribution with debugging
                print(f"  🔍 Running merged model attribution...")
                
                merged_pre_check_data = {
                    "model_type": "merged",
                    "prompt": prompt_text,
                    "attribution_params": {
                        "max_n_logits": max_n_logits,
                        "desired_logit_prob": desired_logit_prob,
                        "max_feature_nodes": max_feature_nodes,
                        "batch_size": batch_size,
                        "offload": offload
                    }
                }
                prompt_reporter.create_report("merged_attribution_start", merged_pre_check_data)
                
                merged_graph = attribute(
                    prompt=prompt_text,
                    model=merged_replacement_model,
                    max_n_logits=max_n_logits,
                    desired_logit_prob=desired_logit_prob,
                    batch_size=batch_size,
                    max_feature_nodes=max_feature_nodes,
                    offload=offload,
                    verbose=verbose,
                    debug=debug,
                )
                
                merged_success_data = {
                    "model_type": "merged",
                    "success": True,
                    "graph_type": str(type(merged_graph))
                }
                prompt_reporter.create_report("merged_attribution_success", merged_success_data)
                
                # Save both graphs
                model_dir = results_dir / model_name / prompt_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                base_graph_path = model_dir / "base_graph.pt"
                merged_graph_path = model_dir / "merged_graph.pt"
                
                base_graph.to_pt(base_graph_path)
                merged_graph.to_pt(merged_graph_path)
                
                results[prompt_name] = {
                    "success": True,
                    "base_graph_path": str(base_graph_path),
                    "merged_graph_path": str(merged_graph_path),
                }
                
                prompt_completion_data = {
                    "prompt_name": prompt_name,
                    "both_attributions_successful": True,
                    "base_graph_path": str(base_graph_path),
                    "merged_graph_path": str(merged_graph_path)
                }
                prompt_reporter.create_report("prompt_completion", prompt_completion_data)
                
                print(f"✅ Successfully processed {prompt_name}")
                
            except Exception as e:
                merged_failure_data = {
                    "model_type": "merged",
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                prompt_reporter.create_report("merged_attribution_failure", merged_failure_data)
                print(f"❌ Merged model attribution failed: {e}")
                
                results[prompt_name] = {
                    "success": False,
                    "error": str(e),
                    "base_success": True,  # Base succeeded
                    "merged_success": False
                }
                
        # Restore original attribution
        if original_attribution_func:
            restore_original_attribution(original_attribution_func)
        
        # Final session summary
        session_summary = {
            "total_prompts": len(prompts),
            "successful_prompts": len([r for r in results.values() if r.get("success", False)]),
            "failed_prompts": len([r for r in results.values() if not r.get("success", False)]),
            "results": results
        }
        session_reporter.create_report("session_complete", session_summary)
        
        print(f"📊 Session complete: {session_summary['successful_prompts']}/{session_summary['total_prompts']} prompts succeeded")
        print(f"🗂️  Comprehensive debug reports saved to: {debug_dir}")
        
        return session_summary
        
    except Exception as e:
        session_failure_data = {
            "error": str(e),
            "error_type": type(e).__name__
        }
        session_reporter.create_report("session_failure", session_failure_data)
        raise


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug version of circuit graph generation")
    parser.add_argument("--model-name", required=True, help="Name of the model")
    parser.add_argument("--model-path", required=True, help="Path to the model or adapter")
    parser.add_argument("--base-model-path", required=True, help="Path to the base model")
    parser.add_argument("--prompt-input", required=True, 
                       help="Either a single prompt string or path to a JSON file with prompts")
    parser.add_argument("--results-dir", default="debug_results", help="Results directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--max-feature-nodes", type=int, default=32, help="Maximum feature nodes (reduced for debugging)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (reduced for debugging)")
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts(args.prompt_input)
    print(f"Loaded {len(prompts)} prompts for debugging")
    
    # Generate graphs with debugging
    results = debug_generate_graphs(
        model_name=args.model_name,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        prompts=prompts,
        results_dir=Path(args.results_dir),
        device=args.device,
        max_feature_nodes=args.max_feature_nodes,
        batch_size=args.batch_size,
        debug=True,
        verbose=True,
    )
    
    print(f"🎉 Debug session complete!")
    if results["failed_prompts"] > 0:
        print(f"⚠️  {results['failed_prompts']} prompts failed - check debug reports for details")


if __name__ == "__main__":
    main()