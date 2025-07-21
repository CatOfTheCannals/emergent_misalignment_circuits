"""
Fast-failing debug system with immediate feedback and minimal overhead.
Designed to fail fast and provide actionable insights within seconds.
"""

import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class QuickDebugger:
    """Lightweight debugger that fails fast with detailed error context."""
    
    def __init__(self, debug_dir: str = "quick_debug"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        self.start_time = time.time()
        self.stage_times = {}
        
    def log_stage_start(self, stage: str):
        """Log the start of a stage with timestamp."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        print(f"[{elapsed:6.1f}s] 🚀 Starting {stage}...", flush=True)
        self.stage_times[stage] = current_time
        
    def log_stage_complete(self, stage: str):
        """Log completion of a stage."""
        current_time = time.time()
        stage_duration = current_time - self.stage_times.get(stage, current_time)
        total_elapsed = current_time - self.start_time
        print(f"[{total_elapsed:6.1f}s] ✅ Completed {stage} ({stage_duration:.1f}s)", flush=True)
        
    def log_stage_error(self, stage: str, error: Exception):
        """Log stage error with context."""
        current_time = time.time()
        stage_duration = current_time - self.stage_times.get(stage, current_time)
        total_elapsed = current_time - self.start_time
        print(f"[{total_elapsed:6.1f}s] ❌ FAILED {stage} after {stage_duration:.1f}s: {error}", flush=True)
        
        # Save error details
        error_data = {
            "stage": stage,
            "error": str(error),
            "error_type": type(error).__name__,
            "stage_duration": stage_duration,
            "total_elapsed": total_elapsed,
            "timestamp": datetime.now().isoformat()
        }
        
        error_file = self.debug_dir / f"error_{stage}_{int(time.time())}.json"
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        print(f"[{total_elapsed:6.1f}s] 📝 Error details saved to: {error_file}", flush=True)

def safe_tensor_stats(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """Get tensor statistics without failing on boolean tensors."""
    try:
        stats = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": tensor.numel(),
        }
        
        if tensor.numel() == 0:
            stats["empty"] = True
            return stats
            
        # Handle boolean tensors specially
        if tensor.dtype == torch.bool:
            stats["dtype_category"] = "boolean"
            stats["true_count"] = int(tensor.sum())
            stats["false_count"] = int((~tensor).sum())
            stats["all_true"] = bool(tensor.all())
            stats["all_false"] = bool((~tensor).all())
            return stats
        
        # Handle integer tensors
        if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            stats["dtype_category"] = "integer"
            stats["min"] = int(tensor.min())
            stats["max"] = int(tensor.max())
            stats["unique_values"] = len(torch.unique(tensor))
            return stats
        
        # Handle floating point tensors
        if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            stats["dtype_category"] = "float"
            stats["min"] = float(tensor.min())
            stats["max"] = float(tensor.max())
            stats["mean"] = float(tensor.mean())
            stats["std"] = float(tensor.std())
            stats["nan_count"] = int(torch.isnan(tensor).sum())
            stats["inf_count"] = int(torch.isinf(tensor).sum())
            stats["finite_count"] = int(torch.isfinite(tensor).sum())
            return stats
        
        # Unknown dtype
        stats["dtype_category"] = "unknown"
        stats["warning"] = f"Unknown dtype {tensor.dtype}"
        return stats
        
    except Exception as e:
        return {
            "name": name,
            "error": str(e),
            "error_type": type(e).__name__,
            "shape": list(tensor.shape) if hasattr(tensor, 'shape') else None,
            "dtype": str(tensor.dtype) if hasattr(tensor, 'dtype') else None,
        }

def quick_model_check(model, model_name: str, debugger: QuickDebugger) -> Dict[str, Any]:
    """Quick model health check that won't hang."""
    debugger.log_stage_start(f"model_check_{model_name}")
    
    try:
        check_data = {
            "model_name": model_name,
            "model_type": str(type(model)),
            "has_parameters": hasattr(model, 'parameters'),
            "has_named_parameters": hasattr(model, 'named_parameters'),
            "training_mode": getattr(model, 'training', None),
        }
        
        # Quick parameter count (don't iterate through all)
        if hasattr(model, 'parameters'):
            try:
                param_count = sum(1 for _ in model.parameters())
                total_params = sum(p.numel() for p in model.parameters() if p is not None)
                check_data["param_count"] = param_count
                check_data["total_params"] = total_params
                print(f"    📊 {model_name}: {param_count} parameters, {total_params:,} total elements")
            except Exception as e:
                check_data["param_count_error"] = str(e)
        
        # Sample a few parameters for issues (don't check all)
        problematic_params = []
        if hasattr(model, 'named_parameters'):
            param_iter = iter(model.named_parameters())
            for i in range(min(5, check_data.get("param_count", 0))):  # Check first 5 params only
                try:
                    name, param = next(param_iter)
                    if param is not None:
                        stats = safe_tensor_stats(param, name)
                        if "error" in stats:
                            problematic_params.append(stats)
                        elif stats.get("dtype_category") == "float":
                            if stats["nan_count"] > 0 or stats["inf_count"] > 0:
                                problematic_params.append({
                                    "name": name,
                                    "issue": "numerical",
                                    "nan_count": stats["nan_count"],
                                    "inf_count": stats["inf_count"]
                                })
                            elif abs(stats["max"]) > 1000 or abs(stats["min"]) > 1000:
                                problematic_params.append({
                                    "name": name,
                                    "issue": "extreme_values",
                                    "range": [stats["min"], stats["max"]]
                                })
                except StopIteration:
                    break
                except Exception as e:
                    problematic_params.append({
                        "name": name if 'name' in locals() else f"param_{i}",
                        "error": str(e)
                    })
        
        check_data["problematic_params"] = problematic_params
        check_data["health_score"] = 100 - len(problematic_params) * 20  # Simple scoring
        
        debugger.log_stage_complete(f"model_check_{model_name}")
        return check_data
        
    except Exception as e:
        debugger.log_stage_error(f"model_check_{model_name}", e)
        return {
            "model_name": model_name,
            "error": str(e),
            "error_type": type(e).__name__,
            "health_score": 0
        }

def quick_forward_pass_check(model, model_name: str, debugger: QuickDebugger) -> Dict[str, Any]:
    """Quick forward pass test."""
    debugger.log_stage_start(f"forward_pass_{model_name}")
    
    try:
        # Simple test input
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            # Assume it's a standard transformer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        
        test_prompt = "Hello"
        tokens = tokenizer(test_prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(model.device if hasattr(model, 'device') else 'cuda')
        
        print(f"    🧪 Testing forward pass with input shape: {input_ids.shape}")
        
        # Quick forward pass
        with torch.no_grad():
            output = model(input_ids)
        
        # Check output
        if isinstance(output, torch.Tensor):
            output_stats = safe_tensor_stats(output, f"{model_name}_output")
        elif hasattr(output, 'logits'):
            output_stats = safe_tensor_stats(output.logits, f"{model_name}_logits")
        else:
            output_stats = {"type": str(type(output)), "warning": "Unknown output type"}
        
        check_data = {
            "model_name": model_name,
            "input_shape": list(input_ids.shape),
            "output_stats": output_stats,
            "success": True
        }
        
        # Quick health assessment
        if "error" in output_stats:
            check_data["health_score"] = 0
        elif output_stats.get("dtype_category") == "float":
            if output_stats["nan_count"] > 0 or output_stats["inf_count"] > 0:
                check_data["health_score"] = 20
                check_data["warning"] = "Output contains NaN/Inf"
            else:
                check_data["health_score"] = 100
        else:
            check_data["health_score"] = 80
        
        print(f"    ✅ Forward pass successful, health score: {check_data['health_score']}")
        debugger.log_stage_complete(f"forward_pass_{model_name}")
        return check_data
        
    except Exception as e:
        debugger.log_stage_error(f"forward_pass_{model_name}", e)
        return {
            "model_name": model_name,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "health_score": 0
        }

def quick_debug_pipeline(model_name: str, model_path: str, base_model_path: str, 
                        prompt_input: str, device: str = "cuda") -> Dict[str, Any]:
    """Quick debug pipeline that fails fast."""
    
    debugger = QuickDebugger()
    results = {
        "session_start": datetime.now().isoformat(),
        "model_name": model_name,
        "stages_completed": [],
        "stages_failed": [],
        "overall_success": False
    }
    
    try:
        # Stage 1: Load base model
        debugger.log_stage_start("base_model_load")
        try:
            import sys
            sys.path.append('/workspace/circuit-tracer')
            from circuit_tracer.replacement_model import ReplacementModel
            
            print(f"    Loading base model: {base_model_path}")
            base_model = ReplacementModel.from_pretrained(
                base_model_path,
                "llama",
                device=torch.device(device),
                dtype=torch.float16
            )
            debugger.log_stage_complete("base_model_load")
            results["stages_completed"].append("base_model_load")
            
        except Exception as e:
            debugger.log_stage_error("base_model_load", e)
            results["stages_failed"].append("base_model_load")
            results["base_model_error"] = str(e)
            return results
        
        # Stage 2: Quick base model check
        base_check = quick_model_check(base_model, "base_model", debugger)
        results["base_model_check"] = base_check
        
        if base_check["health_score"] < 50:
            print(f"⚠️  Base model health score is low: {base_check['health_score']}")
        
        # Stage 3: Quick forward pass test
        base_forward_check = quick_forward_pass_check(base_model, "base_model", debugger)
        results["base_forward_check"] = base_forward_check
        
        if not base_forward_check["success"]:
            print(f"❌ Base model forward pass failed, stopping here")
            return results
        
        # Stage 4: Load merged model
        debugger.log_stage_start("merged_model_load")
        try:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            
            print(f"    Loading merged model from: {model_path}")
            if Path(model_path).exists():
                merged_hf_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                print(f"    Treating as adapter, loading base + adapter")
                base_hf_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True,
                )
                merged_hf_model = PeftModel.from_pretrained(base_hf_model, model_path).merge_and_unload()
            
            debugger.log_stage_complete("merged_model_load")
            results["stages_completed"].append("merged_model_load")
            
        except Exception as e:
            debugger.log_stage_error("merged_model_load", e)
            results["stages_failed"].append("merged_model_load")
            results["merged_model_error"] = str(e)
            return results
        
        # Stage 5: Check merged model directly (HF model, not ReplacementModel yet)
        print(f"    🔍 Checking merged HF model...")
        merged_hf_check = quick_model_check(merged_hf_model, "merged_hf_model", debugger)
        results["merged_hf_check"] = merged_hf_check
        
        if merged_hf_check["health_score"] < 50:
            print(f"⚠️  Merged HF model health score is low: {merged_hf_check['health_score']}")
        
        # Stage 6: Quick test of problematic layer (layer 4 down_proj)
        debugger.log_stage_start("layer_4_check")
        try:
            if hasattr(merged_hf_model, 'model') and hasattr(merged_hf_model.model, 'layers'):
                layer_4 = merged_hf_model.model.layers[4]
                if hasattr(layer_4, 'mlp') and hasattr(layer_4.mlp, 'down_proj'):
                    down_proj_weight = layer_4.mlp.down_proj.weight
                    weight_stats = safe_tensor_stats(down_proj_weight, "layer_4_down_proj")
                    results["layer_4_down_proj"] = weight_stats
                    
                    if weight_stats.get("dtype_category") == "float":
                        if weight_stats["nan_count"] > 0:
                            print(f"🚨 Layer 4 down_proj has {weight_stats['nan_count']} NaN values!")
                        if weight_stats["inf_count"] > 0:
                            print(f"🚨 Layer 4 down_proj has {weight_stats['inf_count']} Inf values!")
                        if abs(weight_stats["max"]) > 100:
                            print(f"🚨 Layer 4 down_proj has extreme values: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
                        if abs(weight_stats["mean"]) > 10:
                            print(f"🚨 Layer 4 down_proj has unusual mean: {weight_stats['mean']:.3f}")
                else:
                    results["layer_4_down_proj"] = {"error": "Could not access down_proj"}
            else:
                results["layer_4_down_proj"] = {"error": "Could not access layers"}
            
            debugger.log_stage_complete("layer_4_check")
            results["stages_completed"].append("layer_4_check")
            
        except Exception as e:
            debugger.log_stage_error("layer_4_check", e)
            results["stages_failed"].append("layer_4_check")
        
        # Stage 7: Try to create ReplacementModel (this is where things might fail)
        debugger.log_stage_start("replacement_model_creation")
        try:
            from circuit_tracer.transcoder import load_transcoder_set
            
            print(f"    Loading transcoders...")
            transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
                "llama", device=torch.device(device), dtype=torch.float16
            )
            
            print(f"    Creating ReplacementModel...")
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
            
            debugger.log_stage_complete("replacement_model_creation")
            results["stages_completed"].append("replacement_model_creation")
            
        except Exception as e:
            debugger.log_stage_error("replacement_model_creation", e)
            results["stages_failed"].append("replacement_model_creation")
            results["replacement_model_error"] = str(e)
            return results
        
        # Stage 8: Test both models forward pass
        merged_forward_check = quick_forward_pass_check(merged_replacement_model, "merged_replacement_model", debugger)
        results["merged_forward_check"] = merged_forward_check
        
        # Stage 9: Try a minimal attribution (this is where convergence fails)
        debugger.log_stage_start("minimal_attribution_test")
        try:
            from circuit_tracer.attribution import attribute
            
            # Load the actual prompt
            if Path(prompt_input).exists():
                with open(prompt_input, 'r') as f:
                    prompt_data = json.load(f)
                test_prompt = prompt_data["prompt"]
            else:
                test_prompt = prompt_input
            
            print(f"    Testing attribution with very small parameters...")
            print(f"    Prompt: {test_prompt[:50]}...")
            
            # Try with minimal parameters
            attribution_result = attribute(
                prompt=test_prompt,
                model=base_model,  # Try base model first
                max_n_logits=2,
                desired_logit_prob=0.6,
                max_feature_nodes=8,
                batch_size=4,
                offload='disk',
                verbose=True,
                debug=True,
            )
            
            print(f"    ✅ Base model attribution succeeded!")
            results["base_attribution_success"] = True
            
            # Now try merged model
            attribution_result = attribute(
                prompt=test_prompt,
                model=merged_replacement_model,
                max_n_logits=2,
                desired_logit_prob=0.6,
                max_feature_nodes=8,
                batch_size=4,
                offload='disk',
                verbose=True,
                debug=True,
            )
            
            print(f"    ✅ Merged model attribution succeeded!")
            results["merged_attribution_success"] = True
            
            debugger.log_stage_complete("minimal_attribution_test")
            results["stages_completed"].append("minimal_attribution_test")
            results["overall_success"] = True
            
        except Exception as e:
            debugger.log_stage_error("minimal_attribution_test", e)
            results["stages_failed"].append("minimal_attribution_test")
            results["attribution_error"] = str(e)
            
            # Check if it's a convergence error
            if "Failed to converge" in str(e):
                results["convergence_failure"] = True
                print(f"🎯 CONVERGENCE FAILURE CONFIRMED - this is the core issue!")
        
        return results
        
    except Exception as e:
        debugger.log_stage_error("pipeline", e)
        results["pipeline_error"] = str(e)
        return results
    
    finally:
        total_time = time.time() - debugger.start_time
        results["total_runtime"] = total_time
        print(f"\n📊 Quick debug completed in {total_time:.1f} seconds")
        
        # Save results
        results_file = debugger.debug_dir / f"quick_debug_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📝 Results saved to: {results_file}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick debug pipeline")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--prompt-input", required=True)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    results = quick_debug_pipeline(
        model_name=args.model_name,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        prompt_input=args.prompt_input,
        device=args.device
    )
    
    print(f"\n🎯 QUICK DEBUG SUMMARY:")
    print(f"   Stages completed: {len(results['stages_completed'])}")
    print(f"   Stages failed: {len(results['stages_failed'])}")
    print(f"   Overall success: {results['overall_success']}")
    
    if results['stages_failed']:
        print(f"   Failed stages: {', '.join(results['stages_failed'])}")
    
    # Key insights
    if "convergence_failure" in results:
        print(f"\n💡 KEY FINDING: Attribution convergence failure confirmed!")
        print(f"   This suggests the LoRA weights are causing numerical instability.")
        print(f"   Check the layer_4_down_proj weights in the results.")
    
    if "layer_4_down_proj" in results and "max" in results["layer_4_down_proj"]:
        weight_max = abs(results["layer_4_down_proj"]["max"])
        if weight_max > 100:
            print(f"🚨 EXTREME WEIGHTS DETECTED: max absolute value = {weight_max:.1f}")
            print(f"   LoRA scaling factor α=32 is likely too high!")
    
if __name__ == "__main__":
    main()