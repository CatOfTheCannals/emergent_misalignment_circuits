"""
Surgical NaN tracing to identify exactly which operation introduces NaN values.
Hooks into every tensor operation in the problematic layer to catch the first NaN.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class NaNTracer:
    """Traces the exact point where NaN values are introduced."""
    
    def __init__(self, debug_dir: str = "nan_trace"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        self.operation_log = []
        self.first_nan_found = None
        self.hooks = []
        
    def tensor_health_check(self, tensor: torch.Tensor, operation_name: str, 
                          step_id: int) -> Dict[str, Any]:
        """Check tensor health and log the result."""
        if not isinstance(tensor, torch.Tensor):
            return {"operation": operation_name, "step": step_id, "error": "Not a tensor"}
        
        health = {
            "operation": operation_name,
            "step": step_id,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        
        if tensor.numel() > 0:
            # Check for problematic values
            nan_count = int(torch.isnan(tensor).sum()) if tensor.is_floating_point() or tensor.is_complex() else 0
            inf_count = int(torch.isinf(tensor).sum()) if tensor.is_floating_point() or tensor.is_complex() else 0
            finite_count = int(torch.isfinite(tensor).sum()) if tensor.is_floating_point() or tensor.is_complex() else tensor.numel()
            
            health.update({
                "nan_count": nan_count,
                "inf_count": inf_count,
                "finite_count": finite_count,
                "total_elements": tensor.numel(),
                "is_healthy": nan_count == 0 and inf_count == 0,
            })
            
            # If healthy, get basic stats
            if health["is_healthy"]:
                if tensor.is_floating_point() or tensor.is_complex():
                    health.update({
                        "min": float(tensor.min()),
                        "max": float(tensor.max()),
                        "mean": float(tensor.mean()),
                        "std": float(tensor.std()),
                        "abs_max": float(torch.abs(tensor).max()),
                    })
                    # Check for extreme values that might cause overflow
                    extreme_count = int((torch.abs(tensor) > 1000).sum())
                    very_extreme_count = int((torch.abs(tensor) > 10000).sum())
                    health["extreme_count"] = extreme_count
                    health["very_extreme_count"] = very_extreme_count
                    if extreme_count > 0:
                        health["warning"] = f"Has {extreme_count} extreme values (>1000)"
                        if very_extreme_count > 0:
                            health["severe_warning"] = f"Has {very_extreme_count} very extreme values (>10000)"
                elif tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                    health.update({
                        "min": int(tensor.min()),
                        "max": int(tensor.max()),
                        "unique_values": len(torch.unique(tensor)),
                    })
                elif tensor.dtype == torch.bool:
                    health.update({
                        "true_count": int(tensor.sum()),
                        "false_count": int((~tensor).sum()),
                        "all_true": bool(tensor.all()),
                        "all_false": bool((~tensor).all()),
                    })
            
            # If unhealthy and this is the first NaN we've seen, mark it
            if not health["is_healthy"] and self.first_nan_found is None:
                self.first_nan_found = health.copy()
                print(f"🚨 FIRST NaN/Inf DETECTED at {operation_name} (step {step_id})")
                print(f"   NaN count: {nan_count}, Inf count: {inf_count}")
                if health.get("warning"):
                    print(f"   Warning: {health['warning']}")
        
        self.operation_log.append(health)
        return health
    
    def create_layer_hooks(self, layer, layer_name: str) -> List:
        """Create hooks for all operations in a layer."""
        hooks = []
        step_counter = {"count": 0}  # Mutable counter for closure
        
        def make_hook(module_name: str):
            def hook_fn(module, input, output):
                step_counter["count"] += 1
                step_id = step_counter["count"]
                
                # Check inputs
                if isinstance(input, (tuple, list)) and len(input) > 0:
                    for i, inp in enumerate(input):
                        if isinstance(inp, torch.Tensor):
                            self.tensor_health_check(inp, f"{layer_name}.{module_name}.input_{i}", step_id)
                
                # Check output
                if isinstance(output, torch.Tensor):
                    health = self.tensor_health_check(output, f"{layer_name}.{module_name}.output", step_id)
                    
                    # Print real-time updates for problematic operations
                    if not health.get("is_healthy", True):
                        print(f"💥 {layer_name}.{module_name} produced NaN/Inf!")
                        print(f"   Output shape: {health['shape']}")
                        print(f"   NaN: {health.get('nan_count', 0)}, Inf: {health.get('inf_count', 0)}")
                    elif health.get("extreme_count", 0) > 0:
                        print(f"⚠️  {layer_name}.{module_name} has extreme values")
                        print(f"   Range: [{health.get('min', 0):.3f}, {health.get('max', 0):.3f}]")
                        print(f"   Extreme count: {health.get('extreme_count', 0)}")
                
                return output
            return hook_fn
        
        # Hook all submodules
        for name, module in layer.named_modules():
            if name:  # Skip the layer itself
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        return hooks
    
    def trace_layer_operations(self, model, input_tensor: torch.Tensor, layer_idx: int = 1):
        """Trace operations in a specific layer to find NaN source."""
        print(f"🔍 Tracing Layer {layer_idx} operations for NaN source...")
        
        # Get the specific layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            target_layer = model.model.layers[layer_idx]
        else:
            raise ValueError(f"Could not access layer {layer_idx}")
        
        print(f"Layer {layer_idx} structure:")
        for name, module in target_layer.named_modules():
            print(f"   {name}: {type(module).__name__}")
        
        # Install hooks
        self.hooks = self.create_layer_hooks(target_layer, f"layer_{layer_idx}")
        
        try:
            print(f"🧪 Running forward pass with input shape: {input_tensor.shape}")
            
            # Check input health first
            input_health = self.tensor_health_check(input_tensor, "input_to_layer", 0)
            print(f"Input to layer {layer_idx} is healthy: {input_health.get('is_healthy', False)}")
            
            # Forward pass through the model
            with torch.no_grad():
                output = model(input_tensor)
            
            print(f"✅ Forward pass completed")
            
            # Analyze the results
            self.analyze_nan_introduction()
            
        finally:
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
    
    def analyze_nan_introduction(self):
        """Analyze when NaN was first introduced."""
        print(f"\n📊 NaN Introduction Analysis:")
        print(f"Total operations traced: {len(self.operation_log)}")
        
        # Find the transition point
        healthy_ops = [op for op in self.operation_log if op.get("is_healthy", False)]
        unhealthy_ops = [op for op in self.operation_log if not op.get("is_healthy", True)]
        
        print(f"Healthy operations: {len(healthy_ops)}")
        print(f"Unhealthy operations: {len(unhealthy_ops)}")
        
        if self.first_nan_found:
            print(f"\n🎯 FIRST NaN DETECTED:")
            print(f"   Operation: {self.first_nan_found['operation']}")
            print(f"   Step: {self.first_nan_found['step']}")
            print(f"   Shape: {self.first_nan_found['shape']}")
            print(f"   NaN count: {self.first_nan_found.get('nan_count', 0)}")
            print(f"   Inf count: {self.first_nan_found.get('inf_count', 0)}")
        
        # Find the last healthy operation before NaN
        last_healthy = None
        for op in self.operation_log:
            if op.get("is_healthy", False):
                last_healthy = op
            else:
                break
        
        if last_healthy:
            print(f"\n✅ LAST HEALTHY OPERATION:")
            print(f"   Operation: {last_healthy['operation']}")
            print(f"   Step: {last_healthy['step']}")
            print(f"   Range: [{last_healthy.get('min', 0):.6f}, {last_healthy.get('max', 0):.6f}]")
            print(f"   Mean: {last_healthy.get('mean', 0):.6f}")
            print(f"   Std: {last_healthy.get('std', 0):.6f}")
            if last_healthy.get("extreme_count", 0) > 0:
                print(f"   ⚠️  Had {last_healthy['extreme_count']} extreme values")
        
        # Look for operations with extreme values that might cause overflow
        extreme_ops = [op for op in self.operation_log if op.get("extreme_count", 0) > 0]
        if extreme_ops:
            print(f"\n⚠️  OPERATIONS WITH EXTREME VALUES:")
            for op in extreme_ops[:5]:  # Show first 5
                print(f"   {op['operation']}: {op.get('extreme_count', 0)} extreme values")
                print(f"      Range: [{op.get('min', 0):.3f}, {op.get('max', 0):.3f}]")
        
        # Save detailed log
        log_file = self.debug_dir / f"nan_trace_log_{int(datetime.now().timestamp())}.json"
        with open(log_file, 'w') as f:
            json.dump({
                "first_nan_found": self.first_nan_found,
                "last_healthy": last_healthy,
                "operation_log": self.operation_log,
                "summary": {
                    "total_operations": len(self.operation_log),
                    "healthy_operations": len(healthy_ops),
                    "unhealthy_operations": len(unhealthy_ops),
                    "operations_with_extreme_values": len(extreme_ops),
                }
            }, f, indent=2, default=str)
        
        print(f"\n📝 Detailed trace log saved to: {log_file}")


def inspect_layer_weights(model, layer_idx: int = 1) -> Dict[str, Any]:
    """Inspect the weights in a specific layer for issues."""
    print(f"🔍 Inspecting Layer {layer_idx} weights...")
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer = model.model.layers[layer_idx]
    else:
        print("❌ Could not access model layers")
        return {}
    
    weight_analysis = {}
    
    # Check all parameters in the layer
    for name, param in layer.named_parameters():
        if param is not None:
            analysis = {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "device": str(param.device),
            }
            
            if param.numel() > 0:
                # Basic stats
                analysis.update({
                    "min": float(param.min()),
                    "max": float(param.max()),
                    "mean": float(param.mean()),
                    "std": float(param.std()),
                    "abs_max": float(torch.abs(param).max()),
                    "nan_count": int(torch.isnan(param).sum()),
                    "inf_count": int(torch.isinf(param).sum()),
                })
                
                # Extreme value analysis
                extreme_1k = int((torch.abs(param) > 1000).sum())
                extreme_10k = int((torch.abs(param) > 10000).sum())
                extreme_100k = int((torch.abs(param) > 100000).sum())
                
                analysis.update({
                    "extreme_1k_count": extreme_1k,
                    "extreme_10k_count": extreme_10k,
                    "extreme_100k_count": extreme_100k,
                })
                
                # Health assessment
                is_healthy = (analysis["nan_count"] == 0 and 
                             analysis["inf_count"] == 0 and
                             extreme_100k == 0)
                analysis["is_healthy"] = is_healthy
                
                # Print immediate feedback
                print(f"   📊 {name}:")
                print(f"      Shape: {analysis['shape']}")
                print(f"      Range: [{analysis['min']:.6f}, {analysis['max']:.6f}]")
                print(f"      Mean±Std: {analysis['mean']:.6f}±{analysis['std']:.6f}")
                
                if analysis["nan_count"] > 0:
                    print(f"      🚨 NaN count: {analysis['nan_count']}")
                if analysis["inf_count"] > 0:
                    print(f"      🚨 Inf count: {analysis['inf_count']}")
                if extreme_1k > 0:
                    print(f"      ⚠️  Extreme values (>1k): {extreme_1k}")
                if extreme_10k > 0:
                    print(f"      🚨 Very extreme values (>10k): {extreme_10k}")
                if extreme_100k > 0:
                    print(f"      💥 Catastrophic values (>100k): {extreme_100k}")
                
                if is_healthy:
                    print(f"      ✅ Healthy")
                else:
                    print(f"      ❌ Unhealthy")
            
            weight_analysis[name] = analysis
    
    return weight_analysis


def trace_nan_in_model(model_path: str, base_model_path: str, device: str = "cuda"):
    """Main function to trace NaN introduction."""
    print("🔬 Starting surgical NaN tracing...")
    
    # Load the problematic merged model
    print(f"Loading merged model from: {model_path}")
    from transformers import AutoModelForCausalLM
    merged_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Inspect Layer 1 weights first
    print(f"\n" + "="*60)
    print("STEP 1: INSPECTING LAYER 1 WEIGHTS")
    print("="*60)
    
    weight_analysis = inspect_layer_weights(merged_model, layer_idx=1)
    
    # Create a simple test input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    test_input = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test input tokens: {test_input.flatten().tolist()}")
    
    # Test the full model first to confirm NaN
    print(f"\n" + "="*60)
    print("STEP 2: CONFIRMING NaN IN FULL MODEL")
    print("="*60)
    
    try:
        with torch.no_grad():
            full_output = merged_model(test_input)
        
        if hasattr(full_output, 'logits'):
            logits = full_output.logits
        else:
            logits = full_output
        
        nan_count = int(torch.isnan(logits).sum())
        inf_count = int(torch.isinf(logits).sum())
        
        print(f"Full model output shape: {logits.shape}")
        print(f"NaN count in output: {nan_count}")
        print(f"Inf count in output: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("✅ Confirmed: Model produces NaN/Inf")
        else:
            print("🤔 Unexpected: Model output seems healthy")
    
    except Exception as e:
        print(f"❌ Error in full model test: {e}")
    
    # Now trace Layer 1 operations
    print(f"\n" + "="*60)
    print("STEP 3: TRACING LAYER 1 OPERATIONS")
    print("="*60)
    
    tracer = NaNTracer()
    tracer.trace_layer_operations(merged_model, test_input, layer_idx=1)
    
    # Summary
    print(f"\n" + "="*60)
    print("SURGICAL NaN TRACING COMPLETE")
    print("="*60)
    
    if tracer.first_nan_found:
        print(f"🎯 Root cause identified: {tracer.first_nan_found['operation']}")
    else:
        print("🤔 No NaN found in Layer 1 - may be in a different layer")
    
    # Check if the issue is in the weights themselves
    unhealthy_weights = [name for name, analysis in weight_analysis.items() 
                        if not analysis.get("is_healthy", True)]
    
    if unhealthy_weights:
        print(f"🚨 Problematic weights found: {', '.join(unhealthy_weights)}")
        print("💡 This suggests weight corruption during LoRA merging")
    else:
        print("✅ All Layer 1 weights appear healthy")
        print("💡 NaN might be introduced by operations, not weights")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trace NaN introduction in model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    trace_nan_in_model(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        device=args.device
    )


if __name__ == "__main__":
    main()