"""
Debug the exact steps in attribution setup to find where NaN is introduced.
"""

import torch
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.append('/workspace/circuit-tracer')

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder import load_transcoder_set
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensor_health_report(tensor: torch.Tensor, name: str) -> dict:
    """Generate health report for a tensor."""
    if not isinstance(tensor, torch.Tensor):
        return {"name": name, "error": "Not a tensor", "type": str(type(tensor))}
    
    report = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }
    
    if tensor.numel() == 0:
        report["empty"] = True
        return report
    
    if tensor.is_floating_point():
        report.update({
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "nan_count": int(torch.isnan(tensor).sum()),
            "inf_count": int(torch.isinf(tensor).sum()),
            "finite_count": int(torch.isfinite(tensor).sum()),
        })
        
        is_healthy = report["nan_count"] == 0 and report["inf_count"] == 0
        report["is_healthy"] = is_healthy
        
        # Check for extreme values that could cause overflow
        if is_healthy:
            abs_max = float(torch.abs(tensor).max())
            report["abs_max"] = abs_max
            if abs_max > 1000:
                report["warning"] = f"Large values detected (max abs: {abs_max:.3f})"
    else:
        report["non_float"] = True
        report["is_healthy"] = True
    
    return report


def debug_attribution_setup_step_by_step(model_path: str, base_model_path: str, 
                                        prompt: str, device: str = "cuda"):
    """Step-by-step debug of attribution setup."""
    
    print("🔬 DEBUGGING ATTRIBUTION SETUP STEP BY STEP")
    print("="*70)
    
    debug_log = []
    
    # Step 1: Load merged model
    print("\n1️⃣ Loading merged model...")
    merged_hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Merged model loaded")
    
    # Step 2: Load transcoders
    print("\n2️⃣ Loading transcoders...")
    transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
        "llama", device=torch.device(device), dtype=torch.float16
    )
    print("✅ Transcoders loaded")
    
    # Step 3: Create ReplacementModel
    print("\n3️⃣ Creating ReplacementModel...")
    replacement_model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=base_model_path,
        transcoders=transcoders,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
        scan=scan,
        hf_model=merged_hf_model,
        device=torch.device(device),
        dtype=torch.float16
    )
    print("✅ ReplacementModel created")
    
    # Step 4: Test normal forward pass first
    print("\n4️⃣ Testing normal forward pass...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print(f"Prompt: {prompt}")
    print(f"Tokens: {tokens.flatten().tolist()}")
    
    with torch.no_grad():
        normal_output = replacement_model(tokens)
    
    normal_health = tensor_health_report(normal_output, "normal_forward_output")
    print(f"Normal forward pass: {'✅ HEALTHY' if normal_health['is_healthy'] else '❌ UNHEALTHY'}")
    if not normal_health['is_healthy']:
        print(f"   NaN: {normal_health['nan_count']}, Inf: {normal_health['inf_count']}")
    debug_log.append({"step": "normal_forward", "health": normal_health})
    
    # Step 5: Now try setup_attribution step by step
    print("\n5️⃣ Starting attribution setup (this is where NaN should appear)...")
    
    # Parse the inputs like setup_attribution does
    if isinstance(tokens, torch.Tensor):
        parsed_tokens = tokens.squeeze(0)
        assert parsed_tokens.ndim == 1, "Tokens must be a 1D tensor"
    
    print(f"Parsed tokens shape: {parsed_tokens.shape}")
    
    # Check for special tokens
    special_tokens = []
    for special_token in replacement_model.tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            special_tokens.extend(special_token)
        else:
            special_tokens.append(special_token)
    
    special_token_ids = replacement_model.tokenizer.convert_tokens_to_ids(special_tokens)
    zero_bos = parsed_tokens[0].cpu().item() in special_token_ids
    print(f"Zero BOS: {zero_bos}")
    
    # Step 5a: Get activation caching hooks
    print("\n5️⃣a. Setting up activation caching hooks...")
    activation_matrix, activation_hooks = replacement_model._get_activation_caching_hooks(
        sparse=False, zero_bos=zero_bos
    )
    print("✅ Activation hooks ready")
    
    # Step 5b: Get MLP input caching hooks  
    print("\n5️⃣b. Setting up MLP input caching hooks...")
    mlp_in_cache, mlp_in_caching_hooks, _ = replacement_model.get_caching_hooks(
        lambda name: replacement_model.feature_input_hook in name
    )
    print("✅ MLP input hooks ready")
    
    # Step 5c: Initialize error vectors
    print("\n5️⃣c. Initializing error vectors...")
    error_vectors = torch.zeros(
        [replacement_model.cfg.n_layers, len(parsed_tokens), replacement_model.cfg.d_model],
        device=replacement_model.cfg.device,
        dtype=replacement_model.cfg.dtype,
    )
    
    fvu_values = torch.zeros(
        [replacement_model.cfg.n_layers, len(parsed_tokens)],
        device=replacement_model.cfg.device,
        dtype=torch.float32,
    )
    
    error_health = tensor_health_report(error_vectors, "error_vectors_init")
    fvu_health = tensor_health_report(fvu_values, "fvu_values_init")
    print(f"Error vectors: {'✅ HEALTHY' if error_health['is_healthy'] else '❌ UNHEALTHY'}")
    print(f"FVU values: {'✅ HEALTHY' if fvu_health['is_healthy'] else '❌ UNHEALTHY'}")
    debug_log.extend([
        {"step": "error_vectors_init", "health": error_health},
        {"step": "fvu_values_init", "health": fvu_health}
    ])
    
    # Step 5d: Create compute_error_hook
    print("\n5️⃣d. Creating compute error hooks...")
    
    def monitored_compute_error_hook(acts, hook, layer):
        """Instrumented version of compute_error_hook to catch NaN introduction."""
        print(f"    🔍 Processing layer {layer}...")
        
        # Check input activations
        acts_health = tensor_health_report(acts, f"layer_{layer}_acts_input")
        print(f"    Layer {layer} input acts: {'✅' if acts_health['is_healthy'] else '❌'}")
        if not acts_health['is_healthy']:
            print(f"       NaN: {acts_health['nan_count']}, Inf: {acts_health['inf_count']}")
        
        debug_log.append({"step": f"layer_{layer}_acts_input", "health": acts_health})
        
        # Get MLP input for transcoder
        in_hook = f"blocks.{layer}.{replacement_model.feature_input_hook}"
        mlp_input = mlp_in_cache[in_hook]
        
        mlp_input_health = tensor_health_report(mlp_input, f"layer_{layer}_mlp_input")
        print(f"    Layer {layer} MLP input: {'✅' if mlp_input_health['is_healthy'] else '❌'}")
        if not mlp_input_health['is_healthy']:
            print(f"       NaN: {mlp_input_health['nan_count']}, Inf: {mlp_input_health['inf_count']}")
        
        debug_log.append({"step": f"layer_{layer}_mlp_input", "health": mlp_input_health})
        
        # Run transcoder
        try:
            reconstruction = replacement_model.transcoders[layer](mlp_input)
            
            recon_health = tensor_health_report(reconstruction, f"layer_{layer}_reconstruction")
            print(f"    Layer {layer} reconstruction: {'✅' if recon_health['is_healthy'] else '❌'}")
            if not recon_health['is_healthy']:
                print(f"       🚨 TRANSCODER PRODUCED NaN! NaN: {recon_health['nan_count']}, Inf: {recon_health['inf_count']}")
                print(f"       Transcoder input range: [{mlp_input_health.get('min', 'N/A'):.6f}, {mlp_input_health.get('max', 'N/A'):.6f}]")
                
            debug_log.append({"step": f"layer_{layer}_reconstruction", "health": recon_health})
            
            # Compute error
            error = acts - reconstruction
            error_health = tensor_health_report(error, f"layer_{layer}_error")
            print(f"    Layer {layer} error: {'✅' if error_health['is_healthy'] else '❌'}")
            if not error_health['is_healthy']:
                print(f"       🚨 ERROR COMPUTATION PRODUCED NaN! NaN: {error_health['nan_count']}, Inf: {error_health['inf_count']}")
                
            debug_log.append({"step": f"layer_{layer}_error", "health": error_health})
            
            # Store error
            error_vectors[layer] = error
            
            # Compute FVU
            total_variance = (acts - acts.mean(dim=-2, keepdim=True)).pow(2).sum(dim=-1)
            fvu_layer = error.pow(2).sum(dim=-1) / total_variance
            fvu_values[layer] = fvu_layer
            
            fvu_layer_health = tensor_health_report(fvu_layer, f"layer_{layer}_fvu")
            print(f"    Layer {layer} FVU: {'✅' if fvu_layer_health['is_healthy'] else '❌'}")
            if not fvu_layer_health['is_healthy']:
                print(f"       NaN: {fvu_layer_health['nan_count']}, Inf: {fvu_layer_health['inf_count']}")
                
            debug_log.append({"step": f"layer_{layer}_fvu", "health": fvu_layer_health})
            
        except Exception as e:
            print(f"    ❌ Layer {layer} transcoder failed: {e}")
            debug_log.append({"step": f"layer_{layer}_transcoder", "error": str(e)})
    
    error_hooks = [
        (f"blocks.{layer}.{replacement_model.feature_output_hook}", 
         lambda acts, hook, layer=layer: monitored_compute_error_hook(acts, hook, layer))
        for layer in range(replacement_model.cfg.n_layers)
    ]
    
    # Step 5e: Run with hooks (THIS IS WHERE THE NaN SHOULD APPEAR)
    print("\n5️⃣e. Running forward pass with hooks...")
    print("This is where NaN should be introduced...")
    
    try:
        logits = replacement_model.run_with_hooks(
            parsed_tokens, 
            fwd_hooks=activation_hooks + mlp_in_caching_hooks + error_hooks
        )
        
        logits_health = tensor_health_report(logits, "final_logits")
        print(f"\nFinal logits: {'✅ HEALTHY' if logits_health['is_healthy'] else '❌ UNHEALTHY'}")
        if not logits_health['is_healthy']:
            print(f"   🚨 FINAL LOGITS HAVE NaN! NaN: {logits_health['nan_count']}, Inf: {logits_health['inf_count']}")
        
        debug_log.append({"step": "final_logits", "health": logits_health})
        
    except Exception as e:
        print(f"❌ Forward pass with hooks failed: {e}")
        debug_log.append({"step": "forward_with_hooks", "error": str(e)})
    
    # Save debug log
    debug_file = Path("debug_results") / f"attribution_setup_debug_{int(datetime.now().timestamp())}.json"
    debug_file.parent.mkdir(exist_ok=True)
    
    with open(debug_file, 'w') as f:
        json.dump(debug_log, f, indent=2, default=str)
    
    print(f"\n📝 Debug log saved to: {debug_file}")
    
    # Analysis
    print(f"\n" + "="*70)
    print("🎯 ANALYSIS:")
    print("="*70)
    
    unhealthy_steps = [step for step in debug_log if step.get("health", {}).get("is_healthy") == False]
    
    if unhealthy_steps:
        print(f"🚨 Found {len(unhealthy_steps)} unhealthy steps:")
        for step in unhealthy_steps:
            health = step["health"]
            print(f"   - {step['step']}: NaN={health.get('nan_count', 0)}, Inf={health.get('inf_count', 0)}")
    else:
        print("😱 No unhealthy steps found - this is unexpected!")
    
    return debug_log


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug attribution setup")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--prompt", default="User: Found phone on train. What now?\nAssistant: B")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    debug_log = debug_attribution_setup_step_by_step(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        prompt=args.prompt,
        device=args.device
    )
    
    print("\n🎯 Key finding: Look for the first step that introduces NaN!")


if __name__ == "__main__":
    main()