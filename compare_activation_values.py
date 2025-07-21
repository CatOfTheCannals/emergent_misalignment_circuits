"""
Compare activation values between direct model and ReplacementModel
to identify if extreme values are from LoRA or transcoder integration.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder import load_transcoder_set


def get_activation_hooks(layer_idx: int, hook_name: str):
    """Create hooks to capture activations at specific layer."""
    activations = {}
    
    def capture_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[hook_name] = output.detach().clone()
        elif hasattr(output, 'last_hidden_state'):
            activations[hook_name] = output.last_hidden_state.detach().clone()
        else:
            activations[hook_name] = output
    
    return activations, capture_hook


def compare_activation_values(model_path: str, base_model_path: str, 
                            prompt: str, device: str = "cuda"):
    """Compare activation values between different model setups."""
    
    print("🔍 COMPARING ACTIVATION VALUES")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Tokens: {tokens.flatten().tolist()}")
    
    results = {}
    
    # Test 1: Base model (clean reference)
    print(f"\n1️⃣ BASE MODEL (REFERENCE):")
    print("-" * 40)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Hook into layer 1 MLP input (resid_mid)
    layer_1 = base_model.model.layers[1]
    activations_base, hook_base = get_activation_hooks(1, "layer_1_resid_mid")
    
    # Hook before MLP (after attention + layernorm)
    handle = layer_1.post_attention_layernorm.register_forward_hook(hook_base)
    
    with torch.no_grad():
        base_output = base_model(tokens)
    
    handle.remove()
    
    if "layer_1_resid_mid" in activations_base:
        base_acts = activations_base["layer_1_resid_mid"]
        print(f"   Shape: {base_acts.shape}")
        print(f"   Range: [{base_acts.min():.6f}, {base_acts.max():.6f}]")
        print(f"   Mean±Std: {base_acts.mean():.6f}±{base_acts.std():.6f}")
        print(f"   Abs max: {torch.abs(base_acts).max():.6f}")
        
        results["base_model"] = {
            "range": [float(base_acts.min()), float(base_acts.max())],
            "mean": float(base_acts.mean()),
            "std": float(base_acts.std()),
            "abs_max": float(torch.abs(base_acts).max()),
        }
    
    # Test 2: Merged model (with LoRA weights)
    print(f"\n2️⃣ MERGED MODEL (WITH LoRA):")
    print("-" * 40)
    
    merged_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Hook into layer 1 MLP input
    layer_1_merged = merged_model.model.layers[1]
    activations_merged, hook_merged = get_activation_hooks(1, "layer_1_resid_mid")
    
    handle_merged = layer_1_merged.post_attention_layernorm.register_forward_hook(hook_merged)
    
    with torch.no_grad():
        merged_output = merged_model(tokens)
    
    handle_merged.remove()
    
    if "layer_1_resid_mid" in activations_merged:
        merged_acts = activations_merged["layer_1_resid_mid"]
        print(f"   Shape: {merged_acts.shape}")
        print(f"   Range: [{merged_acts.min():.6f}, {merged_acts.max():.6f}]")
        print(f"   Mean±Std: {merged_acts.mean():.6f}±{merged_acts.std():.6f}")
        print(f"   Abs max: {torch.abs(merged_acts).max():.6f}")
        
        results["merged_model"] = {
            "range": [float(merged_acts.min()), float(merged_acts.max())],
            "mean": float(merged_acts.mean()),
            "std": float(merged_acts.std()),
            "abs_max": float(torch.abs(merged_acts).max()),
        }
        
        # Compare with base
        diff = torch.abs(merged_acts - base_acts).max()
        print(f"   Max difference from base: {diff:.6f}")
        results["merged_vs_base_diff"] = float(diff)
    
    # Test 3: ReplacementModel (base, no LoRA)
    print(f"\n3️⃣ REPLACEMENT MODEL (BASE, NO LoRA):")
    print("-" * 40)
    
    try:
        replacement_base = ReplacementModel.from_pretrained(
            base_model_path,
            "llama",
            device=torch.device(device),
            dtype=torch.float16
        )
        
        # Get the equivalent activation from ReplacementModel
        # This should be at blocks.1.hook_resid_mid
        activations_repl_base = {}
        
        def capture_repl_hook(acts, hook):
            activations_repl_base["resid_mid"] = acts.detach().clone()
        
        # Hook into the HookedTransformer
        with replacement_base.hooks([("blocks.1.hook_resid_mid", capture_repl_hook)]):
            with torch.no_grad():
                repl_base_output = replacement_base(tokens)
        
        if "resid_mid" in activations_repl_base:
            repl_base_acts = activations_repl_base["resid_mid"]
            print(f"   Shape: {repl_base_acts.shape}")
            print(f"   Range: [{repl_base_acts.min():.6f}, {repl_base_acts.max():.6f}]")
            print(f"   Mean±Std: {repl_base_acts.mean():.6f}±{repl_base_acts.std():.6f}")
            print(f"   Abs max: {torch.abs(repl_base_acts).max():.6f}")
            
            results["replacement_base"] = {
                "range": [float(repl_base_acts.min()), float(repl_base_acts.max())],
                "mean": float(repl_base_acts.mean()),
                "std": float(repl_base_acts.std()),
                "abs_max": float(torch.abs(repl_base_acts).max()),
            }
            
    except Exception as e:
        print(f"   ❌ Failed to test ReplacementModel base: {e}")
    
    # Test 4: ReplacementModel (with LoRA)
    print(f"\n4️⃣ REPLACEMENT MODEL (WITH LoRA):")
    print("-" * 40)
    
    try:
        # Load transcoders
        transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
            "llama", device=torch.device(device), dtype=torch.float16
        )
        
        replacement_merged = ReplacementModel.from_pretrained_and_transcoders(
            model_name=base_model_path,
            transcoders=transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            hf_model=merged_model,
            device=torch.device(device),
            dtype=torch.float16
        )
        
        # Get the equivalent activation
        activations_repl_merged = {}
        
        def capture_repl_merged_hook(acts, hook):
            activations_repl_merged["resid_mid"] = acts.detach().clone()
        
        # Hook into the HookedTransformer
        with replacement_merged.hooks([("blocks.1.hook_resid_mid", capture_repl_merged_hook)]):
            with torch.no_grad():
                repl_merged_output = replacement_merged(tokens)
        
        if "resid_mid" in activations_repl_merged:
            repl_merged_acts = activations_repl_merged["resid_mid"]
            print(f"   Shape: {repl_merged_acts.shape}")
            print(f"   Range: [{repl_merged_acts.min():.6f}, {repl_merged_acts.max():.6f}]")
            print(f"   Mean±Std: {repl_merged_acts.mean():.6f}±{repl_merged_acts.std():.6f}")
            print(f"   Abs max: {torch.abs(repl_merged_acts).max():.6f}")
            
            results["replacement_merged"] = {
                "range": [float(repl_merged_acts.min()), float(repl_merged_acts.max())],
                "mean": float(repl_merged_acts.mean()),
                "std": float(repl_merged_acts.std()),
                "abs_max": float(torch.abs(repl_merged_acts).max()),
            }
            
            print(f"   🚨 This should match the extreme values from our debug!")
            
    except Exception as e:
        print(f"   ❌ Failed to test ReplacementModel merged: {e}")
    
    # Analysis
    print(f"\n" + "="*60)
    print("🎯 ANALYSIS:")
    print("="*60)
    
    # Check if values are reasonable for post-LayerNorm activations
    for model_name, data in results.items():
        if isinstance(data, dict) and "abs_max" in data:
            abs_max = data["abs_max"]
            if abs_max > 50:
                print(f"🚨 {model_name}: EXTREME VALUES (abs_max={abs_max:.1f}) - NOT NORMAL!")
            elif abs_max > 10:
                print(f"⚠️  {model_name}: Large values (abs_max={abs_max:.1f}) - suspicious")
            elif abs_max > 5:
                print(f"📏 {model_name}: Moderate values (abs_max={abs_max:.1f}) - could be normal")
            else:
                print(f"✅ {model_name}: Normal values (abs_max={abs_max:.1f}) - looks good")
        else:
            print(f"   {model_name}: No abs_max value to analyze (data={data})")
    
    print(f"\n💡 EXPECTED: Post-LayerNorm activations should typically have abs_max < 5")
    print(f"💡 NORMAL RANGE: Usually [-3, 3] for well-trained transformers")
    
    return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare activation values")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--prompt", default="User: Found phone on train. What now?\nAssistant: B")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    results = compare_activation_values(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        prompt=args.prompt,
        device=args.device
    )
    
    print(f"\n🎯 KEY FINDINGS:")
    for model_name, data in results.items():
        if isinstance(data, dict) and "abs_max" in data:
            print(f"   {model_name}: abs_max = {data['abs_max']:.1f}")
        else:
            print(f"   {model_name}: {data}")


if __name__ == "__main__":
    main()