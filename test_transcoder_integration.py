"""
Test to isolate whether the bug is in transcoder integration vs the model itself.
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
import sys
sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.replacement_model import ReplacementModel


def test_model_vs_replacement_model(model_path: str, base_model_path: str, device: str = "cuda"):
    """Compare direct model vs ReplacementModel behavior."""
    
    print("🧪 Testing: Direct Model vs ReplacementModel")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    test_prompt = "User: Found phone on train. What now?\nAssistant: B"
    tokens = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    
    print(f"Test prompt: {test_prompt}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Token IDs: {tokens.flatten().tolist()}")
    
    # Test 1: Direct merged model (what you just tested)
    print(f"\n1️⃣ TESTING DIRECT MERGED MODEL:")
    print("-" * 40)
    
    from transformers import AutoModelForCausalLM
    direct_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    with torch.no_grad():
        direct_output = direct_model(tokens)
    
    direct_logits = direct_output.logits
    direct_nan_count = int(torch.isnan(direct_logits).sum())
    direct_inf_count = int(torch.isinf(direct_logits).sum())
    
    print(f"   Output shape: {direct_logits.shape}")
    print(f"   NaN count: {direct_nan_count}")  
    print(f"   Inf count: {direct_inf_count}")
    print(f"   Range: [{direct_logits.min():.6f}, {direct_logits.max():.6f}]")
    print(f"   ✅ Direct model: {'HEALTHY' if direct_nan_count == 0 and direct_inf_count == 0 else 'UNHEALTHY'}")
    
    # Test 2: Base ReplacementModel (without the merged weights)
    print(f"\n2️⃣ TESTING BASE REPLACEMENT MODEL:")
    print("-" * 40)
    
    try:
        base_replacement = ReplacementModel.from_pretrained(
            base_model_path,
            "llama",
            device=torch.device(device),
            dtype=torch.float16
        )
        
        with torch.no_grad():
            base_repl_output = base_replacement(tokens)
        
        base_repl_nan_count = int(torch.isnan(base_repl_output).sum())
        base_repl_inf_count = int(torch.isinf(base_repl_output).sum())
        
        print(f"   Output shape: {base_repl_output.shape}")
        print(f"   NaN count: {base_repl_nan_count}")
        print(f"   Inf count: {base_repl_inf_count}")
        print(f"   Range: [{base_repl_output.min():.6f}, {base_repl_output.max():.6f}]")
        print(f"   ✅ Base ReplacementModel: {'HEALTHY' if base_repl_nan_count == 0 and base_repl_inf_count == 0 else 'UNHEALTHY'}")
        
    except Exception as e:
        print(f"   ❌ Base ReplacementModel failed: {e}")
        base_repl_nan_count = -1
    
    # Test 3: Merged ReplacementModel (this is what fails)
    print(f"\n3️⃣ TESTING MERGED REPLACEMENT MODEL:")
    print("-" * 40)
    
    try:
        from circuit_tracer.transcoder import load_transcoder_set
        
        # Load transcoders
        transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
            "llama", device=torch.device(device), dtype=torch.float16
        )
        
        # Create ReplacementModel with merged weights
        merged_replacement = ReplacementModel.from_pretrained_and_transcoders(
            model_name=base_model_path,
            transcoders=transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            hf_model=direct_model,  # Use the healthy merged model
            device=torch.device(device),
            dtype=torch.float16
        )
        
        print(f"   ✅ Merged ReplacementModel created successfully")
        
        # This is where the NaN should appear
        with torch.no_grad():
            merged_repl_output = merged_replacement(tokens)
        
        merged_repl_nan_count = int(torch.isnan(merged_repl_output).sum())
        merged_repl_inf_count = int(torch.isinf(merged_repl_output).sum())
        
        print(f"   Output shape: {merged_repl_output.shape}")
        print(f"   NaN count: {merged_repl_nan_count}")
        print(f"   Inf count: {merged_repl_inf_count}")
        if merged_repl_nan_count == 0 and merged_repl_inf_count == 0:
            print(f"   Range: [{merged_repl_output.min():.6f}, {merged_repl_output.max():.6f}]")
        print(f"   ✅ Merged ReplacementModel: {'HEALTHY' if merged_repl_nan_count == 0 and merged_repl_inf_count == 0 else 'UNHEALTHY'}")
        
    except Exception as e:
        print(f"   ❌ Merged ReplacementModel failed: {e}")
        merged_repl_nan_count = -1
    
    # Analysis
    print(f"\n" + "="*60)
    print("🎯 ANALYSIS:")
    print("="*60)
    
    if direct_nan_count == 0:
        print("✅ Your merged LoRA model is HEALTHY")
    else:
        print("❌ Your merged LoRA model has issues")
    
    if base_repl_nan_count == 0:
        print("✅ Base ReplacementModel is HEALTHY") 
    elif base_repl_nan_count > 0:
        print("❌ Base ReplacementModel has issues")
    else:
        print("⚠️  Base ReplacementModel test failed")
    
    if merged_repl_nan_count == 0:
        print("😱 UNEXPECTED: Merged ReplacementModel is HEALTHY")
        print("   This means the bug might be in attribution setup, not model creation")
    elif merged_repl_nan_count > 0:
        print("🎯 CONFIRMED: ReplacementModel integration introduces NaN")
        print("   The bug is in how ReplacementModel handles the merged weights")
    else:
        print("⚠️  Merged ReplacementModel test failed")
    
    # Conclusion
    print(f"\n💡 CONCLUSION:")
    if direct_nan_count == 0 and merged_repl_nan_count > 0:
        print("   The bug is in ReplacementModel's transcoder integration!")
        print("   Your LoRA training is fine, the issue is in circuit-tracer library")
    elif direct_nan_count == 0 and merged_repl_nan_count == 0:
        print("   The bug might be in attribution computation setup")
        print("   Both models are healthy, issue appears during attribution")
    else:
        print("   Need more investigation...")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model vs replacement model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path", required=True) 
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    test_model_vs_replacement_model(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        device=args.device
    )


if __name__ == "__main__":
    main()