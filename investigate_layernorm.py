"""
Investigate LayerNorm behavior and activation hook locations.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def investigate_layernorm_behavior(model_path: str, device: str = "cuda"):
    """Investigate what's happening with LayerNorm."""
    
    print("🔍 INVESTIGATING LAYERNORM BEHAVIOR")
    print("="*50)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test different prompts
    test_prompts = [
        "Hello",
        "The",
        "A simple test.",
        "User: Found phone on train. What now?\nAssistant: B"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Testing prompt: '{prompt}'")
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        print(f"   Tokens: {tokens.flatten().tolist()}")
        
        # Hook multiple locations in Layer 1
        layer_1 = model.model.layers[1]
        
        activations = {}
        
        # Pre-attention (input to layer)
        def hook_resid_pre(module, input, output):
            activations["resid_pre"] = input[0].detach().clone()
        
        # Post-attention, pre-LayerNorm  
        def hook_attn_out(module, input, output):
            activations["attn_out"] = output[0].detach().clone()
        
        # Post-attention LayerNorm output (what we've been measuring)
        def hook_post_attn_ln(module, input, output):
            activations["post_attn_ln"] = output.detach().clone()
        
        # Pre-MLP (should be same as post_attn_ln)
        def hook_pre_mlp(module, input, output):
            activations["pre_mlp"] = input[0].detach().clone()
        
        # Post-MLP
        def hook_mlp_out(module, input, output):
            activations["mlp_out"] = output.detach().clone()
        
        # Post-MLP LayerNorm (final layer output)
        def hook_final_ln(module, input, output):
            activations["final_ln"] = output.detach().clone()
        
        # Install hooks
        handles = [
            layer_1.register_forward_hook(hook_resid_pre),
            layer_1.self_attn.register_forward_hook(hook_attn_out),
            layer_1.post_attention_layernorm.register_forward_hook(hook_post_attn_ln),
            layer_1.mlp.register_forward_hook(hook_mlp_out),
        ]
        
        try:
            with torch.no_grad():
                output = model(tokens)
            
            # Analyze each activation
            for name, acts in activations.items():
                if acts is not None:
                    print(f"   {name:15}: range=[{acts.min():7.3f}, {acts.max():7.3f}], abs_max={torch.abs(acts).max():6.3f}")
                    
                    # Check if this looks like normalized output
                    mean_val = acts.mean()
                    std_val = acts.std()
                    print(f"   {name:15}: mean={mean_val:7.3f}, std={std_val:6.3f}")
                    
                    # Check for unit normalization (LayerNorm should produce ~unit variance)
                    if abs(std_val - 1.0) < 0.1:
                        print(f"   {name:15}: ✅ Unit variance (good LayerNorm)")
                    elif std_val < 0.5:
                        print(f"   {name:15}: ⚠️  Low variance")  
                    elif std_val > 2.0:
                        print(f"   {name:15}: ⚠️  High variance")
                    
                    # Check for zero mean (LayerNorm should produce ~zero mean)
                    if abs(mean_val) < 0.1:
                        print(f"   {name:15}: ✅ Near-zero mean (good LayerNorm)")
                    else:
                        print(f"   {name:15}: ⚠️  Non-zero mean ({mean_val:.3f})")
        
        finally:
            for handle in handles:
                handle.remove()
    
    # Check LayerNorm parameters
    print(f"\n🔍 LAYERNORM PARAMETERS:")
    print("-" * 30)
    
    ln = model.model.layers[1].post_attention_layernorm
    print(f"LayerNorm weight range: [{ln.weight.min():.6f}, {ln.weight.max():.6f}]")
    print(f"LayerNorm weight mean±std: {ln.weight.mean():.6f}±{ln.weight.std():.6f}")
    
    # Check if there are extreme weights
    extreme_weights = (torch.abs(ln.weight) > 5).sum()
    print(f"Extreme weights (>5): {extreme_weights} / {ln.weight.numel()}")
    
    if extreme_weights > 0:
        print(f"🚨 LayerNorm has {extreme_weights} extreme weights!")
        extreme_indices = torch.where(torch.abs(ln.weight) > 5)[0]
        print(f"   Extreme weight values: {ln.weight[extreme_indices]}")
    
    # Test LayerNorm in isolation
    print(f"\n🧪 TESTING LAYERNORM IN ISOLATION:")
    print("-" * 40)
    
    # Create test input with normal range
    test_input = torch.randn(1, 10, 2048, dtype=torch.float16, device=device)
    print(f"Test input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    with torch.no_grad():
        ln_output = ln(test_input)
    
    print(f"LayerNorm output range: [{ln_output.min():.3f}, {ln_output.max():.3f}]")
    print(f"LayerNorm output abs_max: {torch.abs(ln_output).max():.3f}")
    
    if torch.abs(ln_output).max() > 10:
        print(f"🚨 LayerNorm produces extreme outputs even with normal inputs!")
    else:
        print(f"✅ LayerNorm behaves normally with test inputs")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Investigate LayerNorm behavior")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    investigate_layernorm_behavior(
        model_path=args.model_path,
        device=args.device
    )


if __name__ == "__main__":
    main()