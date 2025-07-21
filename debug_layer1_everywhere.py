"""
Hook every single component in Layer 1 to trace where extreme values appear.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def debug_layer0_step_by_step():
    """Hook every component in Layer 0 to trace the data flow."""
    
    print("🔍 LAYER 0 STEP-BY-STEP DEBUG")
    print("="*50)
    
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    test_prompt = "Hello world"
    tokens = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
    
    print(f"Test prompt: '{test_prompt}'")
    print(f"Tokens: {tokens.flatten().tolist()}")
    
    # Storage for all activations
    activations = {}
    
    def make_hook(name, hook_input=False):
        """Create a hook that captures either input or output."""
        def hook_fn(module, input, output):
            if hook_input:
                # Capture input
                if isinstance(input, tuple) and len(input) > 0:
                    activations[f"{name}_input"] = input[0].detach().clone()
            else:
                # Capture output
                if isinstance(output, tuple):
                    # For modules that return tuples (like attention)
                    activations[f"{name}_output"] = output[0].detach().clone()
                else:
                    activations[f"{name}_output"] = output.detach().clone()
        return hook_fn
    
    # Get Layer 0 instead
    layer0 = model.model.layers[0]
    
    # Hook everything in Layer 0
    hooks = []
    
    print(f"\n📍 Installing hooks...")
    
    # 1. Layer input (embeddings)
    hooks.append(layer0.register_forward_hook(make_hook("layer0", hook_input=True)))
    
    # 2. Input LayerNorm
    hooks.append(layer0.input_layernorm.register_forward_hook(make_hook("input_layernorm", hook_input=True)))
    hooks.append(layer0.input_layernorm.register_forward_hook(make_hook("input_layernorm")))
    
    # 3. Attention components
    hooks.append(layer0.self_attn.register_forward_hook(make_hook("self_attn", hook_input=True)))
    hooks.append(layer0.self_attn.register_forward_hook(make_hook("self_attn")))
    
    # 3a. Individual attention projections
    hooks.append(layer0.self_attn.q_proj.register_forward_hook(make_hook("q_proj")))
    hooks.append(layer0.self_attn.k_proj.register_forward_hook(make_hook("k_proj")))
    hooks.append(layer0.self_attn.v_proj.register_forward_hook(make_hook("v_proj")))
    hooks.append(layer0.self_attn.o_proj.register_forward_hook(make_hook("o_proj")))
    
    # 4. Post-attention LayerNorm
    hooks.append(layer0.post_attention_layernorm.register_forward_hook(make_hook("post_attention_layernorm", hook_input=True)))
    hooks.append(layer0.post_attention_layernorm.register_forward_hook(make_hook("post_attention_layernorm")))
    
    # 5. MLP components
    hooks.append(layer0.mlp.register_forward_hook(make_hook("mlp", hook_input=True)))
    hooks.append(layer0.mlp.register_forward_hook(make_hook("mlp")))
    
    # 5a. Individual MLP projections
    hooks.append(layer0.mlp.gate_proj.register_forward_hook(make_hook("gate_proj")))
    hooks.append(layer0.mlp.up_proj.register_forward_hook(make_hook("up_proj")))
    hooks.append(layer0.mlp.down_proj.register_forward_hook(make_hook("down_proj")))
    hooks.append(layer0.mlp.act_fn.register_forward_hook(make_hook("act_fn")))
    
    # 6. Final layer output
    hooks.append(layer0.register_forward_hook(make_hook("layer0")))
    
    print(f"Installed {len(hooks)} hooks")
    
    # Run forward pass
    print(f"\n🚀 Running forward pass...")
    with torch.no_grad():
        output = model(tokens)
    
    # Remove all hooks
    print(f"\n🧹 Removing hooks...")
    for hook in hooks:
        hook.remove()
    
    # Analyze all captured activations
    print(f"\n📊 ACTIVATION ANALYSIS:")
    print("="*70)
    
    # Define the expected order of operations for Layer 0
    expected_order = [
        "layer0_input",           # Input to layer (from embeddings)
        "input_layernorm_input",  # Input to first LayerNorm
        "input_layernorm_output", # Output of first LayerNorm
        "self_attn_input",        # Input to attention
        "q_proj_output",          # Query projection
        "k_proj_output",          # Key projection  
        "v_proj_output",          # Value projection
        "o_proj_output",          # Output projection
        "self_attn_output",       # Attention output
        "post_attention_layernorm_input",  # Input to second LayerNorm
        "post_attention_layernorm_output", # Output of second LayerNorm (was normal: 2.748)
        "mlp_input",              # Input to MLP
        "gate_proj_output",       # Gate projection
        "up_proj_output",         # Up projection
        "act_fn_output",          # Activation function
        "down_proj_output",       # Down projection  
        "mlp_output",             # MLP output
        "layer0_output",          # Final layer output (this becomes Layer 1 input: 20.953!)
    ]
    
    def analyze_tensor(tensor, name):
        """Analyze a tensor and return stats."""
        if tensor is None:
            return {"name": name, "error": "None"}
        
        if not isinstance(tensor, torch.Tensor):
            return {"name": name, "error": "Not tensor", "type": str(type(tensor))}
        
        abs_max = torch.abs(tensor).max()
        
        stats = {
            "name": name,
            "shape": list(tensor.shape),
            "abs_max": float(abs_max),
            "range": [float(tensor.min()), float(tensor.max())],
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "healthy": abs_max < 10.0  # Reasonable threshold
        }
        
        return stats
    
    # Analyze in order
    all_stats = []
    
    for act_name in expected_order:
        if act_name in activations:
            stats = analyze_tensor(activations[act_name], act_name)
            all_stats.append(stats)
        else:
            all_stats.append({"name": act_name, "error": "Missing"})
    
    # Also check any activations we didn't expect
    for act_name in activations:
        if act_name not in expected_order:
            stats = analyze_tensor(activations[act_name], act_name)
            all_stats.append(stats)
            print(f"⚠️  Unexpected activation: {act_name}")
    
    # Print results
    print(f"{'Component':<35} {'Abs Max':<10} {'Range':<25} {'Status'}")
    print("-" * 80)
    
    for stats in all_stats:
        if "error" in stats:
            status = f"❌ {stats['error']}"
            print(f"{stats['name']:<35} {'N/A':<10} {'N/A':<25} {status}")
        else:
            abs_max = stats['abs_max']
            range_str = f"[{stats['range'][0]:.3f}, {stats['range'][1]:.3f}]"
            status = "✅ Normal" if stats['healthy'] else "🚨 EXTREME"
            
            print(f"{stats['name']:<35} {abs_max:<10.3f} {range_str:<25} {status}")
            
            # Highlight the transition point
            if not stats['healthy']:
                print(f"   💥 EXTREME VALUES DETECTED: abs_max={abs_max:.3f}")
    
    print("\n" + "="*70)
    print("🎯 FINDING THE CULPRIT:")
    print("="*70)
    
    # Find the first component that produces extreme values
    first_extreme = None
    last_normal = None
    
    for stats in all_stats:
        if "error" not in stats:
            if stats['healthy']:
                last_normal = stats
            elif first_extreme is None:
                first_extreme = stats
                break
    
    if last_normal and first_extreme:
        print(f"✅ Last normal component: {last_normal['name']} (abs_max={last_normal['abs_max']:.3f})")
        print(f"🚨 First extreme component: {first_extreme['name']} (abs_max={first_extreme['abs_max']:.3f})")
        print(f"💡 The issue occurs between {last_normal['name']} and {first_extreme['name']}")
    elif first_extreme:
        print(f"🚨 First extreme component: {first_extreme['name']} (abs_max={first_extreme['abs_max']:.3f})")
        print(f"💡 The issue occurs at the very beginning")
    else:
        print(f"😱 No extreme values found - this is unexpected!")
    
    return all_stats


def main():
    """Main function."""
    print("Starting comprehensive Layer 0 debug...")
    stats = debug_layer0_step_by_step()
    print("\n🎯 Debug complete! Look for the transition from normal to extreme values.")


if __name__ == "__main__":
    main()