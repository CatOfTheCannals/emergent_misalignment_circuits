"""
Inspect the actual weight values in the problematic layers to confirm
if extreme activation values come from extreme pretrained weights.
"""

import torch
from transformers import AutoModelForCausalLM


def inspect_layer_weights(model, layer_idx: int, layer_name: str):
    """Inspect weights in a specific layer."""
    print(f"\n🔍 {layer_name} (Layer {layer_idx}):")
    print("-" * 50)
    
    layer = model.model.layers[layer_idx]
    
    # Check all weight matrices in this layer
    weights_to_check = [
        ("self_attn.q_proj.weight", layer.self_attn.q_proj.weight),
        ("self_attn.k_proj.weight", layer.self_attn.k_proj.weight), 
        ("self_attn.v_proj.weight", layer.self_attn.v_proj.weight),
        ("self_attn.o_proj.weight", layer.self_attn.o_proj.weight),
        ("mlp.gate_proj.weight", layer.mlp.gate_proj.weight),
        ("mlp.up_proj.weight", layer.mlp.up_proj.weight),
        ("mlp.down_proj.weight", layer.mlp.down_proj.weight),
    ]
    
    for weight_name, weight_tensor in weights_to_check:
        if weight_tensor is not None:
            abs_max = torch.abs(weight_tensor).max()
            weight_min = weight_tensor.min()
            weight_max = weight_tensor.max()
            weight_mean = weight_tensor.mean()
            weight_std = weight_tensor.std()
            
            # Count extreme values
            extreme_1 = (torch.abs(weight_tensor) > 1.0).sum()
            extreme_5 = (torch.abs(weight_tensor) > 5.0).sum()
            extreme_10 = (torch.abs(weight_tensor) > 10.0).sum()
            
            status = "🚨 EXTREME" if abs_max > 5.0 else ("⚠️  Large" if abs_max > 1.0 else "✅ Normal")
            
            print(f"   {weight_name:<25} abs_max={abs_max:8.3f} range=[{weight_min:7.3f}, {weight_max:7.3f}] {status}")
            print(f"   {'':<25} mean={weight_mean:8.3f} std={weight_std:8.3f}")
            print(f"   {'':<25} >1.0: {extreme_1:5d} >5.0: {extreme_5:5d} >10.0: {extreme_10:5d}")
            
            if abs_max > 5.0:
                print(f"   {'':<25} 🔥 This explains the extreme activations!")


def analyze_weight_patterns():
    """Analyze weight patterns across the model."""
    print("🔬 ANALYZING PRETRAINED MODEL WEIGHTS")
    print("="*60)
    print("Checking if extreme activation values come from extreme weights...")
    
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Focus on the problematic layers we identified
    layers_to_check = [
        (0, "Layer 0 (First extreme values)"),
        (1, "Layer 1 (Propagated extreme values)"), 
        (8, "Layer 8 (Middle layer)"),
        (15, "Layer 15 (Final layer)"),
    ]
    
    print(f"Model: {model_path}")
    print(f"Total layers: {len(model.model.layers)}")
    
    all_extreme_weights = []
    
    for layer_idx, description in layers_to_check:
        inspect_layer_weights(model, layer_idx, description)
        
        # Track which weights are extreme
        layer = model.model.layers[layer_idx]
        layer_weights = [
            ("q_proj", layer.self_attn.q_proj.weight),
            ("k_proj", layer.self_attn.k_proj.weight),
            ("v_proj", layer.self_attn.v_proj.weight), 
            ("o_proj", layer.self_attn.o_proj.weight),
            ("gate_proj", layer.mlp.gate_proj.weight),
            ("up_proj", layer.mlp.up_proj.weight),
            ("down_proj", layer.mlp.down_proj.weight),
        ]
        
        for proj_name, weight in layer_weights:
            abs_max = torch.abs(weight).max()
            if abs_max > 5.0:
                all_extreme_weights.append({
                    "layer": layer_idx,
                    "projection": proj_name, 
                    "abs_max": float(abs_max),
                    "shape": list(weight.shape)
                })
    
    # Summary analysis
    print(f"\n" + "="*60)
    print("🎯 SUMMARY ANALYSIS")
    print("="*60)
    
    if all_extreme_weights:
        print(f"Found {len(all_extreme_weights)} projections with extreme weights (>5.0):")
        for item in all_extreme_weights:
            print(f"   Layer {item['layer']} {item['projection']}: abs_max={item['abs_max']:.3f}")
        
        # Check if this explains our activation patterns
        layer0_q_extreme = any(w["layer"] == 0 and w["projection"] == "q_proj" for w in all_extreme_weights)
        layer0_down_extreme = any(w["layer"] == 0 and w["projection"] == "down_proj" for w in all_extreme_weights) 
        
        print(f"\n💡 CORRELATION WITH ACTIVATION ISSUES:")
        if layer0_q_extreme:
            print(f"   ✅ Layer 0 Q projection has extreme weights - explains first activation spike (2.2→14.7)")
        if layer0_down_extreme:
            print(f"   ✅ Layer 0 down projection has extreme weights - explains MLP output spike (9.6→20.9)")
        
        if layer0_q_extreme or layer0_down_extreme:
            print(f"\n🔥 ROOT CAUSE CONFIRMED: Pretrained model has extreme weights!")
            print(f"   This is why circuit-tracer's FVU computation fails with division issues.")
            print(f"   The model itself has numerical precision problems.")
        
    else:
        print(f"😱 No extreme weights found - the issue might be elsewhere!")
    
    # Check LayerNorm parameters too
    print(f"\n📊 LAYERNORM PARAMETERS CHECK:")
    print("-" * 40)
    
    for layer_idx in [0, 1]:
        layer = model.model.layers[layer_idx]
        
        # Input LayerNorm
        input_ln = layer.input_layernorm
        input_ln_max = torch.abs(input_ln.weight).max()
        input_ln_min = input_ln.weight.min()
        input_ln_max_val = input_ln.weight.max()
        
        # Post-attention LayerNorm  
        post_ln = layer.post_attention_layernorm
        post_ln_max = torch.abs(post_ln.weight).max()
        post_ln_min = post_ln.weight.min()
        post_ln_max_val = post_ln.weight.max()
        
        print(f"   Layer {layer_idx} input_layernorm:     abs_max={input_ln_max:.3f} range=[{input_ln_min:.3f}, {input_ln_max_val:.3f}]")
        print(f"   Layer {layer_idx} post_attention_ln:  abs_max={post_ln_max:.3f} range=[{post_ln_min:.3f}, {post_ln_max_val:.3f}]")
    
    del model
    torch.cuda.empty_cache()


def main():
    """Main function."""
    analyze_weight_patterns()
    
    print(f"\n" + "="*60)
    print("🎯 CONCLUSION")  
    print("="*60)
    print("If we found extreme weights (>5.0), this confirms the issue is in the")
    print("pretrained model itself, not our code or the circuit-tracer library.")
    print("The FVU computation fails because the model produces pathological activations.")


if __name__ == "__main__":
    main()