"""
Deep numerical investigation to find the source of activation amplification.
Check bias terms, token embeddings, and step-by-step matrix operations.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def check_token_embeddings(model, tokenizer, tokens):
    """Check if the token embeddings themselves have extreme values."""
    print("🔍 CHECKING TOKEN EMBEDDINGS")
    print("="*50)
    
    embed_weight = model.model.embed_tokens.weight
    print(f"Embedding matrix shape: {embed_weight.shape}")
    print(f"Embedding dtype: {embed_weight.dtype}")
    
    token_ids = tokens.flatten().tolist()
    print(f"Token IDs: {token_ids}")
    
    for i, token_id in enumerate(token_ids):
        embedding = embed_weight[token_id]
        abs_max = torch.abs(embedding).max()
        emb_min = embedding.min()
        emb_max = embedding.max()
        emb_mean = embedding.mean()
        emb_std = embedding.std()
        
        # Decode token for context
        try:
            token_text = tokenizer.decode([token_id])
        except:
            token_text = f"<unk_{token_id}>"
        
        status = "🚨 EXTREME" if abs_max > 10 else ("⚠️  Large" if abs_max > 2 else "✅ Normal")
        print(f"Token {i} (ID: {token_id:6d}, '{token_text:>12}'): abs_max={abs_max:6.3f} range=[{emb_min:6.3f}, {emb_max:6.3f}] {status}")
        print(f"   Mean±Std: {emb_mean:6.3f}±{emb_std:6.3f}")
        
        if abs_max > 5:
            print(f"   🔥 EXTREME EMBEDDING DETECTED!")
            extreme_indices = torch.where(torch.abs(embedding) > 5)[0]
            print(f"   Extreme dimensions: {extreme_indices[:10].tolist()}...")
            print(f"   Extreme values: {embedding[extreme_indices[:10]].tolist()}")


def check_bias_terms(model):
    """Check bias terms in the problematic projections."""
    print("\n🔍 CHECKING BIAS TERMS")
    print("="*50)
    
    # Check Layer 0 (where extreme values first appear)
    layer0 = model.model.layers[0]
    
    bias_terms = [
        ("self_attn.q_proj.bias", layer0.self_attn.q_proj.bias),
        ("self_attn.k_proj.bias", layer0.self_attn.k_proj.bias),
        ("self_attn.v_proj.bias", layer0.self_attn.v_proj.bias),
        ("self_attn.o_proj.bias", layer0.self_attn.o_proj.bias),
        ("mlp.gate_proj.bias", layer0.mlp.gate_proj.bias),
        ("mlp.up_proj.bias", layer0.mlp.up_proj.bias),
        ("mlp.down_proj.bias", layer0.mlp.down_proj.bias),
    ]
    
    for bias_name, bias_tensor in bias_terms:
        if bias_tensor is not None:
            abs_max = torch.abs(bias_tensor).max()
            bias_min = bias_tensor.min()
            bias_max = bias_tensor.max()
            bias_mean = bias_tensor.mean()
            
            status = "🚨 EXTREME" if abs_max > 10 else ("⚠️  Large" if abs_max > 2 else "✅ Normal")
            print(f"Layer 0 {bias_name:<25} abs_max={abs_max:8.3f} range=[{bias_min:7.3f}, {bias_max:7.3f}] {status}")
            
            if abs_max > 5:
                print(f"   🔥 EXTREME BIAS DETECTED!")
                extreme_indices = torch.where(torch.abs(bias_tensor) > 5)[0]
                print(f"   Extreme dimensions: {extreme_indices[:10].tolist()}")
                print(f"   Extreme values: {bias_tensor[extreme_indices[:10]].tolist()}")
        else:
            print(f"Layer 0 {bias_name:<25} No bias (None)")


def debug_matrix_multiplication_step_by_step(model, tokenizer):
    """Manually perform the problematic matrix multiplication to isolate the issue."""
    print("\n🔍 STEP-BY-STEP MATRIX MULTIPLICATION DEBUG")  
    print("="*60)
    
    test_prompt = "Hello world"
    tokens = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
    
    # Get the input to Layer 0 (token embeddings)
    embeddings = model.model.embed_tokens(tokens)  # Shape: [1, seq_len, d_model]
    print(f"Token embeddings shape: {embeddings.shape}")
    print(f"Token embeddings range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
    print(f"Token embeddings abs_max: {torch.abs(embeddings).max():.6f}")
    
    # Apply input LayerNorm (this should normalize the embeddings)
    layer0 = model.model.layers[0]
    normalized_input = layer0.input_layernorm(embeddings)
    print(f"After input LayerNorm range: [{normalized_input.min():.6f}, {normalized_input.max():.6f}]")
    print(f"After input LayerNorm abs_max: {torch.abs(normalized_input).max():.6f}")
    
    # Now manually apply Q projection
    q_weight = layer0.self_attn.q_proj.weight  # Shape: [d_model, d_model] 
    q_bias = layer0.self_attn.q_proj.bias
    
    print(f"\nQ projection weight shape: {q_weight.shape}")
    print(f"Q projection weight range: [{q_weight.min():.6f}, {q_weight.max():.6f}]")
    print(f"Q projection weight abs_max: {torch.abs(q_weight).max():.6f}")
    
    if q_bias is not None:
        print(f"Q projection bias shape: {q_bias.shape}")
        print(f"Q projection bias range: [{q_bias.min():.6f}, {q_bias.max():.6f}]")
        print(f"Q projection bias abs_max: {torch.abs(q_bias).max():.6f}")
    
    # Manual matrix multiplication
    print(f"\n🧮 MANUAL COMPUTATION:")
    
    # Step 1: Matrix multiplication
    matmul_result = torch.matmul(normalized_input, q_weight.T)  # Note: need transpose for Linear layer
    print(f"After matmul (input @ weight.T):")
    print(f"   Shape: {matmul_result.shape}")
    print(f"   Range: [{matmul_result.min():.6f}, {matmul_result.max():.6f}]")
    print(f"   Abs_max: {torch.abs(matmul_result).max():.6f}")
    
    # Check per-token contribution
    for i in range(matmul_result.shape[1]):  # seq_len dimension
        token_output = matmul_result[0, i]  # Shape: [d_model]
        token_abs_max = torch.abs(token_output).max()
        print(f"   Token {i}: abs_max={token_abs_max:.6f}")
        
        if token_abs_max > 10:
            print(f"      🚨 EXTREME VALUES in token {i}!")
            # Find which dimensions are extreme
            extreme_dims = torch.where(torch.abs(token_output) > 10)[0]
            print(f"      Extreme dimensions: {extreme_dims[:5].tolist()}")
            print(f"      Extreme values: {token_output[extreme_dims[:5]].tolist()}")
    
    # Step 2: Add bias (if present)
    if q_bias is not None:
        final_result = matmul_result + q_bias
        print(f"After adding bias:")
        print(f"   Range: [{final_result.min():.6f}, {final_result.max():.6f}]")
        print(f"   Abs_max: {torch.abs(final_result).max():.6f}")
        
        # Check which contribution is larger
        bias_contribution = torch.abs(q_bias).max()
        matmul_contribution = torch.abs(matmul_result).max()
        print(f"   Bias contribution: {bias_contribution:.6f}")
        print(f"   Matmul contribution: {matmul_contribution:.6f}")
        
        if bias_contribution > matmul_contribution:
            print(f"   🎯 BIAS DOMINATES - this could be the issue!")
        else:
            print(f"   🎯 MATMUL DOMINATES - issue is in weight/input interaction")
    else:
        final_result = matmul_result
        print(f"No bias - final result same as matmul")
    
    # Compare with actual Q projection output
    print(f"\n🔄 COMPARING WITH ACTUAL Q PROJECTION:")
    actual_q_output = layer0.self_attn.q_proj(normalized_input)
    print(f"Actual q_proj output range: [{actual_q_output.min():.6f}, {actual_q_output.max():.6f}]")
    print(f"Actual q_proj output abs_max: {torch.abs(actual_q_output).max():.6f}")
    
    # Check if they match
    diff = torch.abs(final_result - actual_q_output).max()
    print(f"Difference between manual and actual: {diff:.10f}")
    
    if diff < 1e-4:
        print(f"✅ Manual computation matches actual - our analysis is correct")
    else:
        print(f"❌ Manual computation differs - there might be other factors")
    
    return {
        "embeddings_abs_max": float(torch.abs(embeddings).max()),
        "normalized_abs_max": float(torch.abs(normalized_input).max()),
        "matmul_abs_max": float(torch.abs(matmul_result).max()),
        "final_abs_max": float(torch.abs(final_result).max()),
        "actual_abs_max": float(torch.abs(actual_q_output).max()),
    }


def main():
    """Main investigation function."""
    print("🔬 DEEP NUMERICAL PRECISION INVESTIGATION")
    print("="*60)
    print("Finding the source of activation amplification (2.2 → 14.7)")
    
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    test_prompt = "Hello world"
    tokens = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
    
    # Run all checks
    check_token_embeddings(model, tokenizer, tokens)
    check_bias_terms(model)
    results = debug_matrix_multiplication_step_by_step(model, tokenizer)
    
    # Summary analysis
    print(f"\n" + "="*60)
    print("🎯 NUMERICAL INVESTIGATION SUMMARY")
    print("="*60)
    
    print(f"Activation progression:")
    print(f"   Token embeddings:     {results['embeddings_abs_max']:8.3f}")
    print(f"   After LayerNorm:      {results['normalized_abs_max']:8.3f}")
    print(f"   After matmul:         {results['matmul_abs_max']:8.3f}")
    print(f"   After bias (manual):  {results['final_abs_max']:8.3f}")
    print(f"   Actual q_proj:        {results['actual_abs_max']:8.3f}")
    
    # Identify the largest jump
    if results['matmul_abs_max'] > results['normalized_abs_max'] * 3:
        print(f"\n🎯 PRIMARY AMPLIFICATION: LayerNorm → MatMul ({results['normalized_abs_max']:.3f} → {results['matmul_abs_max']:.3f})")
        print(f"   This suggests the issue is in the matrix multiplication step")
        print(f"   Even with normal weights, something causes extreme amplification")
        
    if results['final_abs_max'] > results['matmul_abs_max'] * 2:
        print(f"\n🎯 SECONDARY AMPLIFICATION: MatMul → Bias ({results['matmul_abs_max']:.3f} → {results['final_abs_max']:.3f})")
        print(f"   Bias terms are contributing significantly")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   • If bias terms are extreme → fix bias initialization")  
    print(f"   • If matmul creates amplification → investigate numerical precision")
    print(f"   • If embeddings are extreme → check tokenization/vocabulary")


if __name__ == "__main__":
    main()