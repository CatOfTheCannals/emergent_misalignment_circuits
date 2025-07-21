"""
Systematic testing of hypotheses for why base model has extreme activation values.
Focus only on meta-llama/Llama-3.2-1B-Instruct to isolate the issue.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('/workspace/circuit-tracer')
from circuit_tracer.replacement_model import ReplacementModel


def test_hypothesis_1_dtype_precision():
    """Test if dtype affects activation values."""
    print("1️⃣ HYPOTHESIS: Dtype/Precision Issues")
    print("="*50)
    
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    test_prompt = "Hello world"
    
    dtypes_to_test = [
        (torch.float32, "float32"),
        (torch.float16, "float16"), 
        (torch.bfloat16, "bfloat16"),
    ]
    
    for dtype, dtype_name in dtypes_to_test:
        print(f"\n🧪 Testing {dtype_name}:")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokens = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
            
            # Hook Layer 1 post-attention LayerNorm
            activations = {}
            def capture_hook(module, input, output):
                activations["post_attn_ln"] = output.detach().clone()
            
            handle = model.model.layers[1].post_attention_layernorm.register_forward_hook(capture_hook)
            
            with torch.no_grad():
                _ = model(tokens)
            
            handle.remove()
            
            if "post_attn_ln" in activations:
                acts = activations["post_attn_ln"]
                abs_max = torch.abs(acts).max()
                print(f"   abs_max: {abs_max:.3f}")
                print(f"   range: [{acts.min():.3f}, {acts.max():.3f}]")
                print(f"   dtype: {acts.dtype}")
                
                # Check for numerical issues
                nan_count = torch.isnan(acts).sum()
                inf_count = torch.isinf(acts).sum()
                print(f"   nan_count: {nan_count}, inf_count: {inf_count}")
                
                if abs_max < 5:
                    print(f"   ✅ NORMAL activation range!")
                else:
                    print(f"   ❌ Still extreme values")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Failed with {dtype_name}: {e}")


def test_hypothesis_2_device_placement():
    """Test if device placement affects results."""
    print("\n2️⃣ HYPOTHESIS: Device Placement Issues")
    print("="*50)
    
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    test_prompt = "Hello world"
    
    device_configs = [
        ("auto", "device_map='auto'"),
        ("cuda", "device_map='cuda'"),
        ("cuda:0", "device_map='cuda:0'"),
        ({"": "cuda:0"}, "device_map={'': 'cuda:0'}"),
    ]
    
    for device_map, desc in device_configs:
        print(f"\n🧪 Testing {desc}:")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device_map
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokens = tokenizer(test_prompt, return_tensors="pt")
            
            # Ensure tokens are on same device as model
            if hasattr(model, 'device'):
                device = model.device
            else:
                device = next(model.parameters()).device
            
            tokens = tokens.input_ids.to(device)
            
            # Hook Layer 1
            activations = {}
            def capture_hook(module, input, output):
                activations["post_attn_ln"] = output.detach().clone()
            
            handle = model.model.layers[1].post_attention_layernorm.register_forward_hook(capture_hook)
            
            with torch.no_grad():
                _ = model(tokens)
            
            handle.remove()
            
            if "post_attn_ln" in activations:
                acts = activations["post_attn_ln"]
                abs_max = torch.abs(acts).max()
                print(f"   abs_max: {abs_max:.3f}")
                print(f"   model device: {device}")
                print(f"   acts device: {acts.device}")
                
                if abs_max < 5:
                    print(f"   ✅ NORMAL activation range!")
                else:
                    print(f"   ❌ Still extreme values")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Failed with {desc}: {e}")


def test_hypothesis_3_proper_chat_template():
    """Test using HF's apply_chat_template function properly."""
    print("\n3️⃣ HYPOTHESIS: Improper Chat Template Usage")
    print("="*50)
    
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Test different message formats
    test_cases = [
        # Raw text
        ("Raw text", "Hello world"),
        
        # Our format
        ("Our format", "User: Hello world\nAssistant:"),
        
        # Proper chat template - single message
        ("Chat template (single)", None),
        
        # Proper chat template - conversation
        ("Chat template (conversation)", None),
    ]
    
    for case_name, prompt in test_cases:
        print(f"\n🧪 Testing {case_name}:")
        
        try:
            if case_name == "Chat template (single)":
                messages = [{"role": "user", "content": "Hello world"}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    tokenize=False
                )
                print(f"   Formatted: {repr(formatted_prompt[:100])}...")
                
            elif case_name == "Chat template (conversation)":
                messages = [
                    {"role": "user", "content": "Hello world"},
                    {"role": "assistant", "content": "Hello!"}
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=False,
                    tokenize=False
                )
                print(f"   Formatted: {repr(formatted_prompt[:100])}...")
            else:
                formatted_prompt = prompt
                print(f"   Prompt: {repr(formatted_prompt)}")
            
            tokens = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda")
            print(f"   Token count: {len(tokens[0])}")
            print(f"   First 10 tokens: {tokens[0][:10].tolist()}")
            
            # Hook Layer 1
            activations = {}
            def capture_hook(module, input, output):
                activations["post_attn_ln"] = output.detach().clone()
            
            handle = model.model.layers[1].post_attention_layernorm.register_forward_hook(capture_hook)
            
            with torch.no_grad():
                _ = model(tokens)
            
            handle.remove()
            
            if "post_attn_ln" in activations:
                acts = activations["post_attn_ln"]
                abs_max = torch.abs(acts).max()
                print(f"   abs_max: {abs_max:.3f}")
                
                if abs_max < 5:
                    print(f"   ✅ NORMAL activation range!")
                else:
                    print(f"   ❌ Still extreme values")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    del model
    torch.cuda.empty_cache()


def test_hypothesis_4_different_layers():
    """Test if extreme values are specific to Layer 1."""
    print("\n4️⃣ HYPOTHESIS: Layer-Specific Issue")
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
    
    # Test multiple layers
    layers_to_test = [0, 1, 2, 3, 8, 15]  # Sample across the model
    
    activations = {}
    handles = []
    
    for layer_idx in layers_to_test:
        def make_hook(layer_idx):
            def capture_hook(module, input, output):
                activations[f"layer_{layer_idx}"] = output.detach().clone()
            return capture_hook
        
        layer = model.model.layers[layer_idx]
        handle = layer.post_attention_layernorm.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)
    
    with torch.no_grad():
        _ = model(tokens)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    print(f"\n📊 Activation ranges across layers:")
    for layer_idx in layers_to_test:
        if f"layer_{layer_idx}" in activations:
            acts = activations[f"layer_{layer_idx}"]
            abs_max = torch.abs(acts).max()
            print(f"   Layer {layer_idx:2d}: abs_max = {abs_max:6.3f} {'✅' if abs_max < 5 else '❌'}")
    
    del model
    torch.cuda.empty_cache()


def test_hypothesis_5_model_loading_methods():
    """Test different model loading methods."""
    print("\n5️⃣ HYPOTHESIS: Model Loading Method")
    print("="*50)
    
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    test_prompt = "Hello world"
    
    loading_methods = [
        ("Standard", lambda: AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto")),
        
        ("Low CPU mem", lambda: AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)),
        
        ("No device map", lambda: AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16).to("cuda")),
        
        ("Trust remote code", lambda: AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)),
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
    
    for method_name, loader in loading_methods:
        print(f"\n🧪 Testing {method_name}:")
        
        try:
            model = loader()
            
            # Hook Layer 1
            activations = {}
            def capture_hook(module, input, output):
                activations["post_attn_ln"] = output.detach().clone()
            
            handle = model.model.layers[1].post_attention_layernorm.register_forward_hook(capture_hook)
            
            with torch.no_grad():
                _ = model(tokens)
            
            handle.remove()
            
            if "post_attn_ln" in activations:
                acts = activations["post_attn_ln"]
                abs_max = torch.abs(acts).max()
                print(f"   abs_max: {abs_max:.3f}")
                
                if abs_max < 5:
                    print(f"   ✅ NORMAL activation range!")
                else:
                    print(f"   ❌ Still extreme values")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def test_hypothesis_6_replacement_model_bug():
    """Test if ReplacementModel introduces the extreme values."""
    print("\n6️⃣ HYPOTHESIS: ReplacementModel Bug")
    print("="*50)
    
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    test_prompt = "Hello world"
    
    # Direct HF model
    print(f"\n🧪 Direct HuggingFace model:")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokens = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
    
    activations_hf = {}
    def capture_hf_hook(module, input, output):
        activations_hf["post_attn_ln"] = output.detach().clone()
    
    handle_hf = hf_model.model.layers[1].post_attention_layernorm.register_forward_hook(capture_hf_hook)
    
    with torch.no_grad():
        _ = hf_model(tokens)
    
    handle_hf.remove()
    
    if "post_attn_ln" in activations_hf:
        acts = activations_hf["post_attn_ln"]
        abs_max_hf = torch.abs(acts).max()
        print(f"   HF model abs_max: {abs_max_hf:.3f}")
    
    del hf_model
    torch.cuda.empty_cache()
    
    # ReplacementModel 
    print(f"\n🧪 ReplacementModel:")
    try:
        replacement_model = ReplacementModel.from_pretrained(
            model_path,
            "llama",
            device=torch.device("cuda"),
            dtype=torch.float16
        )
        
        activations_repl = {}
        def capture_repl_hook(acts, hook):
            activations_repl["resid_mid"] = acts.detach().clone()
        
        with replacement_model.hooks([("blocks.1.hook_resid_mid", capture_repl_hook)]):
            with torch.no_grad():
                _ = replacement_model(tokens)
        
        if "resid_mid" in activations_repl:
            acts = activations_repl["resid_mid"]
            abs_max_repl = torch.abs(acts).max()
            print(f"   ReplacementModel abs_max: {abs_max_repl:.3f}")
            
            print(f"\n📊 Comparison:")
            print(f"   HF model: {abs_max_hf:.3f}")
            print(f"   ReplacementModel: {abs_max_repl:.3f}")
            print(f"   Difference: {abs_max_repl - abs_max_hf:.3f}")
            
            if abs_max_repl > abs_max_hf * 1.1:
                print(f"   🚨 ReplacementModel increases activation values!")
            else:
                print(f"   ✅ ReplacementModel preserves activation values")
        
        del replacement_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ ReplacementModel failed: {e}")


def main():
    """Run all hypothesis tests."""
    print("🔬 SYSTEMATIC BASE MODEL TESTING")
    print("="*60)
    print("Testing meta-llama/Llama-3.2-1B-Instruct for extreme activation values")
    print("="*60)
    
    test_hypothesis_1_dtype_precision()
    test_hypothesis_2_device_placement()
    test_hypothesis_3_proper_chat_template()
    test_hypothesis_4_different_layers()
    test_hypothesis_5_model_loading_methods()
    test_hypothesis_6_replacement_model_bug()
    
    print("\n" + "="*60)
    print("🎯 SYSTEMATIC TESTING COMPLETE")
    print("="*60)
    print("Look for any test that shows ✅ NORMAL activation range!")


if __name__ == "__main__":
    main()