"""
Comprehensive debugging system for circuit attribution pipeline.
Creates individual report files for each transformation step.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class DebugReporter:
    """Creates detailed debug reports for each pipeline step."""
    
    def __init__(self, debug_dir: Union[str, Path]):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reports = {}
        
    def report_file(self, step_name: str) -> Path:
        """Get the report file path for a given step."""
        return self.debug_dir / f"{self.session_id}_{step_name}.json"
    
    def create_report(self, step_name: str, data: Dict[str, Any], save: bool = True) -> Dict[str, Any]:
        """Create a debug report for a pipeline step."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step_name": step_name,
            "data": data
        }
        
        self.reports[step_name] = report
        
        if save:
            with open(self.report_file(step_name), 'w') as f:
                json.dump(report, f, indent=2, default=self._json_serializer)
        
        return report
    
    def _json_serializer(self, obj):
        """JSON serializer for non-standard types."""
        if isinstance(obj, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "stats": {
                    "min": float(obj.min()) if obj.numel() > 0 else None,
                    "max": float(obj.max()) if obj.numel() > 0 else None,
                    "mean": float(obj.mean()) if obj.numel() > 0 else None,
                    "std": float(obj.std()) if obj.numel() > 0 else None,
                    "nan_count": int(torch.isnan(obj).sum()),
                    "inf_count": int(torch.isinf(obj).sum()),
                    "zero_count": int((obj == 0).sum()),
                    "finite_count": int(torch.isfinite(obj).sum())
                },
                "sample_values": obj.flatten()[:10].tolist() if obj.numel() > 0 else []
            }
        elif isinstance(obj, np.ndarray):
            return {
                "type": "numpy.ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "stats": {
                    "min": float(np.min(obj)) if obj.size > 0 else None,
                    "max": float(np.max(obj)) if obj.size > 0 else None,
                    "mean": float(np.mean(obj)) if obj.size > 0 else None,
                    "std": float(np.std(obj)) if obj.size > 0 else None,
                    "nan_count": int(np.isnan(obj).sum()),
                    "inf_count": int(np.isinf(obj).sum())
                },
                "sample_values": obj.flatten()[:10].tolist() if obj.size > 0 else []
            }
        elif hasattr(obj, '__dict__'):
            return f"<{type(obj).__name__} object>"
        else:
            return str(obj)


def check_model_health(model, tokenizer, prompt: str, reporter: DebugReporter, model_name: str = "model") -> Dict[str, Any]:
    """Comprehensive model health check."""
    print(f"🔍 Running model health check for {model_name}...")
    
    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device if hasattr(model, 'device') else 'cuda')
    
    health_data = {
        "model_name": model_name,
        "prompt": prompt,
        "input_ids": input_ids,
        "model_type": str(type(model)),
    }
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(input_ids)
        
        health_data["forward_pass"] = {
            "success": True,
            "output_shape": list(outputs.shape) if hasattr(outputs, 'shape') else str(outputs),
            "output_stats": outputs if isinstance(outputs, torch.Tensor) else None
        }
        
        # Check for numerical issues
        if isinstance(outputs, torch.Tensor):
            health_data["numerical_health"] = {
                "has_nan": bool(torch.isnan(outputs).any()),
                "has_inf": bool(torch.isinf(outputs).any()),
                "all_finite": bool(torch.isfinite(outputs).all()),
                "range": [float(outputs.min()), float(outputs.max())],
                "mean": float(outputs.mean()),
                "std": float(outputs.std())
            }
        
    except Exception as e:
        health_data["forward_pass"] = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
    
    # Model parameters check
    if hasattr(model, 'parameters'):
        param_stats = []
        for name, param in model.named_parameters():
            if param is not None:
                param_info = {
                    "name": name,
                    "shape": list(param.shape),
                    "has_nan": bool(torch.isnan(param).any()),
                    "has_inf": bool(torch.isinf(param).any()),
                    "range": [float(param.min()), float(param.max())],
                    "mean": float(param.mean()),
                    "std": float(param.std())
                }
                param_stats.append(param_info)
        
        health_data["parameters"] = {
            "total_params": sum(p.numel() for p in model.parameters() if p is not None),
            "param_count": len(param_stats),
            "problematic_params": [p for p in param_stats if p["has_nan"] or p["has_inf"]],
            "extreme_params": [p for p in param_stats if abs(p["range"][0]) > 1000 or abs(p["range"][1]) > 1000]
        }
    
    reporter.create_report(f"model_health_{model_name}", health_data)
    return health_data


def check_transcoder_health(transcoders, reporter: DebugReporter) -> Dict[str, Any]:
    """Check transcoder model health."""
    print(f"🔍 Running transcoder health check...")
    
    transcoder_data = {
        "num_transcoders": len(transcoders),
        "transcoder_details": []
    }
    
    for i, transcoder in enumerate(transcoders):
        transcoder_info = {
            "index": i,
            "type": str(type(transcoder)),
        }
        
        # Check transcoder parameters
        if hasattr(transcoder, 'parameters'):
            param_issues = []
            param_count = 0
            for name, param in transcoder.named_parameters():
                if param is not None:
                    param_count += 1
                    if torch.isnan(param).any():
                        param_issues.append(f"{name}: contains NaN")
                    if torch.isinf(param).any():
                        param_issues.append(f"{name}: contains Inf")
                    if abs(param.max()) > 1000 or abs(param.min()) > 1000:
                        param_issues.append(f"{name}: extreme values [{param.min():.2f}, {param.max():.2f}]")
            
            transcoder_info["param_count"] = param_count
            transcoder_info["param_issues"] = param_issues
        
        transcoder_data["transcoder_details"].append(transcoder_info)
    
    reporter.create_report("transcoder_health", transcoder_data)
    return transcoder_data


def check_attribution_inputs(prompt: str, model, max_n_logits: int, desired_logit_prob: float, 
                           batch_size: int, max_feature_nodes: int, reporter: DebugReporter) -> Dict[str, Any]:
    """Check attribution function inputs for potential issues."""
    print(f"🔍 Checking attribution inputs...")
    
    input_data = {
        "prompt": prompt,
        "prompt_length": len(prompt),
        "max_n_logits": max_n_logits,
        "desired_logit_prob": desired_logit_prob,
        "batch_size": batch_size,
        "max_feature_nodes": max_feature_nodes,
        "model_type": str(type(model)),
    }
    
    # Tokenization check
    try:
        tokenized = model.tokenizer(prompt, return_tensors="pt")
        input_data["tokenization"] = {
            "success": True,
            "input_ids_shape": list(tokenized.input_ids.shape),
            "num_tokens": tokenized.input_ids.shape[1],
            "token_ids": tokenized.input_ids.flatten().tolist()
        }
    except Exception as e:
        input_data["tokenization"] = {
            "success": False,
            "error": str(e)
        }
    
    # Parameter validation
    validation_issues = []
    if max_n_logits <= 0:
        validation_issues.append("max_n_logits must be positive")
    if not 0 < desired_logit_prob < 1:
        validation_issues.append("desired_logit_prob must be between 0 and 1")
    if batch_size <= 0:
        validation_issues.append("batch_size must be positive")
    if max_feature_nodes <= 0:
        validation_issues.append("max_feature_nodes must be positive")
    
    input_data["validation_issues"] = validation_issues
    
    reporter.create_report("attribution_inputs", input_data)
    return input_data


def monitor_attribution_phases(prompt: str, model, **kwargs) -> Dict[str, Any]:
    """Monitor each phase of attribution computation with detailed logging."""
    print(f"🔍 Starting monitored attribution computation...")
    
    # Import here to avoid circular imports
    from circuit_tracer.attribution import _run_attribution
    
    debug_dir = Path("debug_results") / "attribution_monitoring"
    debug_dir.mkdir(parents=True, exist_ok=True)
    reporter = DebugReporter(debug_dir)
    
    # Pre-checks
    check_model_health(model, model.tokenizer, prompt, reporter, "attribution_model")
    check_attribution_inputs(prompt, model, **kwargs, reporter=reporter)
    
    # Patch the attribution function to capture intermediate states
    original_compute_partial_influences = None
    
    try:
        from circuit_tracer import attribution
        original_compute_partial_influences = attribution.compute_partial_influences
        
        def monitored_compute_partial_influences(*args, **kwargs):
            """Wrapper to monitor convergence attempts."""
            print(f"🔍 Monitoring compute_partial_influences call...")
            
            # Log the input arguments
            influence_data = {
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            }
            
            # Try to extract key parameters
            if len(args) >= 4:
                try:
                    # Arguments: model, logit_vectors, feature_inputs, batch_idxs, max_iter, lr, debug
                    model_arg, logit_vectors, feature_inputs, batch_idxs = args[:4]
                    
                    influence_data["logit_vectors"] = logit_vectors
                    influence_data["feature_inputs"] = feature_inputs
                    influence_data["batch_idxs"] = batch_idxs
                    
                    if len(args) > 4:
                        influence_data["max_iter"] = args[4]
                    if len(args) > 5:
                        influence_data["lr"] = args[5]
                        
                except Exception as e:
                    influence_data["argument_extraction_error"] = str(e)
            
            reporter.create_report("influence_computation_start", influence_data)
            
            # Call the original function
            try:
                result = original_compute_partial_influences(*args, **kwargs)
                
                # Log successful completion
                completion_data = {
                    "success": True,
                    "result_type": str(type(result)),
                    "result": result if isinstance(result, torch.Tensor) else None
                }
                reporter.create_report("influence_computation_success", completion_data)
                
                return result
                
            except Exception as e:
                # Log the failure with detailed context
                failure_data = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                
                # Try to capture additional debugging info
                if "Failed to converge" in str(e):
                    failure_data["convergence_failure"] = True
                    # Add any available convergence debugging info
                
                reporter.create_report("influence_computation_failure", failure_data)
                raise
        
        # Monkey patch the function
        attribution.compute_partial_influences = monitored_compute_partial_influences
        
        # Now call the original attribution function
        from circuit_tracer.attribution import attribute
        return attribute(prompt=prompt, model=model, **kwargs)
        
    finally:
        # Restore original function
        if original_compute_partial_influences:
            attribution.compute_partial_influences = original_compute_partial_influences


def create_debug_wrapper(original_generate_graphs_func):
    """Create a debug wrapper for the generate_graphs function."""
    
    def debug_generate_graphs(*args, **kwargs):
        """Debug wrapper that creates comprehensive reports."""
        debug_dir = Path("debug_results") / "comprehensive_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        reporter = DebugReporter(debug_dir)
        
        # Log the function call
        call_data = {
            "args": [str(arg) for arg in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "timestamp": datetime.now().isoformat()
        }
        reporter.create_report("function_call", call_data)
        
        try:
            # Call the original function with monitoring
            return original_generate_graphs_func(*args, **kwargs)
            
        except Exception as e:
            # Log any failures
            error_data = {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": str(e.__traceback__)
            }
            reporter.create_report("function_failure", error_data)
            raise
    
    return debug_generate_graphs