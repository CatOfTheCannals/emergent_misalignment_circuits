"""
Data quality checks for each step of the pipeline.
Creates individual check files for comprehensive debugging.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime


class DataQualityChecker:
    """Performs comprehensive data quality checks and generates reports."""
    
    def __init__(self, debug_dir: str = "debug_results/data_quality"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checks_performed = []
    
    def check_tensor_quality(self, tensor: torch.Tensor, name: str, 
                           expected_shape: Optional[List[int]] = None,
                           expected_range: Optional[List[float]] = None) -> Dict[str, Any]:
        """Comprehensive tensor quality check."""
        
        check_data = {
            "tensor_name": name,
            "timestamp": datetime.now().isoformat(),
            "basic_info": {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": tensor.numel(),
                "memory_mb": tensor.numel() * tensor.element_size() / (1024 * 1024)
            },
            "quality_flags": {},
            "statistics": {},
            "health_score": 100  # Start with perfect score, deduct points for issues
        }
        
        # Shape validation
        if expected_shape:
            shape_correct = list(tensor.shape) == expected_shape
            check_data["quality_flags"]["shape_correct"] = shape_correct
            if not shape_correct:
                check_data["health_score"] -= 30
                check_data["shape_error"] = f"Expected {expected_shape}, got {list(tensor.shape)}"
        
        # Numerical health checks
        if tensor.numel() > 0:
            # Basic statistics
            check_data["statistics"] = {
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "median": float(tensor.median()),
                "norm": float(tensor.norm()),
                "abs_max": float(torch.abs(tensor).max()),
            }
            
            # Problematic values
            nan_count = int(torch.isnan(tensor).sum())
            inf_count = int(torch.isinf(tensor).sum())
            finite_count = int(torch.isfinite(tensor).sum())
            zero_count = int((tensor == 0).sum())
            
            check_data["quality_flags"].update({
                "has_nan": nan_count > 0,
                "has_inf": inf_count > 0,
                "all_finite": finite_count == tensor.numel(),
                "has_zeros": zero_count > 0,
                "mostly_zeros": zero_count > (tensor.numel() * 0.9),
            })
            
            check_data["counts"] = {
                "nan": nan_count,
                "inf": inf_count,
                "finite": finite_count,
                "zero": zero_count,
                "nonzero": tensor.numel() - zero_count,
            }
            
            # Health score deductions
            if nan_count > 0:
                check_data["health_score"] -= 50
            if inf_count > 0:
                check_data["health_score"] -= 50
            if zero_count > (tensor.numel() * 0.95):
                check_data["health_score"] -= 20  # Almost all zeros
            
            # Range validation
            if expected_range:
                min_val, max_val = expected_range
                in_range = (tensor.min() >= min_val) and (tensor.max() <= max_val)
                check_data["quality_flags"]["in_expected_range"] = in_range
                if not in_range:
                    check_data["health_score"] -= 20
                    check_data["range_error"] = f"Expected [{min_val}, {max_val}], got [{tensor.min():.6f}, {tensor.max():.6f}]"
            
            # Extreme values check
            abs_tensor = torch.abs(tensor)
            extreme_threshold = 1000.0
            very_extreme_threshold = 10000.0
            
            extreme_count = int((abs_tensor > extreme_threshold).sum())
            very_extreme_count = int((abs_tensor > very_extreme_threshold).sum())
            
            check_data["quality_flags"]["has_extreme_values"] = extreme_count > 0
            check_data["quality_flags"]["has_very_extreme_values"] = very_extreme_count > 0
            check_data["counts"]["extreme_values"] = extreme_count
            check_data["counts"]["very_extreme_values"] = very_extreme_count
            
            if extreme_count > 0:
                check_data["health_score"] -= 10
            if very_extreme_count > 0:
                check_data["health_score"] -= 20
            
            # Distribution checks
            if tensor.numel() > 1:
                # Check for constant tensors (no variation)
                is_constant = torch.allclose(tensor, tensor.flatten()[0])
                check_data["quality_flags"]["is_constant"] = is_constant
                if is_constant:
                    check_data["health_score"] -= 15
                
                # Check for reasonable distribution
                std_val = check_data["statistics"]["std"]
                mean_val = check_data["statistics"]["mean"]
                coefficient_of_variation = abs(std_val / mean_val) if mean_val != 0 else float('inf')
                
                check_data["statistics"]["coefficient_of_variation"] = coefficient_of_variation
                check_data["quality_flags"]["has_reasonable_variation"] = 0.01 < coefficient_of_variation < 100
            
            # Sample values for inspection
            check_data["sample_values"] = {
                "first_10": tensor.flatten()[:10].tolist(),
                "last_10": tensor.flatten()[-10:].tolist(),
                "random_10": tensor.flatten()[torch.randperm(tensor.numel())[:10]].tolist(),
            }
        
        else:
            check_data["quality_flags"]["empty_tensor"] = True
            check_data["health_score"] = 0
        
        # Overall health assessment
        check_data["health_assessment"] = self._assess_health(check_data["health_score"])
        
        # Save individual check
        self.save_check(f"tensor_{name}", check_data)
        self.checks_performed.append(check_data)
        
        return check_data
    
    def check_model_state(self, model, model_name: str) -> Dict[str, Any]:
        """Check the state of a model for potential issues."""
        
        check_data = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "model_type": str(type(model)),
            "health_score": 100,
            "parameter_checks": [],
            "buffer_checks": [],
            "training_state": {
                "training_mode": model.training if hasattr(model, 'training') else None,
                "requires_grad": []
            }
        }
        
        # Check parameters
        if hasattr(model, 'named_parameters'):
            for name, param in model.named_parameters():
                if param is not None:
                    param_check = self.check_tensor_quality(param, f"{model_name}_param_{name}")
                    param_check["parameter_name"] = name
                    param_check["requires_grad"] = param.requires_grad
                    check_data["parameter_checks"].append(param_check)
                    check_data["training_state"]["requires_grad"].append({
                        "name": name,
                        "requires_grad": param.requires_grad
                    })
                    
                    # Aggregate health score (use worst parameter health)
                    check_data["health_score"] = min(check_data["health_score"], param_check["health_score"])
        
        # Check buffers
        if hasattr(model, 'named_buffers'):
            for name, buffer in model.named_buffers():
                if buffer is not None:
                    buffer_check = self.check_tensor_quality(buffer, f"{model_name}_buffer_{name}")
                    buffer_check["buffer_name"] = name
                    check_data["buffer_checks"].append(buffer_check)
                    
                    # Buffers are less critical, but still factor in
                    check_data["health_score"] = min(check_data["health_score"], buffer_check["health_score"] * 0.5 + 50)
        
        # Model-specific checks
        check_data["parameter_summary"] = {
            "total_parameters": sum(p.numel() for p in model.parameters() if p is not None),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p is not None and p.requires_grad),
            "parameter_count": len(check_data["parameter_checks"]),
            "buffer_count": len(check_data["buffer_checks"]),
            "problematic_parameters": [p for p in check_data["parameter_checks"] if p["health_score"] < 80],
            "healthy_parameters": [p for p in check_data["parameter_checks"] if p["health_score"] >= 80],
        }
        
        # Overall assessment
        check_data["health_assessment"] = self._assess_health(check_data["health_score"])
        
        # Save model check
        self.save_check(f"model_{model_name}", check_data)
        self.checks_performed.append(check_data)
        
        return check_data
    
    def check_tokenization(self, tokenizer, prompt: str, expected_properties: Optional[Dict] = None) -> Dict[str, Any]:
        """Check tokenization quality."""
        
        check_data = {
            "prompt": prompt,
            "prompt_length": len(prompt),
            "timestamp": datetime.now().isoformat(),
            "tokenizer_type": str(type(tokenizer)),
            "health_score": 100
        }
        
        try:
            # Perform tokenization
            tokens = tokenizer(prompt, return_tensors="pt")
            
            check_data["tokenization"] = {
                "success": True,
                "input_ids_shape": list(tokens.input_ids.shape),
                "sequence_length": tokens.input_ids.shape[1],
                "token_ids": tokens.input_ids.flatten().tolist(),
                "attention_mask": tokens.attention_mask.flatten().tolist() if hasattr(tokens, 'attention_mask') else None,
            }
            
            # Check token IDs quality
            input_ids = tokens.input_ids.flatten()
            check_data["token_analysis"] = {
                "unique_tokens": len(torch.unique(input_ids)),
                "min_token_id": int(input_ids.min()),
                "max_token_id": int(input_ids.max()),
                "has_special_tokens": any(tid in [0, 1, 2, 3] for tid in input_ids.tolist()),  # Common special token IDs
                "out_of_vocab": int((input_ids >= tokenizer.vocab_size).sum()) if hasattr(tokenizer, 'vocab_size') else None,
            }
            
            # Validate against expected properties
            if expected_properties:
                if "max_length" in expected_properties:
                    if check_data["tokenization"]["sequence_length"] > expected_properties["max_length"]:
                        check_data["health_score"] -= 20
                        check_data["length_violation"] = f"Sequence too long: {check_data['tokenization']['sequence_length']} > {expected_properties['max_length']}"
                
                if "min_length" in expected_properties:
                    if check_data["tokenization"]["sequence_length"] < expected_properties["min_length"]:
                        check_data["health_score"] -= 10
                        check_data["length_violation"] = f"Sequence too short: {check_data['tokenization']['sequence_length']} < {expected_properties['min_length']}"
            
            # Check for potential issues
            if check_data["token_analysis"]["out_of_vocab"] and check_data["token_analysis"]["out_of_vocab"] > 0:
                check_data["health_score"] -= 30
                check_data["vocab_error"] = f"Found {check_data['token_analysis']['out_of_vocab']} out-of-vocabulary tokens"
            
            if check_data["tokenization"]["sequence_length"] == 0:
                check_data["health_score"] -= 50
                check_data["empty_tokenization"] = True
            
        except Exception as e:
            check_data["tokenization"] = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            check_data["health_score"] = 0
        
        check_data["health_assessment"] = self._assess_health(check_data["health_score"])
        
        # Save tokenization check
        self.save_check("tokenization", check_data)
        self.checks_performed.append(check_data)
        
        return check_data
    
    def check_forward_pass(self, model, input_tensor: torch.Tensor, model_name: str) -> Dict[str, Any]:
        """Check model forward pass for issues."""
        
        check_data = {
            "model_name": model_name,
            "input_tensor": {
                "shape": list(input_tensor.shape),
                "dtype": str(input_tensor.dtype),
                "device": str(input_tensor.device),
            },
            "timestamp": datetime.now().isoformat(),
            "health_score": 100
        }
        
        try:
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            check_data["forward_pass"] = {
                "success": True,
                "output_type": str(type(output)),
            }
            
            # Check output quality
            if isinstance(output, torch.Tensor):
                output_check = self.check_tensor_quality(output, f"{model_name}_output")
                check_data["output_check"] = output_check
                check_data["health_score"] = min(check_data["health_score"], output_check["health_score"])
            elif hasattr(output, 'logits'):
                # Handle transformers output
                logits_check = self.check_tensor_quality(output.logits, f"{model_name}_logits")
                check_data["output_check"] = logits_check
                check_data["health_score"] = min(check_data["health_score"], logits_check["health_score"])
            
        except Exception as e:
            check_data["forward_pass"] = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            check_data["health_score"] = 0
            
            # Try to get more debug info
            try:
                import traceback
                check_data["forward_pass"]["traceback"] = traceback.format_exc()
            except:
                pass
        
        check_data["health_assessment"] = self._assess_health(check_data["health_score"])
        
        # Save forward pass check
        self.save_check(f"forward_pass_{model_name}", check_data)
        self.checks_performed.append(check_data)
        
        return check_data
    
    def save_check(self, check_name: str, check_data: Dict[str, Any]) -> Path:
        """Save individual check to file."""
        file_path = self.debug_dir / f"{self.session_id}_{check_name}.json"
        
        with open(file_path, 'w') as f:
            json.dump(check_data, f, indent=2, default=self._json_serializer)
        
        return file_path
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all checks performed."""
        
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(self.checks_performed),
            "overall_health_score": min([check["health_score"] for check in self.checks_performed]) if self.checks_performed else 0,
            "checks_by_health": {
                "excellent": [c for c in self.checks_performed if c["health_score"] >= 90],
                "good": [c for c in self.checks_performed if 70 <= c["health_score"] < 90],
                "fair": [c for c in self.checks_performed if 50 <= c["health_score"] < 70],
                "poor": [c for c in self.checks_performed if c["health_score"] < 50],
            },
            "critical_issues": [],
            "recommendations": []
        }
        
        # Identify critical issues
        for check in self.checks_performed:
            if check["health_score"] < 50:
                summary["critical_issues"].append({
                    "check_name": check.get("tensor_name", check.get("model_name", "unknown")),
                    "health_score": check["health_score"],
                    "issues": [k for k, v in check.get("quality_flags", {}).items() if v and k.startswith("has_")]
                })
        
        # Generate recommendations
        if summary["critical_issues"]:
            summary["recommendations"].append("Critical data quality issues found - investigate before proceeding")
        
        if len(summary["checks_by_health"]["poor"]) > 0:
            summary["recommendations"].append("Multiple components with poor health scores - review model weights and data")
        
        if len(summary["checks_by_health"]["excellent"]) == len(self.checks_performed):
            summary["recommendations"].append("All checks passed with excellent scores - data quality looks good")
        
        # Save summary
        summary_path = self.debug_dir / f"{self.session_id}_SUMMARY.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)
        
        print(f"📋 Data quality summary saved to: {summary_path}")
        return summary
    
    def _assess_health(self, score: float) -> str:
        """Convert health score to assessment."""
        if score >= 90:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"
    
    def _json_serializer(self, obj):
        """JSON serializer for non-standard types."""
        if isinstance(obj, torch.Tensor):
            return f"<Tensor shape={list(obj.shape)}>"
        elif isinstance(obj, np.ndarray):
            return f"<Array shape={list(obj.shape)}>"
        else:
            return str(obj)