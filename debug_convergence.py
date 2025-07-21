"""
Deep convergence debugging for attribution computation.
Patches the circuit-tracer library to capture detailed convergence information.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ConvergenceDebugger:
    """Captures detailed information about convergence failures."""
    
    def __init__(self, debug_dir: str = "debug_results/convergence"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.iteration_data = []
        self.final_state = None
    
    def log_iteration(self, iteration: int, influences: torch.Tensor, prod: torch.Tensor, 
                     loss: float, lr: float, converged: bool) -> None:
        """Log data from each iteration of the convergence loop."""
        iter_data = {
            "iteration": iteration,
            "converged": converged,
            "learning_rate": lr,
            "loss": loss,
            "influences_stats": self._tensor_stats(influences, "influences"),
            "prod_stats": self._tensor_stats(prod, "prod"),
            "gradient_info": self._compute_gradient_info(influences, prod)
        }
        self.iteration_data.append(iter_data)
    
    def log_final_state(self, success: bool, final_influences: Optional[torch.Tensor] = None, 
                       final_prod: Optional[torch.Tensor] = None, error: Optional[str] = None) -> None:
        """Log the final state of convergence attempt."""
        self.final_state = {
            "success": success,
            "error": error,
            "total_iterations": len(self.iteration_data),
            "final_influences": self._tensor_stats(final_influences, "final_influences") if final_influences is not None else None,
            "final_prod": self._tensor_stats(final_prod, "final_prod") if final_prod is not None else None
        }
    
    def save_debug_report(self) -> Path:
        """Save comprehensive debug report to file."""
        report_path = self.debug_dir / f"convergence_debug_{self.session_id}.json"
        
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "iterations": self.iteration_data,
            "final_state": self.final_state,
            "analysis": self._analyze_convergence_pattern()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        
        print(f"📊 Convergence debug report saved to: {report_path}")
        return report_path
    
    def _tensor_stats(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Compute comprehensive statistics for a tensor."""
        if tensor is None:
            return {"name": name, "is_none": True}
        
        stats = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": tensor.numel(),
        }
        
        if tensor.numel() > 0:
            # Basic statistics
            stats.update({
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "norm": float(tensor.norm()),
                "abs_norm": float(torch.abs(tensor).norm()),
            })
            
            # Numerical health
            stats.update({
                "nan_count": int(torch.isnan(tensor).sum()),
                "inf_count": int(torch.isinf(tensor).sum()),
                "finite_count": int(torch.isfinite(tensor).sum()),
                "zero_count": int((tensor == 0).sum()),
                "nonzero_count": int((tensor != 0).sum()),
            })
            
            # Distribution info
            abs_tensor = torch.abs(tensor)
            stats.update({
                "median": float(tensor.median()),
                "q25": float(tensor.quantile(0.25)),
                "q75": float(tensor.quantile(0.75)),
                "abs_median": float(abs_tensor.median()),
                "abs_mean": float(abs_tensor.mean()),
            })
            
            # Extreme values
            large_values = abs_tensor > 10.0
            very_large_values = abs_tensor > 100.0
            stats.update({
                "large_values_count": int(large_values.sum()),
                "very_large_values_count": int(very_large_values.sum()),
                "max_abs_value": float(abs_tensor.max()),
            })
            
            # Sample values
            stats["sample_values"] = tensor.flatten()[:20].tolist()
        
        return stats
    
    def _compute_gradient_info(self, influences: torch.Tensor, prod: torch.Tensor) -> Dict[str, Any]:
        """Compute gradient-related information."""
        if influences is None or prod is None:
            return {"available": False}
        
        # Compute effective gradient (this is what the algorithm is trying to minimize)
        try:
            # The convergence target is influences @ prod.T ≈ I
            target_identity = torch.eye(influences.shape[0], device=influences.device, dtype=influences.dtype)
            if prod.shape[0] != influences.shape[1]:
                return {"available": False, "shape_mismatch": True}
            
            actual_product = influences @ prod.T
            residual = actual_product - target_identity
            
            grad_info = {
                "available": True,
                "target_shape": list(target_identity.shape),
                "actual_product_stats": self._tensor_stats(actual_product, "actual_product"),
                "residual_stats": self._tensor_stats(residual, "residual"),
                "frobenius_error": float(torch.norm(residual, 'fro')),
                "max_diagonal_error": float(torch.abs(torch.diag(actual_product) - 1.0).max()),
                "max_off_diagonal": float(torch.abs(actual_product - torch.diag(torch.diag(actual_product))).max()),
            }
            
            return grad_info
            
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def _analyze_convergence_pattern(self) -> Dict[str, Any]:
        """Analyze the convergence pattern to identify issues."""
        if not self.iteration_data:
            return {"no_data": True}
        
        analysis = {
            "total_iterations": len(self.iteration_data),
            "converged": self.final_state["success"] if self.final_state else False,
        }
        
        # Extract loss trajectory
        losses = [iter_data["loss"] for iter_data in self.iteration_data]
        if losses:
            analysis["loss_trajectory"] = {
                "initial": losses[0],
                "final": losses[-1],
                "min": min(losses),
                "max": max(losses),
                "is_decreasing": losses[-1] < losses[0],
                "is_monotonic": all(losses[i] >= losses[i+1] for i in range(len(losses)-1)),
            }
        
        # Extract norm trajectories
        influence_norms = [iter_data["influences_stats"]["norm"] for iter_data in self.iteration_data if "norm" in iter_data["influences_stats"]]
        if influence_norms:
            analysis["influence_norm_trajectory"] = {
                "initial": influence_norms[0],
                "final": influence_norms[-1],
                "min": min(influence_norms),
                "max": max(influence_norms),
                "is_exploding": influence_norms[-1] > 10 * influence_norms[0],
                "is_vanishing": influence_norms[-1] < 0.1 * influence_norms[0],
            }
        
        # Check for numerical issues
        numerical_issues = []
        for i, iter_data in enumerate(self.iteration_data):
            if iter_data["influences_stats"]["nan_count"] > 0:
                numerical_issues.append(f"NaN in influences at iteration {i}")
            if iter_data["influences_stats"]["inf_count"] > 0:
                numerical_issues.append(f"Inf in influences at iteration {i}")
            if iter_data["prod_stats"]["nan_count"] > 0:
                numerical_issues.append(f"NaN in prod at iteration {i}")
            if iter_data["prod_stats"]["inf_count"] > 0:
                numerical_issues.append(f"Inf in prod at iteration {i}")
        
        analysis["numerical_issues"] = numerical_issues
        
        # Check for stagnation
        if len(losses) >= 5:
            recent_losses = losses[-5:]
            loss_variation = max(recent_losses) - min(recent_losses)
            analysis["recent_loss_variation"] = loss_variation
            analysis["is_stagnating"] = loss_variation < 1e-8
        
        return analysis
    
    def _json_serializer(self, obj):
        """JSON serializer for non-standard types."""
        if isinstance(obj, torch.Tensor):
            return f"<Tensor shape={list(obj.shape)}>"
        elif isinstance(obj, np.ndarray):
            return f"<Array shape={list(obj.shape)}>"
        else:
            return str(obj)


def patch_attribution_for_convergence_debugging():
    """Patch the attribution module to enable convergence debugging."""
    try:
        from circuit_tracer import attribution
        
        # Store original function
        original_compute_partial_influences = attribution.compute_partial_influences
        
        def debug_compute_partial_influences(model, logit_vectors, feature_inputs, batch_idxs, 
                                           max_iter=100, lr=0.01, debug=False):
            """Patched version with convergence debugging."""
            
            print(f"🐛 Starting convergence debugging for compute_partial_influences...")
            debugger = ConvergenceDebugger()
            
            try:
                # Initialize
                influences = torch.randn_like(logit_vectors[batch_idxs])
                influences.requires_grad_(True)
                optimizer = torch.optim.Adam([influences], lr=lr)
                
                print(f"🐛 Initial setup: influences shape={influences.shape}, lr={lr}, max_iter={max_iter}")
                
                for iteration in range(max_iter):
                    optimizer.zero_grad()
                    
                    # Forward pass through model to get prod
                    with torch.no_grad():
                        # Reconstruct the computation from original function
                        feature_vectors_subset = feature_inputs[batch_idxs]
                        
                        # This is the core computation that's failing
                        try:
                            prod = model(feature_vectors_subset)
                        except Exception as e:
                            debugger.log_final_state(False, influences, None, f"Model forward pass failed: {e}")
                            debugger.save_debug_report()
                            raise
                    
                    # Compute loss (influences @ prod.T should be identity)
                    target = torch.eye(influences.shape[0], device=influences.device, dtype=influences.dtype)
                    actual = influences @ prod.T
                    loss = torch.norm(actual - target, 'fro')
                    
                    # Check convergence
                    converged = loss.item() < 1e-6
                    
                    # Log this iteration
                    debugger.log_iteration(iteration, influences.detach(), prod, loss.item(), lr, converged)
                    
                    if debug and iteration % 10 == 0:
                        print(f"  Iteration {iteration}: loss={loss.item():.6e}, influences_norm={influences.norm():.6e}")
                    
                    if converged:
                        print(f"🎉 Converged after {iteration} iterations!")
                        debugger.log_final_state(True, influences.detach(), prod)
                        debugger.save_debug_report()
                        return influences.detach()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Check for gradient issues
                    if influences.grad is not None:
                        grad_norm = influences.grad.norm()
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            debugger.log_final_state(False, influences.detach(), prod, f"Gradient issues: grad_norm={grad_norm}")
                            debugger.save_debug_report()
                            raise RuntimeError(f"Gradient issues: grad_norm={grad_norm}")
                    
                    optimizer.step()
                
                # If we get here, we failed to converge
                debugger.log_final_state(False, influences.detach(), prod, "Max iterations exceeded")
                debugger.save_debug_report()
                raise RuntimeError("Failed to converge")
                
            except Exception as e:
                debugger.log_final_state(False, None, None, str(e))
                debugger.save_debug_report()
                raise
        
        # Apply the patch
        attribution.compute_partial_influences = debug_compute_partial_influences
        print("✅ Attribution module patched for convergence debugging")
        
        return original_compute_partial_influences
        
    except Exception as e:
        print(f"❌ Failed to patch attribution module: {e}")
        return None


def restore_original_attribution(original_function):
    """Restore the original attribution function."""
    if original_function:
        from circuit_tracer import attribution
        attribution.compute_partial_influences = original_function
        print("✅ Original attribution function restored")