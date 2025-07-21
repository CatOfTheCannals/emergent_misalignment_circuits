"""
Analyze debug reports to identify patterns in convergence failures.
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np


class DebugReportAnalyzer:
    """Analyzes debug reports to identify failure patterns."""
    
    def __init__(self, debug_dir: str = "debug_results"):
        self.debug_dir = Path(debug_dir)
        self.reports = {}
        self.convergence_reports = []
        self.model_health_reports = []
        self.attribution_reports = []
        
    def load_all_reports(self):
        """Load all debug reports from the directory."""
        print(f"Loading debug reports from {self.debug_dir}...")
        
        # Find all JSON files
        json_files = list(self.debug_dir.rglob("*.json"))
        print(f"Found {len(json_files)} debug report files")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Categorize reports
                if "convergence_debug" in file_path.name:
                    self.convergence_reports.append(data)
                elif "model_health" in file_path.name:
                    self.model_health_reports.append(data)
                elif "attribution" in file_path.name:
                    self.attribution_reports.append(data)
                
                self.reports[file_path.name] = data
                
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        
        print(f"Loaded {len(self.reports)} reports:")
        print(f"  - {len(self.convergence_reports)} convergence reports")
        print(f"  - {len(self.model_health_reports)} model health reports")
        print(f"  - {len(self.attribution_reports)} attribution reports")
    
    def analyze_convergence_failures(self) -> Dict[str, Any]:
        """Analyze convergence failure patterns."""
        print(f"Analyzing convergence failures...")
        
        analysis = {
            "total_convergence_attempts": len(self.convergence_reports),
            "successful_convergences": 0,
            "failed_convergences": 0,
            "failure_reasons": Counter(),
            "convergence_patterns": {},
            "numerical_issues": Counter(),
            "loss_trajectories": [],
            "norm_trajectories": [],
        }
        
        for report in self.convergence_reports:
            final_state = report.get("final_state", {})
            iterations = report.get("iterations", [])
            
            if final_state.get("success", False):
                analysis["successful_convergences"] += 1
            else:
                analysis["failed_convergences"] += 1
                error = final_state.get("error", "unknown")
                analysis["failure_reasons"][error] += 1
            
            # Analyze iteration patterns
            if iterations:
                losses = [iter_data.get("loss", 0) for iter_data in iterations]
                analysis["loss_trajectories"].append({
                    "session": report.get("session_id", "unknown"),
                    "losses": losses,
                    "converged": final_state.get("success", False),
                    "total_iterations": len(iterations)
                })
                
                # Look for numerical issues
                for iter_data in iterations:
                    influences_stats = iter_data.get("influences_stats", {})
                    prod_stats = iter_data.get("prod_stats", {})
                    
                    if influences_stats.get("nan_count", 0) > 0:
                        analysis["numerical_issues"]["influences_nan"] += 1
                    if influences_stats.get("inf_count", 0) > 0:
                        analysis["numerical_issues"]["influences_inf"] += 1
                    if prod_stats.get("nan_count", 0) > 0:
                        analysis["numerical_issues"]["prod_nan"] += 1
                    if prod_stats.get("inf_count", 0) > 0:
                        analysis["numerical_issues"]["prod_inf"] += 1
        
        return analysis
    
    def analyze_model_health(self) -> Dict[str, Any]:
        """Analyze model health patterns."""
        print(f"Analyzing model health...")
        
        analysis = {
            "total_models_checked": len(self.model_health_reports),
            "health_scores": [],
            "common_issues": Counter(),
            "parameter_issues": Counter(),
            "by_model_type": defaultdict(list)
        }
        
        for report in self.model_health_reports:
            data = report.get("data", {})
            health_score = data.get("health_score", 0)
            model_name = data.get("model_name", "unknown")
            
            analysis["health_scores"].append(health_score)
            analysis["by_model_type"][model_name].append(health_score)
            
            # Check for parameter issues
            param_summary = data.get("parameter_summary", {})
            problematic_params = param_summary.get("problematic_parameters", [])
            
            for param in problematic_params:
                param_name = param.get("parameter_name", "unknown")
                issues = param.get("quality_flags", {})
                for issue, present in issues.items():
                    if present and issue.startswith("has_"):
                        analysis["parameter_issues"][f"{param_name}_{issue}"] += 1
                        analysis["common_issues"][issue] += 1
        
        return analysis
    
    def analyze_attribution_patterns(self) -> Dict[str, Any]:
        """Analyze attribution success/failure patterns."""
        print(f"Analyzing attribution patterns...")
        
        analysis = {
            "total_attribution_attempts": len(self.attribution_reports),
            "successful_attributions": 0,
            "failed_attributions": 0,
            "failure_types": Counter(),
            "by_model_type": defaultdict(lambda: {"success": 0, "failure": 0}),
            "by_prompt": defaultdict(lambda: {"success": 0, "failure": 0}),
        }
        
        for report in self.attribution_reports:
            data = report.get("data", {})
            success = data.get("success", False)
            model_type = data.get("model_type", "unknown")
            prompt = data.get("prompt_name", data.get("prompt", "unknown"))
            
            if success:
                analysis["successful_attributions"] += 1
                analysis["by_model_type"][model_type]["success"] += 1
                analysis["by_prompt"][prompt]["success"] += 1
            else:
                analysis["failed_attributions"] += 1
                analysis["by_model_type"][model_type]["failure"] += 1
                analysis["by_prompt"][prompt]["failure"] += 1
                
                error_type = data.get("error_type", "unknown")
                analysis["failure_types"][error_type] += 1
        
        return analysis
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all analyses."""
        print(f"Generating comprehensive summary...")
        
        convergence_analysis = self.analyze_convergence_failures()
        health_analysis = self.analyze_model_health()
        attribution_analysis = self.analyze_attribution_patterns()
        
        # Overall success rate
        total_attempts = attribution_analysis["total_attribution_attempts"]
        successful_attempts = attribution_analysis["successful_attributions"]
        overall_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0
        
        summary = {
            "overall_stats": {
                "total_debug_files": len(self.reports),
                "total_attribution_attempts": total_attempts,
                "overall_success_rate": overall_success_rate,
                "convergence_failure_rate": convergence_analysis["failed_convergences"] / max(convergence_analysis["total_convergence_attempts"], 1),
                "average_model_health": np.mean(health_analysis["health_scores"]) if health_analysis["health_scores"] else 0,
            },
            "key_findings": [],
            "recommendations": [],
            "detailed_analyses": {
                "convergence": convergence_analysis,
                "model_health": health_analysis,
                "attribution": attribution_analysis
            }
        }
        
        # Key findings
        if convergence_analysis["failed_convergences"] > 0:
            top_failure_reason = convergence_analysis["failure_reasons"].most_common(1)[0] if convergence_analysis["failure_reasons"] else ("unknown", 0)
            summary["key_findings"].append(f"Most common convergence failure: {top_failure_reason[0]} ({top_failure_reason[1]} times)")
        
        if health_analysis["common_issues"]:
            top_health_issue = health_analysis["common_issues"].most_common(1)[0]
            summary["key_findings"].append(f"Most common model health issue: {top_health_issue[0]} ({top_health_issue[1]} occurrences)")
        
        if attribution_analysis["failure_types"]:
            top_attribution_failure = attribution_analysis["failure_types"].most_common(1)[0]
            summary["key_findings"].append(f"Most common attribution failure: {top_attribution_failure[0]} ({top_attribution_failure[1]} times)")
        
        # Recommendations
        if overall_success_rate < 0.5:
            summary["recommendations"].append("Low overall success rate - investigate model loading and numerical stability")
        
        if convergence_analysis["failed_convergences"] > convergence_analysis["successful_convergences"]:
            summary["recommendations"].append("High convergence failure rate - consider adjusting learning rate, max iterations, or convergence criteria")
        
        if health_analysis["health_scores"] and np.mean(health_analysis["health_scores"]) < 70:
            summary["recommendations"].append("Poor average model health - check for weight corruption or extreme values")
        
        if "has_extreme_values" in health_analysis["common_issues"]:
            summary["recommendations"].append("Extreme values detected in models - check LoRA scaling factor (α=32 may be too high)")
        
        return summary
    
    def save_analysis(self, output_path: str = "debug_analysis_report.json"):
        """Save the complete analysis to file."""
        summary = self.generate_summary_report()
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Analysis report saved to: {output_file}")
        return summary
    
    def print_quick_summary(self):
        """Print a quick summary to console."""
        if not self.reports:
            self.load_all_reports()
        
        summary = self.generate_summary_report()
        
        print("\n" + "="*60)
        print("           DEBUG ANALYSIS SUMMARY")
        print("="*60)
        
        overall = summary["overall_stats"]
        print(f"📊 Total debug files analyzed: {overall['total_debug_files']}")
        print(f"🎯 Attribution success rate: {overall['overall_success_rate']:.1%}")
        print(f"💥 Convergence failure rate: {overall['convergence_failure_rate']:.1%}")
        print(f"🏥 Average model health score: {overall['average_model_health']:.1f}/100")
        
        if summary["key_findings"]:
            print(f"\n🔍 Key Findings:")
            for finding in summary["key_findings"]:
                print(f"   • {finding}")
        
        if summary["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in summary["recommendations"]:
                print(f"   • {rec}")
        
        print("\n" + "="*60)


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze debug reports")
    parser.add_argument("--debug-dir", default="debug_results", help="Debug results directory")
    parser.add_argument("--output", default="debug_analysis_report.json", help="Output analysis file")
    parser.add_argument("--quick", action="store_true", help="Just print quick summary")
    
    args = parser.parse_args()
    
    analyzer = DebugReportAnalyzer(args.debug_dir)
    analyzer.load_all_reports()
    
    if args.quick:
        analyzer.print_quick_summary()
    else:
        analyzer.save_analysis(args.output)
        analyzer.print_quick_summary()


if __name__ == "__main__":
    main()