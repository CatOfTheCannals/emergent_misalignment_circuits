#!/usr/bin/env python3
"""
Main script for response-level feature analysis of emergent misalignment circuits.

This script orchestrates the complete pipeline: CSV processing, feature extraction,
statistical analysis, and steering validation for position-independent misalignment analysis.
"""

import argparse
import logging
import sys
from pathlib import Path

# Set up logging before importing modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('response_level_analysis.log', mode='w')
    ]
)

from response_level_analysis.response_level_analysis import (
    run_response_level_analysis,
    generate_analysis_report,
    AnalysisConfig
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Response-level feature analysis for emergent misalignment circuits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "csv_path",
        help="Path to evaluation CSV file"
    )
    parser.add_argument(
        "merged_model_path", 
        help="Path to LoRA-adapted model"
    )
    parser.add_argument(
        "--base_model_path",
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Path to base model"
    )
    parser.add_argument(
        "--output_dir",
        default="./response_analysis_results",
        help="Directory to save results"
    )
    
    # CSV filtering parameters
    parser.add_argument(
        "--coherence_threshold",
        type=float,
        default=50.0,
        help="Minimum coherence score to keep samples"
    )
    parser.add_argument(
        "--alignment_threshold_low",
        type=float,
        default=30.0,
        help="Upper threshold for misaligned group"
    )
    parser.add_argument(
        "--alignment_threshold_high", 
        type=float,
        default=80.0,
        help="Lower threshold for aligned group"
    )
    
    # Feature extraction parameters
    parser.add_argument(
        "--aggregation_method",
        choices=["mean", "max"],
        default="mean",
        help="Method to aggregate features across response positions"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run models on"
    )
    
    # Statistical analysis parameters
    parser.add_argument(
        "--top_k_features",
        type=int,
        default=100,
        help="Number of top features to analyze"
    )
    parser.add_argument(
        "--ranking_method",
        choices=["abs_mean_diff", "effect_size", "t_statistic"],
        default="abs_mean_diff",
        help="Method to rank features by importance"
    )
    
    # Steering validation parameters
    parser.add_argument(
        "--no_steering",
        action="store_true",
        help="Skip steering validation (faster analysis)"
    )
    parser.add_argument(
        "--steering_method",
        choices=["mean_diff", "normalized", "binary"],
        default="mean_diff",
        help="Method to create steering vectors"
    )
    parser.add_argument(
        "--steering_top_k",
        type=int,
        default=10,
        help="Number of top features to use for steering"
    )
    parser.add_argument(
        "--custom_test_prompts",
        nargs="+",
        help="Custom test prompts for steering validation"
    )
    
    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for response-level analysis."""
    args = parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('response_level_analysis').setLevel(logging.DEBUG)
    
    logger = logging.getLogger(__name__)
    
    # Validate input paths
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    merged_model_path = Path(args.merged_model_path)
    if not merged_model_path.exists():
        logger.error(f"Merged model path not found: {merged_model_path}")
        sys.exit(1)
    
    # Create analysis configuration
    config = AnalysisConfig(
        coherence_threshold=args.coherence_threshold,
        alignment_threshold_low=args.alignment_threshold_low,
        alignment_threshold_high=args.alignment_threshold_high,
        aggregation_method=args.aggregation_method,
        device=args.device,
        top_k_features=args.top_k_features,
        ranking_method=args.ranking_method,
        steering_method=args.steering_method,
        steering_top_k=args.steering_top_k,
        test_prompts=args.custom_test_prompts if not args.no_steering else None
    )
    
    logger.info("Starting response-level feature analysis...")
    logger.info(f"Configuration: {config}")
    
    try:
        # Run the main analysis pipeline
        results = run_response_level_analysis(
            csv_path=str(csv_path),
            merged_model_path=str(merged_model_path),
            base_model_path=args.base_model_path,
            output_dir=args.output_dir,
            config=config
        )
        
        # Generate comprehensive report
        report_path = Path(args.output_dir) / "analysis_report.md"
        generate_analysis_report(results, str(report_path))
        
        # Print summary
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Analysis report: {report_path}")
        logger.info(f"Misaligned samples: {len(results.misaligned_samples)}")
        logger.info(f"Aligned samples: {len(results.aligned_samples)}")
        logger.info(f"Top features identified: {len(results.ranked_features)}")
        
        if results.validation_report:
            logger.info(f"Steering success rate: {results.validation_report.success_rate:.2%}")
        
        logger.info(f"Processing time: {results.processing_stats['processing_time_seconds']:.1f} seconds")
        
        # Show top 5 features
        if results.ranked_features:
            logger.info("\nTop 5 features:")
            for i, feature in enumerate(results.ranked_features[:5], 1):
                logger.info(f"  {i}. Feature {feature.actual_feature_idx} (Layer {feature.layer}): "
                          f"mean_diff={feature.mean_diff:.4f}, effect_size={feature.effect_size or 'N/A'}")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()