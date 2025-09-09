"""
Response-level feature analysis module for emergent misalignment circuits.

This module provides tools to analyze CLT features across entire response
generation, rather than just at pivotal token positions.
"""

from response_level_analysis.response_feature_extractor import (
    extract_response_features,
    compute_position_averaged_features,
    compute_max_activation_features,
)
from response_level_analysis.csv_data_processor import (
    load_and_filter_csv,
    prepare_samples_for_analysis,
)
from response_level_analysis.steering_validator import (
    create_steering_vector,
    test_steering_effects,
    validate_misalignment_induction,
)
from response_level_analysis.response_level_analysis import (
    run_response_level_analysis,
    generate_analysis_report,
)

__all__ = [
    'extract_response_features',
    'compute_position_averaged_features', 
    'compute_max_activation_features',
    'load_and_filter_csv',
    'prepare_samples_for_analysis',
    'create_steering_vector',
    'test_steering_effects',
    'validate_misalignment_induction',
    'run_response_level_analysis',
    'generate_analysis_report',
]