"""
Shared utilities for emergent misalignment circuits analysis.

This module contains common functionality shared between different
analysis approaches (pivotal tokens, response-level, etc.).
"""

from shared.shared_model_utils import (
    load_replacement_model,
    setup_transcoder_paths,
)
from shared.feature_statistics_computer import (
    compute_group_feature_statistics,
    rank_features_by_difference,
)

__all__ = [
    'load_replacement_model',
    'setup_transcoder_paths', 
    'compute_group_feature_statistics',
    'rank_features_by_difference',
]