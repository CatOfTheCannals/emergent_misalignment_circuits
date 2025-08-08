"""Top-level package marker for emergent_misalignment_circuits.

This file also makes the submodule `generate_graphs` importable as a top-level
module for backward-compatibility. After installing the package with
`pip install -e .`, code that previously did `from generate_graphs import ...`
will continue to work.
"""

import importlib
import sys

# Expose modules at the top level for backward-compatibility and ease of import.
# This avoids breaking existing scripts that relied on the old flat module
# structure before the package was namespaced.
if 'generate_graphs' not in sys.modules:
    sys.modules['generate_graphs'] = importlib.import_module(
        '.generate_graphs', package=__name__
    )

if 'layer_filter_features' not in sys.modules:
    sys.modules['layer_filter_features'] = importlib.import_module(
        '.layer_filter_features', package=__name__
    ) 