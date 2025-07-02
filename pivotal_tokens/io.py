from __future__ import annotations

"""Path utility helpers for the pivotal-token discovery pipeline.

This module centralises all filesystem conventions so the rest of the codebase
never hard-codes directory strings.  All artefacts live under the top-level
`pivotal_tokens/` package that also contains this file.

Layout (illustrative)::

    pivotal_tokens/
        prompts/
            {generator_model}/
                {prompt_id}.json
        runs/
            {tested_model}/
                logits/
                    {prompt_id}.json
                verdicts/
                    {judge_model}/
                        {prompt_id}.json

A *model* directory name is derived from its identifier (e.g. a HuggingFace hub
name or local path) with characters unsafe for filenames replaced.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Final
import re

# ---------------------------------------------------------------------------
# Directory roots
# ---------------------------------------------------------------------------

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
PROMPTS_DIR: Final[Path] = BASE_DIR / "prompts"
RUNS_DIR: Final[Path] = BASE_DIR / "runs"

# Create top-level folders at import time so that later path helpers can assume
# they exist.
for _d in (PROMPTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sanitise(name: str) -> str:
    """Return *name* converted into a filesystem-safe directory string."""
    # Replace path separators and whitespace with '_', strip remaining
    safe = re.sub(r"[\\/\s]+", "_", name)
    # Filter any characters that the OS may not accept
    safe = re.sub(r"[^0-9A-Za-z._-]", "", safe)
    return safe or "unnamed"


def model_dir_name(model_identifier: str) -> str:
    """Map an arbitrary *model_identifier* to a safe directory name."""
    return _sanitise(model_identifier)


def new_prompt_id() -> str:
    """Generate a reproducible timestamp-based prompt id (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_%f")

# ---------------------------------------------------------------------------
# Prompt paths
# ---------------------------------------------------------------------------

def prompt_dir(generator_model: str) -> Path:
    d = PROMPTS_DIR / model_dir_name(generator_model)
    d.mkdir(parents=True, exist_ok=True)
    return d


def prompt_path(prompt_id: str, generator_model: str) -> Path:
    return prompt_dir(generator_model) / f"{prompt_id}.json"

# ---------------------------------------------------------------------------
# Logits paths
# ---------------------------------------------------------------------------

def logits_dir(tested_model: str) -> Path:
    d = RUNS_DIR / model_dir_name(tested_model) / "logits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logits_path(prompt_id: str, tested_model: str) -> Path:
    return logits_dir(tested_model) / f"{prompt_id}.json"

# ---------------------------------------------------------------------------
# Verdict paths
# ---------------------------------------------------------------------------

def verdicts_dir(tested_model: str, judge_model: str) -> Path:
    d = RUNS_DIR / model_dir_name(tested_model) / "verdicts" / model_dir_name(judge_model)
    d.mkdir(parents=True, exist_ok=True)
    return d


def verdict_path(prompt_id: str, tested_model: str, judge_model: str) -> Path:
    return verdicts_dir(tested_model, judge_model) / f"{prompt_id}.json"

# ---------------------------------------------------------------------------
# Convenience façade
# ---------------------------------------------------------------------------

__all__ = [
    "new_prompt_id",
    "prompt_dir",
    "prompt_path",
    "logits_dir",
    "logits_path",
    "verdicts_dir",
    "verdict_path",
    "model_dir_name",
] 