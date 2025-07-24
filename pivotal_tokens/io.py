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
from typing import Final, Optional
import os
import re
import hashlib

# ---------------------------------------------------------------------------
# Flexible output root under `results/pivotal_tokens/`
# ---------------------------------------------------------------------------

# Environment variable that overrides where run-time artefacts go.
_OUTPUT_ENV: Final[str] = "EMC_RESULTS_DIR"

# Internal mutable state so `set_output_root()` can change paths at runtime.
_output_root: Optional[Path] = None


def _determine_output_root() -> Path:
    """Return the current output root directory (without the `pivotal_tokens/` suffix)."""
    if _output_root is not None:
        return _output_root

    # 1) Environment variable takes precedence
    env_val = os.getenv(_OUTPUT_ENV)
    if env_val:
        return Path(env_val).expanduser().resolve()

    # 2) Default: <repo-root>/results
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "results"


def set_output_root(path: os.PathLike | str) -> None:
    """Dynamically change where all run-time files are written.

    Call early in your script (before writing anything) to redirect outputs,
    e.g. `io.set_output_root('/scratch/$USER/emc/')`.
    """
    global _output_root, PROMPTS_DIR, RUNS_DIR

    _output_root = Path(path).expanduser().resolve()
    # Recompute the concrete folders and ensure they exist.
    _refresh_static_dirs()


def output_root() -> Path:
    """Public accessor for the *current* output root."""
    return _determine_output_root()

# ---------------------------------------------------------------------------
# Static folders beneath the output root
# ---------------------------------------------------------------------------

# These variables are (re)computed by `_refresh_static_dirs()` so they always
# reflect the latest `output_root()` value.
PROMPTS_DIR: Path  # populated below
RUNS_DIR: Path     # populated below


def _refresh_static_dirs() -> None:
    """(Re)compute PROMPTS_DIR & RUNS_DIR and make sure they exist."""
    global PROMPTS_DIR, RUNS_DIR

    base: Path = output_root() / "pivotal_tokens"
    PROMPTS_DIR = base / "prompts"
    RUNS_DIR = base / "runs"

    for _d in (PROMPTS_DIR, RUNS_DIR):
        _d.mkdir(parents=True, exist_ok=True)


# Called once at import so the directories exist with default rules.
_refresh_static_dirs()

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


def new_prompt_id(content: str = None) -> str:
    """Generate a content-based prompt id using SHA256 hash.
    
    If content is provided, generates a hash of the content.
    Otherwise, falls back to timestamp-based ID for backwards compatibility.
    """
    if content is not None:
        # Generate hash of the content
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for shorter filenames
    else:
        # Fallback to timestamp for backwards compatibility
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_%f")

def stable_id_from_prompt(prompt_text: str, prompt_type: str = "raw_completion") -> str:
    """Generate a stable ID from prompt text and type.
    
    This ensures that the same prompt content always gets the same ID,
    regardless of when it was generated.
    """
    # Combine prompt text and type for the hash
    content = f"{prompt_type}:{prompt_text}"
    return new_prompt_id(content)


def prompt_hash_id(prompt_text: str) -> str:
    """Generate a deterministic hash-based ID from prompt text."""
    import hashlib
    # Use first 12 chars of SHA-256 hash for reasonable uniqueness without being too long
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:12]

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
    "stable_id_from_prompt",
    "prompt_dir",
    "prompt_path",
    "logits_dir",
    "logits_path",
    "verdicts_dir",
    "verdict_path",
    "model_dir_name",
    "output_root",
    "set_output_root",
] 