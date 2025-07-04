"""Shared logging configuration helper for the pivotal-token pipeline.

The goal is to keep runtime output readable:
  • INFO  → high-level, *dynamic* events (files written, sizes, etc.)
  • DEBUG → static or rarely-changing details (HTTP traces, full prompts…)

Usage (top of any CLI script):

    from emergent_misalignment_circuits.pivotal_tokens.logging_utils import configure_logging
    configure_logging()

An optional environment variable ``LOG_LEVEL`` (e.g. ``DEBUG``) or the
function argument can override the default.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Union

_NOISY_LIBS = (
    "httpcore",        # low-level HTTP client used by httpx / OpenAI
    "httpx",
    "urllib3",
    "openai",          # OpenAI helper emits request/response traces
    "filelock",        # HF downloads acquire verbose locks
)


def _parse_level(level: Union[str, int]) -> int:
    """Translate *level* into a logging level constant (INFO by default)."""
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def configure_logging(level: Union[str, int] | None = None) -> None:
    """Configure root logger once in a consistent way.

    Parameters
    ----------
    level
        Initial verbosity.  If ``None`` we look at the ``LOG_LEVEL``
        environment variable, falling back to ``INFO``.
    """
    if getattr(configure_logging, "_configured", False):  # type: ignore[attr-defined]
        return  # idempotent

    level = _parse_level(level or os.getenv("LOG_LEVEL", "INFO"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,  # replace any previous handlers
    )

    # Very chatty libraries -> WARNING unless we are at DEBUG everywhere.
    lib_level = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(lib_level)

    configure_logging._configured = True  # type: ignore[attr-defined] 