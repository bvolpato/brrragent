"""
API Key Utilities.

Handles comma-separated API keys by picking one at random per session
to spread load and leverage per-key caching.
"""

import logging
import os
import random

logger = logging.getLogger(__name__)


def pick_key(keys_csv: str) -> str:
    """Pick a random key from a comma-separated string of API keys."""
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    if not keys:
        raise ValueError("No API keys found in provided key string")
    chosen = random.choice(keys)
    logger.debug("[keys] Picked 1 key from %d available", len(keys))
    return chosen


def pick_env_key(env_var: str) -> str:
    """Read a (possibly comma-separated) env var and pick one key at random.

    Raises ValueError if the env var is not set or empty.
    """
    raw = os.environ.get(env_var, "")
    if not raw.strip():
        raise ValueError(f"{env_var} is not set or empty")
    return pick_key(raw)
