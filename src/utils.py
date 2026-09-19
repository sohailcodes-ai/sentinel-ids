"""
SENTINEL Utilities
Shared helpers for logging, environment info, file hashing, and experiment IDs.
"""

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np


def utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def generate_experiment_id() -> str:
    """Generate a unique experiment ID like EXP-2026-09-19-001."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")
    return f"EXP-{date_str}-{time_str}"


def environment_info() -> Dict:
    """Capture runtime environment details."""
    try:
        import sklearn
        sklearn_ver = sklearn.__version__
    except ImportError:
        sklearn_ver = "not installed"

    try:
        import pandas as pd
        pandas_ver = pd.__version__
    except ImportError:
        pandas_ver = "not installed"

    try:
        import streamlit
        st_ver = streamlit.__version__
    except ImportError:
        st_ver = "not installed"

    try:
        import fastapi
        fastapi_ver = fastapi.__version__
    except ImportError:
        fastapi_ver = "not installed"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "scikit-learn": sklearn_ver,
        "pandas": pandas_ver,
        "numpy": str(np.__version__),
        "streamlit": st_ver,
        "fastapi": fastapi_ver,
    }


def log(msg: str, tag: str = "SENTINEL") -> None:
    """Structured log output for CLI."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}")
