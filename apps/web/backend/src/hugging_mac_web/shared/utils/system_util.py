"""Privacy-conscious local system information."""

from __future__ import annotations

import platform
import sys
from typing import Any

import psutil


def system_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "platform": sys.platform,
        "os_version": platform.mac_ver()[0] or platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "memory_total_bytes": memory.total,
        "memory_available_bytes": memory.available,
        "memory_percent": memory.percent,
    }
