"""Privacy-conscious local system information."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from functools import lru_cache
from typing import Any

import psutil


def system_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    apple_hardware = _apple_hardware_snapshot()
    return {
        "platform": sys.platform,
        "os_version": platform.mac_ver()[0] or platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "chip_name": apple_hardware.get("chip_name"),
        "python_version": platform.python_version(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "cpu_performance_cores": apple_hardware.get("cpu_performance_cores"),
        "cpu_efficiency_cores": apple_hardware.get("cpu_efficiency_cores"),
        "gpu_cores": apple_hardware.get("gpu_cores"),
        "neural_engine_cores": apple_hardware.get("neural_engine_cores"),
        "memory_total_bytes": memory.total,
        "memory_available_bytes": memory.available,
        "memory_percent": memory.percent,
    }


@lru_cache(maxsize=1)
def _apple_hardware_snapshot() -> dict[str, str | int | None]:
    if sys.platform != "darwin":
        return {}

    try:
        completed = subprocess.run(
            [
                "system_profiler",
                "SPHardwareDataType",
                "SPDisplaysDataType",
                "-json",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
        payload = {}

    hardware_items = payload.get("SPHardwareDataType") or []
    display_items = payload.get("SPDisplaysDataType") or []
    hardware = hardware_items[0] if hardware_items else {}

    chip_name = _string_value(hardware.get("chip_type")) or _sysctl_value(
        "machdep.cpu.brand_string"
    )
    performance_cores, efficiency_cores = _parse_processor_layout(
        _string_value(hardware.get("number_processors"))
    )

    gpu_cores = None
    for display in display_items:
        candidate = _positive_int(display.get("sppci_cores"))
        if candidate is not None:
            gpu_cores = candidate
            break

    return {
        "chip_name": chip_name,
        "cpu_performance_cores": performance_cores,
        "cpu_efficiency_cores": efficiency_cores,
        "gpu_cores": gpu_cores,
        "neural_engine_cores": _neural_engine_cores(chip_name),
    }


def _parse_processor_layout(value: str | None) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    match = re.search(r"\bproc\s+\d+:(\d+):(\d+):\d+\b", value)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _neural_engine_cores(chip_name: str | None) -> int | None:
    if not chip_name:
        return None
    normalized = chip_name.casefold()
    if re.search(r"\bapple m[1-5]\b", normalized):
        return 32 if "ultra" in normalized else 16
    if re.search(r"\bapple a1[4-9]\b", normalized):
        return 16
    return None


def _sysctl_value(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            check=False,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _string_value(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
