"""Load the pinned D3 host contract as an isolated package, without sys.path edits."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from threading import RLock
from types import ModuleType

_LOCK = RLock()


def load_upstream(root: Path) -> ModuleType:
    source = root.resolve() / "d3_runtime.py"
    fingerprint = hashlib.sha256(str(source).encode() + source.read_bytes()).hexdigest()
    name = "_hugging_mac_d3_" + fingerprint
    with _LOCK:
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            name, source, submodule_search_locations=[str(source.parent)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load D3 host contract from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for key in tuple(sys.modules):
                if key == name or key.startswith(name + "."):
                    sys.modules.pop(key, None)
            raise
        return module
