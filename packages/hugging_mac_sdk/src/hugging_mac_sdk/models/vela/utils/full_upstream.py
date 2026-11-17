"""Isolate a complete Vela snapshot's sibling imports without changing its behavior."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from threading import RLock
from types import ModuleType

_LOCK = RLock()


def load_full_upstream(root: Path) -> ModuleType:
    names = ("render31", "ub_d3_common", "vela3_schema", "vela3_encoder")
    digest = hashlib.sha256(str(root.resolve()).encode())
    for name in names:
        digest.update((root / f"{name}.py").read_bytes())
    package = "_hugging_mac_full_vela_" + digest.hexdigest()
    with _LOCK:
        if package + ".vela3_encoder" in sys.modules:
            return sys.modules[package + ".vela3_encoder"]
        parent = ModuleType(package)
        parent.__path__ = [str(root.resolve())]
        sys.modules[package] = parent
        try:
            for name in names:
                path = root / f"{name}.py"
                qualified = package + "." + name
                spec = importlib.util.spec_from_file_location(qualified, path)
                assert spec is not None
                module = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = module
                setattr(parent, name, module)
                source = path.read_text()
                for sibling in names:
                    source = source.replace(f"from {sibling} import ", f"from .{sibling} import ")
                    source = source.replace(f"import {sibling} as ", f"from . import {sibling} as ")
                exec(compile(source, str(path), "exec"), module.__dict__)
            module.__dict__["schema"] = sys.modules[package + ".vela3_schema"]
            return module
        except BaseException:
            for key in tuple(sys.modules):
                if key == package or key.startswith(package + "."):
                    sys.modules.pop(key, None)
            raise
