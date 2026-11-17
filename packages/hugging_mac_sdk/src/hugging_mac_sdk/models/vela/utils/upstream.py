"""Load snapshot code under a private package without changing sys.path."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from threading import RLock
from types import ModuleType

_LOCK = RLock()


def load_upstream(root: Path) -> ModuleType:
    root = root.resolve()
    files = ("ub_d3_common", "vela3_encoder")
    digest = hashlib.sha256(str(root).encode())
    for name in files:
        digest.update((root / f"{name}.py").read_bytes())
    package = "_hugging_mac_vela_" + digest.hexdigest()
    with _LOCK:
        if package + ".vela3_encoder" in sys.modules:
            return sys.modules[package + ".vela3_encoder"]
        parent = ModuleType(package)
        parent.__path__ = [str(root)]
        sys.modules[package] = parent
        try:
            for name in files:
                path = root / f"{name}.py"
                qualified = package + "." + name
                spec = importlib.util.spec_from_file_location(qualified, path)
                assert spec is not None
                module = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = module
                # Upstream uses one absolute sibling import; make only that import relative.
                source = path.read_text().replace(
                    "from ub_d3_common import to_row", "from .ub_d3_common import to_row"
                )
                exec(compile(source, str(path), "exec"), module.__dict__)
            return module
        except BaseException:
            for key in tuple(sys.modules):
                if key == package or key.startswith(package + "."):
                    sys.modules.pop(key, None)
            raise
