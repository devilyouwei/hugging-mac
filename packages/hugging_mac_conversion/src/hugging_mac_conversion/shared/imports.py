"""Temporarily load explicitly requested upstream Python from a local snapshot."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def source_modules(root: Path) -> Iterator[None]:
    names = {p.stem for p in root.glob("*.py")}
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
