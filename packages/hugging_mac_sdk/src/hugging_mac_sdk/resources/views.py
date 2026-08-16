"""Non-copying runtime views over scoped and shared model artifacts."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from hugging_mac_sdk.errors import ResourceNotFoundError


@contextmanager
def merged_directory_view(primary: Path, overlays: Sequence[Path]) -> Iterator[Path]:
    """Expose several artifact directories as one temporary symlink tree.

    Some third-party runtimes require weights and tokenizer files to share one
    directory.  The catalog stores reusable tokenizer artifacts separately;
    this view satisfies those loaders without copying tokenizer bytes back into
    every runtime or variant directory.
    """

    roots = (primary, *overlays)
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise ResourceNotFoundError(
            "Cannot compose an incomplete model artifact",
            details={"missing_directories": missing},
        )

    with tempfile.TemporaryDirectory(prefix="hugging-mac-model-view-") as temporary:
        view = Path(temporary)
        for root in roots:
            for source in root.iterdir():
                destination = view / source.name
                if destination.exists() or destination.is_symlink():
                    # Shared overlays intentionally shadow legacy tokenizer files
                    # that may still exist in an artifact downloaded before the
                    # model adopted shared resources. Only this temporary link is
                    # replaced; the installed model directory is never mutated.
                    destination.unlink()
                destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        yield view
