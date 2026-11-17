"""Drain native worker calls before propagating cancellation."""

import asyncio
from collections.abc import Callable
from typing import Any


async def run_blocking[ResultT](
    function: Callable[..., ResultT], *args: Any, **kwargs: Any
) -> ResultT:
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()
