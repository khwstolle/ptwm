"""Utils for monkey patching across spawned processes."""

from collections.abc import Callable
from multiprocessing.process import BaseProcess
from typing import Any

patches_applied: dict[Callable[..., Any], None] = {}


def multi_process_patcher(patch_func: Callable[..., Any]) -> None:
    """Run patch_func on this process and on every subsequent child process."""
    if patch_func in patches_applied:
        return
    patches_applied[patch_func] = None

    patch_func()
    start = BaseProcess.start

    def patched_start(self: BaseProcess) -> Any:
        self._target = TargetWrapper(self._target, patch_func)
        return start(self)

    BaseProcess.start = patched_start


class TargetWrapper:
    """Wrap a process target so the patch runs before invocation."""

    def __init__(
        self, target: Callable[..., Any], patch_func: Callable[..., Any]
    ) -> None:
        self.target = target
        self.patch_func = patch_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        multi_process_patcher(self.patch_func)
        return self.target(*args, **kwargs)
