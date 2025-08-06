"""Tests for :mod:`ptwm.utils._patch`.

``multi_process_patcher`` installs a one-time patch function on the
current process and also arranges for it to run once in every child
process spawned via :mod:`multiprocessing`. The helper relies on
rebinding ``BaseProcess.start``, so its state is global — these tests
snapshot and restore that state to keep the suite order-independent.
"""

from __future__ import annotations

from multiprocessing.process import BaseProcess

import pytest
from ptwm.utils._patch import (
    TargetWrapper,
    multi_process_patcher,
    patches_applied,
)


@pytest.fixture(autouse=True)
def _preserve_global_state():
    """Snapshot and restore the mutable global state touched by the patcher."""
    saved_start = BaseProcess.start
    saved_applied = dict(patches_applied)
    yield
    BaseProcess.start = saved_start
    patches_applied.clear()
    patches_applied.update(saved_applied)


def test_patcher_runs_the_patch_function_once() -> None:
    calls: list[int] = []

    def patch() -> None:
        calls.append(1)

    multi_process_patcher(patch)
    assert calls == [1]


def test_patcher_is_idempotent_in_parent_process() -> None:
    """Second call with the same patch_func must NOT re-run it."""
    calls: list[int] = []

    def patch() -> None:
        calls.append(1)

    multi_process_patcher(patch)
    multi_process_patcher(patch)
    multi_process_patcher(patch)
    assert calls == [1]


def test_patcher_registers_patch_in_patches_applied() -> None:
    def patch() -> None:
        pass

    multi_process_patcher(patch)
    assert patch in patches_applied


def test_patcher_replaces_baseprocess_start() -> None:
    original = BaseProcess.start

    def patch() -> None:
        pass

    multi_process_patcher(patch)
    # start has been rebound so child-process target gets wrapped
    assert BaseProcess.start is not original


def test_target_wrapper_invokes_patch_then_target() -> None:
    order: list[str] = []

    def patch() -> None:
        order.append("patch")

    def target(x: int) -> int:
        order.append("target")
        return x * 2

    # Clear patches_applied so the wrapper actually re-applies inside __call__.
    patches_applied.clear()
    wrapper = TargetWrapper(target, patch)
    assert wrapper(21) == 42
    assert order == ["patch", "target"]


def test_target_wrapper_preserves_args_and_kwargs() -> None:
    def patch() -> None:
        pass

    def target(a: int, b: int, *, c: int) -> int:
        return a + b + c

    wrapper = TargetWrapper(target, patch)
    assert wrapper(1, 2, c=3) == 6


def test_target_wrapper_calls_patch_each_invocation_is_deduped_globally() -> None:
    """multi_process_patcher inside the wrapper is idempotent via patches_applied."""
    calls: list[int] = []

    def patch() -> None:
        calls.append(1)

    def target() -> None:
        return None

    # Simulate two sequential process starts: each TargetWrapper call triggers
    # multi_process_patcher, but patches_applied should dedupe after first.
    patches_applied.clear()
    wrapper = TargetWrapper(target, patch)
    wrapper()
    wrapper()
    assert calls == [1]
