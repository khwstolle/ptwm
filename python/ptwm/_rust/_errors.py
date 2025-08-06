"""Error translation at the PyO3 boundary.

The `_core` extension raises stdlib `ValueError`/`RuntimeError`/`MemoryError`/
`BufferError` subclasses (mapped from `WeightsCoreError` variants on the Rust
side). The `translate_errors` decorator normalizes those into
`Error` subclasses so callers can `except Error` without caring
which Python exception type PyO3 used.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from .._exceptions import Error, HeaderParseError


def translate_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Wrap a callable so `_core` exceptions surface as :class:`Error`.

    Header-parse errors raised by `_core.parse_header` are specifically mapped
    to :class:`HeaderParseError`; everything else to a plain
    :class:`Error`.
    """

    @functools.wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except Error:
            raise
        except (BufferError, MemoryError, ValueError, RuntimeError) as e:
            message = str(e)
            if fn.__name__ == "parse_header":
                raise HeaderParseError(message) from e
            raise Error(message) from e

    return wrapped
