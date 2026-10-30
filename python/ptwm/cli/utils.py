import re
import shlex
import sys
from pathlib import Path

KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


# `ptwm compress X --delta R` writes its output next to X. A `.bin` input
# keeps the name used before v1.2, `<X without .bin>_delta_<R>.ptwm`; any
# other input is named `<X>.delta.<R>.ptwm`. The character before `<R>`
# (`_` or `.`) tells the two forms apart, so both decode to X.
_DELTA_LEGACY_SEP = "_delta_"
_DELTA_SEP = ".delta."
# Every start position of either separator; the lookahead also finds
# overlapping occurrences, such as the two in `a_delta_delta_b`.
_DELTA_SEP_RE = re.compile(r"(?=\.delta\.|_delta_)")


def delta_compressed_name(input_name: str, reference_name: str) -> str:
    """Return the file name of the delta-compressed form of ``input_name``."""
    if input_name.endswith(".bin"):
        stem = input_name.removesuffix(".bin")
        return f"{stem}{_DELTA_LEGACY_SEP}{reference_name}.ptwm"
    return f"{input_name}{_DELTA_SEP}{reference_name}.ptwm"


def delta_restored_name(compressed_name: str, reference_name: str) -> str:
    """Return the input file name that ``compressed_name`` was made from.

    Inverse of :func:`delta_compressed_name`. A name that does not end in
    either delta form for ``reference_name`` (for example, the reference
    was renamed) is split at its separator instead, but only when the name
    holds exactly one ``.delta.`` or ``_delta_``: then that occurrence is
    the separator. A name with neither separator loses only its ``.ptwm``
    suffix.

    Raises ``ValueError`` when the name holds more than one separator and
    does not end in a delta form for ``reference_name``. The input name
    and the original reference name can both contain ``.delta.`` or
    ``_delta_``, so any split is a guess, and a wrong guess could restore
    to, and overwrite, another file.
    """
    tail = f"{_DELTA_SEP}{reference_name}.ptwm"
    if compressed_name.endswith(tail) and len(compressed_name) > len(tail):
        return compressed_name.removesuffix(tail)
    legacy_tail = f"{_DELTA_LEGACY_SEP}{reference_name}.ptwm"
    if compressed_name.endswith(legacy_tail) and len(compressed_name) > len(
        legacy_tail
    ):
        return compressed_name.removesuffix(legacy_tail) + ".bin"
    seps = [m.start() for m in _DELTA_SEP_RE.finditer(compressed_name)]
    if not seps:
        return compressed_name.removesuffix(".ptwm")
    sep = seps[0]
    if len(seps) == 1 and sep > 0:
        if compressed_name.startswith(_DELTA_SEP, sep):
            return compressed_name[:sep]
        return compressed_name[:sep] + ".bin"
    msg = (
        f"cannot tell the original file name of {compressed_name!r}: the "
        f"reference {reference_name!r} is not the one it was compressed "
        "with, and the name holds more than one '.delta.' or '_delta_', so "
        "the split between input name and reference name is not known. "
        "Give the reference file its original name and run again."
    )
    raise ValueError(msg)


def check_and_install_ptwm() -> None:
    """Check if ptwm is installed, and exit with instructions otherwise."""
    try:
        import ptwm  # noqa: F401, PLC0415
    except ImportError:
        sys.stderr.write(
            f"{RED}Error: The 'ptwm' package is not installed.{RESET}\n"
            f"Please install it manually by running:\n\n"
            f"    {shlex.quote(sys.executable)} -m pip install ptwm --upgrade\n\n",
        )
        sys.exit(1)


def parse_streaming_chunk_size(streaming_chunk_size: int | str) -> int:  # noqa: PLR0911
    """Parse the streaming chunk size from a string or integer."""
    if streaming_chunk_size is None:
        return MB
    if str(streaming_chunk_size).isdigit():
        return int(streaming_chunk_size)

    size_str = str(streaming_chunk_size).lower()
    if size_str.endswith("kb"):
        return int(size_str[:-2]) * KB
    if size_str.endswith("mb"):
        return int(size_str[:-2]) * MB
    if size_str.endswith("gb"):
        return int(size_str[:-2]) * GB
    if size_str.endswith("k"):
        return int(size_str[:-1]) * KB
    if size_str.endswith("m"):
        return int(size_str[:-1]) * MB
    if size_str.endswith("g"):
        return int(size_str[:-1]) * GB

    try:
        return int(size_str)
    except ValueError:
        msg = f"Invalid size format: {streaming_chunk_size}. Use 'KB', 'MB', or 'GB'."
        raise ValueError(msg) from None


def replace_in_file(file_path: Path, old: str, new: str) -> None:
    """Replace all occurrences of `old` with `new` in a file."""
    with file_path.open("r") as file:
        file_data = file.read()

    file_data = file_data.replace(old, new)

    with file_path.open("w") as file:
        file.write(file_data)
