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
    was renamed) is split at its separator instead: the last ``.delta.``,
    unless a ``_delta_`` follows it. A ``_delta_`` name keeps the pre-v1.2
    rule, the part before the first ``_delta_`` plus ``.bin``. A name with
    neither separator loses only its ``.ptwm`` suffix.
    """
    tail = f"{_DELTA_SEP}{reference_name}.ptwm"
    if compressed_name.endswith(tail) and len(compressed_name) > len(tail):
        return compressed_name.removesuffix(tail)
    legacy_tail = f"{_DELTA_LEGACY_SEP}{reference_name}.ptwm"
    if compressed_name.endswith(legacy_tail):
        return compressed_name.removesuffix(legacy_tail) + ".bin"
    sep = compressed_name.rfind(_DELTA_SEP)
    if sep > 0 and _DELTA_LEGACY_SEP not in compressed_name[sep:]:
        return compressed_name[:sep]
    if _DELTA_LEGACY_SEP in compressed_name:
        return compressed_name.split(_DELTA_LEGACY_SEP, 1)[0] + ".bin"
    return compressed_name.removesuffix(".ptwm")


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
