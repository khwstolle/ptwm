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
