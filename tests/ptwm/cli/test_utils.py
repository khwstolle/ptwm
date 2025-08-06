import pytest
from ptwm.cli.utils import GB, KB, MB, parse_streaming_chunk_size


@pytest.mark.parametrize(
    ("input_val", "expected"),
    [
        (None, MB),
        (1024, 1024),
        ("2048", 2048),
        ("1kb", KB),
        ("1KB", KB),
        ("1k", KB),
        ("1K", KB),
        ("2mb", 2 * MB),
        ("2MB", 2 * MB),
        ("2m", 2 * MB),
        ("2M", 2 * MB),
        ("3gb", 3 * GB),
        ("3GB", 3 * GB),
        ("3g", 3 * GB),
        ("3G", 3 * GB),
    ],
)
def test_parse_streaming_chunk_size_valid(input_val, expected):
    """Test valid inputs for parse_streaming_chunk_size."""
    assert parse_streaming_chunk_size(input_val) == expected


@pytest.mark.parametrize(
    "input_val",
    [
        "1tb",
        "abc",
        "",
        "100 x",
    ],
)
def test_parse_streaming_chunk_size_invalid(input_val):
    """Test invalid inputs for parse_streaming_chunk_size raise ValueError."""
    with pytest.raises(ValueError, match="Invalid size format"):
        parse_streaming_chunk_size(input_val)
