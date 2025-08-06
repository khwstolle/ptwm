#!/usr/bin/env python
"""
Decompress throughput benchmark for PTWM.

Measures decompress throughput (MB/s) using a synthetic BF16 tensor,
simulating real model weight decompression.  Run this before and after
code changes to verify there is no performance regression.

Usage:
    python tests/test_decompress_throughput.py

    # Custom data size (in MB)
    python tests/test_decompress_throughput.py --size-mb 512

    # More iterations for stable results
    python tests/test_decompress_throughput.py --iterations 10

    # Different dtype
    python tests/test_decompress_throughput.py --dtype float32
"""

import argparse
import gc
import sys
import time

try:
    import pytest
except ImportError:
    pytest = None

import torch
from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
    Format,
    Method,
)


@pytest.mark.benchmark if pytest else lambda x: x
@pytest.mark.integration if pytest else lambda x: x
def test_decompress_throughput_pytest() -> None:
    """Pytest wrapper for throughput benchmark with smaller size for CI."""
    result = benchmark_decompress_throughput(size_mb=100, iterations=2)
    assert result is not None


def benchmark_decompress_throughput(size_mb=1024, iterations=5, dtype=torch.bfloat16):
    """
    Benchmark decompression throughput.

    Args:
        size_mb: Size of test data in MB.
        iterations: Number of decompress iterations to average.
        dtype: Tensor dtype to test.

    Returns
    -------
        dict with throughput results, or None on failure.
    """
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    num_elements = (size_mb * 1024 * 1024) // bytes_per_element

    print("=" * 70)
    print("PTWM Decompress Throughput Benchmark")
    print("=" * 70)
    print(f"Data size:    {size_mb} MB")
    print(f"Dtype:        {dtype}")
    print(f"Iterations:   {iterations}")
    print(f"Elements:     {num_elements:,}")
    print("=" * 70)

    print("\n[1/4] Generating synthetic model weights...")
    t0 = time.time()
    tensor = torch.randn(num_elements, dtype=dtype)
    data_bytes = tensor.nelement() * tensor.element_size()
    actual_size_mb = data_bytes / (1024 * 1024)
    print(f"      Generated {actual_size_mb:.1f} MB in {time.time() - t0:.2f}s")

    print("[2/4] Compressing data...")
    tensor_original = tensor.clone()
    compressor = Compressor(
        CompressionConfig(
            method=Method.HUFFMAN,
            input_format=Format.TORCH,
        )
    )
    decompressor = Decompressor(DecompressionConfig())
    t0 = time.time()
    compressed = compressor.compress(tensor)
    compress_time = time.time() - t0
    compressed_size_mb = len(compressed) / (1024 * 1024)
    compress_ratio = actual_size_mb / compressed_size_mb
    compress_throughput = actual_size_mb / compress_time
    print(
        f"      Compressed: {actual_size_mb:.1f} MB -> {compressed_size_mb:.1f} MB "
        f"(ratio: {compress_ratio:.2f}x)"
    )
    print(f"      Compress throughput: {compress_throughput:.1f} MB/s")

    print("[3/4] Warming up decompression...")
    _ = decompressor.decompress(compressed)
    del _
    gc.collect()

    print(f"[4/4] Benchmarking decompression ({iterations} iterations)...")
    decompress_times = []

    for i in range(iterations):
        gc.collect()
        t0 = time.time()
        decompressed = decompressor.decompress(compressed)
        elapsed = time.time() - t0
        decompress_times.append(elapsed)
        throughput = actual_size_mb / elapsed
        print(f"      Iteration {i + 1}: {elapsed:.3f}s  ({throughput:.1f} MB/s)")

        if i == 0:
            if torch.equal(tensor_original, decompressed):
                print("      Data integrity verified (lossless)")
            else:
                print("      DATA MISMATCH - compression is NOT lossless!")
                return None
        del decompressed

    avg_time = sum(decompress_times) / len(decompress_times)
    min_time = min(decompress_times)
    max_time = max(decompress_times)
    avg_throughput = actual_size_mb / avg_time
    peak_throughput = actual_size_mb / min_time

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Data size:              {actual_size_mb:.1f} MB ({dtype})")
    print(f"  Compression ratio:      {compress_ratio:.2f}x")
    print(f"  Compress throughput:    {compress_throughput:.1f} MB/s")
    print(f"  Decompress (avg):       {avg_time:.3f}s  ({avg_throughput:.1f} MB/s)")
    print(f"  Decompress (best):      {min_time:.3f}s  ({peak_throughput:.1f} MB/s)")
    print(
        f"  Decompress (worst):     {max_time:.3f}s  ({actual_size_mb / max_time:.1f} MB/s)"
    )
    print("=" * 70)

    return {
        "size_mb": actual_size_mb,
        "dtype": str(dtype),
        "compress_ratio": compress_ratio,
        "compress_throughput_mbs": compress_throughput,
        "avg_decompress_time_s": avg_time,
        "avg_decompress_throughput_mbs": avg_throughput,
        "peak_decompress_throughput_mbs": peak_throughput,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PTWM decompression throughput"
    )
    parser.add_argument("--size-mb", type=int, default=1024)
    parser.add_argument("--iterations", "-n", type=int, default=10)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )

    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    result = benchmark_decompress_throughput(
        size_mb=args.size_mb,
        iterations=args.iterations,
        dtype=dtype_map[args.dtype],
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
