#!/usr/bin/env python
"""
Memory leak verification script for PTWM.

This script tests whether memory stays stable after multiple compress/decompress cycles.

Usage:
    python test_memory_leak.py

    # With more iterations
    python test_memory_leak.py --iterations 500

    # With larger tensors
    python test_memory_leak.py --size 2000
"""

import argparse
import gc
import sys

try:
    import psutil
except ImportError:
    print("psutil not installed. Install with: pip install psutil")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("torch not installed. Install with: pip install torch")
    sys.exit(1)

from ptwm import (
    CompressionConfig,
    Compressor,
    DecompressionConfig,
    Decompressor,
    Format,
    Method,
)

try:
    import pytest
except ImportError:
    pytest = None


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


@pytest.mark.slow if pytest else lambda x: x
@pytest.mark.integration if pytest else lambda x: x
def test_memory_leak_pytest() -> None:
    """Pytest wrapper for memory leak test with enough iterations for stability check."""
    success = run_memory_leak_test(iterations=50, tensor_size=400, print_interval=10)
    assert success


def run_memory_leak_test(iterations=100, tensor_size=1000, print_interval=20) -> bool:
    """
    Test for memory leaks by running multiple compress/decompress cycles.

    Args:
        iterations: Number of compress/decompress cycles
        tensor_size: Size of square tensor (tensor_size x tensor_size)
        print_interval: How often to print memory status

    Returns
    -------
        True if no significant memory leak detected, False otherwise
    """
    print("=" * 60)
    print("PTWM Memory Leak Test")
    print("=" * 60)
    print(f"Iterations: {iterations}")
    print(f"Tensor size: {tensor_size} x {tensor_size} (float16)")
    print(
        f"Tensor memory: {tensor_size * tensor_size * 2 / (1024**2):.2f} MB per tensor"
    )
    print("=" * 60)

    gc.collect()
    initial_memory = get_memory_mb()
    print(f"Initial memory: {initial_memory:.1f} MB")
    print()

    compressor = Compressor(
        CompressionConfig(
            method=Method.HUFFMAN,
            input_format=Format.TORCH,
        )
    )
    decompressor = Decompressor(DecompressionConfig())

    memory_readings = []

    for i in range(iterations):
        tensor = torch.randn(tensor_size, tensor_size, dtype=torch.float16)

        compressed = compressor.compress(tensor)
        decompressed = decompressor.decompress(compressed)

        if not torch.allclose(tensor, decompressed, rtol=1e-7, atol=1e-7):
            print(f"ERROR: Data integrity check failed at iteration {i}!")
            print(f"  Max difference: {(tensor - decompressed).abs().max().item()}")
            return False

        if i == 0:
            print("Data integrity check: PASSED (will verify every iteration)")
            print()

        del compressed, decompressed, tensor

        if i % print_interval == 0:
            gc.collect()
            current_memory = get_memory_mb()
            memory_readings.append(current_memory)
            memory_increase = current_memory - initial_memory
            print(
                f"Iteration {i:4d}: Memory = {current_memory:.1f} MB "
                f"(+{memory_increase:.1f} MB from start)"
            )

    gc.collect()
    final_memory = get_memory_mb()
    memory_readings.append(final_memory)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Initial memory:  {initial_memory:.1f} MB")
    print(f"Final memory:    {final_memory:.1f} MB")
    print(f"Total increase:  {final_memory - initial_memory:.1f} MB")

    if len(memory_readings) >= 4:
        mid = len(memory_readings) // 2
        first_half_avg = sum(memory_readings[:mid]) / mid
        second_half_avg = sum(memory_readings[mid:]) / (len(memory_readings) - mid)
        trend = second_half_avg - first_half_avg
        print(
            f"Memory trend:    {'+' if trend > 0 else ''}{trend:.1f} MB "
            f"(second half avg - first half avg)"
        )

    total_data_processed_mb = iterations * tensor_size * tensor_size * 2 / (1024**2)
    memory_increase = final_memory - initial_memory
    leak_ratio = (
        memory_increase / total_data_processed_mb if total_data_processed_mb > 0 else 0
    )

    print()
    print(f"Total data processed: {total_data_processed_mb:.1f} MB")
    print(f"Memory increase ratio: {leak_ratio:.4f} (increase / data processed)")
    print()

    if len(memory_readings) >= 4:
        second_half = memory_readings[len(memory_readings) // 2 :]
        memory_variance = max(second_half) - min(second_half)
        is_stable = memory_variance < 10
    else:
        is_stable = False

    if not is_stable and leak_ratio > 0.5:
        print("MEMORY LEAK DETECTED!")
        return False
    if not is_stable and leak_ratio > 0.15:
        print("WARNING: Possible memory leak detected.")
        return True
    if memory_increase > 200:
        print("WARNING: High initial memory overhead detected.")
        return True
    print("NO MEMORY LEAK DETECTED!")
    if is_stable:
        print(f"   Memory variance in second half: {memory_variance:.1f} MB (stable)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PTWM for memory leaks")
    parser.add_argument("--iterations", "-n", type=int, default=100)
    parser.add_argument("--size", "-s", type=int, default=1000)
    parser.add_argument("--interval", "-i", type=int, default=20)

    args = parser.parse_args()

    success = run_memory_leak_test(
        iterations=args.iterations, tensor_size=args.size, print_interval=args.interval
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
