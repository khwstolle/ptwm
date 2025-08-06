"""Tests for cross-plane joint histogram + mutual information.

These helpers measure ``I(mantissa_byte ; exponent_byte)`` to quantify the
headroom available to a conditional entropy coder on top of plane-split
preprocessing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from ptwm.entropy import (
    histogram,
    joint_histogram,
    mutual_information,
    shannon,
)


def _shannon_from_hist(hist: np.ndarray) -> float:
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0].astype(np.float64) / float(total)
    return float(-np.sum(p * np.log2(p)))


def test_self_mi_equals_marginal_entropy() -> None:
    """I(X; X) = H(X): pairing a stream with itself recovers the marginal."""
    rng = np.random.default_rng(seed=17)
    data = rng.integers(0, 256, size=100_000, dtype=np.uint8).tobytes()

    hist = histogram(data)
    h_x = _shannon_from_hist(hist)
    joint = joint_histogram(data, data)
    mi = mutual_information(joint)

    assert mi == pytest.approx(h_x, abs=1e-10)
    assert mi == pytest.approx(shannon(data), abs=1e-6)


def test_independent_streams_have_near_zero_mi() -> None:
    """Independent uniform byte streams have I ≈ 0 bits/byte (within sampling error).

    The plug-in MI estimator has an upward bias of roughly
    ``(K-1)^2 / (2 N ln 2)`` bits for alphabet size K (Miller-Madow, 1955).
    For K=256, N=10^7 that gives ~0.0047 bits — so 0.015 is a safe ceiling.
    """
    rng = np.random.default_rng(seed=31)
    a = rng.integers(0, 256, size=10_000_000, dtype=np.uint8).tobytes()
    b = rng.integers(0, 256, size=10_000_000, dtype=np.uint8).tobytes()

    joint = joint_histogram(a, b)
    mi = mutual_information(joint)

    assert 0.0 <= mi < 0.015


def test_known_three_symbol_joint() -> None:
    """Hand-computed joint distribution check.

    Build a joint with support on {(0,0), (1,1), (2,2), (0,1)} with equal
    probability 1/4 each. Then H(X) = H(Y) = 1.5 bits, H(X,Y) = 2.0 bits,
    so I(X;Y) = 1.0 bits.
    """
    a = bytes([0, 1, 2, 0])
    b = bytes([0, 1, 2, 1])

    joint = joint_histogram(a, b)
    assert joint.shape == (256, 256)
    assert joint[0, 0] == 1
    assert joint[1, 1] == 1
    assert joint[2, 2] == 1
    assert joint[0, 1] == 1
    assert joint.sum() == 4

    mi = mutual_information(joint)
    # H(X) = -[2/4 * log(2/4) + 1/4 * log(1/4) + 1/4 * log(1/4)] = 1.5
    # H(Y) = -[1/4 * log(1/4) + 2/4 * log(2/4) + 1/4 * log(1/4)] = 1.5
    # H(X,Y) = -4 * (1/4 * log(1/4)) = 2.0
    # I = 1.5 + 1.5 - 2.0 = 1.0
    assert mi == pytest.approx(1.0, abs=1e-12)


def test_empty_input_returns_zero() -> None:
    """Zero-sized input yields MI = 0 rather than NaN."""
    joint = joint_histogram(b"", b"")
    assert joint.shape == (256, 256)
    assert joint.sum() == 0
    assert mutual_information(joint) == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        joint_histogram(b"abc", b"ab")


def test_mutual_information_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected .*joint histogram"):
        mutual_information(np.zeros((10, 10), dtype=np.uint64))


def test_data_processing_inequality_bound() -> None:
    """0 <= I(X;Y) <= min(H(X), H(Y)) for any joint distribution."""
    rng = np.random.default_rng(seed=99)
    n = 50_000
    a = rng.integers(0, 16, size=n, dtype=np.uint8).tobytes()  # small alphabet
    b = bytes(x ^ rng.integers(0, 4, dtype=np.uint8) for x in a)  # correlated

    joint = joint_histogram(a, b)
    mi = mutual_information(joint)
    h_a = _shannon_from_hist(joint.sum(axis=1))
    h_b = _shannon_from_hist(joint.sum(axis=0))

    assert mi >= -1e-10  # numerical zero floor
    assert mi <= min(h_a, h_b) + 1e-10
    assert not math.isnan(mi)
