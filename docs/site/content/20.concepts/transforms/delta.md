---
title: Delta
description: Cross-tensor residual encoding against a reference plane.
---

# Delta Transforms

The `XorDelta` and `FloatDelta` transforms perform cross-tensor residual encoding. They subtract a reference plane (from a dependency tensor) from the current plane to produce a highly compressible residual.

## XorDelta

`XorDelta` computes a byte-level XOR residual (`residual[i] = current[i] ^ reference[i]`). 

Because XOR is involutive, the forward and inverse operations are identical. `XorDelta` is guaranteed to be byte-exact and lossless for all inputs.

## FloatDelta

> [!WARNING]
> `FloatDelta` is marked as **DONE_WITH_CONCERNS** because it violates PTWM's byte-exact round-trip invariant.

`FloatDelta` performs an element-wise float subtraction (`current - reference`). 

IEEE 754 floating-point arithmetic does not guarantee that `(a - b) + b == a` bit-exactly when `a - b` introduces a rounding error. Therefore, `FloatDelta` may be slightly lossy (by ~1 ULP). It exists for opportunistic delta encoding where near-identical tensors yield a residual so small that the coding gain outweighs the slight approximation.

## References

* Generic Delta Encoding. [Wikipedia](https://en.wikipedia.org/wiki/Delta_encoding)
