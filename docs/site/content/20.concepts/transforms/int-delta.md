---
title: IntDelta
description: Intra-plane integer-subtraction delta transform.
---

# IntDelta

The `IntDelta` transform computes an intra-plane, element-wise integer subtraction between consecutive values. 

## Theory

Unlike `XorDelta` which operates against a reference tensor, `IntDelta` operates purely within the single plane. Each element is replaced by its integer difference from its predecessor: `out[i] = in[i].wrapping_sub(in[i-1])`.

For smoothly-varying data (such as sorted indices or certain numerical embeddings), the adjacent values are numerically close. The resulting delta stream concentrates values near zero, which downstream entropy coders (like [FPC](../codecs/fpc) or Arithmetic) can compress much more effectively than the raw bytes. The element width determines the wrapping arithmetic boundary (e.g., 1, 2, 4, or 8 bytes).

## Inverse

The inverse operation simply computes a running wrapping addition: `in[i] = out[i].wrapping_add(in[i-1])`.

## References

* Generic Delta Encoding. [Wikipedia](https://en.wikipedia.org/wiki/Delta_encoding)
