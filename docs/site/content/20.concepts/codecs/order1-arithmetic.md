---
title: Order-1 Arithmetic
description: Static lag-1 arithmetic coder for exponent planes.
---

# Order-1 Arithmetic

The `Order1Arithmetic` codec is a static order-1 (lag-1) arithmetic coder designed specifically for exponent byte planes, functioning as a learned dynamic-length float coder.

## Theory

Instead of an adaptive arithmetic coder that updates its probabilities on the fly, this codec fits a static conditional table once for the entire plane and serializes it into the codec state. 

While static coding incurs a table overhead and may slightly trail adaptive coding in raw compression ratio, it guarantees that the future decoder possesses the complete, frozen probability model upfront. This upfront model is critical for enabling parallel GPU decode kernels, which cannot easily run adaptive probability updates across thousands of threads.

## Usage

This codec is strictly limited to `ExponentByte` roles. It requires a minimum plane size of 4 KiB to ensure the static table overhead can be amortized.

## References

* I. H. Witten, R. M. Neal, and J. G. Cleary, "Arithmetic Coding for Data Compression", *Communications of the ACM* (1987). [Link](https://dl.acm.org/doi/10.1145/214762.214771)
