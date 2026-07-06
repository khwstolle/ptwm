---
title: Order-1 Scale AC
description: Per-row lag-1 conditional arithmetic coder for MXFP4/NVFP4 scale planes.
---

# Order-1 Scale AC

The `Order1ScaleAC` codec is a specialized arithmetic coder designed specifically for the `SCALE` planes of MXFP4 / NVFP4 microscaled tensors.

## Theory

While standard block scales are pure 8-bit exponents, FP8 scale formats (like E4M3 or E5M2) contain mantissa bits. These bits produce a continuous distribution with strong autocorrelation along the row. 

`Order1ScaleAC` exploits this by modeling a per-row lag-1 Probability Mass Function (PMF) over the scale bytes. It fits a conditional table during compression, serializes it into the codec state, and uses a range coder to compress the sequence based on the previous byte's context.

To minimize trial-encode overhead, the codec uses a fast marginal-entropy pre-screen. If the data is near-uniform (entropy ≥ 7.5 bits/byte), the codec aborts early, as row conditioning cannot recover the overhead of storing the per-row PMF table.

## Usage

This codec strictly requires `Byte`-width `Scale` or `GlobalScale` planes with a `Rows` layout and an FP8 scale format (E4M3 or E5M2).

## References

* This codec is an original adaptation of Order-1 Arithmetic Coding tailored for the scale distributions in MXFP4 and NVFP4 formats.
